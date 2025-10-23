"""
Stock Analysis Agents
주식 분석 에이전트 - 뉴스 중심 전략

전략 변경:
- SEC 공시는 부가적 소스로 활용 (접근 제약)
- Bloomberg, Reuters, WSJ, FT 등 공신력 있는 뉴스를 primary로 사용
- 여러 소스를 종합하여 신뢰도 높은 분석 제공
- 에이전트별 Evaluator 패턴으로 할루시네이션 방지
"""

from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv(usecwd=True), override=True)

import os
import json
import re
import requests
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from bs4 import BeautifulSoup
from urllib.parse import urlparse
from datetime import date, timedelta, datetime
from tavily import TavilyClient
from openai import OpenAI
import FinanceDataReader as fdr
from pdfminer.high_level import extract_text as pdf_extract_text
from typing import List, Dict, Any, Optional, Tuple
import time

# ============================================================================
# CONFIGURATION
# ============================================================================

AGENT_NAMES = {
    "FILING": "Business & News Analysis (뉴스 중심 분석)",
    "FUNDAMENTAL": "Fundamental (Finviz 재무/밸류 분석)",
    "TECHNICAL": "Technical (FDR 차트 해석)"
}

def _clean(s: str) -> str:
    return (s or "").strip().strip('"').strip("'").replace("\uFEFF", "").replace("\r", "").replace("\n", "")

# API 초기화
API_KEY = _clean(os.getenv("OPENAI_API_KEY", ""))
if not API_KEY:
    raise RuntimeError("OPENAI_API_KEY가 비어있습니다. .env를 확인하세요.")

OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
OPENAI_PROJECT = _clean(os.getenv("OPENAI_PROJECT", "")) or None
OPENAI = OpenAI(api_key=API_KEY, project=OPENAI_PROJECT)

TAVILY = TavilyClient(api_key=os.environ["TAVILY_API_KEY"])
SEC_UA = os.environ.get("SEC_USER_AGENT", "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36")

FINVIZ_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36",
    "Referer": "https://finviz.com/"
}

# 신뢰할 수 있는 도메인 (우선순위 높음)
PREMIUM_SOURCES = {
    "bloomberg.com": 10,
    "reuters.com": 10,
    "wsj.com": 9,
    "ft.com": 9,
    "cnbc.com": 8,
    "finance.yahoo.com": 7,
    "marketwatch.com": 7,
    "seekingalpha.com": 6,
    "barrons.com": 8,
    "fool.com": 5
}

# 차단할 도메인
BLOCKED = {
    "dictionary.com",
    "investopedia.com",
    "wikipedia.org",
    "reddit.com",
    "quora.com",
    "shiftcomm.com"
}

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def _domain(url: str) -> str:
    """URL에서 도메인 추출"""
    try:
        return urlparse(url).netloc.lower()
    except:
        return ""

def _get_source_priority(url: str) -> int:
    """소스 우선순위 점수 계산"""
    domain = _domain(url)
    for premium_domain, score in PREMIUM_SOURCES.items():
        if premium_domain in domain:
            return score
    return 1  # 기본 점수

def _ensure_source(state: dict, new_sources: list):
    """중복 없이 소스 추가"""
    known = {s["url"] for s in state["sources"]}
    for s in new_sources:
        if s.get("url") and s["url"] not in known:
            state["sources"].append(s)

def _save_price_chart(df: pd.DataFrame, ticker: str, outdir="artifacts") -> str:
    """주가 차트 저장"""
    os.makedirs(outdir, exist_ok=True)
    path = os.path.join(outdir, f"{ticker}_daily.png")
    plt.figure(figsize=(9, 4))
    close_col = "Close" if "Close" in df.columns else "close"
    plt.plot(df.index, df[close_col], color='blue', label='Close')
    
    # SMA 계산 및 플롯
    try:
        sma20 = df[close_col].rolling(window=20).mean()
        sma60 = df[close_col].rolling(window=60).mean()
        plt.plot(df.index, sma20, color='orange', linestyle='--', linewidth=1.5, label='SMA20', alpha=0.8)
        plt.plot(df.index, sma60, color='green', linestyle='--', linewidth=1.5, label='SMA60', alpha=0.8)
    except Exception as e:
        print(f"    ⚠️  SMA calculation failed: {e}")
    plt.title(f"{ticker} Daily Close")
    plt.xlabel("Date")
    plt.ylabel("Close Price")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path)
    plt.close()
    return path

def _binary_eval(text: str, needed_topics: list, sources: list) -> dict:
    """분석 품질 평가"""
    covered = sum(1 for m in needed_topics if m in text)
    coverage_ok = (len(needed_topics) == 0) or (covered / len(needed_topics) >= 0.8)
    grounded_ok = len(sources) >= 1
    contradiction_ok = ("모순" not in text and "오류" not in text)
    safety_style_ok = (len(text) > 300 and "주민등록번호" not in text)
    domains = {s.get("domain") for s in sources}
    sources_ok = (len(sources) >= 3 and len(domains) >= 2)
    
    return {
        "sufficient": coverage_ok and grounded_ok and contradiction_ok and safety_style_ok and sources_ok,
        "coverage_ok": coverage_ok,
        "grounded_ok": grounded_ok,
        "contradiction_ok": contradiction_ok,
        "safety_style_ok": safety_style_ok,
        "sources_ok": sources_ok,
        "missing_points": [m for m in needed_topics if m not in text],
        "fixes": []
    }

# ============================================================================
# DATE FILTERING EVALUATORS (NEW)
# ============================================================================

def extract_year_from_date(date_str: str) -> int | None:
    """날짜 문자열에서 연도 추출"""
    if not date_str:
        return None
    
    # ISO 형식: 2024-10-23
    if '-' in date_str and len(date_str) >= 10:
        try:
            return int(date_str[:4])
        except:
            pass
    
    # 다른 형식들
    patterns = [
        r'(\d{4})-\d{2}-\d{2}',  # 2024-10-23
        r'(\d{4})/\d{2}/\d{2}',  # 2024/10/23
        r'\b(202[0-9])\b',       # 2020~2029
    ]
    
    for pattern in patterns:
        match = re.search(pattern, date_str)
        if match:
            try:
                return int(match.group(1))
            except:
                pass
    
    return None


def extract_year_from_text(text: str) -> int | None:
    """URL이나 제목에서 연도 추출"""
    # URL 패턴: .../2024/10/article-name
    patterns = [
        r'/(202[4-5])/',
        r'-(202[4-5])-',
        r'\b(202[4-5])\b'
    ]
    
    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            try:
                return int(match.group(1))
            except:
                pass
    
    return None


def evaluate_source_freshness(sources: list[dict], cutoff_year: int = 2024) -> tuple[int, list[str]]:
    """
    소스의 최신성 평가 - 2024년 이후 기사만 허용
    
    Returns:
        (score, old_sources): score는 0(재시도) 또는 1(통과), old_sources는 오래된 소스 리스트
    """
    old_sources = []
    
    for src in sources:
        date_str = src.get("date", "")
        url = src.get("url", "")
        title = src.get("title", "")
        
        # 1. 날짜 문자열 파싱
        year = extract_year_from_date(date_str)
        
        # 2. URL/제목에서도 연도 추출 시도
        if not year or year < cutoff_year:
            year_from_text = extract_year_from_text(f"{url} {title}")
            if year_from_text:
                year = year_from_text
        
        # 3. 여전히 오래되었으면 기록
        if year and year < cutoff_year:
            old_sources.append({
                "title": title[:60],
                "date": date_str,
                "year": year,
                "url": url
            })
    
    if old_sources:
        return 0, old_sources
    
    return 1, []

# ============================================================================
# CONTENT HALLUCINATION EVALUATORS (NEW)
# ============================================================================

def evaluate_filing_content(
    business_text: str, 
    product_info: dict,
    openai_client,
    ticker: str
) -> tuple[int, list[str]]:
    """
    FILING 에이전트 출력 평가 - 할루시네이션 검증
    
    Returns:
        (score, issues): score는 0(재시도) 또는 1(통과), issues는 문제점 리스트
    """
    issues = []
    
    sys = f"""You are a fact-checker for investment reports. Analyze the following text for:

1. **Tense Errors**: Check if past/present/future tenses are used correctly
   - Events before 2024: past tense
   - Current events (2024-2025): present tense
   - Future events (2026+): future tense
   - Be lenient with financial reporting (e.g., "Q4 2024 reported..." is acceptable)

2. **Suspicious Model Names**: Flag model names that seem CLEARLY FABRICATED
   - ONLY flag: obviously fake names like "Stealth Edition", "Ultra Plus", "Performance Max", made-up suffixes
   - IGNORE: Korean/translated versions of real models (e.g., "머스탱 마하-E" = valid translation of "Mustang Mach-E")
   - IGNORE: Official model names even if unusual (e.g., "Mach-E", "Lightning", "Maverick" are real Ford models)
   - Compare with known {ticker} product lineup, but be conservative

3. **Invalid Promotions**: Check promotion validity
   - Must have specific dates/periods showing it's active in 2024-2025
   - Must have concrete benefits (discount amounts, not vague statements)
   - Flag "0% financing" without context as ambiguous
   - Expired promotions (before 2024) should be flagged

4. **Duplicate Information**: Check for repeated information with slight variations

Return JSON:
{{
  "tense_errors": [list of specific examples with clear errors ONLY],
  "fake_models": [list of CLEARLY FABRICATED model names ONLY - exclude translations],
  "invalid_promos": [list of invalid promotions],
  "duplicates": [list of duplicate entries],
  "score": 0 or 1
}}

Score should be 1 if there are NO MAJOR issues. Minor issues (translations, acceptable tense variations) should not affect score.
Be CONSERVATIVE - only flag clear, obvious problems.
"""
    
    content = f"""Business Text:
{business_text[:3000]}

Product Info:
{json.dumps(product_info, ensure_ascii=False)[:2000]}
"""
    
    try:
        resp = openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": sys},
                {"role": "user", "content": content}
            ],
            response_format={"type": "json_object"},
            temperature=0.0
        )
        
        result = json.loads(resp.choices[0].message.content)
        
        if result.get("tense_errors"):
            issues.append(f"❌ Tense errors: {len(result['tense_errors'])} found")
            for err in result["tense_errors"][:2]:
                issues.append(f"  → {err}")
        
        if result.get("fake_models"):
            issues.append(f"❌ Suspicious models: {len(result['fake_models'])} found")
            for model in result["fake_models"][:2]:
                issues.append(f"  → {model}")
        
        if result.get("invalid_promos"):
            issues.append(f"❌ Invalid promotions: {len(result['invalid_promos'])} found")
            for promo in result["invalid_promos"][:2]:
                issues.append(f"  → {promo}")
        
        if result.get("duplicates"):
            issues.append(f"⚠️  Duplicate info: {len(result['duplicates'])} found")
        
        score = result.get("score", 0)
        
        return score, issues
        
    except Exception as e:
        print(f"⚠️ Content evaluation failed: {e}")
        return 1, []  # 평가 실패 시 일단 통과


# ============================================================================
# ROUTING & PLANNING
# ============================================================================

def rewrite_query_for_routing(query: str) -> str:
    """쿼리를 라우팅 친화적으로 리라이팅"""
    try:
        resp = OPENAI.chat.completions.create(
            model=OPENAI_MODEL,
            temperature=0.1,
            messages=[
                {"role": "system", "content":
                 "Rewrite the user's request into a compact, routing-friendly English query. "
                 "Include company/ticker if present, and a hint: value or technical. "
                 "Output one line only."},
                {"role": "user", "content": query}
            ]
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        print(f"⚠️  Query rewrite failed: {e}")
        return query

def _lock_entity(query_or_rewritten: str) -> dict:
    """엔티티(회사) 정보 추출"""
    sys = "Extract company entity (name,ticker,exchange) from the text. JSON {entity:{name,ticker,exchange}}"
    try:
        resp = OPENAI.chat.completions.create(
            model=OPENAI_MODEL,
            temperature=0.0,
            messages=[{"role": "system", "content": sys}, {"role": "user", "content": query_or_rewritten}],
            response_format={"type": "json_object"}
        )
        ent = json.loads(resp.choices[0].message.content).get("entity", {})
    except:
        ent = {
            "name": query_or_rewritten.strip(),
            "ticker": query_or_rewritten.strip().upper(),
            "exchange": "UNKNOWN"
        }
    ent["confidence"] = 0.9
    return ent

def route_with_llm(query: str, rewritten_query: str | None = None, context: dict | None = None) -> dict:
    """LLM 기반 라우팅 결정"""
    ctx = context or {}
    forced_persp = ctx.get("forced_perspective")
    forced_ticker = ctx.get("forced_ticker")

    ent = _lock_entity(rewritten_query or query)
    if forced_ticker:
        ent["ticker"] = forced_ticker
        ent["confidence"] = 0.99

    system = (
        "Determine the investment perspective (value or tech) for the user's query and choose which agents to run.\n"
        "\n"
        "ROUTING RULES:\n"
        "1. ALWAYS include 'FILING' (news & business analysis)\n"
        "2. If query mentions 'value', 'fundamental', '가치', '재무', '밸류에이션':\n"
        "   → Add 'FUNDAMENTAL' (financial analysis)\n"
        "3. If query mentions 'tech', 'technical', '기술적', '차트', '시세':\n"
        "   → Add 'TECHNICAL' (chart analysis)\n"
        "4. If unsure, default to perspective='value' and route=['FILING', 'FUNDAMENTAL']\n"
        "\n"
        "Return JSON: {\n"
        "  'perspective': 'value' or 'tech',\n"
        "  'route': ['FILING', 'FUNDAMENTAL'] or ['FILING', 'TECHNICAL'],\n"
        "  'must_cover': [list of topics],\n"
        "  'search_seeds': [list],\n"
        "  'entity': {name, ticker, exchange}\n"
        "}"
    )
    user = json.dumps(
        {"query": query, "rewritten": rewritten_query, "entity_hint": ent, "context": ctx},
        ensure_ascii=False,
    )
    
    try:
        resp = OPENAI.chat.completions.create(
            model=OPENAI_MODEL,
            temperature=0.1,
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
            response_format={"type": "json_object"},
        )
        obj = json.loads(resp.choices[0].message.content)
    except Exception as e:
        print(f"⚠️  Routing failed: {e}")
        obj = {
            "perspective": "value",
            "route": ["FILING", "FUNDAMENTAL"],
            "must_cover": [
                "business structure & catalysts",
                "revenue comparison",
                "risk factors",
                "investment recommendation",
            ],
            "search_seeds": [],
            "entity": ent,
        }

    # 강제 설정 적용
    if forced_persp in ("value", "tech"):
        obj["perspective"] = forced_persp
        obj["route"] = ["FILING", "FUNDAMENTAL"] if forced_persp == "value" else ["FILING", "TECHNICAL"]

    obj["entity"] = obj.get("entity") or ent

    # route 정규화
    route_list = obj.get("route", ["FILING", "FUNDAMENTAL"])
    if isinstance(route_list, str):
        route_list = [route_list]
    obj["route"] = route_list

    # must_cover 정규화
    must_cover = obj.get("must_cover", [])
    if isinstance(must_cover, str):
        must_cover = [must_cover]
    elif not isinstance(must_cover, list):
        must_cover = list(must_cover)
    obj["must_cover"] = must_cover + [
        "business structure & catalysts",
        "revenue comparison",
        "risk factors",
        "investment recommendation",
    ]

    # search_seeds 정규화
    search_seeds = obj.get("search_seeds", [])
    if isinstance(search_seeds, str):
        search_seeds = [search_seeds]
    elif not isinstance(search_seeds, list):
        search_seeds = list(search_seeds)
    obj["search_seeds"] = search_seeds

    return obj

def decompose_tasks_with_llm(route_obj: dict) -> dict:
    """작업 분해 계획 생성"""
    sys = (
        "너는 리서치 플래너다. 라우팅 정보(FILING/FUNDAMENTAL/TECHNICAL)를 바탕으로 "
        "최소 단계의 작업 분해를 4~8개 제안하라. 각 단계는 agent 코드('FILING','FUNDAMENTAL','TECHNICAL')와 한 문장 desc. "
        "마지막에 완료 기준 3~5개. JSON만."
    )
    user = json.dumps({
        "entity": route_obj.get("entity"),
        "perspective": route_obj.get("perspective"),
        "route": route_obj.get("route"),
        "must_cover": route_obj.get("must_cover")
    }, ensure_ascii=False)
    
    try:
        resp = OPENAI.chat.completions.create(
            model=OPENAI_MODEL,
            temperature=0.1,
            messages=[{"role": "system", "content": sys}, {"role": "user", "content": user}],
            response_format={"type": "json_object"}
        )
        return json.loads(resp.choices[0].message.content)
    except Exception as e:
        print(f"⚠️  Task decomposition failed: {e}")
        route = route_obj.get("route", ["FILING", "FUNDAMENTAL"])
        steps = [
            {"id": "S1", "agent": "FILING", "desc": "뉴스 및 공시에서 사업 동향과 매출 비교 추출"},
            {"id": "S2", "agent": "FUNDAMENTAL", "desc": "Finviz에서 재무/밸류에이션 지표 수집 및 요약"} if "FUNDAMENTAL" in route else None,
            {"id": "S3", "agent": "TECHNICAL", "desc": "FinanceDataReader로 주가 데이터 분석 및 차트 생성"} if "TECHNICAL" in route else None,
            {"id": "S4", "agent": "REPORT", "desc": "섹션 합성 및 투자 추천(Buy/Hold/Sell) 생성"}
        ]
        steps = [s for s in steps if s is not None]
        acceptance = [
            "사업 전개/동향 서술 포함",
            "분기 매출 비교 표 포함",
            "출처 3개 이상 및 도메인 다양성",
            "모순/오류 없음",
            "투자 추천(Buy/Hold/Sell) 포함"
        ]
        return {"steps": steps, "acceptance": acceptance}

# ============================================================================
# NEWS & DATA COLLECTION (뉴스 중심 전략)
# ============================================================================

def fetch_comprehensive_news(ticker: str, max_results: int = 15, debug: bool = True) -> list[dict]:
    """
    종합 뉴스 수집 - 다양한 쿼리로 풍부한 정보 확보
    
    전략:
    1. 실적/재무 뉴스
    2. 애널리스트 리포트
    3. 최근 동향
    4. 경쟁사/산업 동향
    """
    if debug:
        print(f"[NEWS] 📰 Comprehensive news search for {ticker}...")
    
    queries = [
        # 실적 & 재무
        f"{ticker} latest earnings report revenue profit 2024 2025",
        f"{ticker} quarterly results financial performance 2024",
        
        # 제품 & 모델 (연도 명시!)
        f"{ticker} new models launch vehicles cars 2024 2025",
        f"{ticker} electric vehicle EV battery hybrid models 2024",
        f"{ticker} vehicle lineup product portfolio 2024",
        
        # 애널리스트 & 등급
        f"{ticker} analyst rating price target upgrade downgrade 2024",
        f"{ticker} stock recommendation buy sell hold 2024",
        
        # 최근 동향 & 전략
        f"{ticker} latest news updates announcement 2024 2025",
        f"{ticker} business development strategic initiative expansion 2024",
        f"{ticker} technology innovation autonomous software 2024",
        
        # 산업 & 경쟁
        f"{ticker} industry trends market share competition 2024",
    ]
    
    all_articles = []
    seen_urls = set()
    successful_queries = 0
    
    for i, q in enumerate(queries, 1):
        try:
            if debug:
                print(f"[NEWS]   Query {i}/{len(queries)}: {q[:50]}...")
            
            # Tavily API 속도 제한 회피
            if i > 1:
                time.sleep(1.5)
            
            results = TAVILY.search(
                query=q,
                max_results=max(2, max_results // len(queries)),
                search_depth="basic"
            )
            
            successful_queries += 1
            
            for it in results.get("results", []):
                url = it.get("url", "")
                title = it.get("title", "")
                
                if not url or not title or url in seen_urls:
                    continue
                
                domain = _domain(url)
                
                # 차단 도메인 필터링
                if domain in BLOCKED:
                    continue
                
                # 우선순위 점수 계산
                priority = _get_source_priority(url)
                
                article = {
                    "url": url,
                    "domain": domain,
                    "title": title,
                    "content": it.get("content", ""),
                    "type": "news",
                    "date": it.get("published_date") or str(date.today()),
                    "priority": priority,
                    "score": it.get("score", 0.0)
                }
                
                all_articles.append(article)
                seen_urls.add(url)
                
            # 조기 종료
            if len(all_articles) >= max_results:
                if debug:
                    print(f"[NEWS]   ✅ Sufficient sources, stopping early")
                break
                
        except Exception as e:
            error_msg = str(e)
            if "excessive requests" in error_msg.lower() or "rate limit" in error_msg.lower():
                if debug:
                    print(f"[NEWS]   ⚠️  Rate limit hit, waiting 5s...")
                time.sleep(5)
                try:
                    time.sleep(2)
                    results = TAVILY.search(query=q, max_results=2, search_depth="basic")
                    successful_queries += 1
                    for it in results.get("results", []):
                        url = it.get("url", "")
                        if url and url not in seen_urls and _domain(url) not in BLOCKED:
                            all_articles.append({
                                "url": url,
                                "domain": _domain(url),
                                "title": it.get("title", ""),
                                "content": it.get("content", ""),
                                "type": "news",
                                "date": it.get("published_date") or str(date.today()),
                                "priority": _get_source_priority(url),
                                "score": it.get("score", 0.0)
                            })
                            seen_urls.add(url)
                except:
                    pass
            else:
                if debug:
                    print(f"[NEWS]   ⚠️  Query failed: {str(e)[:60]}")
            continue
    
    # 날짜 추출 및 파싱
    def parse_date(date_str):
        """날짜 문자열을 파싱하여 정렬 키 반환 (최신순)"""
        try:
            # ISO 형식 (2025-10-23)
            if '-' in date_str and len(date_str) >= 10:
                return date_str[:10]
            # 기타 형식은 오늘 날짜 반환
            return str(datetime.now().date())
        except:
            return "2000-01-01"  # 파싱 실패 시 오래된 날짜
    
    # 각 기사에 정렬용 날짜 추가
    for article in all_articles:
        article["sort_date"] = parse_date(article.get("date", ""))
    
    # 우선순위 + 날짜 + 점수로 정렬 (날짜가 최신일수록 우선)
    all_articles.sort(
        key=lambda x: (x["priority"], x["sort_date"], x["score"]), 
        reverse=True
    )
    
    # 상위 결과 선택
    top_articles = all_articles[:max_results]
    
    if debug:
        print(f"[NEWS] ✅ Retrieved {len(top_articles)} articles from {len(set(a['domain'] for a in top_articles))} sources")
        print(f"[NEWS]   Successful queries: {successful_queries}/{len(queries)}")
        if top_articles:
            print(f"[NEWS]   Top sources: {', '.join(list(set(a['domain'] for a in top_articles[:5]))[:3])}")
    
    return top_articles

def try_fetch_sec_filings(ticker: str, limit: int = 3) -> list[dict]:
    """
    SEC 공시 시도 (부가적 소스)
    
    실패해도 괜찮음 - 뉴스로 충분히 커버 가능
    """
    print(f"[SEC] 📋 Attempting SEC filings for {ticker} (optional)...")
    
    try:
        queries = [
            f"{ticker} 10-K annual report site:sec.gov",
            f"{ticker} 10-Q quarterly filing site:sec.gov",
        ]
        
        filings = []
        seen_urls = set()
        
        for q in queries:
            if len(filings) >= limit:
                break
                
            try:
                results = TAVILY.search(
                    query=q,
                    max_results=limit,
                    search_depth="basic"
                )
                
                for it in results.get("results", []):
                    url = it.get("url", "")
                    if "sec.gov" in url and url not in seen_urls:
                        filings.append({
                            "title": it.get("title", "SEC Filing"),
                            "url": url,
                            "date": it.get("published_date", ""),
                            "type": "filing",
                            "priority": 8  # SEC는 높은 우선순위
                        })
                        seen_urls.add(url)
            except:
                continue
        
        if filings:
            print(f"[SEC] ✅ Retrieved {len(filings)} SEC filings")
        else:
            print(f"[SEC] ℹ️  No SEC filings found (continuing with news)")
        
        return filings[:limit]
        
    except Exception as e:
        print(f"[SEC] ℹ️  SEC access unavailable: {str(e)[:60]}")
        print(f"[SEC] ℹ️  Relying on news sources instead")
        return []

def _fetch_text_with_retry(url: str, timeout: int = 30, max_retries: int = 2) -> str:
    """텍스트 fetch with retry"""
    for attempt in range(max_retries):
        try:
            headers = {"User-Agent": SEC_UA}
            r = requests.get(url, headers=headers, timeout=timeout)
            r.raise_for_status()
            return r.text[:50000]  # 50K로 제한
        except Exception as e:
            if attempt < max_retries - 1:
                time.sleep(1)
            else:
                return ""
    return ""

def extract_revenue_from_text(text: str, ticker: str, openai_client) -> dict:
    """텍스트에서 매출 정보 추출"""
    if not text:
        return {}
    
    sys = (
        f"Extract latest quarter revenue for {ticker}. "
        "CRITICAL: Return revenue as STRING with unit (B for billions, M for millions, K for thousands). "
        "Examples: '48.3B' (not 48.3), '500M' (not 500), '1.2B' (not 1.2). "
        "Output JSON: {"
        "  'quarter_label': str,  // e.g., 'Q4 2024'"
        "  'last_quarter': str,   // e.g., '48.3B'"
        "  'prev_quarter': str,   // e.g., '46.2B' (previous quarter)"
        "  'yoy': str,            // e.g., '44.1B' (year-over-year comparison)"
        "  'yoy_quarter_label': str  // e.g., 'Q4 2023'"
        "}. "
        "If revenue is in billions, use 'B'. If millions, use 'M'. "
        "Return empty dict if no data found."
    )
    user = json.dumps({"snippet": text[:8000]}, ensure_ascii=False)
    
    try:
        resp = openai_client.chat.completions.create(
            model=OPENAI_MODEL,
            temperature=0.0,
            messages=[{"role": "system", "content": sys}, {"role": "user", "content": user}],
            response_format={"type": "json_object"}
        )
        result = json.loads(resp.choices[0].message.content)
        
        # 단위가 있는 문자열을 숫자로 변환하는 헬퍼
        def parse_revenue(rev_str):
            if not rev_str or not isinstance(rev_str, str):
                return None
            try:
                rev_str = rev_str.strip().upper()
                if 'B' in rev_str:
                    return float(rev_str.replace('B', '').strip()) * 1e9
                elif 'M' in rev_str:
                    return float(rev_str.replace('M', '').strip()) * 1e6
                elif 'K' in rev_str:
                    return float(rev_str.replace('K', '').strip()) * 1e3
                else:
                    # 단위 없으면 그대로 float 시도
                    return float(rev_str)
            except:
                return None
        
        # 문자열을 숫자로 변환 (기존 코드 호환성)
        if result:
            result['last_quarter'] = parse_revenue(result.get('last_quarter'))
            result['prev_quarter'] = parse_revenue(result.get('prev_quarter'))
            result['yoy'] = parse_revenue(result.get('yoy'))
        
        return result
    except Exception as e:
        return {}

def extract_key_developments(text: str, title: str, openai_client) -> list[str]:
    """주요 사업 개발사항 추출 (상세 버전)"""
    if not text:
        return []
    
    sys = (
        "Extract 5-8 key business developments with SPECIFIC DETAILS. "
        "Focus on:\n"
        "1. New products/models (names, specs, launch dates)\n"
        "2. Strategic initiatives (partnerships, investments, expansions)\n"
        "3. Financial performance (revenue, profit, guidance)\n"
        "4. Market developments (competition, market share, trends)\n"
        "5. Technology updates (EV, autonomous, software)\n"
        "6. Production/capacity changes (factories, output)\n"
        "7. Executive/organizational changes\n"
        "8. Regulatory/policy impacts\n\n"
        "Output JSON: {'developments': [str]}. "
        "Each should be a DETAILED sentence in Korean with specific names, numbers, and dates when available."
    )
    user = json.dumps({"title": title, "content": text[:12000]}, ensure_ascii=False)
    
    try:
        resp = openai_client.chat.completions.create(
            model=OPENAI_MODEL,
            temperature=0.0,
            messages=[{"role": "system", "content": sys}, {"role": "user", "content": user}],
            response_format={"type": "json_object"}
        )
        return json.loads(resp.choices[0].message.content).get("developments", [])
    except Exception as e:
        return []


def extract_product_portfolio(text: str, title: str, ticker: str, openai_client) -> dict:
    """제품 포트폴리오 및 신제품 정보 추출 (자동차 산업 특화)"""
    if not text:
        return {}
    
    sys = (
        f"Extract {ticker}'s product portfolio and new model information.\n\n"
        "CRITICAL - USE CORRECT TENSE:\n"
        "- Past events (2023 or earlier): past tense (e.g., 'launched in 2023')\n"
        "- Current events (2024-2025): present tense (e.g., 'is launching', 'offers')\n"
        "- Future events (2026+): future tense (e.g., 'will launch')\n"
        "- Promotions: ONLY include if CURRENTLY ACTIVE in 2024-2025. Exclude expired promotions.\n\n"
        "Focus on:\n"
        "1. Current popular models (names, segments, features)\n"
        "2. Upcoming new models (names, launch dates, specs)\n"
        "3. EV/Hybrid models (range, charging, battery)\n"
        "4. Technology features (autonomous, connectivity, software)\n"
        "5. Promotions/pricing updates (ONLY if currently active with specific benefits)\n\n"
        "CRITICAL for Promotions:\n"
        "- ONLY include if:\n"
        "  * Has specific dates showing it's active in 2024-2025\n"
        "  * Has concrete benefits (discount amount, free item, etc.)\n"
        "  * NOT vague statements like '0% financing available'\n"
        "- If no clearly active promotions found, return empty list\n\n"
        "Output JSON:\n"
        "{\n"
        "  'current_models': [{'name': str, 'segment': str, 'details': str}],\n"
        "  'upcoming_models': [{'name': str, 'launch_date': str, 'details': str}],\n"
        "  'ev_updates': [str],\n"
        "  'technology': [str],\n"
        "  'promotions': [str]\n"
        "}\n\n"
        "Use Korean. Include specific names and numbers. Be conservative - when in doubt, exclude questionable information."
    )
    user = json.dumps({"title": title, "content": text[:12000]}, ensure_ascii=False)
    
    try:
        resp = openai_client.chat.completions.create(
            model=OPENAI_MODEL,
            temperature=0.0,
            messages=[{"role": "system", "content": sys}, {"role": "user", "content": user}],
            response_format={"type": "json_object"}
        )
        return json.loads(resp.choices[0].message.content)
    except Exception as e:
        return {}

# ============================================================================
# FILING BUSINESS AGENT (뉴스 중심 리팩토링 + EVALUATOR)
# ============================================================================


def _merge_duplicate_models(models: list) -> list:
    """중복된 모델명을 병합하여 하나로 합침"""
    if not models:
        return []
    
    merged = {}
    for model in models:
        # 모델명 추출 (괄호 앞까지)
        if isinstance(model, dict):
            name = model.get("name", "")
            details = model.get("details", "")
        else:
            # 문자열인 경우
            name = model.split("(")[0].strip() if "(" in model else model.split(":")[0].strip()
            details = model
        
        # 정규화 (공백, 대소문자)
        normalized_name = name.lower().strip()
        
        if normalized_name in merged:
            # 이미 있으면 상세 정보 병합
            if isinstance(model, dict):
                merged[normalized_name]["details"] += f" {details}"
            else:
                merged[normalized_name] += f" {model.split(':', 1)[-1].strip() if ':' in model else ''}"
        else:
            merged[normalized_name] = model if isinstance(model, dict) else model
    
    return list(merged.values())

def filing_business_agent(
    state: dict, 
    progress=lambda *_: None, 
    max_retries: int = 2
) -> dict:
    """
    Filing & Business Agent (뉴스 중심 + Evaluator 패턴)
    
    전략:
    1. 고품질 뉴스 소스에서 풍부한 정보 수집
    2. 날짜 필터링으로 최신 기사만 선택 (2024년 이후)
    3. 할루시네이션 검증 (시제, 모델명, 프로모션)
    4. SEC 공시는 보조적으로 활용 (실패해도 OK)
    """
    e = state["entity"]
    ticker = e["ticker"]
    openai_client = state["openai_client"]
    
    retry_count = 0
    
    while retry_count <= max_retries:
        progress(f"  └─ [FILING] 📰 News collection (attempt {retry_count + 1}/{max_retries + 1})...")
        
        # Step 1: 종합 뉴스 수집 (PRIMARY)
        news_sources = fetch_comprehensive_news(ticker, max_results=15, debug=True)
        
        # Step 2: 날짜 필터링 평가 (2024년 이후만)
        date_score, old_sources = evaluate_source_freshness(news_sources, cutoff_year=2024)
        
        if date_score == 0 and old_sources:
            print(f"    ⚠️  Found {len(old_sources)} old articles:")
            for src in old_sources[:3]:
                print(f"      🚫 {src['year']} - {src['title']}")
            
            # 오래된 소스 제거
            news_sources = [
                src for src in news_sources 
                if extract_year_from_date(src.get("date", "")) >= 2024
                or extract_year_from_text(f"{src.get('url', '')} {src.get('title', '')}") >= 2024
            ]
            
            print(f"      → Filtered to {len(news_sources)} recent articles")
        
        # Step 3: 소스가 충분한지 확인
        if len(news_sources) < 5:
            if retry_count < max_retries:
                print(f"    ⚠️  Insufficient recent sources ({len(news_sources)}), retrying...")
                retry_count += 1
                time.sleep(2)
                continue
            else:
                print(f"    ⚠️  Proceeding with {len(news_sources)} sources (max retries reached)")
        
        _ensure_source(state, news_sources)
        print(f"    → Collected {len(news_sources)} high-quality news articles")

        # Step 4: SEC 공시 시도 (OPTIONAL)
        progress("  └─ [FILING] 📋 SEC filings (optional)...")
        sec_filings = try_fetch_sec_filings(ticker, limit=3)
        _ensure_source(state, sec_filings)

        # 모든 소스 통합
        all_sources = news_sources + sec_filings

        # Step 5: 소스별 텍스트 추출 및 분석
        progress("  └─ [FILING] 📄 Analyzing sources...")
        
        developments = []
        rev_candidates = []
        filings_table = []
        product_info = {
            "current_models": [],
            "upcoming_models": [],
            "ev_updates": [],
            "technology": [],
            "promotions": []
        }
        
        for i, src in enumerate(all_sources[:10], 1):  # 상위 10개만 상세 분석
            print(f"    Processing {i}/10: [{src['domain']}] {src['title'][:40]}...")
            
            # 이미 content가 있으면 사용, 없으면 fetch
            text = src.get("content", "")
            if not text:
                text = _fetch_text_with_retry(src["url"])
            
            if not text:
                continue
            
            # 주요 개발사항 추출 (더 상세하게)
            devs = extract_key_developments(text, src["title"], openai_client)
            if devs:
                print(f"      → Found {len(devs)} developments")
                developments.extend(devs)
            
            # 제품 포트폴리오 추출 (첫 3개 소스에서만)
            if i <= 3:
                portfolio = extract_product_portfolio(text, src["title"], ticker, openai_client)
                if portfolio:
                    print(f"      → Extracting product portfolio...")
                    for key in product_info:
                        if key in portfolio and portfolio[key]:
                            product_info[key].extend(portfolio[key])
            
            # 매출 정보 추출
            rev_data = extract_revenue_from_text(text, ticker, openai_client)
            if rev_data and rev_data.get("last_quarter"):
                rev_candidates.append(rev_data)
                print(f"      → Found revenue data: {rev_data.get('quarter_label', 'N/A')}")
            
            # SEC 공시 테이블
            if src["type"] == "filing" and "sec.gov" in src["url"]:
                filings_table.append({
                    "Form": src["title"],
                    "Date": src.get("date", ""),
                    "URL": src["url"]
                })

        # Step 6: 매출 데이터 선택 (가장 최근 것)
        rev_best = {}
        if rev_candidates:
            rev_best = max(rev_candidates, key=lambda x: x.get("last_quarter", 0))
        
        state.setdefault("tables", {})["filings"] = filings_table
        
        # Step 7: 비즈니스 텍스트 생성
        def _fmt(v):
            try:
                return f"${v/1e9:.2f}B" if abs(v) >= 1e9 else (f"${v/1e6:.2f}M" if abs(v) >= 1e6 else f"${v:,.0f}")
            except:
                return "N/A"

        business_text = "### 사업 전개 및 동향 분석\n\n"
        
        # 제품 포트폴리오 섹션
        if any(product_info.values()):
            business_text += "#### 제품 및 사업 현황\n\n"
            
            # 현재 주력 모델
            if product_info["current_models"]:
                business_text += "**현재 주력 모델**:\n"
                merged_models = _merge_duplicate_models(product_info["current_models"])
                for model in merged_models[:5]:
                    if isinstance(model, dict):
                        name = model.get("name", "")
                        segment = model.get("segment", "")
                        details = model.get("details", "")
                        business_text += f"- **{name}** ({segment}): {details}\n"
                    else:
                        business_text += f"- {model}\n"
                business_text += "\n"
            
            # 신모델 및 출시 예정
            if product_info["upcoming_models"]:
                business_text += "**신모델 및 출시 예정**:\n"
                for model in product_info["upcoming_models"][:5]:
                    if isinstance(model, dict):
                        name = model.get("name", "")
                        launch = model.get("launch_date", "")
                        details = model.get("details", "")
                        launch_str = f" (출시: {launch})" if launch else ""
                        business_text += f"- **{name}**{launch_str}: {details}\n"
                    else:
                        business_text += f"- {model}\n"
                business_text += "\n"
            
            # 전기차 동향
            if product_info["ev_updates"]:
                business_text += "**전기차(EV) 동향**:\n"
                unique_ev = list(dict.fromkeys(product_info["ev_updates"]))[:6]
                for update in unique_ev:
                    business_text += f"- {update}\n"
                business_text += "\n"
            
            # 기술 및 혁신
            if product_info["technology"]:
                business_text += "**기술 및 혁신**:\n"
                for tech in product_info["technology"][:4]:
                    business_text += f"- {tech}\n"
                business_text += "\n"
            
            # 프로모션 (명확한 혜택이 있는 것만)
            if product_info["promotions"]:
                business_text += "**프로모션 및 가격 정책**:\n"
                for promo in product_info["promotions"][:3]:
                    business_text += f"- {promo}\n"
                business_text += "\n"
        
        # 매출 정보
        if rev_best:
            business_text += "#### 재무 성과\n\n"
            quarter_label = rev_best.get("quarter_label", "최근 분기")
            lq = rev_best.get("last_quarter")
            pq = rev_best.get("prev_quarter")
            yoy = rev_best.get("yoy")
            
            comp = "전년 동분기 대비" if yoy else "전분기 대비"
            try:
                percentage = ((lq - yoy) / yoy * 100) if yoy else ((lq - pq) / pq * 100)
                business_text += f"**매출 현황**: {quarter_label} 매출은 {_fmt(lq)}로, {comp} {percentage:.1f}% 변화를 기록했습니다.\n\n"
            except:
                business_text += f"**매출 현황**: {quarter_label} 매출은 {_fmt(lq)}로 확인됩니다.\n\n"
            
            state["tables"]["rev_compare"] = [{
                "Quarter": quarter_label,
                "Last": lq,
                "Prev/YoY": pq or yoy,
                "Type": "QoQ" if pq else "YoY"
            }]
        else:
            business_text += "#### 재무 성과\n\n"
            business_text += "**매출 현황**: 최근 분기 매출 데이터를 확인 중입니다.\n\n"
            state["tables"]["rev_compare"] = []
        
        # 주요 개발사항 및 시장 동향
        if developments:
            business_text += "#### 주요 사업 동향\n\n"
            unique_devs = list(dict.fromkeys(developments))[:10]
            for dev in unique_devs:
                business_text += f"- {dev}\n"
            business_text += "\n"
        
        # 소스 요약
        business_text += f"**분석 기반**: {len(all_sources)}개 소스 (뉴스 {len(news_sources)}개"
        if sec_filings:
            business_text += f", SEC 공시 {len(sec_filings)}개"
        business_text += ")\n"

        # Step 8: 내용 평가 (할루시네이션 체크)
        progress("  └─ [FILING] 🔍 Content validation...")
        content_score, content_issues = evaluate_filing_content(
            business_text,
            product_info,
            openai_client,
            ticker
        )
        
        if content_score == 1:
            print(f"    ✅ FILING content validation passed")
            break
        else:
            print(f"    ⚠️  Content issues found (attempt {retry_count + 1}/{max_retries + 1}):")
            for issue in content_issues:
                print(f"      {issue}")
            
            retry_count += 1
            if retry_count > max_retries:
                print(f"    ⚠️  Max retries reached, proceeding with current content")
                print(f"    ℹ️  User should review the report carefully for accuracy")
                break
            
            # 재시도 전 대기
            time.sleep(2)

    # Step 9: 리스크 요약 생성
    progress("  └─ [FILING] ⚠️  Risk analysis...")
    
    sys_risk = f"Based on {ticker} recent news and filings, summarize 3-5 key investment risks in Korean bullets. Focus on {state['perspective']} perspective."
    user_risk = json.dumps({
        "sources": [s["title"] for s in all_sources[:10]],
        "perspective": state["perspective"]
    }, ensure_ascii=False)
    
    try:
        resp_risk = openai_client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[{"role": "system", "content": sys_risk}, {"role": "user", "content": user_risk}]
        )
        risks = "### 리스크 분석\n\n" + resp_risk.choices[0].message.content
    except Exception as e:
        print(f"    ⚠️  Risk summary failed: {e}")
        risks = "### 리스크 분석\n\n- 산업 변동성: 시장 환경 변화에 따른 리스크\n- 경쟁 심화: 업계 경쟁 상황 모니터링 필요\n- 규제 변화: 정책 및 규제 리스크\n"

    state["sections"]["business"] = business_text
    state["sections"]["risks"] = risks

    # Step 10: 평가
    needed = ["사업 전개", "매출", "리스크", "투자 추천"]
    eval_res = _binary_eval(business_text + "\n" + risks, needed, state["sources"])
    eval_res["retries"] = retry_count
    eval_res["date_filtering_passed"] = date_score == 1
    eval_res["content_validation_passed"] = content_score == 1
    state["evals"].append({"agent": "FILING", **eval_res})
    
    print(f"    ✅ FILING completed with {len(state['sources'])} total sources")
    print(f"       Retries: {retry_count}, Date Filter: {'✅' if date_score == 1 else '⚠️'}, Content Valid: {'✅' if content_score == 1 else '⚠️'}")
    
    return state

# ============================================================================
# FUNDAMENTAL AGENT
# ============================================================================

def _fetch_finviz_metrics(ticker: str) -> dict:
    """Finviz에서 재무 지표 수집"""
    url = f"https://finviz.com/quote.ashx?t={ticker}"
    try:
        r = requests.get(url, headers=FINVIZ_HEADERS, timeout=30)
        r.raise_for_status()
        soup = BeautifulSoup(r.text, "lxml")
        table = soup.find("table", class_="snapshot-table2")
        if not table:
            return {"raw": {}}
        
        cells = [c.get_text(strip=True) for c in table.select("td")]
        kv = {}
        for i in range(0, len(cells)-1, 2):
            kv[cells[i]] = cells[i+1]
        
        def g(k): return kv.get(k, "N/A")
        
        return {"raw": {
            "MarketCap": g("Market Cap"),
            "P/E": g("P/E"),
            "Forward P/E": g("Forward P/E"),
            "PEG": g("PEG"),
            "P/S": g("P/S"),
            "P/B": g("P/B"),
            "EV/EBITDA": g("EV/EBITDA"),
            "ProfitMargin(%)": g("Profit Margin"),
            "OperMargin(%)": g("Oper. Margin"),
            "ROE(%)": g("ROE"),
            "Debt/Eq": g("Debt/Eq")
        }}
    except Exception as e:
        print(f"⚠️  Finviz fetch failed: {e}")
        return {"raw": {}}

def fundamental_agent(state: dict, progress=lambda *_: None) -> dict:
    """Fundamental Analysis Agent"""
    e = state["entity"]
    ticker = e["ticker"]
    
    progress("  └─ [FUNDAMENTAL] 📊 Finviz metrics collection...")
    finviz = _fetch_finviz_metrics(ticker)
    
    blocks = {
        "valuation": {
            k: finviz["raw"].get(k, "N/A") 
            for k in ["P/E", "Forward P/E", "PEG", "P/S", "P/B", "EV/EBITDA"]
        },
        "profit_leverage": {
            k: finviz["raw"].get(k, "N/A") 
            for k in ["ProfitMargin(%)", "OperMargin(%)", "ROE(%)", "Debt/Eq"]
        }
    }
    
    state.setdefault("tables", {})["finviz_blocks"] = blocks

    val_txt = (
        "### 가치투자 분석\n\n"
        "**밸류에이션 멀티플**\n"
        f"- P/E: {blocks['valuation'].get('P/E', 'N/A')} "
        f"(Forward P/E: {blocks['valuation'].get('Forward P/E', 'N/A')})\n"
        f"- PEG: {blocks['valuation'].get('PEG', 'N/A')}, "
        f"P/S: {blocks['valuation'].get('P/S', 'N/A')}, "
        f"P/B: {blocks['valuation'].get('P/B', 'N/A')}\n"
        f"- EV/EBITDA: {blocks['valuation'].get('EV/EBITDA', 'N/A')}\n\n"
        "**수익성 지표**\n"
        f"- Profit Margin: {blocks['profit_leverage'].get('ProfitMargin(%)', 'N/A')}, "
        f"Oper. Margin: {blocks['profit_leverage'].get('OperMargin(%)', 'N/A')}\n"
        f"- ROE: {blocks['profit_leverage'].get('ROE(%)', 'N/A')}\n\n"
        "**재무 건전성**\n"
        f"- Debt/Eq: {blocks['profit_leverage'].get('Debt/Eq', 'N/A')}\n"
    )

    
    state["sections"]["value"] = val_txt
    
    _ensure_source(state, [{
        "url": f"https://finviz.com/quote.ashx?t={ticker}",
        "domain": "finviz.com",
        "title": f"Finviz Financial Snapshot - {ticker}",
        "type": "stats",
        "date": str(date.today()),
        "priority": 7
    }])
    
    state["evals"].append({
        "agent": "FUNDAMENTAL",
        **_binary_eval(val_txt, ["밸류에이션", "재무"], state["sources"]),
        "retries": 0
    })
    
    print(f"    ✅ FUNDAMENTAL completed")
    return state

# ============================================================================
# TECHNICAL AGENT
# ============================================================================

def sma(series: pd.Series, window: int) -> pd.Series:
    return series.rolling(window).mean()

def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    up = delta.clip(lower=0).ewm(alpha=1/period, adjust=False).mean()
    down = (-delta.clip(upper=0)).ewm(alpha=1/period, adjust=False).mean()
    rs = up / down.replace(0, np.nan)
    return 100 - (100 / (1 + rs))

def macd(series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9):
    ema_fast = series.ewm(span=fast, adjust=False).mean()
    ema_slow = series.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    return macd_line, signal_line, macd_line - signal_line

def technical_agent(state: dict, progress=lambda *_: None) -> dict:
    """Technical Analysis Agent"""
    e = state["entity"]
    ticker = e["ticker"]
    
    progress("  └─ [TECHNICAL] 📈 Chart analysis with FDR...")
    
    try:
        df = fdr.DataReader(ticker, start=date.today() - timedelta(days=365))
        state["artifacts"]["price_chart"] = _save_price_chart(df, ticker)
        
        close = df["Close"]
        rsi_v = rsi(close).iloc[-1]
        macd_line, signal_line, hist = macd(close)
        sma20 = sma(close, 20).iloc[-1]
        sma60 = sma(close, 60).iloc[-1]
        
        recent = close.tail(60)
        support = float(recent.min())
        resistance = float(recent.max())
        
        state.setdefault("tables", {})["technicals"] = {
            "RSI(14)": float(rsi_v),
            "SMA20": float(sma20),
            "SMA60": float(sma60)
        }
        state["tables"]["trend"] = {
            "Support (60d)": support,
            "Resistance (60d)": resistance
        }
        
        price = float(close.iloc[-1])
        
        lines = []
        
        # 이동평균 분석
        if pd.notna(sma20) and pd.notna(sma60):
            if sma20 > sma60:
                lines.append("**단기 이동평균(20일)이 중기(60일)보다 높아** 단기 상승 추세를 보이고 있습니다.")
            else:
                lines.append("**단기 이동평균(20일)이 중기(60일)보다 낮아** 조정 국면에 있습니다.")
        
        # RSI 분석
        if pd.notna(rsi_v):
            if rsi_v >= 70:
                lines.append(f"**RSI {rsi_v:.1f}**: 과매수 구간으로 단기 조정 가능성이 있습니다.")
            elif rsi_v <= 30:
                lines.append(f"**RSI {rsi_v:.1f}**: 과매도 구간으로 반등 가능성을 고려할 수 있습니다.")
            else:
                lines.append(f"**RSI {rsi_v:.1f}**: 중립 구간으로 추세가 명확하지 않습니다.")
        
        # MACD 분석
        if pd.notna(macd_line.iloc[-1]) and pd.notna(signal_line.iloc[-1]):
            if macd_line.iloc[-1] > signal_line.iloc[-1]:
                lines.append("**MACD가 시그널선 위**에 있어 상승 모멘텀이 감지됩니다.")
            else:
                lines.append("**MACD가 시그널선 아래**에 있어 하락/조정 모멘텀입니다.")
        
        # 지지/저항
        dist_sup = (price - support) / support * 100 if support else None
        dist_res = (resistance - price) / price * 100 if price else None
        if dist_sup is not None and dist_res is not None:
            lines.append(f"현재가 ${price:.2f}는 지지선(${support:.2f})으로부터 {dist_sup:.1f}%, 저항선(${resistance:.2f})까지 {dist_res:.1f}% 거리입니다.")
        
        tech_text = "### 기술적 분석\n\n" + "\n\n".join(lines)
        
    except Exception as e:
        print(f"⚠️  Technical analysis failed: {e}")
        tech_text = "### 기술적 분석\n\n데이터 제한으로 차트 분석을 수행할 수 없습니다.\n"
    
    state["sections"]["tech"] = tech_text
    state["evals"].append({
        "agent": "TECHNICAL",
        **_binary_eval(tech_text, ["차트", "지표"], state["sources"]),
        "retries": 0
    })
    
    print(f"    ✅ TECHNICAL completed")
    return state
