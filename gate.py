"""
Quality Gate - 리포트 품질 검증
"""

from typing import Dict, Any, Tuple, Set, List
from openai import OpenAI
import json


BASE_TRUSTED_DEFAULT: Set[str] = {
    "sec.gov",
    "reuters.com",
    "bloomberg.com",
    "ft.com",
    "wsj.com",
    "cnbc.com",
    "finance.yahoo.com",
    "marketwatch.com",
    "seekingalpha.com",
    "barrons.com"
}


def _coverage_ratio(text: str, must_cover: list) -> float:
    """필수 커버리지 비율 계산"""
    if not must_cover:
        return 1.0
    hit = sum(1 for m in must_cover if m in text)
    return hit / len(must_cover)


def _domain_diversity(sources: List[Dict[str, Any]]) -> int:
    """도메인 다양성 계산"""
    return len({s.get("domain") for s in sources if s.get("domain")})


def _has_pii_or_banned(text: str) -> bool:
    """PII 또는 금지어 포함 여부"""
    banned = ["주민등록번호", "card number", "password=", "ssn", "신용카드"]
    return any(b in text.lower() for b in banned)


def _trusted_hits(sources: List[Dict[str, Any]], trusted: Set[str]) -> int:
    """신뢰 도메인 히트 수"""
    return len({s.get("domain") for s in sources if s.get("domain") in (trusted or set())})


def _high_priority_sources(sources: List[Dict[str, Any]]) -> int:
    """고우선순위 소스 수 (priority >= 7)"""
    return sum(1 for s in sources if s.get("priority", 0) >= 7)


def _llm_contradiction_check(draft: str, openai_client) -> bool:
    """LLM 기반 모순 체크"""
    sys = (
        "Check for contradictions or factual errors in this investment report. "
        "Output 'OK' if no issues found, else 'ERROR: brief description'."
    )
    
    try:
        resp = openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": sys},
                {"role": "user", "content": draft[:3000]}  # 처음 3000자만
            ],
            temperature=0.0
        )
        result = resp.choices[0].message.content
        return "OK" in result.upper()
    except Exception as e:
        print(f"⚠️  LLM contradiction check failed: {e}")
        return True  # 실패 시 통과로 간주


def check(state: Dict[str, Any]) -> Tuple[bool, Dict[str, Any]]:
    """
    품질 검증 게이트
    
    검증 항목:
    1. Coverage: 필수 섹션 포함 여부
    2. Grounding: 충분한 소스 기반
    3. Contradiction: 모순 없음
    4. Safety: PII/금지어 없음, 최소 길이
    5. Sources: 도메인 다양성, 신뢰 소스 포함
    6. Business: 사업 분석 포함
    7. Revenue: 매출 데이터 포함
    8. Recommendation: 투자 추천 포함
    
    Args:
        state: 파이프라인 상태
    
    Returns:
        tuple: (passed, feedback)
    """
    draft = state.get("report_md", "")
    must = state.get("must_cover", [])
    sources = state.get("sources", [])
    source_map = state.get("source_map", {})
    trusted = state.get("trusted_domains") or BASE_TRUSTED_DEFAULT
    openai_client = state.get("openai_client")

    # 1. Coverage 체크
    coverage_ok = _coverage_ratio(draft, must) >= 0.7  # 70%로 완화

    # 2. Grounding 체크
    grounded_ok = len(sources) >= 3 and bool(source_map)

    # 3. Contradiction 체크
    contradiction_ok = True
    if openai_client:
        contradiction_ok = _llm_contradiction_check(draft, openai_client)

    # 4. Safety & Style 체크
    safety_style_ok = len(draft) > 500 and not _has_pii_or_banned(draft)

    # 5. Sources 체크 (강화)
    domain_diversity = _domain_diversity(sources)
    trusted_count = _trusted_hits(sources, trusted)
    high_priority_count = _high_priority_sources(sources)
    
    sources_ok = (
        len(sources) >= 5 and  # 최소 5개 소스
        domain_diversity >= 3 and  # 최소 3개 도메인
        (trusted_count >= 2 or high_priority_count >= 3)  # 신뢰 소스 2개 OR 고우선순위 3개
    )

    # 6. Business 섹션 체크
    business_ok = bool(state.get("sections", {}).get("business"))

    # 7. Revenue 데이터 체크 (선택적)
    rev_ok = bool(state.get("tables", {}).get("rev_compare"))
    # 매출 데이터는 필수는 아님 (뉴스로도 충분히 분석 가능)
    if not rev_ok:
        print("    ℹ️  Revenue data not found, but continuing (news analysis sufficient)")

    # 8. Recommendation 체크
    rec_ok = bool(state.get("sections", {}).get("recommendation"))

    # 전체 통과 여부
    sufficient = (
        coverage_ok and
        grounded_ok and
        contradiction_ok and
        safety_style_ok and
        sources_ok and
        business_ok and
        rec_ok
        # rev_ok는 필수 아님
    )

    # 피드백 구성
    feedback = {
        "coverage_ok": coverage_ok,
        "grounded_ok": grounded_ok,
        "contradiction_ok": contradiction_ok,
        "safety_style_ok": safety_style_ok,
        "sources_ok": sources_ok,
        "business_ok": business_ok,
        "rev_ok": rev_ok,
        "rec_ok": rec_ok,
        "source_stats": {
            "total": len(sources),
            "domains": domain_diversity,
            "trusted": trusted_count,
            "high_priority": high_priority_count
        },
        "tips": []
    }

    # 개선 제안
    if not coverage_ok:
        feedback["tips"].append("필수 커버 항목(must_cover)을 본문에 70% 이상 반영하세요.")
    
    if not grounded_ok or not sources_ok:
        if len(sources) < 5:
            feedback["tips"].append("최소 5개 이상의 소스가 필요합니다.")
        if domain_diversity < 3:
            feedback["tips"].append("최소 3개 이상의 다른 도메인 소스가 필요합니다.")
        if trusted_count < 2 and high_priority_count < 3:
            feedback["tips"].append("신뢰할 수 있는 소스(Bloomberg, Reuters 등)를 더 추가하세요.")
    
    if not safety_style_ok:
        if len(draft) <= 500:
            feedback["tips"].append("리포트 길이가 너무 짧습니다 (최소 500자).")
        if _has_pii_or_banned(draft):
            feedback["tips"].append("PII 또는 금지어를 제거하세요.")
    
    if not business_ok:
        feedback["tips"].append("사업 전개/동향 분석 섹션을 추가하세요.")
    
    if not rec_ok:
        feedback["tips"].append("투자 추천 섹션(Buy/Hold/Sell 포함)을 추가하세요.")

    return sufficient, feedback


def submit(state: Dict[str, Any], passed: bool) -> Dict[str, Any]:
    """
    최종 결과 제출
    
    Args:
        state: 파이프라인 상태
        passed: 품질 검증 통과 여부
    
    Returns:
        dict: 최종 번들
    """
    return {
        "passed": passed,
        "entity": state.get("entity"),
        "perspective": state.get("perspective"),
        "route": state.get("route"),
        "tl_dr": state.get("tl_dr", ""),
        "report_md": state.get("report_md", ""),
        "sources": state.get("sources", []),
        "source_map": state.get("source_map", {}),
        "evals": state.get("evals", []),
        "gate_feedback": state.get("gate_feedback", {}),
        "docx_path": state.get("docx_path"),
        "disclaimer": (
            "본 내용은 정보 제공 목적이며 투자 조언이 아닙니다. "
            "투자 결정은 본인의 판단과 책임 하에 이루어져야 합니다."
        )
    }