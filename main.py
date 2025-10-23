"""
Stock Analysis Agent System - Main Pipeline
주식 분석 자동화 시스템 메인 파이프라인
"""

from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv(usecwd=True), override=True)

from agents import (
    rewrite_query_for_routing,
    route_with_llm,
    decompose_tasks_with_llm,
    filing_business_agent,
    fundamental_agent,
    technical_agent,
    AGENT_NAMES
)
from report_agent import render as render_report, save_docx
from gate import check as gate_check, submit as submit_final
from datetime import datetime
import os
import json
from openai import OpenAI


def pipeline(query: str, context: dict | None = None) -> dict:
    """
    주식 분석 파이프라인 실행
    
    Args:
        query: 사용자 질의 (예: "TSLA 가치투자 분석")
        context: 컨텍스트 정보 (forced_perspective, forced_ticker 등)
    
    Returns:
        dict: 분석 결과 번들
            - passed: 품질 검증 통과 여부
            - report_md: Markdown 리포트
            - docx_path: Word 문서 경로
            - sources: 출처 목록
            - entity: 기업 정보
    """
    def progress(msg: str):
        """진행 상황 출력"""
        print(msg, flush=True)

    # OpenAI 클라이언트 초기화
    openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    # 상태 초기화
    state = {
        "query": query,
        "rewritten_query": None,
        "entity": None,
        "perspective": None,
        "route": [],
        "must_cover": [],
        "search_seeds": [],
        "sources": [],
        "sections": {},
        "artifacts": {},
        "evals": [],
        "tables": {},
        "log": [],
        "openai_client": openai_client
    }

    try:
        # Step 1: 쿼리 리라이팅
        progress("🛠️  [Rewrite] 라우팅 전 쿼리 리라이팅...")
        rewritten = rewrite_query_for_routing(query)
        state["rewritten_query"] = rewritten
        progress(f"✅ [Rewrite] {rewritten}")

        # Step 2: 라우팅 결정
        progress("🔎 [Router] 쿼리 해석 및 라우팅...")
        route = route_with_llm(query, rewritten_query=rewritten, context=context or {})
        state["entity"] = route.get("entity", {})
        state["perspective"] = route.get("perspective", "value")
        state["route"] = route.get("route", ["FILING", "FUNDAMENTAL"])
        state["must_cover"] = route.get("must_cover", [])
        state["search_seeds"] = route.get("search_seeds", [])
        
        pretty_route = [AGENT_NAMES.get(code, code) for code in state["route"]]
        progress(f"✅ [Router] route={pretty_route} perspective={state.get('perspective')}")

        # Step 3: 작업 분해
        progress("🧭 [Plan] 작업 분해 생성")
        state["plan"] = decompose_tasks_with_llm(route)
        for i, st in enumerate(state["plan"].get("steps", []), 1):
            progress(f"  · Step{i} [{AGENT_NAMES.get(st.get('agent'), st.get('agent'))}] {st.get('desc')}")

        # Step 4: 에이전트 실행
        progress(f"📋 [Run] {AGENT_NAMES['FILING']}")
        state = run_agents_parallel(state, progress)

        # Step 5: 리포트 생성
        progress("📝 [Report] 텍스트 합성")
        report_md, tl_dr = render_report(state)
        state["report_md"], state["tl_dr"] = report_md, tl_dr

        # Step 6: 품질 검증
        progress("🔒 [Gate] 최종 품질 점검")
        passed, feedback = gate_check(state)
        state["gate_feedback"] = feedback
        progress(f"✅ [Gate] passed={passed}")

        # Step 7: Word 문서 저장
        progress("💾 [Export] Word(.docx) 저장")
        docx_path = save_docx(state)
        state["docx_path"] = docx_path
        progress(f"📄 [Export] saved: {docx_path}")

        # 최종 결과 반환
        final_bundle = submit_final(state, passed)
        final_bundle["docx_path"] = docx_path
        return final_bundle

    except Exception as e:
        print(f"❌ Pipeline error: {str(e)}")
        import traceback
        traceback.print_exc()
        return {"error": str(e), "passed": False}


def run_agents_parallel(state: dict, progress):
    """
    에이전트 실행 (병렬 아님, 순차 실행으로 단순화)
    
    FILING은 항상 먼저 실행 (기본 정보 수집)
    FUNDAMENTAL, TECHNICAL은 순차 실행
    """
    try:
        # FILING 에이전트 먼저 실행 (뉴스 + 공시)
        state = filing_business_agent(state, progress=progress)
        
        # FUNDAMENTAL, TECHNICAL 순차 실행
        if "FUNDAMENTAL" in state["route"]:
            try:
                result = fundamental_agent(state.copy(), progress=progress)
                state["sources"].extend(result.get("sources", []))
                state["tables"].update(result.get("tables", {}))
                state["sections"].update(result.get("sections", {}))
                state["evals"].extend(result.get("evals", []))
            except Exception as e:
                print(f"⚠️  FUNDAMENTAL agent error: {e}")
        
        if "TECHNICAL" in state["route"]:
            try:
                result = technical_agent(state.copy(), progress=progress)
                state["sources"].extend(result.get("sources", []))
                state["tables"].update(result.get("tables", {}))
                state["sections"].update(result.get("sections", {}))
                state["evals"].extend(result.get("evals", []))
            except Exception as e:
                print(f"⚠️  TECHNICAL agent error: {e}")
                
    except Exception as e:
        print(f"❌ Agent execution error: {e}")
        import traceback
        traceback.print_exc()
    
    return state


def chat():
    """
    대화형 콘솔 인터페이스
    
    Commands:
        /exit, /q, quit, exit: 종료
        /mode value|tech: 관점 강제 지정
        /ticker TSLA: 티커 강제 지정
        /savemd [name]: 리포트를 Markdown으로 저장
        /help, /?: 도움말 표시
    """
    print("💬 Stock Agent Chat — '/exit'로 종료, '/help'로 도움말")
    
    session_ctx = {
        "forced_perspective": None,
        "forced_ticker": None,
        "last_entity": None
    }
    last_bundle = None

    while True:
        try:
            q = input("\nYou> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n👋 Bye!")
            break
            
        if not q:
            continue
            
        # 명령어 처리
        if q in ("/help", "/?"):
            print(chat.__doc__)
            continue
            
        if q in ("/exit", "exit", "quit", "/q"):
            print("👋 Bye!")
            break
            
        if q.startswith("/mode"):
            parts = q.split()
            if len(parts) >= 2 and parts[1] in ("value", "tech"):
                session_ctx["forced_perspective"] = parts[1]
                print(f"→ 관점 고정: {parts[1]}")
            else:
                print("사용법: /mode value|tech")
            continue
            
        if q.startswith("/ticker"):
            parts = q.split()
            if len(parts) >= 2:
                session_ctx["forced_ticker"] = parts[1].upper()
                print(f"→ 티커 고정: {session_ctx['forced_ticker']}")
            else:
                print("사용법: /ticker TSLA")
            continue
            
        if q.startswith("/savemd"):
            if not last_bundle:
                print("먼저 질의를 실행하세요.")
                continue
            parts = q.split(maxsplit=1)
            ticker = (last_bundle.get("entity") or {}).get("ticker", "REPORT")
            ts = datetime.now().strftime("%Y%m%d_%H%M")
            name = parts[1] if len(parts) > 1 else f"{ticker}_{ts}.md"
            os.makedirs("reports", exist_ok=True)
            with open(os.path.join("reports", name), "w", encoding="utf-8") as f:
                f.write(last_bundle.get("report_md", ""))
            print(f"✅ MD 저장 완료: reports/{name}")
            continue

        # 컨텍스트 구성
        ctx = {
            "last_entity": session_ctx["last_entity"],
            "forced_perspective": session_ctx["forced_perspective"],
            "forced_ticker": session_ctx["forced_ticker"]
        }
        
        # 파이프라인 실행
        bundle = pipeline(q, context=ctx)
        last_bundle = bundle
        
        if bundle.get("entity"):
            session_ctx["last_entity"] = bundle["entity"]

        # 결과 출력
        ent = bundle.get("entity") or {}
        pretty_route = [AGENT_NAMES.get(c, c) for c in bundle.get("route", [])]
        
        print(f"\n▶ 티커: {ent.get('ticker','?')} | 관점: {bundle.get('perspective')} | 경로: {pretty_route}")
        print("▶ TL;DR:", bundle.get("tl_dr",""))
        print("▶ Gate Passed:", bundle.get("passed"))
        print("▶ Word:", bundle.get("docx_path"))
        print("▶ Sources (top 5):")
        for i, s in enumerate(bundle.get("sources", [])[:5], 1):
            print(f"  {i}. [{s.get('domain','')}] {s.get('title','')} - {s.get('url','')}")


if __name__ == "__main__":
    chat()
