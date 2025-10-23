"""
Stock Analysis Agent System - LangGraph Implementation
주식 분석 자동화 시스템 - LangGraph 기반 파이프라인

주요 변경사항:
1. 게이트 실패 시 루프 제거 - 에이전트별 평가자에 의한 루프만 유지
2. 시제 문제 해결을 위한 프롬프팅 개선
"""

from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv(usecwd=True), override=True)

import os
from datetime import datetime
from typing import TypedDict, Annotated, Literal
from langgraph.graph import StateGraph, END, START
from langgraph.checkpoint.memory import MemorySaver
from openai import OpenAI

# 기존 agents.py의 함수들 임포트
from agents import (
    rewrite_query_for_routing,
    route_with_llm,
    decompose_tasks_with_llm,
    filing_business_agent,
    fundamental_agent,
    technical_agent,
    evaluate_source_freshness,
    evaluate_filing_content,
    AGENT_NAMES
)
from report_agent import render as render_report, save_docx
from gate import check as gate_check, submit as submit_final


# ============================================================================
# STATE DEFINITION
# ============================================================================

class AgentState(TypedDict):
    """LangGraph 상태 정의"""
    # Input
    query: str
    context: dict
    
    # Routing
    rewritten_query: str
    entity: dict
    perspective: str
    route: list[str]
    must_cover: list[str]
    search_seeds: list[str]
    plan: dict
    
    # Agent Outputs
    sources: list[dict]
    sections: dict
    artifacts: dict
    tables: dict
    
    # Evaluations (에이전트별 평가만 유지)
    evals: list[dict]
    filing_retries: int
    filing_passed: bool
    date_filter_passed: bool
    content_valid: bool
    
    # Final Output
    report_md: str
    tl_dr: str
    gate_feedback: dict
    gate_passed: bool  # 참고용으로만 사용 (루프 없음)
    docx_path: str
    
    # System
    log: list[str]


# ============================================================================
# GRAPH NODES
# ============================================================================

def rewrite_node(state: AgentState) -> AgentState:
    """Step 1: 쿼리 리라이팅"""
    print("🛠️  [Rewrite] 라우팅 전 쿼리 리라이팅...")
    
    rewritten = rewrite_query_for_routing(state["query"])
    state["rewritten_query"] = rewritten
    state["log"].append(f"[Rewrite] {rewritten}")
    
    print(f"✅ [Rewrite] {rewritten}")
    return state


def router_node(state: AgentState) -> AgentState:
    """Step 2: 라우팅 결정"""
    print("🔎 [Router] 쿼리 해석 및 라우팅...")
    
    route = route_with_llm(
        state["query"],
        rewritten_query=state["rewritten_query"],
        context=state.get("context", {})
    )
    
    state["entity"] = route.get("entity", {})
    state["perspective"] = route.get("perspective", "value")
    state["route"] = route.get("route", ["FILING", "FUNDAMENTAL"])
    state["must_cover"] = route.get("must_cover", [])
    state["search_seeds"] = route.get("search_seeds", [])
    
    pretty_route = [AGENT_NAMES.get(code, code) for code in state["route"]]
    print(f"✅ [Router] route={pretty_route} perspective={state['perspective']}")
    
    state["log"].append(f"[Router] route={state['route']} perspective={state['perspective']}")
    return state


def planning_node(state: AgentState) -> AgentState:
    """Step 3: 작업 분해 계획"""
    print("🧭 [Plan] 작업 분해 생성")
    
    plan = decompose_tasks_with_llm({
        "entity": state["entity"],
        "perspective": state["perspective"],
        "route": state["route"],
        "must_cover": state["must_cover"]
    })
    
    state["plan"] = plan
    
    for i, st in enumerate(plan.get("steps", []), 1):
        agent_name = AGENT_NAMES.get(st.get("agent"), st.get("agent"))
        print(f"  · Step{i} [{agent_name}] {st.get('desc')}")
    
    state["log"].append(f"[Plan] {len(plan.get('steps', []))} steps created")
    return state


def filing_node(state: AgentState) -> AgentState:
    """Step 4: Filing & Business Agent with Retry
    
    ⚠️ 중요: 이 노드는 내부적으로 retry 로직을 가지고 있음
    에이전트별 평가자가 실패 시 자동으로 재시도
    """
    print(f"📋 [Run] {AGENT_NAMES['FILING']}")
    
    # 초기화
    if "filing_retries" not in state:
        state["filing_retries"] = 0
    if "sources" not in state:
        state["sources"] = []
    if "sections" not in state:
        state["sections"] = {}
    if "tables" not in state:
        state["tables"] = {}
    if "artifacts" not in state:
        state["artifacts"] = {}
    if "evals" not in state:
        state["evals"] = []
    
    # OpenAI 클라이언트 생성 (매번 새로 생성)
    state["openai_client"] = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    
    # Filing 실행 (max_retries는 내부에서 처리)
    state = filing_business_agent(state, progress=print, max_retries=2)
    
    # 결과 평가
    last_eval = state["evals"][-1] if state["evals"] else {}
    state["filing_passed"] = last_eval.get("content_validation_passed", False)
    state["date_filter_passed"] = last_eval.get("date_filtering_passed", False)
    state["filing_retries"] = last_eval.get("retries", 0)
    
    # OpenAI 클라이언트 제거 (직렬화 방지)
    del state["openai_client"]
    
    return state


def fundamental_node(state: AgentState) -> AgentState:
    """Step 5a: Fundamental Analysis"""
    print(f"📊 [Run] {AGENT_NAMES['FUNDAMENTAL']}")
    
    result = fundamental_agent(state.copy(), progress=print)
    
    # 결과 병합
    state["sources"].extend(result.get("sources", []))
    state["tables"].update(result.get("tables", {}))
    state["sections"].update(result.get("sections", {}))
    state["evals"].extend(result.get("evals", []))
    
    return state


def technical_node(state: AgentState) -> AgentState:
    """Step 5b: Technical Analysis"""
    print(f"📈 [Run] {AGENT_NAMES['TECHNICAL']}")
    
    result = technical_agent(state.copy(), progress=print)
    
    # 결과 병합
    state["sources"].extend(result.get("sources", []))
    state["tables"].update(result.get("tables", {}))
    state["sections"].update(result.get("sections", {}))
    state["evals"].extend(result.get("evals", []))
    
    return state


def report_node(state: AgentState) -> AgentState:
    """Step 6: 리포트 생성"""
    print("📝 [Report] 텍스트 합성")
    
    # OpenAI 클라이언트 임시 생성
    state["openai_client"] = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    
    report_md, tl_dr = render_report(state)
    state["report_md"] = report_md
    state["tl_dr"] = tl_dr
    
    # OpenAI 클라이언트 제거
    del state["openai_client"]
    
    state["log"].append(f"[Report] Generated ({len(report_md)} chars)")
    return state


def gate_node(state: AgentState) -> AgentState:
    """Step 7: 품질 검증 (참고용, 루프 없음)
    
    ⚠️ 중요 변경사항:
    - 게이트 실패 시 루프를 돌지 않음
    - 품질 피드백만 제공하고 무조건 진행
    - 에이전트별 평가자만 재시도 수행
    """
    print("🔒 [Gate] 최종 품질 점검")
    
    # OpenAI 클라이언트 임시 생성
    state["openai_client"] = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    
    passed, feedback = gate_check(state)
    state["gate_passed"] = passed
    state["gate_feedback"] = feedback
    
    # OpenAI 클라이언트 제거
    del state["openai_client"]
    
    if not passed:
        print(f"⚠️  [Gate] Quality check failed, but proceeding to export")
        print(f"    Issues: {', '.join(feedback.get('tips', []))}")
        print(f"    Note: 에이전트별 평가는 이미 완료되었으므로 진행합니다.")
    else:
        print(f"✅ [Gate] Passed")
    
    state["log"].append(f"[Gate] passed={passed} (no retry)")
    
    return state


def export_node(state: AgentState) -> AgentState:
    """Step 8: Word 문서 저장"""
    print("💾 [Export] Word(.docx) 저장")
    
    docx_path = save_docx(state)
    state["docx_path"] = docx_path
    
    print(f"📄 [Export] saved: {docx_path}")
    state["log"].append(f"[Export] {docx_path}")
    
    return state


# ============================================================================
# ROUTING FUNCTIONS
# ============================================================================

def route_after_filing(state: AgentState) -> str:
    """Filing 후 다음 노드 결정"""
    route = state.get("route", [])
    
    # FUNDAMENTAL 우선
    if "FUNDAMENTAL" in route:
        return "fundamental"
    # TECHNICAL
    elif "TECHNICAL" in route:
        return "technical"
    # 둘 다 없으면 바로 리포트
    else:
        return "report"


def route_after_fundamental(state: AgentState) -> str:
    """FUNDAMENTAL 후 다음 노드 결정"""
    route = state.get("route", [])
    
    # TECHNICAL도 있으면 실행
    if "TECHNICAL" in route:
        return "technical"
    # 없으면 리포트
    else:
        return "report"


# ============================================================================
# GRAPH CONSTRUCTION
# ============================================================================

def create_stock_analysis_graph():
    """주식 분석 LangGraph 생성
    
    중요 변경사항:
    - Gate 실패 시 루프 제거
    - Gate → Export로 직접 연결
    - 에이전트별 평가자만 재시도 수행
    """
    
    # StateGraph 생성
    workflow = StateGraph(AgentState)
    
    # 노드 추가
    workflow.add_node("rewrite", rewrite_node)
    workflow.add_node("router", router_node)
    workflow.add_node("planning", planning_node)
    workflow.add_node("filing", filing_node)
    workflow.add_node("fundamental", fundamental_node)
    workflow.add_node("technical", technical_node)
    workflow.add_node("report", report_node)
    workflow.add_node("gate", gate_node)
    workflow.add_node("export", export_node)
    
    # 엣지 추가 (순차 실행)
    workflow.add_edge(START, "rewrite")
    workflow.add_edge("rewrite", "router")
    workflow.add_edge("router", "planning")
    workflow.add_edge("planning", "filing")
    
    # 조건부 엣지 (라우팅에 따라 분기)
    workflow.add_conditional_edges(
        "filing",
        route_after_filing,
        {
            "fundamental": "fundamental",
            "technical": "technical",
            "report": "report"
        }
    )
    
    workflow.add_conditional_edges(
        "fundamental",
        route_after_fundamental,
        {
            "technical": "technical",
            "report": "report"
        }
    )
    
    workflow.add_edge("technical", "report")
    workflow.add_edge("report", "gate")
    
    # ⚠️ 중요: Gate → Export 직접 연결 (루프 제거)
    workflow.add_edge("gate", "export")
    workflow.add_edge("export", END)
    
    # 메모리 체크포인터 추가
    memory = MemorySaver()
    
    return workflow.compile(checkpointer=memory)


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def run_analysis(query: str, context: dict | None = None) -> dict:
    """
    주식 분석 실행 (LangGraph)
    
    Args:
        query: 사용자 질의
        context: 컨텍스트 정보
    
    Returns:
        dict: 분석 결과
    """
    # 초기 상태 구성
    initial_state: AgentState = {
        "query": query,
        "context": context or {},
        "rewritten_query": "",
        "entity": {},
        "perspective": "",
        "route": [],
        "must_cover": [],
        "search_seeds": [],
        "plan": {},
        "sources": [],
        "sections": {},
        "artifacts": {},
        "tables": {},
        "evals": [],
        "filing_retries": 0,
        "filing_passed": False,
        "date_filter_passed": False,
        "content_valid": False,
        "report_md": "",
        "tl_dr": "",
        "gate_feedback": {},
        "gate_passed": False,
        "docx_path": "",
        "log": []
    }
    
    # 그래프 생성
    app = create_stock_analysis_graph()
    
    # 실행
    config = {"configurable": {"thread_id": "stock-analysis-1"}}
    
    try:
        final_state = app.invoke(initial_state, config)
        
        # 최종 결과 반환
        return submit_final(final_state, final_state["gate_passed"])
        
    except Exception as e:
        print(f"❌ Pipeline error: {str(e)}")
        import traceback
        traceback.print_exc()
        return {"error": str(e), "passed": False}


# ============================================================================
# INTERACTIVE CHAT
# ============================================================================

def chat():
    """
    대화형 콘솔 인터페이스
    
    Commands:
        /exit, /q, quit, exit: 종료
        /mode value|tech: 관점 강제 지정
        /ticker TSLA: 티커 강제 지정
        /savemd [name]: 리포트를 Markdown으로 저장
        /graph: 그래프 구조 시각화
        /help, /?: 도움말 표시
    """
    print("💬 Stock Agent Chat (LangGraph) — '/exit'로 종료, '/help'로 도움말")
    print("⚠️  Note: Gate 실패 시 루프는 제거되었습니다. 에이전트별 평가만 재시도합니다.")
    
    session_ctx = {
        "forced_perspective": None,
        "forced_ticker": None,
        "last_entity": None
    }
    last_bundle = None
    
    # 그래프 생성 (재사용)
    app = create_stock_analysis_graph()

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
            
        if q == "/graph":
            try:
                # 그래프 시각화 (Mermaid)
                print("\n📊 Graph Structure:")
                print(app.get_graph().draw_mermaid())
            except Exception as e:
                print(f"⚠️  Graph visualization failed: {e}")
            continue
            
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
        
        # LangGraph 실행
        bundle = run_analysis(q, context=ctx)
        last_bundle = bundle
        
        if bundle.get("entity"):
            session_ctx["last_entity"] = bundle["entity"]

        # 결과 출력
        if "error" in bundle:
            print(f"❌ Error: {bundle['error']}")
            continue
            
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
