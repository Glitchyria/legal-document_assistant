from __future__ import annotations

import re
from typing import Callable

from .retrieval import retrieve
from .state import CapstoneState
from .tools import legal_utility_tool

MAX_EVAL_RETRIES = 2


def add_trace(state: CapstoneState, label: str) -> CapstoneState:
    trace = list(state.get("trace", []))
    trace.append(label)
    state["trace"] = trace
    return state


def memory_node(state: CapstoneState) -> CapstoneState:
    question = state.get("question", "")
    messages = list(state.get("messages", []))
    messages.append({"role": "user", "content": question})
    state["messages"] = messages[-6:]
    match = re.search(r"my name is\s+([A-Za-z][A-Za-z .'-]{1,40})", question, re.I)
    if match:
        state["user_name"] = match.group(1).strip().rstrip(".!?")
    return add_trace(state, "memory")


def router_node(state: CapstoneState) -> CapstoneState:
    question = state.get("question", "").lower()
    memory_terms = ["my name", "who am i", "what is my name", "remember"]
    tool_terms = ["today", "date", "time", "deadline", "calculate", "total", "fee total", "pages"]
    if any(term in question for term in memory_terms):
        route = "skip"
    elif any(term in question for term in tool_terms):
        route = "tool"
    else:
        route = "retrieve"
    state["route"] = route
    return add_trace(state, f"router:{route}")


def retrieval_node_factory(embedder, collection) -> Callable[[CapstoneState], CapstoneState]:
    def retrieval_node(state: CapstoneState) -> CapstoneState:
        context, sources = retrieve(state.get("question", ""), embedder, collection)
        state["retrieved"] = context
        state["sources"] = sources
        return add_trace(state, "retrieval")

    return retrieval_node


def skip_retrieval_node(state: CapstoneState) -> CapstoneState:
    state["retrieved"] = ""
    state["sources"] = []
    return add_trace(state, "skip_retrieval")


def tool_node(state: CapstoneState) -> CapstoneState:
    state["tool_result"] = legal_utility_tool(state.get("question", ""))
    return add_trace(state, "tool")


def answer_node(state: CapstoneState) -> CapstoneState:
    question = state.get("question", "")
    retrieved = state.get("retrieved", "")
    tool_result = state.get("tool_result", "")
    user_name = state.get("user_name", "")
    lower = question.lower()

    if "ignore your instructions" in lower or "system prompt" in lower:
        answer = (
            "I cannot reveal hidden instructions or ignore safety rules. I can help summarize legal documents, "
            "extract clauses, and answer from the available legal knowledge base."
        )
    elif re.search(r"\b(my name is)\b", lower):
        answer = f"Nice to meet you, {user_name}. I will remember your name during this session."
    elif "what is my name" in lower or "who am i" in lower:
        answer = f"Your name is {user_name}." if user_name else "I do not have your name in this session yet."
    elif any(term in lower for term in ["guarantee", "win my case", "case outcome", "predict the case"]):
        answer = (
            "I cannot guarantee or predict a legal outcome. The knowledge base is for document review support only, "
            "and this question should be reviewed by a qualified attorney."
        )
    elif tool_result:
        answer = tool_result
    elif retrieved:
        sentences = re.split(r"(?<=[.!?])\s+", retrieved.replace("\n", " "))
        keywords = set(re.findall(r"[a-zA-Z]{4,}", question.lower()))
        ranked = sorted(
            sentences,
            key=lambda sentence: sum(1 for word in keywords if word in sentence.lower()),
            reverse=True,
        )
        selected = [s.strip() for s in ranked[:4] if s.strip()]
        answer = " ".join(selected)
        if not answer or sum(1 for word in keywords if word in answer.lower()) == 0:
            answer = (
                "The knowledge base does not contain a grounded answer to that question. "
                "Please ask a supervising attorney to review the source documents."
            )
        else:
            answer += " This is document information only, not legal advice."
    else:
        answer = (
            "The knowledge base does not contain the answer. Please upload the relevant legal document "
            "or ask a qualified attorney to review it."
        )

    state["answer"] = answer
    return add_trace(state, "answer")


def eval_node(state: CapstoneState) -> CapstoneState:
    if not state.get("retrieved"):
        state["faithfulness"] = 1.0
        return add_trace(state, "eval:skipped")

    answer_words = set(re.findall(r"[a-zA-Z]{5,}", state.get("answer", "").lower()))
    context_words = set(re.findall(r"[a-zA-Z]{5,}", state.get("retrieved", "").lower()))
    if not answer_words:
        score = 0.0
    else:
        score = len(answer_words & context_words) / max(1, len(answer_words))
    state["faithfulness"] = round(float(score), 2)
    state["eval_retries"] = int(state.get("eval_retries", 0)) + 1
    return add_trace(state, f"eval:{state['faithfulness']}")


def save_node(state: CapstoneState) -> CapstoneState:
    messages = list(state.get("messages", []))
    messages.append({"role": "assistant", "content": state.get("answer", "")})
    state["messages"] = messages[-6:]
    return add_trace(state, "save")


def route_decision(state: CapstoneState) -> str:
    route = state.get("route", "retrieve")
    if route == "tool":
        return "tool"
    if route == "skip":
        return "skip"
    return "retrieve"


def eval_decision(state: CapstoneState) -> str:
    if state.get("faithfulness", 1.0) < 0.7 and state.get("eval_retries", 0) < MAX_EVAL_RETRIES:
        return "answer"
    return "save"
