from __future__ import annotations

from typing import Any, Dict, List, TypedDict


class CapstoneState(TypedDict, total=False):
    question: str
    messages: List[Dict[str, str]]
    route: str
    retrieved: str
    sources: List[str]
    tool_result: str
    answer: str
    faithfulness: float
    eval_retries: int
    user_name: str
    trace: List[str]
    raw: Dict[str, Any]


def initial_state(question: str) -> CapstoneState:
    return {
        "question": question,
        "messages": [],
        "route": "",
        "retrieved": "",
        "sources": [],
        "tool_result": "",
        "answer": "",
        "faithfulness": 0.0,
        "eval_retries": 0,
        "user_name": "",
        "trace": [],
        "raw": {},
    }

