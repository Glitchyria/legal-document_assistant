from __future__ import annotations

from copy import deepcopy
from typing import Dict, Tuple

from .nodes import (
    answer_node,
    eval_decision,
    eval_node,
    memory_node,
    retrieval_node_factory,
    route_decision,
    router_node,
    save_node,
    skip_retrieval_node,
    tool_node,
)
from .retrieval import build_collection, load_embedder
from .state import CapstoneState, initial_state


class SimpleApp:
    def __init__(self, retrieval_node):
        self.retrieval_node = retrieval_node
        self.memory: Dict[str, CapstoneState] = {}

    def invoke(self, state: CapstoneState, config=None) -> CapstoneState:
        thread_id = "default"
        if config:
            thread_id = config.get("configurable", {}).get("thread_id", "default")

        previous = deepcopy(self.memory.get(thread_id, {}))
        merged = initial_state(state.get("question", ""))
        for key in ("messages", "user_name"):
            if previous.get(key):
                merged[key] = previous[key]
        merged["question"] = state.get("question", merged["question"])
        merged["eval_retries"] = 0
        merged["tool_result"] = ""
        merged["trace"] = []

        merged = memory_node(merged)
        merged = router_node(merged)
        decision = route_decision(merged)
        if decision == "retrieve":
            merged = self.retrieval_node(merged)
        elif decision == "tool":
            merged = tool_node(merged)
        else:
            merged = skip_retrieval_node(merged)

        merged = answer_node(merged)
        merged = eval_node(merged)
        while eval_decision(merged) == "answer":
            merged = answer_node(merged)
            merged = eval_node(merged)
        merged = save_node(merged)
        self.memory[thread_id] = deepcopy(merged)
        return merged


def build_app():
    embedder = load_embedder()
    collection = build_collection(embedder)
    retrieval_node = retrieval_node_factory(embedder, collection)

    try:
        from langgraph.checkpoint.memory import MemorySaver
        from langgraph.graph import END, StateGraph

        graph = StateGraph(CapstoneState)
        graph.add_node("memory", memory_node)
        graph.add_node("router", router_node)
        graph.add_node("retrieval", retrieval_node)
        graph.add_node("skip_retrieval", skip_retrieval_node)
        graph.add_node("tool", tool_node)
        graph.add_node("answer", answer_node)
        graph.add_node("eval", eval_node)
        graph.add_node("save", save_node)
        graph.set_entry_point("memory")
        graph.add_edge("memory", "router")
        graph.add_conditional_edges(
            "router",
            route_decision,
            {"retrieve": "retrieval", "skip": "skip_retrieval", "tool": "tool"},
        )
        graph.add_edge("retrieval", "answer")
        graph.add_edge("skip_retrieval", "answer")
        graph.add_edge("tool", "answer")
        graph.add_edge("answer", "eval")
        graph.add_conditional_edges("eval", eval_decision, {"answer": "answer", "save": "save"})
        graph.add_edge("save", END)
        return graph.compile(checkpointer=MemorySaver()), embedder, collection
    except Exception:
        return SimpleApp(retrieval_node), embedder, collection


APP, EMBEDDER, COLLECTION = build_app()


def ask(question: str, thread_id: str = "demo") -> CapstoneState:
    return APP.invoke(initial_state(question), config={"configurable": {"thread_id": thread_id}})
