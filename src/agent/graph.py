"""Construye el StateGraph del agente Agroposta."""
from __future__ import annotations

from langgraph.graph import END, START, StateGraph

from agent.nodes.answerer import answerer_node
from agent.nodes.classifier import classifier_node
from agent.nodes.retriever import retriever_node
from agent.state import AgentState


def make_graph():
    g = StateGraph(AgentState)
    g.add_node("classifier", classifier_node)
    g.add_node("retriever", retriever_node)
    g.add_node("answerer", answerer_node)

    g.add_edge(START, "classifier")
    g.add_edge("classifier", "retriever")
    g.add_edge("retriever", "answerer")
    g.add_edge("answerer", END)

    return g.compile()


graph = make_graph()
