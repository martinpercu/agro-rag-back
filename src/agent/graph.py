"""Construye el StateGraph del agente Agroposta."""
from __future__ import annotations

from langgraph.graph import END, START, StateGraph

from agent.nodes.answerer import answerer_node
from agent.nodes.classifier import classifier_node
from agent.nodes.field_collector import field_collector_node
from agent.nodes.plan_intent import plan_intent_node
from agent.nodes.retriever import retriever_node
from agent.state import AgentState


def make_graph():
    g = StateGraph(AgentState)
    g.add_node("classifier", classifier_node)
    g.add_node("plan_intent", plan_intent_node)
    g.add_node("field_collector", field_collector_node)
    g.add_node("retriever", retriever_node)
    g.add_node("answerer", answerer_node)

    # Fase 2 baby: classifier → plan_intent → field_collector → retriever → answerer
    # Siempre pasa por los 3 primeros; son livianos y no bloquean (divisions puede quedar []).
    g.add_edge(START, "classifier")
    g.add_edge("classifier", "plan_intent")
    g.add_edge("plan_intent", "field_collector")
    g.add_edge("field_collector", "retriever")
    g.add_edge("retriever", "answerer")
    g.add_edge("answerer", END)

    return g.compile()


graph = make_graph()
