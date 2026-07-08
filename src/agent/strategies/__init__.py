"""Modulo de estrategias de retrieval para el comparador RAG.

Cada estrategia es una pieza que toma (question, history) y devuelve un
StrategyResult con los chunks recuperados y metadata de como se recuperaron.

El runner (strategies/runner.py) corre varias estrategias en paralelo y
el endpoint /compare las expone lado a lado.
"""
