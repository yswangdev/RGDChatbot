"""RAG evaluation harness for the RGD chatbot.

Scores retrieval and generation with reference-based metrics and an
LLM-as-judge, plus robustness stress testing, and combines them into a single
overall score. See ``eval/run_eval.py`` for the orchestrator.
"""
