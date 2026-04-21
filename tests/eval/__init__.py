"""External evaluation framework for the MTSS RAG agent.

Two-phase architecture:
    Phase 1 (run):   execute golden questions through the agent, log everything
    Phase 2 (score): compute auto-grader metrics from logged results + diff runs

Humans do the qualitative judging — the framework deliberately has no
LLM-as-judge step (the user is the judge).

CLI: `mtss eval run | score | diff | list`
"""
