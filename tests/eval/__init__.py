"""External evaluation framework for the MTSS RAG agent.

Two-phase architecture:
    Phase 1 (run):    execute golden questions through the agent, log everything
    Phase 2 (judge):  score logged results with auto-graders + LLM judge
    Phase 3 (diff):   compare two runs to enable iterative tuning

CLI: `mtss eval run | judge | diff | list`
"""
