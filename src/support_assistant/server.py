"""Graph entry point for `langgraph dev` (Agent Chat UI, LangGraph Studio); see langgraph.json.

The server loads this file by path, so imports are absolute.
"""

from support_assistant.config import load_kb, load_model, load_settings, model_name, tickets_path
from support_assistant.graph import build_graph

load_settings()
# No checkpointer: the server persists threads and interrupts itself
graph = build_graph(load_model(model_name()), load_kb(), tickets_path())
