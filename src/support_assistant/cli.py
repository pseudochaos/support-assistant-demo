"""Command-line chat: streams the answer and asks the operator to confirm tickets."""

import argparse
import logging
import os
import sys
from uuid import uuid4

from langchain_core.callbacks import UsageMetadataCallbackHandler
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import HumanMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

from .config import load_kb, load_model, load_settings, model_name, tickets_path
from .graph import build_graph


def print_usage(handler: UsageMetadataCallbackHandler) -> None:
    """Token accounting feeds the cost metrics; prices come from config, not code."""
    tokens_in = sum(u.get("input_tokens", 0) for u in handler.usage_metadata.values())
    tokens_out = sum(u.get("output_tokens", 0) for u in handler.usage_metadata.values())
    line = f"[токени] вхід {tokens_in}, вихід {tokens_out}"
    price_in, price_out = os.getenv("PRICE_INPUT_PER_MTOK"), os.getenv("PRICE_OUTPUT_PER_MTOK")
    if price_in and price_out:
        cost = (tokens_in * float(price_in) + tokens_out * float(price_out)) / 1_000_000
        line += f", ≈ ${cost:.5f}"
    print(line)


def ask_operator(request: dict) -> dict:
    """Terminal version of the review card: approve or reject (editing is in Agent Chat UI)."""
    action = request["action_requests"][0]
    print(f"\n[ескалація] {action.get('description', '')}")
    print(f"  Тема: {action['args'].get('subject')}\n  Опис: {action['args'].get('description')}")
    approved = input("Створити тикет? [y/N] ").strip().lower() in {"y", "т", "так"}
    return {"decisions": [{"type": "approve"} if approved else {"type": "reject"}]}


class Printer:
    """Keeps streamed tokens and status lines from running into each other."""

    def __init__(self):
        self.mid_line = False

    def tokens(self, text: str) -> None:
        print(text, end="", flush=True)
        self.mid_line = True

    def line(self, text: str = "") -> None:
        if self.mid_line:
            print()
            self.mid_line = False
        if text:
            print(text)


def run(graph, question: str) -> None:
    usage = UsageMetadataCallbackHandler()
    config = {"configurable": {"thread_id": uuid4().hex}, "callbacks": [usage]}
    payload = {"messages": [HumanMessage(question)]}
    out = Printer()

    while True:
        pending = None
        # subgraphs=True streams tokens from inside the agent, not only its final state
        for _, mode, chunk in graph.stream(payload, config, stream_mode=["messages", "updates"], subgraphs=True):
            if mode == "messages":
                message, meta = chunk
                if meta.get("langgraph_node") == "model" and message.text:
                    out.tokens(message.text)
                elif meta.get("langgraph_node") == "tools":
                    out.line(f"[інструмент] {message.text}")
                continue
            if "__interrupt__" in chunk:
                pending = chunk["__interrupt__"][0].value
            if update := chunk.get("classify"):
                r = update["routing"]
                label = f"тема={r['topic']}, терміновість={r['urgency']}, мова={r['language']}" if r else "не вдалася"
                out.line(f"[класифікація] {label}")
            if update := chunk.get("retrieve"):
                out.line("[знайдені фрагменти] " + (", ".join(c["source"] for c in update["chunks"]) or "немає"))
        out.line()
        if pending is None:
            break
        payload = Command(resume=ask_operator(pending))

    print_usage(usage)


def main() -> None:
    parser = argparse.ArgumentParser(description="Асистент технічної підтримки — навчальне демо")
    parser.add_argument("question", nargs="?", help="звернення клієнта; без нього — діалог у терміналі")
    parser.add_argument("--graph", action="store_true", help="вивести граф у форматі Mermaid і завершити")
    args = parser.parse_args()

    load_settings()
    logging.basicConfig(format="[попередження] %(message)s")
    kb = load_kb()

    if args.graph:
        # Drawing needs no provider: a placeholder model is never called
        placeholder = GenericFakeChatModel(messages=iter(()))
        print(build_graph(placeholder, kb, tickets_path()).get_graph(xray=1).draw_mermaid())
        return

    name = model_name()
    try:
        model = load_model(name)
    except Exception as e:
        sys.exit(f"Не вдалося створити модель {name}: {e}\nПеревірте .env (див. .env.example).")
    print(f"Модель: {name}")
    graph = build_graph(model, kb, tickets_path(), checkpointer=InMemorySaver())

    if args.question:
        run(graph, args.question)
        return
    print("Порожній рядок або Ctrl+D — вихід.")
    while True:
        try:
            question = input("\nЗвернення клієнта: ").strip()
        except EOFError:
            break
        if not question:
            break
        run(graph, question)
