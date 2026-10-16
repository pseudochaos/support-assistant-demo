"""Settings shared by the terminal chat and the `langgraph dev` entry point."""

import os
from pathlib import Path

from dotenv import load_dotenv

from .kb import KnowledgeBase

DEFAULT_MODEL = "google_genai:gemini-3.5-flash-lite"
REPO_ROOT = Path(__file__).resolve().parents[2]


def load_settings() -> None:
    load_dotenv(REPO_ROOT / ".env")


def model_name() -> str:
    return os.getenv("ASSISTANT_MODEL", DEFAULT_MODEL)


def load_model(name: str):
    from langchain.chat_models import init_chat_model

    # Without a timeout a stalled request waits forever and the fallback to a human
    # never fires; retries multiply the wait, so both stay small and configurable
    return init_chat_model(
        name,
        timeout=float(os.getenv("MODEL_TIMEOUT", "60")),
        max_retries=int(os.getenv("MODEL_MAX_RETRIES", "2")),
    )


def load_kb() -> KnowledgeBase:
    return KnowledgeBase.from_dir(Path(os.getenv("KB_DIR", REPO_ROOT / "kb")))


def tickets_path() -> Path:
    return Path(os.getenv("TICKETS_FILE", REPO_ROOT / "tickets.jsonl"))
