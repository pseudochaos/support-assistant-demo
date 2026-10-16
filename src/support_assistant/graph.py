"""The assistant as a LangGraph graph with a LangChain agent inside.

    START ─┬─ classify ─┬─ agent ─ END
           └─ retrieve ─┘

classify and retrieve run in parallel. The agent answers from the retrieved
fragments or calls create_ticket; HumanInTheLoopMiddleware pauses the graph
before the tool runs, so a ticket exists only after a human approves it.

The state keeps the chat under "messages" and the pause uses LangChain's
human-in-the-loop request format, so the same graph works in the terminal,
in Agent Chat UI and in LangGraph Studio.
"""

import json
import logging
import textwrap
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal, NotRequired
from uuid import uuid4

from langchain.agents import AgentState, create_agent
from langchain.agents.middleware import AgentMiddleware, HumanInTheLoopMiddleware, ModelRequest, dynamic_prompt
from langchain.tools import ToolRuntime, tool
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field

from .kb import KnowledgeBase

log = logging.getLogger(__name__)

SYSTEM = (
    "Ти — асистент технічної підтримки онлайн-сервісу бухгалтерського обліку для ФОП. "
    "Відповідай мовою клієнта, стисло. "
    "Відповідай ВИКЛЮЧНО на основі наданих фрагментів бази знань і нічого не вигадуй. "
    "Якщо у фрагментах немає відповіді на запитання, не відповідай сам, "
    "а виклич інструмент create_ticket, щоб передати звернення оператору. "
    "Наприкінці відповіді переліч використані джерела в квадратних дужках."
)

TICKET_TOOL = "create_ticket"


class Routing(BaseModel):
    """Classification of a customer message."""

    topic: Literal["account", "billing", "reporting", "technical", "other"] = Field(
        description="account: access and settings; billing: subscription and payments; "
        "reporting: taxes and reports; technical: errors and outages; other: anything else"
    )
    urgency: int = Field(ge=1, le=5, description="1: can wait, 5: blocks the customer's work right now")
    language: str = Field(description="ISO 639-1 code of the message language, e.g. uk, en")


class SupportState(AgentState):
    # Both fields are rewritten for every new question in the thread
    routing: NotRequired[dict | None]  # Routing as a dict; None if classification failed
    chunks: NotRequired[list[dict]]  # {"source": ..., "text": ...}


def brief(error: Exception) -> str:
    """Provider errors carry whole JSON payloads; one line is enough for a log."""
    return textwrap.shorten(f"{type(error).__name__}: {error}", width=200, placeholder=" …")


def question_of(state) -> str:
    """The question is the latest customer message in the thread."""
    return next(m.text for m in reversed(state["messages"]) if isinstance(m, HumanMessage))


def current_turn(messages: list) -> list:
    """Messages after the latest customer message."""
    last_human = max(i for i, m in enumerate(messages) if isinstance(m, HumanMessage))
    return messages[last_human + 1 :]


def format_context(chunks: list[dict]) -> str:
    if not chunks:
        return "(релевантних фрагментів не знайдено)"
    return "\n\n".join(f"[{c['source']}]\n{c['text']}" for c in chunks)


def save_ticket(path: Path, subject: str, description: str, routing: dict | None) -> str:
    """Stand-in for a ticketing system: one JSON line per ticket."""
    ticket = {
        "id": uuid4().hex[:8],
        "created": datetime.now(UTC).isoformat(timespec="seconds"),
        "subject": subject,
        "description": description,
        "routing": routing,
    }
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(ticket, ensure_ascii=False) + "\n")
    return ticket["id"]


def make_ticket_tool(tickets_path: Path):
    @tool(TICKET_TOOL, parse_docstring=True)
    def create_ticket(subject: str, description: str, runtime: ToolRuntime) -> str:
        """Create a ticket for a human operator when the knowledge base fragments do not answer the customer's question.

        Args:
            subject: Short subject of the ticket.
            description: What the customer asks and what the knowledge base lacks.
        """
        # The routing comes from the graph state, not from the model's arguments
        ticket_id = save_ticket(tickets_path, subject, description, runtime.state.get("routing"))
        return f"Тикет {ticket_id} створено й передано оператору."

    return create_ticket


def describe_ticket(tool_call, state, runtime) -> str:
    """Text of the review card: what the model proposes and how the message was classified."""
    routing = state.get("routing")
    note = f"Тема: {routing['topic']}, терміновість: {routing['urgency']}." if routing else "Класифікація не вдалася."
    return f"Асистент пропонує передати звернення оператору. {note}"


@dynamic_prompt
def grounded_prompt(request: ModelRequest) -> str:
    """RAG: the fragments for the current question go into the system prompt, not the chat."""
    return f"{SYSTEM}\n\nФРАГМЕНТИ БАЗИ ЗНАНЬ:\n{format_context(request.state.get('chunks', []))}"


class FallbackToHuman(AgentMiddleware):
    """On service failure: a safe fallback to a human, not silence.

    A failed model call becomes a create_ticket call, so it goes through the
    same human review. Broad on purpose: each provider raises its own errors.
    Both sync and async hooks are needed: the terminal runs the graph
    synchronously, `langgraph dev` asynchronously.
    """

    def wrap_model_call(self, request: ModelRequest, handler):
        try:
            return handler(request)
        except Exception as e:
            return self.fallback(request, e)

    async def awrap_model_call(self, request: ModelRequest, handler):
        try:
            return await handler(request)
        except Exception as e:
            return self.fallback(request, e)

    @staticmethod
    def fallback(request: ModelRequest, error: Exception) -> AIMessage:
        log.warning("Model call failed: %s", brief(error))
        results = [m for m in current_turn(request.state["messages"]) if isinstance(m, ToolMessage) and m.name == TICKET_TOOL]
        if results:
            # The ticket is already decided on; tell the customer instead of asking again
            if results[-1].status == "error":
                return AIMessage("Асистент тимчасово недоступний, тикет не створено. Спробуйте, будь ласка, пізніше.")
            return AIMessage(f"{results[-1].text} Асистент тимчасово недоступний, тож відповідь надасть оператор.")
        draft = {"subject": "Асистент недоступний", "description": question_of(request.state)}
        return AIMessage("", tool_calls=[{"name": TICKET_TOOL, "args": draft, "id": f"fallback-{uuid4().hex[:8]}"}])


def build_graph(
    model: BaseChatModel,
    kb: KnowledgeBase,
    tickets_path: Path,
    checkpointer: BaseCheckpointSaver | None = None,
):
    """interrupt() needs a checkpointer to keep the paused state: the terminal passes
    an in-memory one, while `langgraph dev` supplies its own and must get None.
    """

    def classify(state: SupportState) -> dict:
        try:
            # The tag keeps this call's tokens out of chat UIs; the result is not a reply
            routing = (
                model.with_structured_output(Routing)
                .with_config(tags=["langsmith:nostream"])
                .invoke([HumanMessage(f"Класифікуй звернення клієнта: «{question_of(state)}»")])
            )
        except Exception as e:
            # Classification only enriches the ticket; the answer does not depend on it
            log.warning("Classification failed: %s", brief(e))
            return {"routing": None}
        return {"routing": routing.model_dump()}

    def retrieve(state: SupportState) -> dict:
        return {"chunks": [{"source": c.source, "text": c.text} for c in kb.search(question_of(state))]}

    agent = create_agent(
        model,
        tools=[make_ticket_tool(tickets_path)],
        state_schema=SupportState,
        middleware=[
            grounded_prompt,
            FallbackToHuman(),
            # Human control point: the tool runs only after an operator decides
            HumanInTheLoopMiddleware(
                interrupt_on={
                    TICKET_TOOL: {"allowed_decisions": ["approve", "edit", "reject"], "description": describe_ticket}
                }
            ),
        ],
        name="agent",
    )

    graph = StateGraph(SupportState)
    graph.add_node("classify", classify)
    graph.add_node("retrieve", retrieve)
    graph.add_node("agent", agent)
    graph.add_edge(START, "classify")
    graph.add_edge(START, "retrieve")
    # Both branches finish in the same step, so the agent runs once. Not a join edge
    # (["classify", "retrieve"]): with it the LangGraph server drops the agent's pending
    # interrupt from the thread state, and Agent Chat UI shows no review card
    graph.add_edge("classify", "agent")
    graph.add_edge("retrieve", "agent")
    graph.add_edge("agent", END)
    return graph.compile(checkpointer=checkpointer)
