"""Offline tests: a fake chat model stands in for the provider, no API key needed."""

import asyncio
import json
from pathlib import Path

import pytest
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.runnables import RunnableLambda
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

from support_assistant import cli
from support_assistant.config import load_model
from support_assistant.graph import Routing, build_graph
from support_assistant.kb import KnowledgeBase

KB_DIR = Path(__file__).resolve().parents[1] / "kb"
ROUTING = Routing(topic="billing", urgency=3, language="uk")
APPROVE = {"decisions": [{"type": "approve"}]}
REJECT = {"decisions": [{"type": "reject"}]}


class FakeModel(BaseChatModel):
    """Returns scripted messages in order and records the prompts it received."""

    responses: list[AIMessage] = []
    prompts: list[list] = []
    fail: bool = False

    @property
    def _llm_type(self) -> str:
        return "fake"

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        if self.fail:
            raise RuntimeError("service unavailable")
        self.prompts.append(messages)
        return ChatResult(generations=[ChatGeneration(message=self.responses.pop(0))])

    def bind_tools(self, tools, **kwargs):
        return self

    def with_structured_output(self, schema, **kwargs):
        def parse(_):
            if self.fail:
                raise RuntimeError("service unavailable")
            return ROUTING

        return RunnableLambda(parse)


def ticket_call(subject="Повернення коштів", description="Клієнт просить повернути оплату"):
    return AIMessage(
        content="",
        tool_calls=[{"name": "create_ticket", "args": {"subject": subject, "description": description}, "id": "call_1"}],
    )


@pytest.fixture
def kb():
    return KnowledgeBase.from_dir(KB_DIR)


@pytest.fixture
def tickets(tmp_path):
    return tmp_path / "tickets.jsonl"


def make_graph(model, kb, tickets):
    return build_graph(model, kb, tickets, checkpointer=InMemorySaver())


def config():
    return {"configurable": {"thread_id": "test"}}


def ask(graph, question):
    return graph.invoke({"messages": [HumanMessage(question)]}, config())


def resume(graph, decision):
    return graph.invoke(Command(resume=decision), config())


def saved_tickets(tickets):
    return [json.loads(line) for line in tickets.read_text(encoding="utf-8").splitlines()] if tickets.exists() else []


def test_search_finds_password_article(kb):
    sources = [c.source for c in kb.search("Не приходить лист для скидання пароля")]
    assert sources and sources[0].startswith("password-reset.md")


def test_search_returns_nothing_for_unrelated_query(kb):
    assert kb.search("xyzzy") == []


def test_answers_from_knowledge_base(kb, tickets):
    model = FakeModel(responses=[AIMessage("Натисніть «Забули пароль?» [password-reset.md]")])

    result = ask(make_graph(model, kb, tickets), "Як скинути пароль?")

    assert result["messages"][-1].text.startswith("Натисніть")
    assert result["routing"] == ROUTING.model_dump()
    assert "__interrupt__" not in result
    assert saved_tickets(tickets) == []


def test_fragments_go_to_system_prompt_not_chat(kb, tickets):
    model = FakeModel(responses=[AIMessage("Відповідь")])

    result = ask(make_graph(model, kb, tickets), "Як скинути пароль?")

    system = model.prompts[0][0]
    assert isinstance(system, SystemMessage) and "password-reset.md" in system.text
    assert all("password-reset.md" not in m.text for m in result["messages"] if isinstance(m, HumanMessage))


def test_escalation_waits_for_human_in_hitl_format(kb, tickets):
    paused = ask(make_graph(FakeModel(responses=[ticket_call()]), kb, tickets), "Поверніть гроші за підписку")

    request = paused["__interrupt__"][0].value
    assert request["action_requests"][0]["name"] == "create_ticket"
    assert request["action_requests"][0]["args"]["subject"] == "Повернення коштів"
    assert "billing" in request["action_requests"][0]["description"]
    assert request["review_configs"][0]["allowed_decisions"] == ["approve", "edit", "reject"]
    assert saved_tickets(tickets) == [], "nothing is created before the human decides"


def test_approved_ticket_carries_routing_and_model_replies(kb, tickets):
    model = FakeModel(responses=[ticket_call(), AIMessage("Ваше звернення передано оператору.")])
    graph = make_graph(model, kb, tickets)
    ask(graph, "Поверніть гроші за підписку")

    result = resume(graph, APPROVE)

    [ticket] = saved_tickets(tickets)
    assert ticket["routing"]["topic"] == "billing"
    tool_result = next(m for m in result["messages"] if isinstance(m, ToolMessage))
    assert ticket["id"] in tool_result.text
    assert result["messages"][-1].text == "Ваше звернення передано оператору."


def test_operator_can_edit_ticket(kb, tickets):
    model = FakeModel(responses=[ticket_call(), AIMessage("Передано.")])
    graph = make_graph(model, kb, tickets)
    ask(graph, "Поверніть гроші за підписку")
    edited = {"name": "create_ticket", "args": {"subject": "Повернення за березень", "description": "Перевірити оплату"}}

    resume(graph, {"decisions": [{"type": "edit", "edited_action": edited}]})

    assert saved_tickets(tickets)[0]["subject"] == "Повернення за березень"


def test_rejected_ticket_creates_nothing(kb, tickets):
    model = FakeModel(responses=[ticket_call(), AIMessage("Тикет не створено.")])
    graph = make_graph(model, kb, tickets)
    ask(graph, "Поверніть гроші за підписку")

    result = resume(graph, REJECT)

    assert saved_tickets(tickets) == []
    tool_result = next(m for m in result["messages"] if isinstance(m, ToolMessage))
    assert tool_result.status == "error"
    assert result["messages"][-1].text == "Тикет не створено."


def test_next_question_in_thread_starts_clean(kb, tickets):
    model = FakeModel(responses=[ticket_call(), AIMessage("Передано."), AIMessage("Натисніть «Забули пароль?»")])
    graph = make_graph(model, kb, tickets)
    ask(graph, "Поверніть гроші за підписку")
    resume(graph, APPROVE)

    result = ask(graph, "Як скинути пароль?")

    assert "__interrupt__" not in result
    assert result["messages"][-1].text.startswith("Натисніть")
    assert any(c["source"].startswith("password-reset.md") for c in result["chunks"])


def test_service_failure_goes_to_human_review(kb, tickets, caplog):
    paused = ask(make_graph(FakeModel(fail=True), kb, tickets), "Як скинути пароль?")

    assert paused["routing"] is None
    assert "Classification failed" in caplog.text, "a failed node must not stay silent"
    args = paused["__interrupt__"][0].value["action_requests"][0]["args"]
    assert args == {"subject": "Асистент недоступний", "description": "Як скинути пароль?"}


def test_async_run_matches_sync(kb, tickets):
    """`langgraph dev` runs the graph asynchronously: every middleware needs an async path."""
    graph = make_graph(FakeModel(fail=True), kb, tickets)

    paused = asyncio.run(graph.ainvoke({"messages": [HumanMessage("Як скинути пароль?")]}, config()))
    result = asyncio.run(graph.ainvoke(Command(resume=APPROVE), config()))

    assert paused["__interrupt__"]
    assert saved_tickets(tickets)[0]["id"] in result["messages"][-1].text


def test_service_failure_approved_ticket_is_reported(kb, tickets):
    graph = make_graph(FakeModel(fail=True), kb, tickets)
    ask(graph, "Як скинути пароль?")

    result = resume(graph, APPROVE)

    [ticket] = saved_tickets(tickets)
    assert ticket["id"] in result["messages"][-1].text
    assert "__interrupt__" not in result, "the fallback must not ask for a second ticket"


def test_service_failure_rejected_ticket_is_reported(kb, tickets):
    graph = make_graph(FakeModel(fail=True), kb, tickets)
    ask(graph, "Як скинути пароль?")

    result = resume(graph, REJECT)

    assert saved_tickets(tickets) == []
    assert "тикет не створено" in result["messages"][-1].text


def test_cli_run_prints_answer_and_tokens(kb, tickets, capsys):
    reply = AIMessage(
        "Натисніть «Забули пароль?»",
        usage_metadata={"input_tokens": 120, "output_tokens": 30, "total_tokens": 150},
        response_metadata={"model_name": "fake"},
    )
    cli.run(make_graph(FakeModel(responses=[reply]), kb, tickets), "Як скинути пароль?")

    out = capsys.readouterr().out
    assert "тема=billing" in out
    assert "Натисніть «Забули пароль?»" in out
    assert "вхід 120, вихід 30" in out


def test_cli_run_asks_operator(kb, tickets, monkeypatch, capsys):
    monkeypatch.setattr("builtins.input", lambda _: "y")
    model = FakeModel(responses=[ticket_call(), AIMessage("Ваше звернення передано оператору.")])
    cli.run(make_graph(model, kb, tickets), "Поверніть гроші")

    out = capsys.readouterr().out
    assert "Тема: Повернення коштів" in out
    assert "[інструмент] Тикет" in out
    assert "передано оператору" in out
    assert len(saved_tickets(tickets)) == 1


def test_model_gets_timeout_and_retry_limits(monkeypatch):
    monkeypatch.setenv("GOOGLE_API_KEY", "test-key")
    monkeypatch.setenv("MODEL_TIMEOUT", "15")
    monkeypatch.delenv("MODEL_MAX_RETRIES", raising=False)

    model = load_model("google_genai:gemini-3.5-flash-lite")

    assert (model.timeout, model.max_retries) == (15, 2)
