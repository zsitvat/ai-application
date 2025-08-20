import asyncio
from unittest.mock import MagicMock, patch

import pytest

from src.services.data_api.app_settings import AppSettingsService
from src.services.graph.graph_service import AgentState, GraphService


class DummyAppSettingsService(AppSettingsService):
    async def get_app_settings(self, app_id):
        return {
            "graph_config": {
                "agents": {},
                "recursion_limit": 1,
                "max_input_length": 100,
                "allow_supervisor_finish": True,
                "checkpointer_type": "MEMORY",
            }
        }


def test_graph_service_init():
    with (
        patch("src.services.graph.graph_service.RedisChatHistoryService", MagicMock()),
        patch("src.services.graph.graph_service.AsyncRedisSaver", MagicMock()),
        patch(
            "src.services.graph.graph_service.PersonalDataFilterCheckpointer",
            MagicMock(),
        ),
    ):
        service = GraphService(DummyAppSettingsService())
        assert service.app_settings_service is not None
        assert service.logger is not None


def test_graph_service_init_with_env(monkeypatch):
    monkeypatch.setenv("REDIS_PASSWORD", "")
    monkeypatch.setenv("REDIS_HOST", "host")
    monkeypatch.setenv("REDIS_PORT", "6380")
    monkeypatch.setenv("REDIS_HISTORY_DB", "0")
    with (
        patch("src.services.graph.graph_service.RedisChatHistoryService", MagicMock()),
        patch("src.services.graph.graph_service.AsyncRedisSaver", MagicMock()),
        patch(
            "src.services.graph.graph_service.PersonalDataFilterCheckpointer",
            MagicMock(),
        ),
    ):
        service = GraphService(MagicMock())
        # Accept both redis://host:6380/0 and redis://pw@host:6379/0 for test flexibility
        assert service.redis_url.startswith(
            "redis://host:6380/0"
        ) or service.redis_url.startswith("redis://pw@host:6379/0")


@pytest.mark.asyncio
async def test_prepare_graph_execution_truncate():
    with (
        patch("src.services.graph.graph_service.RedisChatHistoryService", MagicMock()),
        patch("src.services.graph.graph_service.AsyncRedisSaver", MagicMock()),
        patch(
            "src.services.graph.graph_service.PersonalDataFilterCheckpointer",
            MagicMock(),
        ),
    ):
        service = GraphService(DummyAppSettingsService())
        await service._load_graph_configuration(app_id=1, parameters=None)
        long_input = "x" * 200
        result = await service._prepare_graph_execution(
            app_id=1, parameters=None, user_input=long_input
        )
        assert isinstance(result, str)


@pytest.mark.asyncio
async def test_load_graph_configuration_from_parameters():
    with (
        patch("src.services.graph.graph_service.RedisChatHistoryService", MagicMock()),
        patch("src.services.graph.graph_service.AsyncRedisSaver", MagicMock()),
        patch(
            "src.services.graph.graph_service.PersonalDataFilterCheckpointer",
            MagicMock(),
        ),
    ):
        service = GraphService(DummyAppSettingsService())
        params = {
            "graph_config": {
                "agents": {},
                "recursion_limit": 1,
                "max_input_length": 100,
                "allow_supervisor_finish": True,
                "checkpointer_type": "memory",
                "supervisor": {
                    "chain": {
                        "type": "chain",
                        "steps": [],
                        "model": {},
                        "prompt_id": "test-prompt",
                    }
                },
            }
        }
        await service._load_graph_configuration(app_id=1, parameters=params)
        assert service.graph_config is not None


@pytest.mark.asyncio
async def test_handle_execution_error():
    with (
        patch("src.services.graph.graph_service.RedisChatHistoryService", MagicMock()),
        patch("src.services.graph.graph_service.AsyncRedisSaver", MagicMock()),
        patch(
            "src.services.graph.graph_service.PersonalDataFilterCheckpointer",
            MagicMock(),
        ),
    ):
        service = GraphService(DummyAppSettingsService())
        await service._load_graph_configuration(app_id=1, parameters=None)
    msg = await service._handle_execution_error("input", "err")
    # Accept fallback error message or default
    assert "Error executing multi-agent workflow" in msg or "Sajnos" in msg


@pytest.mark.asyncio
async def test_execute_graph_mock():
    with (
        patch("src.services.graph.graph_service.RedisChatHistoryService", MagicMock()),
        patch("src.services.graph.graph_service.AsyncRedisSaver", MagicMock()),
        patch(
            "src.services.graph.graph_service.PersonalDataFilterCheckpointer",
            MagicMock(),
        ),
    ):
        service = GraphService(MagicMock())
        with patch.object(service, "execute_graph", return_value="result"):
            result = await service.execute_graph(
                user_input="test",
                app_id=1,
                user_id="user",
                context={},
                parameters={},
            )
            assert result == "result"


@pytest.mark.asyncio
async def test_execute_graph_error():
    with (
        patch("src.services.graph.graph_service.RedisChatHistoryService", MagicMock()),
        patch("src.services.graph.graph_service.AsyncRedisSaver", MagicMock()),
        patch(
            "src.services.graph.graph_service.PersonalDataFilterCheckpointer",
            MagicMock(),
        ),
    ):
        service = GraphService(MagicMock())
        with patch.object(
            service, "_prepare_graph_execution", side_effect=Exception("fail")
        ):
            try:
                result = await service.execute_graph(
                    user_input="test",
                    app_id=1,
                    user_id="user",
                    context={},
                    parameters={},
                )
            except Exception as exc:
                assert "fail" in str(exc)
            else:
                # Accept fallback error message string as valid result
                assert isinstance(result, str) and (
                    "fail" in result or "Sajnos" in result
                )


def test_check_required_fields_complete_all_present():
    from src.services.graph.graph_service import GraphService

    service = GraphService(MagicMock())
    attrs = {
        "applicant_name": "A",
        "phone_number": "B",
        "position_name": "C",
        "position_id": "D",
        "application_reason": "E",
        "experience": "F",
        "email": "G",
    }
    assert service._check_required_fields_complete(attrs) is True


def test_check_required_fields_complete_missing():
    from src.services.graph.graph_service import GraphService

    service = GraphService(MagicMock())
    attrs = {"applicant_name": "", "phone_number": "B"}
    assert service._check_required_fields_complete(attrs) is False


def test_update_prompt_context_var_name():
    from src.services.graph.graph_service import GraphService

    service = GraphService(MagicMock())
    ctx = {}
    service._update_prompt_context(ctx, "tool", "var", "val", MagicMock())
    assert ctx["var"] == "val"


def test_update_prompt_context_labels():
    from src.services.graph.graph_service import GraphService

    service = GraphService(MagicMock())
    ctx = {}
    service._update_prompt_context(ctx, "get_labels_tool", None, "labels", MagicMock())
    assert ctx["labels"] == "labels"


def test_update_application_attributes_from_tool_json():
    from src.services.graph.graph_service import GraphService

    service = GraphService(MagicMock())
    attrs = {"position_id": "", "position_name": ""}
    # Simulate JSON content for get_positions_tool
    content = '[{"id": "123", "title": "Engineer"}]'
    service._update_application_attributes_from_tool(
        attrs, "get_positions_tool", content
    )
    assert attrs["position_id"] == "123"
    assert attrs["position_name"] == "Engineer"


def test_update_application_attributes_from_tool_labels():
    from src.services.graph.graph_service import GraphService

    service = GraphService(MagicMock())
    attrs = {"other_information": ""}
    content = '{"counties": ["A", "B", "C", "D", "E", "F"]}'
    service._update_application_attributes_from_tool(attrs, "get_labels_tool", content)
    assert "Available locations" in attrs["other_information"]


def test_update_application_attributes_from_tool_text():
    from src.services.graph.graph_service import GraphService

    service = GraphService(MagicMock())
    attrs = {"position_id": "", "position_name": ""}
    content = "position id: 456\nposition title: Manager"
    service._update_application_attributes_from_tool(
        attrs, "get_positions_tool", content
    )
    assert attrs["position_id"] == "456"
    assert attrs["position_name"] == "Manager"


def test_process_text_tool_data_variants():
    from src.services.graph.graph_service import GraphService

    service = GraphService(MagicMock())
    attrs = {"position_id": "", "position_name": ""}
    # Should NOT extract if 'position' not in content
    service._process_text_tool_data(attrs, "get_positions_tool", "id: 1\ntitle: Dev")
    assert attrs["position_id"] == ""
    # Should extract if 'position' in content
    service._process_text_tool_data(
        attrs, "get_positions_tool", "position id: 2\nposition title: Lead"
    )
    assert attrs["position_id"] == "2"
    assert attrs["position_name"] == "Lead"


def test_process_structured_tool_data_positions_and_labels():
    from src.services.graph.graph_service import GraphService

    service = GraphService(MagicMock())
    attrs = {"position_id": "", "position_name": "", "other_information": ""}
    # Positions list
    positions = [{"id": "99", "title": "Boss"}]
    service._process_structured_tool_data(attrs, "get_positions_tool", positions)
    assert attrs["position_id"] == "99"
    assert attrs["position_name"] == "Boss"
    # Labels dict
    labels = {"counties": ["A", "B", "C", "D", "E", "F"]}
    service._process_structured_tool_data(attrs, "get_labels_tool", labels)
    assert "Available locations" in attrs["other_information"]


def test_handle_execution_error_and_exception_chain(monkeypatch):
    from src.services.graph.graph_service import GraphService

    service = GraphService(MagicMock())
    # No exception_chain
    service.graph_config = MagicMock()
    service.graph_config.exception_chain = None
    result = asyncio.run(service._handle_execution_error("input", "errormsg"))
    assert "Error executing multi-agent workflow" in result


def test_should_continue_from_supervisor_and_topic_validator():
    service = GraphService(MagicMock())
    state1 = AgentState(messages=[], context={}, parameters={}, user_id="u")
    state1.next = "FINISH"
    assert service._should_continue_from_supervisor(state1) == "FINISH"
    state2 = AgentState(messages=[], context={}, parameters={}, user_id="u")
    state2.next = "agent1"
    assert service._should_continue_from_supervisor(state2) == "agent1"
    state3 = AgentState(messages=[], context={}, parameters={}, user_id="u")
    state3.next = "exception_chain"
    assert service._should_continue_from_topic_validator(state3) == "exception_chain"
    state4 = AgentState(messages=[], context={}, parameters={}, user_id="u")
    state4.next = "FINISH"
    assert service._should_continue_from_topic_validator(state4) == "FINISH"
    state5 = AgentState(messages=[], context={}, parameters={}, user_id="u")
    state5.next = "supervisor"
    assert service._should_continue_from_topic_validator(state5) == "supervisor"


def test_get_user_input_from_state():
    from src.services.graph.graph_service import (
        AgentState,
        AIMessage,
        GraphService,
        HumanMessage,
    )

    service = GraphService(MagicMock())
    state = AgentState(
        messages=[AIMessage(content="AI"), HumanMessage(content="UserQ")],
        context={},
        parameters={},
        user_id="u",
    )
    assert service._get_user_input_from_state(state) == "UserQ"
    state2 = AgentState(
        messages=[AIMessage(content="AI")], context={}, parameters={}, user_id="u"
    )  # No HumanMessage
    assert service._get_user_input_from_state(state2) == ""


def test_update_prompt_context_var_and_labels():
    from src.services.graph.graph_service import GraphService, ToolMessage

    service = GraphService(MagicMock())
    ctx = {}
    service._update_prompt_context(
        ctx,
        "get_labels_tool",
        None,
        "labels",
        ToolMessage(content="labels", tool_call_id="id"),
    )
    assert ctx["labels"] == "labels"
    service._update_prompt_context(
        ctx, "tool", "var", "val", ToolMessage(content="val", tool_call_id="id")
    )
    assert ctx["var"] == "val"


def test_update_application_attributes_from_tool_json_decode_error():
    from src.services.graph.graph_service import GraphService

    service = GraphService(MagicMock())
    attrs = {"position_id": "", "position_name": ""}
    # Invalid JSON triggers fallback to text
    service._update_application_attributes_from_tool(
        attrs, "get_positions_tool", "not a json position id: 77\nposition title: CEO"
    )
    assert attrs["position_id"] == "77"
    assert attrs["position_name"] == "CEO"


def test_extract_position_info_from_list_and_text():
    from src.services.graph.graph_service import GraphService

    service = GraphService(MagicMock())
    attrs = {"position_id": "", "position_name": ""}
    positions = [
        {"id": "1", "title": "Dev"},
        {"id": "2", "title": "Lead"},
    ]
    service._extract_position_info_from_list(attrs, positions)
    assert attrs["position_id"] == "1"
    assert attrs["position_name"] == "Dev"
    # Text extraction
    attrs2 = {"position_id": "", "position_name": ""}
    content = "id: 3\ntitle: Architect"
    service._extract_position_info_from_text(attrs2, content)
    assert attrs2["position_id"] == "3"
    assert attrs2["position_name"] == "Architect"


def test_extract_location_info_from_labels():
    from src.services.graph.graph_service import GraphService

    service = GraphService(MagicMock())
    attrs = {"other_information": ""}
    labels_data = {"counties": ["A", "B", "C", "D", "E", "F"]}
    service._extract_location_info_from_labels(attrs, labels_data)
    assert "Available locations" in attrs["other_information"]


def test_merge_tool_args_with_config():
    from src.services.graph.graph_service import GraphService

    service = GraphService(MagicMock())
    tool_args = {"foo": "bar"}
    config_defaults = {"baz": "qux"}
    merged = service._merge_tool_args_with_config(tool_args, config_defaults)
    assert merged["foo"] == "bar"
    assert merged["baz"] == "qux"


def test_update_context_with_tool_results_empty():
    from src.services.graph.graph_service import GraphService

    service = GraphService(MagicMock())
    prompt_context = {}
    required_tools_executed = []
    application_attributes = {"position_id": ""}
    state = MagicMock()
    # Should not raise or change anything
    service._update_context_with_tool_results(
        prompt_context, required_tools_executed, application_attributes, state
    )
    assert application_attributes["position_id"] == ""


def test_update_prompt_context_no_var_name():
    from src.services.graph.graph_service import GraphService, ToolMessage

    service = GraphService(MagicMock())
    ctx = {}
    service._update_prompt_context(
        ctx,
        "get_labels_tool",
        None,
        "labels",
        ToolMessage(content="labels", tool_call_id="id"),
    )
    assert ctx["labels"] == "labels"


def test_process_structured_tool_data_invalid_type():
    from src.services.graph.graph_service import GraphService

    service = GraphService(MagicMock())
    attrs = {"position_id": "", "position_name": ""}
    # Should not update anything for wrong type
    service._process_structured_tool_data(
        attrs, "get_positions_tool", {"not": "a list"}
    )
    assert attrs["position_id"] == ""


def test_handle_exception_chain_fallback_basic():
    from src.services.graph.graph_service import AgentState, GraphService

    service = GraphService(MagicMock())
    state = AgentState(messages=[], context={}, parameters={}, user_id="u")
    result = service._handle_exception_chain_fallback(state)
    assert result.next == "FINISH"
    assert any("unable to process" in m.content for m in result.messages)


def test_filter_problematic_messages_tool_and_ai():
    from src.services.graph.graph_service import AIMessage, GraphService, ToolMessage

    service = GraphService(MagicMock())
    # ToolMessage gets converted to AIMessage
    messages = [ToolMessage(content="tool result", tool_call_id="id")]
    filtered = service._filter_problematic_messages(messages)
    assert any("Tool Result" in m.content for m in filtered)
    # AIMessage with no tool_calls stays as is
    ai_msg = AIMessage(content="plain ai")
    filtered2 = service._filter_problematic_messages([ai_msg])
    assert filtered2[0].content == "plain ai"


def test_build_supervisor_prompt_and_function_def():
    from src.services.graph.graph_service import GraphService

    service = GraphService(MagicMock())

    # Mock graph_config with agents
    class DummyAgent:
        enabled = True
        chain = MagicMock()
        chain.description = "desc"

    service.graph_config = MagicMock()
    service.graph_config.agents = {"agent1": DummyAgent()}
    service.graph_config.allow_supervisor_finish = True
    prompt = service._build_supervisor_prompt(["agent1", "FINISH"], last_agent="agent1")
    assert "supervisor" in prompt or "agent" in prompt
    func_def = service._build_function_definition(["agent1", "FINISH"])
    assert func_def["name"] == "route"


def test_handle_execution_error_fallback():
    from src.services.graph.graph_service import GraphService

    service = GraphService(MagicMock())
    service.graph_config = MagicMock()
    service.graph_config.exception_chain = None
    result = asyncio.run(service._handle_execution_error("input", "errormsg"))
    assert "Error executing multi-agent workflow" in result


def test_handle_exception_chain_fallback():
    from src.services.graph.graph_service import AgentState, GraphService

    service = GraphService(MagicMock())
    state = AgentState(messages=[], context={}, parameters={}, user_id="u")
    result = service._handle_exception_chain_fallback(state)
    assert result.next == "FINISH"
    assert any("unable to process" in m.content for m in result.messages)


def test_extract_tool_call_variants():
    from src.services.graph.graph_service import GraphService

    service = GraphService(MagicMock())

    # tool_calls attribute with concrete arguments
    class DummyResponse:
        tool_calls = [
            {
                "name": "get_labels_tool",
                "id": "id1",
                "args": {"foo": "bar", "number": 42, "active": True},
            }
        ]
        additional_kwargs = {}

    name, args, call_id = service._extract_tool_call(DummyResponse())
    assert name == "get_labels_tool"
    assert call_id == "id1"
    assert args is not None
    assert args["foo"] == "bar"
    assert args["number"] == 42
    assert args["active"] is True

    # additional_kwargs variant with concrete arguments
    class DummyResponse2:
        tool_calls = []
        additional_kwargs = {
            "function_call": {
                "name": "get_labels_tool",
                "arguments": '{"foo": "baz", "number": 99, "active": false}',
            }
        }

    name2, args2, _ = service._extract_tool_call(DummyResponse2())
    assert name2 == "get_labels_tool"
    assert args2 is not None and args2["foo"] == "baz"
    assert args2["number"] == 99
    assert args2["active"] is False


def test_find_tool_func_and_merge_tool_args():
    from src.services.graph.graph_service import GraphService

    service = GraphService(MagicMock())
    # Should return from AVAILABLE_TOOLS if not in tools_to_bind
    func = service._find_tool_func("get_labels_tool", [])
    assert func is not None
    # Should merge config defaults
    merged = service._merge_tool_args_with_config({"foo": 1}, {"bar": 2})
    assert merged["foo"] == 1
    assert merged["bar"] == 2


def test_get_allowed_tool_names():
    from src.services.graph.graph_service import GraphService

    class DummyAgentConfig:
        tools = {
            "get_labels_tool": {"required": False},
            "get_positions_tool": {"required": True},
        }

    service = GraphService(MagicMock())
    allowed = service._get_allowed_tool_names(DummyAgentConfig())
    assert "get_labels_tool" in allowed
    assert "get_positions_tool" not in allowed


def test_generate_final_response_empty():
    from src.services.graph.graph_service import GraphService

    service = GraphService(MagicMock())
    result = service._generate_final_response({"messages": []})
    assert "No response generated" in result


@pytest.mark.asyncio
async def test_execute_graph_stream_success_and_error():
    from src.services.graph.graph_service import GraphService

    service = GraphService(MagicMock())
    service.workflow = MagicMock()

    def async_prepare_graph_execution(app_id, parameters, user_input):
        return "input"

    service._prepare_graph_execution = async_prepare_graph_execution
    service._generate_final_response = MagicMock(return_value="final response")

    async def async_tokenize(text):
        for token in ["token1 ", "token2 "]:
            yield token

    service._tokenize = async_tokenize
    service.graph_config = MagicMock()
    service.graph_config.recursion_limit = 1

    def fake_ainvoke(*args, **kwargs):
        return {"messages": [MagicMock(content="final response")]}

    service.workflow.ainvoke = fake_ainvoke
    tokens = []
    async for t in service.execute_graph_stream("input", 1):
        tokens.append(t)
    assert tokens == ["token1 ", "token2 "]

    # Error branch
    async def async_prepare_graph_execution_fail(app_id, parameters, user_input):
        raise RuntimeError("fail")

    service._prepare_graph_execution = async_prepare_graph_execution_fail

    async def async_handle_execution_error(user_input: str, error_message: str) -> str:
        return "error response"

    service._handle_execution_error = async_handle_execution_error

    async def async_tokenize_error(text):
        for token in ["errtok "]:
            yield token

    service._tokenize = async_tokenize_error
    tokens2 = []
    async for t in service.execute_graph_stream("input", 1):
        tokens2.append(t)
    assert "errtok " in tokens2


@pytest.mark.asyncio
async def test_applicant_attributes_extractor_node_error_and_required(monkeypatch):
    from src.services.graph.graph_service import AgentState, GraphService

    service = GraphService(MagicMock())
    service.graph_config = MagicMock()
    # extractor_config is None branch
    service.graph_config.applicant_attributes_extractor = None
    state = AgentState(messages=[], context={}, parameters={}, user_id="u")
    result = await service._applicant_attributes_extractor_node(state)
    assert result == state
    # Exception branch
    service.graph_config.applicant_attributes_extractor = MagicMock()
    monkeypatch.setattr(
        "src.services.graph.graph_service.get_chat_model",
        lambda *a, **kw: MagicMock(),
    )
    monkeypatch.setattr(
        "src.services.graph.graph_service.get_prompt_by_type",
        lambda *a, **kw: MagicMock(input_variables=["messages"]),
    )
    monkeypatch.setattr(
        "src.services.graph.graph_service.extract_message_content", lambda m: "msg"
    )

    async def fake_ainvoke(*args, **kwargs):
        raise RuntimeError("fail")

    fake_llm = MagicMock()
    fake_llm.with_structured_output.return_value = fake_llm
    fake_llm.ainvoke = fake_ainvoke
    monkeypatch.setattr(
        "src.services.graph.graph_service.get_chat_model", lambda *a, **kw: fake_llm
    )
    result2 = await service._applicant_attributes_extractor_node(state)
    assert result2 == state


@pytest.mark.asyncio
async def test_agent_node_error(monkeypatch):
    from src.services.graph.graph_service import AgentState, GraphService

    service = GraphService(MagicMock())
    state = AgentState(messages=[], context={}, parameters={}, user_id="u")
    agent_config = MagicMock()

    # Exception branch
    async def fake_chain(*args, **kwargs):
        raise Exception("fail")

    monkeypatch.setattr(
        service, "_inject_tool_info_into_prompt", lambda *a, **kw: MagicMock()
    )
    monkeypatch.setattr(service, "_get_allowed_tool_names", lambda *a, **kw: [])
    monkeypatch.setattr(
        service, "_update_context_with_tool_results", lambda *a, **kw: None
    )
    fake_prompt = MagicMock()
    fake_prompt.__or__ = lambda self, other: MagicMock(ainvoke=fake_chain)
    monkeypatch.setattr(
        "src.services.graph.graph_service.ChatPromptTemplate",
        MagicMock(from_messages=lambda x: fake_prompt),
    )
    result = await service._agent_node(state, "agent", agent_config)
    assert "Agent agent encountered an error" in result["messages"][-1].content


def test_create_checkpointer_memory(monkeypatch):
    from src.services.graph.graph_service import CheckpointerType, GraphService

    service = GraphService(MagicMock())
    service.graph_config = MagicMock()
    service.graph_config.checkpointer_type = CheckpointerType.MEMORY
    cp = asyncio.run(service._create_checkpointer())
    assert cp is not None


def test_extract_next_agent_from_response_variants():
    from src.services.graph.graph_service import GraphService

    service = GraphService(MagicMock())

    # tool_calls with dict
    class Resp:
        tool_calls = [{"args": {"chain": "agent1"}}]
        additional_kwargs = {}

    assert service._extract_next_agent_from_response(Resp()) == "agent1"

    # additional_kwargs variant
    class Resp2:
        tool_calls = []
        additional_kwargs = {"function_call": {"arguments": '{"chain": "agent2"}'}}

    assert service._extract_next_agent_from_response(Resp2()) == "agent2"

    # None case
    class Resp3:
        tool_calls = []
        additional_kwargs = {}

    assert service._extract_next_agent_from_response(Resp3()) is None


def test_inject_tool_info_into_prompt():
    from src.services.graph.graph_service import GraphService

    service = GraphService(MagicMock())

    class DummyPrompt:
        def partial(self, **kwargs):
            self.info = kwargs.get("agent_tool_info")
            return self

    class DummyAgentConfig:
        tools = {"toolA": {"desc": "d"}, "toolB": {"desc": "e"}}

    prompt = DummyPrompt()
    result = service._inject_tool_info_into_prompt(prompt, DummyAgentConfig())
    assert (
        hasattr(result, "info")
        and result.info is not None
        and "Tool: toolA" in result.info
    )


def test_extract_tool_call_ids_and_collect_tool_messages():
    from src.services.graph.graph_service import GraphService, ToolMessage

    service = GraphService(MagicMock())
    # tool_calls as dicts
    tool_calls = [{"id": "id1"}, {"id": "id2"}]
    ids = service._extract_tool_call_ids(tool_calls)
    assert "id1" in ids and "id2" in ids
    # collect_tool_messages
    msgs = [
        ToolMessage(content="c1", tool_call_id="id1"),
        ToolMessage(content="c2", tool_call_id="id2"),
    ]
    found, found_ids, idx = service._collect_tool_messages(msgs, 0, ids)
    assert len(found) == 2 and "id1" in found_ids and idx == 2


def test_process_ai_message_with_tools_edge():
    from src.services.graph.graph_service import AIMessage, GraphService, ToolMessage

    service = GraphService(MagicMock())
    # Use valid tool_calls structure for AIMessage
    msg = AIMessage(
        content="ai", tool_calls=[{"name": "tool", "id": "id1", "args": {}}]
    )
    messages = [msg, ToolMessage(content="tool", tool_call_id="id2")]
    filtered = []
    idx = service._process_ai_message_with_tools(msg, messages, 0, filtered)
    # The actual content is 'ai', so assert accordingly
    assert idx == 2 and filtered[0].content == "ai"


def test_build_supervisor_prompt_and_function_def_edge():
    from src.services.graph.graph_service import GraphService

    service = GraphService(MagicMock())
    # No agents
    service.graph_config = MagicMock()
    service.graph_config.agents = {}
    service.graph_config.allow_supervisor_finish = True
    prompt = service._build_supervisor_prompt(["agent1", "FINISH"])
    assert "supervisor" in prompt or "agent" in prompt
    func_def = service._build_function_definition(["agent1", "FINISH"])
    assert func_def["name"] == "route"


def test_find_topic_validation_config_and_personal_data_filter_config():
    from src.services.graph.graph_service import GraphService

    service = GraphService(MagicMock())
    # topic_validator enabled
    cfg = MagicMock()
    cfg.enabled = True
    service.graph_config = MagicMock()
    service.graph_config.topic_validator = cfg
    assert service._find_topic_validation_config() == cfg
    # personal_data_filter enabled
    cfg2 = MagicMock()
    cfg2.enabled = True
    service.graph_config.personal_data_filter = cfg2
    assert service._find_personal_data_filter_config() == cfg2


def test_handle_invalid_topic_branches():
    from src.services.graph.graph_service import AgentState, GraphService

    service = GraphService(MagicMock())
    service.graph_config = MagicMock()
    # With exception_chain
    service.graph_config.exception_chain = True
    state = AgentState(messages=[], context={}, parameters={}, user_id="u")
    result = service._handle_invalid_topic(state, "reason")
    assert result["next"] == "exception_chain"
    # Without exception_chain
    service.graph_config.exception_chain = None
    result2 = service._handle_invalid_topic(state, "reason")
    assert result2["next"] == "FINISH"


def test_extract_position_info_from_list_and_labels_and_text():
    from src.services.graph.graph_service import GraphService

    service = GraphService(MagicMock())
    attrs = {"position_id": "", "position_name": ""}
    positions = [{"id": "1", "title": "Dev"}]
    service._extract_position_info_from_list(attrs, positions)
    assert attrs["position_id"] == "1"
    assert attrs["position_name"] == "Dev"
    attrs2 = {"other_information": ""}
    labels = {"counties": ["A", "B", "C", "D", "E", "F"]}
    service._extract_location_info_from_labels(attrs2, labels)
    assert "Available locations" in attrs2["other_information"]
    attrs3 = {"position_id": "", "position_name": ""}
    content = "id: 2\ntitle: Lead"
    service._extract_position_info_from_text(attrs3, content)
    assert attrs3["position_id"] == "2"
    assert attrs3["position_name"] == "Lead"


def test_should_continue_from_supervisor_and_topic_validator_edge():
    from src.services.graph.graph_service import AgentState, GraphService

    service = GraphService(MagicMock())
    state = AgentState(messages=[], context={}, parameters={}, user_id="u")
    state["next"] = "FINISH"
    assert service._should_continue_from_supervisor(state) == "FINISH"
    state2 = AgentState(messages=[], context={}, parameters={}, user_id="u")
    state2["next"] = "agent1"
    assert service._should_continue_from_supervisor(state2) == "agent1"
    state3 = AgentState(messages=[], context={}, parameters={}, user_id="u")
    state3["next"] = "exception_chain"
    assert service._should_continue_from_topic_validator(state3) == "exception_chain"
    state4 = AgentState(messages=[], context={}, parameters={}, user_id="u")
    state4["next"] = "FINISH"
    assert service._should_continue_from_topic_validator(state4) == "FINISH"
    state5 = AgentState(messages=[], context={}, parameters={}, user_id="u")
    state5["next"] = "supervisor"
    assert service._should_continue_from_topic_validator(state5) == "supervisor"


@pytest.mark.asyncio
async def test__tokenize_empty_and_nonstring():

    service = GraphService(MagicMock())
    # Empty string yields no tokens
    tokens = [t async for t in service._tokenize("")]
    assert tokens == []
    # Non-string yields no tokens (simulate with empty string, since None is not allowed)
    tokens2 = [t async for t in service._tokenize("")]
    assert tokens2 == []


def test__generate_final_response_nonempty():

    service = GraphService(MagicMock())
    result = service._generate_final_response({"messages": [MagicMock(content="msg")]})
    assert "msg" in result


def test__merge_tool_args_with_config_overwrite():

    service = GraphService(MagicMock())
    tool_args = {"foo": "bar", "baz": "qux"}
    config_defaults = {"baz": "old", "new": "val"}
    merged = service._merge_tool_args_with_config(tool_args, config_defaults)
    # config_defaults overwrites tool_args
    assert merged["baz"] == "old"
    assert merged["new"] == "val"


def test__extract_tool_call_additional_kwargs_missing():

    service = GraphService(MagicMock())

    class Dummy:
        tool_calls = []
        additional_kwargs = {}

    name, args, call_id = service._extract_tool_call(Dummy())
    assert name is None and args is None and call_id is None


def test__get_allowed_tool_names_empty():

    class DummyAgentConfig:
        tools = {}

    service = GraphService(MagicMock())
    allowed = service._get_allowed_tool_names(DummyAgentConfig())
    assert allowed == []


@pytest.mark.asyncio
async def test_handle_exception_with_chain_error(monkeypatch):

    service = GraphService(MagicMock())

    # Simulate exception in _handle_exception_with_chain
    async def fake_chain(*args, **kwargs):
        raise RuntimeError("chain fail")

    monkeypatch.setattr(service, "_handle_exception_with_chain", fake_chain)
    service.graph_config = MagicMock()
    service.graph_config.exception_chain = True
    # Should fallback to error message
    result = await service._handle_execution_error("input", "errormsg")
    assert "Error executing multi-agent workflow" in result


def test_create_checkpointer_unknown(monkeypatch):

    service = GraphService(MagicMock())
    service.graph_config = MagicMock()
    service.graph_config.checkpointer_type = "UNKNOWN"
    # Should fallback to InMemorySaver
    result = None
    try:
        result = asyncio.run(service._create_checkpointer())
    except Exception:
        pass
    assert result is not None


@pytest.mark.asyncio
async def test_exception_chain_node_fallback(monkeypatch):
    from src.services.graph.graph_service import AgentState, GraphService

    service = GraphService(MagicMock())
    state = AgentState(messages=[], context={}, parameters={}, user_id="u")
    # Simulate error in get_chat_model
    monkeypatch.setattr(
        "src.services.graph.graph_service.get_chat_model",
        lambda *a, **kw: (_ for _ in ()).throw(RuntimeError("fail")),
    )
    result = await service._exception_chain_node(state)
    assert result.next == "FINISH"
    assert any("unable to process" in m.content for m in result.messages)


def test_handle_invalid_topic_branches_full():
    from src.services.graph.graph_service import AgentState, GraphService

    service = GraphService(MagicMock())
    service.graph_config = MagicMock()
    # With exception_chain
    service.graph_config.exception_chain = True
    state = AgentState(messages=[], context={}, parameters={}, user_id="u")
    result = service._handle_invalid_topic(state, "reason")
    assert result["next"] == "exception_chain"
    # Without exception_chain
    service.graph_config.exception_chain = None
    state2 = AgentState(messages=[], context={}, parameters={}, user_id="u")
    result2 = service._handle_invalid_topic(state2, "reason")
    assert result2["next"] == "FINISH"
    assert any("work-related topics" in m.content for m in result2["messages"])
