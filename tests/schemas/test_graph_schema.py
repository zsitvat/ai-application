import pytest

from src.schemas.graph_schema import AgentState, create_empty_application_attributes


def test_create_empty_application_attributes():
    attrs = create_empty_application_attributes()
    expected_keys = [
        "applicant_name",
        "phone_number",
        "email",
        "position_name",
        "position_id",
        "application_reason",
        "language_skills",
        "experience",
        "other_information",
    ]
    assert set(attrs.keys()) == set(expected_keys)
    assert all(v == "" for v in attrs.values())


def test_agent_state_dict_access():
    state = AgentState()
    # __getitem__
    assert state["next"] == ""
    # __setitem__
    state["next"] = "test_next"
    assert state.next == "test_next"
    # get
    assert state.get("next") == "test_next"
    assert state.get("nonexistent", "default") == "default"


@pytest.mark.skip(reason="AI-dependent functionality should be skipped.")
def test_agent_state_ai_related():
    pass


def test_application_identifier_schema():
    from src.schemas.graph_schema import ApplicationIdentifierSchema

    obj = ApplicationIdentifierSchema(tenantIdentifier=1, applicationIdentifier="app1")
    assert obj.tenantIdentifier == 1
    assert obj.applicationIdentifier == "app1"


def test_checkpointer_type_enum():
    from src.schemas.graph_schema import CheckpointerType

    assert CheckpointerType.MEMORY == "memory"
    assert CheckpointerType.REDIS == "redis"


def test_chain_model():
    from src.schemas.graph_schema import Chain
    from src.schemas.schema import Model

    chain = Chain(model=Model(), prompt_id="pid", description="desc")
    assert chain.prompt_id == "pid"
    assert chain.description == "desc"


def test_embedding_model():
    from src.schemas.graph_schema import Embedding
    from src.schemas.schema import Model

    emb = Embedding(model=Model())
    assert emb.search_type == "mmr"
    assert emb.search_kwargs["k"] == 4


def test_agent_model():
    from src.schemas.graph_schema import Agent, Chain
    from src.schemas.schema import Model

    chain = Chain(model=Model(), prompt_id="pid")
    agent = Agent(chain=chain)
    assert agent.enabled is True
    assert agent.tool_choice == "auto"


def test_topic_validator_config():
    from src.schemas.graph_schema import TopicValidatorConfig
    from src.schemas.schema import Model

    config = TopicValidatorConfig(
        model=Model(), allowed_topics=["A"], invalid_topics=["B"]
    )
    assert config.enabled is True
    assert "A" in config.allowed_topics


def test_personal_data_filter_config():
    from src.schemas.graph_schema import Chain, PersonalDataFilterConfig
    from src.schemas.schema import Model

    chain = Chain(model=Model(), prompt_id="pid")
    config = PersonalDataFilterConfig(chain=chain, sensitive_data_types=["email"])
    assert config.enabled is True
    assert config.mask_char == "*"


def test_extractor_config():
    from src.schemas.graph_schema import ExtractorConfig
    from src.schemas.schema import Model

    config = ExtractorConfig(model=Model(), prompt_id="pid")
    assert config.prompt_id == "pid"


def test_graph_config_minimal():
    from src.schemas.graph_schema import Agent, Chain, GraphConfig
    from src.schemas.schema import Model

    chain = Chain(model=Model(), prompt_id="pid")
    agent = Agent(chain=chain)
    config = GraphConfig(agents={"main": agent}, supervisor=agent)
    assert "main" in config.agents
    assert config.supervisor == agent


def test_application_attributes_typed_dict():
    from src.schemas.graph_schema import ApplicationAttributes

    attrs: ApplicationAttributes = {
        "applicant_name": "Test",
        "phone_number": "123",
        "email": "test@example.com",
        "position_name": "Dev",
        "position_id": "1",
        "application_reason": "Reason",
        "language_skills": "EN",
        "experience": "2 years",
        "other_information": "None",
    }
    assert isinstance(attrs, dict)
    assert attrs["applicant_name"] == "Test"


def test_agent_state_defaults():
    from src.schemas.graph_schema import AgentState

    state = AgentState()
    assert isinstance(state.messages, list)
    assert state.next == ""
    assert isinstance(state.context, dict)
    assert isinstance(state.parameters, dict)
    assert isinstance(state.application_attributes, dict)


def test_graph_config_all_options():
    from src.schemas.graph_schema import (
        Agent,
        Chain,
        CheckpointerType,
        ExtractorConfig,
        GraphConfig,
        PersonalDataFilterConfig,
        TopicValidatorConfig,
    )
    from src.schemas.schema import Model

    chain = Chain(model=Model(), prompt_id="pid")
    agent = Agent(chain=chain)
    topic_validator = TopicValidatorConfig(model=Model())
    personal_data_filter = PersonalDataFilterConfig(chain=chain)
    extractor = ExtractorConfig(model=Model(), prompt_id="pid")
    config = GraphConfig(
        agents={"main": agent},
        supervisor=agent,
        exception_chain=agent,
        topic_validator=topic_validator,
        personal_data_filter=personal_data_filter,
        applicant_attributes_extractor=extractor,
        max_input_length=100,
        chat_memory_db=1,
        chat_memory_index_name="index",
        chat_memory_type="redis",
        chat_history_max_length=20,
        tags=["tag1"],
        enable_checkpointer=False,
        checkpointer_type=CheckpointerType.REDIS,
        allow_supervisor_finish=True,
        recursion_limit=5,
    )
    assert config.exception_chain == agent
    assert config.topic_validator == topic_validator
    assert config.personal_data_filter == personal_data_filter
    assert config.applicant_attributes_extractor == extractor
    assert config.max_input_length == 100
    assert config.chat_memory_db == 1
    assert config.tags == ["tag1"]
    assert config.enable_checkpointer is False
    assert config.checkpointer_type == CheckpointerType.REDIS
    assert config.allow_supervisor_finish is True
    assert config.recursion_limit == 5
