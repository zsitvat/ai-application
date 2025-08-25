import pydantic_core
import pytest

from src.services.graph.tools.issue_tracker_tool import issue_tracker_tool


@pytest.mark.asyncio
async def test_issue_tracker_tool_validation():
    # Should raise validation error for missing required fields
    with pytest.raises(pydantic_core.ValidationError):
        await issue_tracker_tool.ainvoke({"issue": "", "priority": "", "category": ""})


@pytest.mark.asyncio
async def test_issue_tracker_tool_success(monkeypatch):
    # Patch httpx.AsyncClient to simulate a successful response
    class DummyResponse:
        status_code = 200
        text = "ok"

    class DummyClient:
        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            pass

        async def post(self, url, json):
            return DummyResponse()

    monkeypatch.setattr("httpx.AsyncClient", lambda timeout: DummyClient())
    result = await issue_tracker_tool.ainvoke(
        {"issue": "desc", "priority": "high", "category": "bug_report"}
    )
    assert "Issue saved and submitted" in result
