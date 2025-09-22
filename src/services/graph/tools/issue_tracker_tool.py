import os

import httpx
from langchain_core.tools import tool
from pydantic import BaseModel, Field

from src.services.logger.logger_service import LoggerService

logger = LoggerService().get_logger(__name__)


class IssueInput(BaseModel):
    issue: str = Field(..., min_length=1, description="Issue description")
    priority: str = Field(
        ..., min_length=1, description="Priority level (low, medium, high, critical)"
    )
    category: str = Field(
        ...,
        min_length=1,
        description="Issue category (technical, account, billing, feature_request, bug_report)",
    )


@tool(args_schema=IssueInput)
async def issue_tracker_tool(
    issue: str, priority: str, category: str, app_id: str | int = None
) -> str:
    """
    Categorize and save issues using LLM.

    Parameters:
    - issue: Description of the issue.
    - priority: Priority level (low, medium, high, critical).
    - category: Category of the issue (technical, account, billing, feature_request, bug_report).
    - app_id: Optional application ID to associate with the issue.
    Returns:
    - Confirmation message indicating whether the issue was saved and submitted or saved.
    """

    base_url = os.getenv("DATA_API_BASE_URL", "").rstrip("/")
    issue_url = os.getenv("ISSUE_TRACKER_ENDPOINT")
    endpoint_url = f"{base_url}/{issue_url.lstrip('/')}"

    payload = {
        "description": issue,
        "priority": priority,
        "category": category,
        "app_id": app_id,
    }

    try:
        async with httpx.AsyncClient(timeout=10) as client:
            response = await client.post(endpoint_url, json=payload)

            if response.status_code == 200:
                logger.info(
                    f"[IssueTrackerTool] Issue successfully submitted to {endpoint_url}"
                )
                return f"Issue saved and submitted: {issue}, priority: {priority}, category: {category}"

            else:
                logger.error(
                    f"[IssueTrackerTool] Failed to submit issue. Status: {response.status_code}, Response: {response.text}"
                )
                return f"Issue logged, but failed to submit: {issue}, priority: {priority}, category: {category}"

    except Exception as ex:
        logger.error(
            f"[IssueTrackerTool] Error submitting issue to {endpoint_url}: {str(ex)}"
        )
        return f"Issue logged, but error submitting: {issue}, priority: {priority}, category: {category}"
