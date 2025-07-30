from langchain_core.tools import tool
from pydantic import BaseModel, Field


class IssueInput(BaseModel):
    description: str = Field(..., description="Issue description")
    priority: str = Field(
        ..., description="Priority level (low, medium, high, critical)"
    )
    category: str = Field(
        ...,
        description="Issue category (technical, account, billing, feature_request, bug_report)",
    )


@tool(args_schema=IssueInput)
def issue_tracker_tool(description: str, priority: str, category: str) -> str:
    """
    Categorize and save issues using LLM.
    """

    return f"Issue saved: {description}, priority: {priority}, category: {category}"
