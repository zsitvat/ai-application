from langchain_core.tools import tool


@tool
def create_issue_tracker_tool(
    name="issue_tracker",
    description="Categorize and save issues using LLM.",
    priority_levels=None,
    categories=None,
    required_fields=None,
):
    """
    Create an issue tracker tool configuration for LLM-based issue categorization and saving.

    Args:
        name: Name of the tool
        description: Description of the tool
        priority_levels: List of allowed priority levels
        categories: List of allowed categories
        required_fields: List of required fields for an issue

    Returns:
        dict: Tool configuration dictionary
    """
    return {
        "type": name,
        "description": description,
        "priority_levels": priority_levels or ["low", "medium", "high", "critical"],
        "categories": categories
        or ["technical", "account", "billing", "feature_request", "bug_report"],
        "required_fields": required_fields or ["description", "priority", "category"],
    }
