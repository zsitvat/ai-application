from fastapi import APIRouter, HTTPException, Depends
import logging


from schemas.agent_request_post_schema import AgentRequestPostSchema
from schemas.response_schema import AgentAnswerResponseSchema
from services.agent.agent_service import AgentService

router = APIRouter()


def get_agent_service(request: AgentRequestPostSchema):
    return AgentService(
        prompt=request.prompt,
        model=request.model,
        tools_config=request.tools,
        user_id=request.user_id,
    )


@router.post("/api/agent/answer", response_model=AgentAnswerResponseSchema)
async def agent_answer(
    request: AgentRequestPostSchema,
    agent_service: AgentService = Depends(get_agent_service),
):
    """
    Get an answer from the agent model.

    Args:
        request (AgentRequestPostSchema): Request body
        agent_service (AgentService): Agent service instance

    Returns:
            AgentAnswerResponseSchema
    """

    try:

        result = await agent_service.get_agent_answer(user_input=request.user_input)

        return {"answer": result}

    except Exception as ex:
        logging.getLogger("logger").error(f"Error in agent answer route: {str(ex)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error processing the request in agent answer route: {str(ex)}",
        )
