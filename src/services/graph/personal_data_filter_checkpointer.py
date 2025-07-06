import copy

from services.validators.personal_data.personal_data_filter_service import (
    PersonalDataFilterService,
)


class PersonalDataFilterCheckpointer:
    """Wrapper for checkpointers that filters personal data before database operations."""
    
    def __init__(self, base_checkpointer, personal_data_service: PersonalDataFilterService, personal_data_config, logger):
        self.base_checkpointer = base_checkpointer
        self.personal_data_service = personal_data_service
        self.personal_data_config = personal_data_config
        self.logger = logger
    
    async def _filter_state_data(self, state_data: dict) -> dict:
        """Filter personal data from state before saving to database."""
        if not self.personal_data_config:
            return state_data
            
        try:
            # Create a deep copy to avoid modifying the original
            filtered_state = copy.copy(state_data)
            
            if "messages" in filtered_state and filtered_state["messages"]:
                filtered_state["messages"] = copy.copy(filtered_state["messages"])
                for i, message in enumerate(filtered_state["messages"]):
                    if hasattr(message, 'content') and message.content:
                        filtered_content, _ = await self.personal_data_service.filter_personal_data(
                            text=message.content,
                            config=self.personal_data_config.config
                        )
                        filtered_state["messages"][i] = copy.copy(message)
                        filtered_state["messages"][i].content = filtered_content
            
            if "user_input" in filtered_state and filtered_state["user_input"]:
                filtered_input, _ = await self.personal_data_service.filter_personal_data(
                    text=filtered_state["user_input"],
                    config=self.personal_data_config.config
                )
                filtered_state["user_input"] = filtered_input
            
            self.logger.debug("Personal data filtering applied before database save")
            return filtered_state
            
        except Exception as ex:
            self.logger.error(f"Error filtering personal data: {str(ex)}")
            return state_data
    
    async def aput(self, config, checkpoint, metadata):
        """Save checkpoint with personal data filtering."""
        # Create a deep copy to avoid modifying the original
        checkpoint_copy = copy.deepcopy(checkpoint)
        
        if hasattr(checkpoint_copy, 'channel_values'):
            filtered_values = await self._filter_state_data(checkpoint_copy.channel_values)
            checkpoint_copy.channel_values = filtered_values
        
        return await self.base_checkpointer.aput(config, checkpoint_copy, metadata)
    
    def put(self, config, checkpoint, metadata):
        """Save checkpoint with personal data filtering (sync version)."""
        import asyncio
        try:
            loop = asyncio.get_event_loop()
            return loop.run_until_complete(self.aput(config, checkpoint, metadata))
        except RuntimeError:
            return asyncio.run(self.aput(config, checkpoint, metadata))
    
    def __getattr__(self, name):
        return getattr(self.base_checkpointer, name)
