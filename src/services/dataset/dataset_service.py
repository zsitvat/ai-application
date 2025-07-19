import asyncio
import logging
from datetime import datetime

import httpx
from langchain.smith.evaluation import run_on_dataset
from langsmith import Client

from schemas.dataset_schema import (
    DatasetCreationError,
    DatasetNotFoundError,
    DatasetRunConfigSchema,
    DatasetRunError,
    DatasetUpdateError,
)


class DatasetService:
    """Service for managing datasets using LangSmith integration."""

    def __init__(self):
        """Initialize the dataset service with logger and LangSmith client."""
        self.logger = logging.getLogger(__name__)
        self.client = Client()
        self.background_tasks = set()

    # Public methods
    def create_dataset(
        self, name: str, description: str | None, test_cases: list[dict]
    ) -> dict:
        """Create a new dataset in LangSmith.

        Args:
            name: The name of the dataset
            description: Optional description for the dataset
            test_cases: List of test cases to associate with the dataset

        Returns:
            Dictionary containing dataset information

        Raises:
            DatasetCreationError: If dataset creation fails
        """
        try:
            self.logger.info(f"Creating dataset: {name}")

            dataset = self.client.create_dataset(
                dataset_name=name, description=description or f"Test dataset: {name}"
            )

            if test_cases:
                self._update_test_cases(dataset.id, test_cases)

            result = {
                "id": str(dataset.id),
                "name": dataset.name,
                "description": dataset.description,
                "test_cases_count": len(test_cases) if test_cases else 0,
                "created_at": (
                    dataset.created_at.isoformat()
                    if dataset.created_at
                    else datetime.now().isoformat()
                ),
                "url": dataset.url,
                "tags": dataset.tags,
                "metadata": dataset.extra,
            }

            self.logger.info(
                f"Dataset created successfully: {name} with {len(test_cases) if test_cases else 0} test cases"
            )
            return result

        except Exception as e:
            self.logger.error(f"Error creating dataset {name}: {str(e)}")
            raise DatasetCreationError(f"Failed to create dataset: {str(e)}")

    def get_dataset(self, dataset_name: str) -> dict:
        """Get dataset information from LangSmith.

        Args:
            dataset_name: The name of the dataset to retrieve

        Returns:
            Dictionary containing dataset information

        Raises:
            DatasetNotFoundError: If dataset is not found
        """
        try:
            self.logger.info(f"Retrieving dataset: {dataset_name}")

            datasets = self.client.list_datasets(dataset_name=dataset_name)
            dataset_list = list(datasets)

            if not dataset_list:
                raise DatasetNotFoundError(f"Dataset '{dataset_name}' not found")

            dataset = dataset_list[0]

            examples = list(self.client.list_examples(dataset_id=dataset.id))

            result = {
                "id": str(dataset.id),
                "name": dataset.name,
                "description": dataset.description,
                "test_cases_count": len(examples),
                "test_cases": [
                    {
                        "id": str(example.id),
                        "inputs": example.inputs,
                        "outputs": example.outputs,
                        "metadata": example.metadata,
                        "created_at": (
                            example.created_at.isoformat()
                            if example.created_at
                            else None
                        ),
                    }
                    for example in examples
                ],
                "created_at": (
                    dataset.created_at.isoformat() if dataset.created_at else None
                ),
                "modified_at": (
                    dataset.modified_at.isoformat() if dataset.modified_at else None
                ),
                "url": dataset.url,
                "tags": dataset.tags,
                "metadata": dataset.extra,
            }

            self.logger.info(
                f"Dataset retrieved successfully: {dataset_name} with {len(examples)} test cases"
            )
            return result

        except DatasetNotFoundError:
            raise
        except Exception as e:
            self.logger.error(f"Error retrieving dataset {dataset_name}: {str(e)}")
            raise DatasetNotFoundError(f"Failed to retrieve dataset: {str(e)}")

    def update_dataset(
        self,
        dataset_name: str,
        name: str | None = None,
        description: str | None = None,
        test_cases: list[dict] | None = None,
    ) -> dict:
        """Update an existing dataset in LangSmith.

        Args:
            dataset_name: The current name of the dataset
            name: New name for the dataset (optional)
            description: New description for the dataset (optional)
            test_cases: New test cases to replace existing ones (optional)

        Returns:
            Dictionary containing updated dataset information

        Raises:
            DatasetNotFoundError: If dataset is not found
            DatasetUpdateError: If update fails
        """
        try:
            self.logger.info(f"Updating dataset: {dataset_name}")

            datasets = self.client.list_datasets(dataset_name=dataset_name)
            dataset_list = list(datasets)

            if not dataset_list:
                raise DatasetNotFoundError(f"Dataset '{dataset_name}' not found")

            dataset = dataset_list[0]

            if name or description:
                self.client.update_dataset(
                    dataset_id=dataset.id,
                    name=name or dataset.name,
                    description=description or dataset.description,
                )

            if test_cases is not None:
                self._update_test_cases(dataset.id, test_cases)

            return self.get_dataset(name or dataset_name)

        except DatasetNotFoundError:
            raise
        except Exception as e:
            self.logger.error(f"Error updating dataset {dataset_name}: {str(e)}")
            raise DatasetUpdateError(f"Failed to update dataset: {str(e)}")

    def run_dataset(
        self,
        dataset_name: str,
        config: DatasetRunConfigSchema | dict | None = None,
        uuid: str | None = None,
    ) -> dict:
        """Run a dataset against the configured API endpoint using LangChain's run_on_dataset.

        This method starts the dataset run in the background and returns immediately.
        The actual execution happens asynchronously and results are logged.

        Args:
            dataset_name: Name of the dataset to run
            config: Optional configuration for API calls including endpoint and parameters
            uuid: Optional UUID to use for API calls. If provided, will override any UUID in config

        Returns:
            Dictionary containing run information (run starts immediately but executes in background)

        Raises:
            DatasetRunError: If the dataset run initialization fails
        """
        try:
            self.logger.info(f"Starting background dataset run: {dataset_name}")

            run_id = f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{dataset_name}"

            # Start the background task
            task = asyncio.create_task(
                self._run_dataset_background(dataset_name, config, uuid, run_id)
            )
            self.background_tasks.add(task)
            task.add_done_callback(self.background_tasks.discard)

            # Don't wait for the task to complete
            return {
                "dataset_name": dataset_name,
                "run_id": run_id,
                "status": "started",
                "message": "Dataset run started in background",
                "config": config,
                "timestamp": datetime.now().isoformat(),
            }

        except Exception as e:
            self.logger.error(f"Error starting dataset run {dataset_name}: {str(e)}")
            raise DatasetRunError(f"Failed to start dataset run: {str(e)}")

    def delete_dataset(self, dataset_name: str) -> dict:
        """Delete a dataset from LangSmith.

        Args:
            dataset_name: The name of the dataset to delete

        Returns:
            Dictionary containing confirmation of deletion

        Raises:
            DatasetNotFoundError: If dataset is not found
            DatasetUpdateError: If deletion fails
        """
        try:
            self.logger.info(f"Deleting dataset: {dataset_name}")

            datasets = self.client.list_datasets(dataset_name=dataset_name)
            dataset_list = list(datasets)

            if not dataset_list:
                raise DatasetNotFoundError(f"Dataset '{dataset_name}' not found")

            dataset = dataset_list[0]
            self.client.delete_dataset(dataset_id=dataset.id)

            return {
                "message": f"Dataset '{dataset_name}' successfully deleted",
                "dataset_name": dataset_name,
                "deleted_at": datetime.now().isoformat(),
            }

        except DatasetNotFoundError:
            raise
        except Exception as e:
            self.logger.error(f"Error deleting dataset {dataset_name}: {str(e)}")
            raise DatasetUpdateError(f"Failed to delete dataset: {str(e)}")

    # Private methods
    def _prepare_examples_from_test_cases(self, test_cases: list[dict]) -> list[dict]:
        """Prepare examples from test cases for LangSmith dataset."""
        examples = []
        for test_case in test_cases:
            example = {
                "inputs": test_case.get("inputs", {}),
                "outputs": test_case.get(
                    "outputs", test_case.get("expected_output", {})
                ),
                "metadata": test_case.get("metadata", {}),
            }
            examples.append(example)
        return examples

    def _update_test_cases(self, dataset_id: str, test_cases: list[dict]):
        """Update test cases for a dataset by replacing all existing examples."""
        examples = self._prepare_examples_from_test_cases(test_cases)
        for example in examples:
            self.client.create_example(
                inputs=example["inputs"],
                outputs=example["outputs"],
                dataset_id=dataset_id,
                metadata=example["metadata"],
            )

    async def _call_graph_api(
        self, question: str, config: DatasetRunConfigSchema | dict | None
    ) -> dict:
        """Make an HTTP call to the graph API endpoint.

        Args:
            question: The question to send to the API
            config: Configuration containing API endpoint and parameters

        Returns:
            Dictionary containing the API response

        Raises:
            Exception: If the API call fails
        """
        try:
            # Default configuration
            default_config = {
                "url": "http://localhost:8000/api/v1/graph/execute",
                "timeout": 300,
                "method": "POST",
                "headers": {"Content-Type": "application/json"},
                "app_id": 0,
                "user_id": "test_user",
                "parameters": {},
            }

            # Use provided config or defaults
            if config:
                if hasattr(config, "model_dump"):
                    config_dict = config.model_dump()
                else:
                    config_dict = dict(config)

                api_config = {**default_config, **config_dict}
            else:
                api_config = default_config

            payload = {
                "user_input": question,
                "app_id": api_config.get("app_id", 0),
                "user_id": api_config.get("user_id", "test_user"),
                "parameters": api_config.get("parameters", {}),
            }

            # Add UUID to parameters if provided
            if "uuid" in api_config:
                payload["parameters"]["uuid"] = api_config["uuid"]

            self.logger.debug(
                f"Making API call to {api_config['url']} with payload: {payload}"
            )

            async with httpx.AsyncClient(
                timeout=api_config.get("timeout", 300)
            ) as client:
                response = await client.post(
                    api_config["url"],
                    json=payload,
                    headers=api_config.get("headers", {}),
                )
                response.raise_for_status()
                return response.json()

        except Exception as e:
            self.logger.error(f"Error calling graph API: {str(e)}")
            raise

    def _extract_question_from_test_case(
        self, test_case: dict, config: DatasetRunConfigSchema | dict | None
    ) -> str:
        """Extract the question from a test case based on configuration."""
        if not test_case:
            return ""

        question_key = "user_input"
        if config and hasattr(config, "question_key"):
            question_key = config.question_key
        elif config and isinstance(config, dict) and "question_key" in config:
            question_key = config["question_key"]

        return test_case.get(question_key, "")

    def _create_result_dict(
        self,
        question: str,
        api_response: dict,
        error: Exception | None = None,
        config: DatasetRunConfigSchema | dict | None = None,
    ) -> dict:
        """Create a standardized result dictionary."""
        result = {
            "question": question,
            "timestamp": datetime.now().isoformat(),
            "config": config,
        }

        if error:
            result.update(
                {
                    "success": False,
                    "error": str(error),
                    "response": None,
                }
            )
        else:
            result.update(
                {
                    "success": True,
                    "error": None,
                    "response": api_response,
                }
            )

        return result

    def _create_llm_function(self, config: DatasetRunConfigSchema | dict | None):
        """Create an LLM function for use with run_on_dataset."""

        def llm_function(inputs):
            """LLM function that processes inputs and returns outputs."""
            try:
                question = self._extract_question_from_test_case(inputs, config)

                if not question:
                    self.logger.warning(f"No question found in inputs: {inputs}")
                    return {"error": "No question found in test case"}

                # Use asyncio to call the async API function
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                try:
                    api_response = loop.run_until_complete(
                        self._call_graph_api(question, config)
                    )
                    return self._create_result_dict(
                        question, api_response, None, config
                    )
                finally:
                    loop.close()

            except Exception as e:
                self.logger.error(f"Error in LLM function: {str(e)}")
                return self._create_result_dict(
                    question if "question" in locals() else "", {}, e, config
                )

        return llm_function

    async def _run_dataset_background(
        self,
        dataset_name: str,
        config: DatasetRunConfigSchema | dict | None,
        uuid: str | None,
        run_id: str,
    ) -> None:
        """Execute the dataset run in the background and log results."""
        try:
            self.logger.info(f"Executing background dataset run: {run_id}")

            config = config or {}

            if uuid and (not config or "uuid" not in config):
                if hasattr(config, "model_dump"):
                    config_dict = config.model_dump()
                    config_dict["uuid"] = uuid
                    config = config_dict
                else:
                    config = dict(config) if config else {}
                    config["uuid"] = uuid

            llm_function = self._create_llm_function(config)

            loop = asyncio.get_event_loop()
            results = await loop.run_in_executor(
                None,
                lambda: run_on_dataset(
                    dataset_name=dataset_name,
                    llm_or_chain_factory=llm_function,
                    client=self.client,
                    evaluation=None,
                    input_mapper=None,
                    tags=[f"dataset_run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"],
                ),
            )

            self.logger.info(
                f"Dataset run completed successfully: {run_id}. "
                f"Results ID: {results.id if hasattr(results, 'id') else 'N/A'}"
            )

            self.logger.debug(f"Dataset run {run_id} detailed results: {results}")

        except Exception as e:
            self.logger.error(f"Error in background dataset run {run_id}: {str(e)}")
