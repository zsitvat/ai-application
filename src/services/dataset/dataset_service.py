import logging
import os
import asyncio
from datetime import datetime
from uuid import uuid4
import httpx

from langchain.smith.evaluation import run_on_dataset
from langsmith import Client

from schemas.dataset_schema import (
    DatasetNotFoundError,
    DatasetCreationError,
    DatasetUpdateError,
    DatasetRunError,
    DatasetRunConfigSchema,
)
from schemas.graph_schema import RestOperationPostSchema, ApplicationIdentifierSchema


class DatasetService:
    """Service for managing datasets using LangSmith integration."""

    def __init__(self):
        """Initialize the dataset service with logger and LangSmith client."""
        self.logger = logging.getLogger(__name__)
        self.client = Client()
        self.background_tasks = set()

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

            self.client.create_examples(dataset_id=dataset.id, examples=test_cases)

            return {
                "id": str(dataset.id),
                "name": dataset.name,
                "description": dataset.description,
                "test_cases": test_cases,
                "created_at": (
                    dataset.created_at.isoformat()
                    if dataset.created_at
                    else datetime.now().isoformat()
                ),
                "updated_at": (
                    dataset.modified_at.isoformat()
                    if dataset.modified_at
                    else datetime.now().isoformat()
                ),
            }

        except Exception as e:
            self.logger.error(f"Error creating dataset {name}: {str(e)}")
            raise DatasetCreationError(f"Failed to create dataset: {str(e)}")

    def get_dataset(self, dataset_name: str) -> dict:
        """Retrieve a dataset by name from LangSmith.

        Args:
            dataset_name: The name of the dataset to retrieve

        Returns:
            Dictionary containing dataset information and test cases

        Raises:
            DatasetNotFoundError: If dataset is not found
        """
        try:
            self.logger.info(f"Getting dataset: {dataset_name}")

            datasets = self.client.list_datasets(dataset_name=dataset_name)
            dataset_list = list(datasets)

            if not dataset_list:
                raise DatasetNotFoundError(f"Dataset '{dataset_name}' not found")

            dataset = dataset_list[0]

            examples = list(self.client.list_examples(dataset_id=dataset.id))

            test_cases = []
            for example in examples:
                test_case = {
                    "inputs": example.inputs,
                    "outputs": example.outputs,
                    "metadata": example.metadata or {},
                }
                test_cases.append(test_case)

            return {
                "id": str(dataset.id),
                "name": dataset.name,
                "description": dataset.description,
                "test_cases": test_cases,
                "created_at": (
                    dataset.created_at.isoformat()
                    if dataset.created_at
                    else datetime.now().isoformat()
                ),
                "updated_at": (
                    dataset.modified_at.isoformat()
                    if dataset.modified_at
                    else datetime.now().isoformat()
                ),
            }

        except Exception as e:
            self.logger.error(f"Error getting dataset {dataset_name}: {str(e)}")
            raise DatasetNotFoundError(f"Failed to get dataset: {str(e)}")

    def _prepare_examples_from_test_cases(self, test_cases: list[dict]) -> list[dict]:
        """Convert test cases to LangSmith example format."""
        examples = []
        for test_case in test_cases:
            example = {
                "inputs": test_case.get("inputs", {}),
                "outputs": test_case.get("outputs", {}),
                "metadata": test_case.get("metadata", {}),
            }
            if "input" in test_case:
                example["inputs"] = {"question": test_case["input"]}
            if "expected_output" in test_case:
                example["outputs"] = {"answer": test_case["expected_output"]}

            examples.append(example)
        return examples

    def _update_test_cases(self, dataset_id: str, test_cases: list[dict]):
        """Replace all examples in a LangSmith dataset with new ones."""
        existing_examples = list(self.client.list_examples(dataset_id=dataset_id))
        for example in existing_examples:
            self.client.delete_example(example.id)

        examples = self._prepare_examples_from_test_cases(test_cases)
        self.client.create_examples(dataset_id=dataset_id, examples=examples)

    def update_dataset(
        self,
        dataset_name: str,
        description: str | None = None,
        test_cases: list[dict] | None = None,
    ) -> dict:
        """Update an existing dataset in LangSmith.

        Args:
            dataset_name: The name of the dataset to update
            description: Optional new description (warning: not fully supported by LangSmith)
            test_cases: Optional new test cases to replace existing ones

        Returns:
            Dictionary containing updated dataset information

        Raises:
            DatasetNotFoundError: If dataset is not found
            DatasetUpdateError: If update operation fails
        """
        try:
            self.logger.info(f"Updating dataset: {dataset_name}")

            datasets = self.client.list_datasets(dataset_name=dataset_name)
            dataset_list = list(datasets)

            if not dataset_list:
                raise DatasetNotFoundError(f"Dataset '{dataset_name}' not found")

            dataset = dataset_list[0]

            if description is not None:
                self.logger.warning(
                    "Description update not directly supported by LangSmith SDK"
                )

            if test_cases is not None:
                self._update_test_cases(dataset.id, test_cases)

            return self.get_dataset(dataset_name)

        except DatasetNotFoundError:
            raise
        except Exception as e:
            self.logger.error(f"Error updating dataset {dataset_name}: {str(e)}")
            raise DatasetUpdateError(f"Failed to update dataset: {str(e)}")

    async def _call_graph_api(
        self, question: str, config: DatasetRunConfigSchema | dict | None
    ) -> tuple[str | None, bool, str | None]:
        """Call the graph API and return the response, success status, and error if any."""
        try:
            if hasattr(config, "model_dump"):
                config_dict = config.model_dump()
            else:
                config_dict = config or {}

            if "endpoint" in config_dict and config_dict["endpoint"]:
                endpoint_url = config_dict["endpoint"]
                request_data = {"question": question}
            else:
                port = os.getenv("PORT")
                base_url = f"http://localhost:{port}"
                endpoint_url = f"{base_url}/api/graph"
                graph_request = RestOperationPostSchema(
                    uuid=config_dict.get("uuid", str(uuid4())),
                    applicationIdentifier=config_dict.get("applicationIdentifier"),
                    platform=config_dict.get("platform", "dataset_evaluation"),
                    user_input=question,
                    context=config_dict.get("context"),
                    parameters=config_dict.get("parameters"),
                )
                request_data = graph_request.model_dump()

            async with httpx.AsyncClient(timeout=300.0) as client:
                response = await client.post(
                    endpoint_url,
                    json=request_data,
                    headers={"Content-Type": "application/json"},
                )

                if response.status_code == 200:
                    return response.text.strip().strip('"'), True, None
                else:
                    error = f"HTTP {response.status_code}: {response.text}"
                    return None, False, error

        except Exception as e:
            return None, False, str(e)

    def _extract_question_from_test_case(
        self, test_case: dict, index: int
    ) -> str | None:
        """Extract question from test case, handling both input formats."""
        if "inputs" in test_case and "question" in test_case["inputs"]:
            return test_case["inputs"]["question"]
        elif "input" in test_case:
            return test_case["input"]
        else:
            self.logger.warning(f"No question found in test case {index}")
            return None

    def _create_result_dict(
        self,
        index: int,
        question: str,
        test_case: dict,
        actual_output: str | None,
        success: bool,
        error: str | None = None,
    ) -> dict:
        """Create a result dictionary for a test case."""
        result = {
            "test_case_index": index,
            "input": question,
            "expected_output": test_case.get(
                "outputs", test_case.get("expected_output")
            ),
            "actual_output": actual_output,
            "success": success,
            "metadata": test_case.get("metadata", {}),
        }

        if error:
            result["error"] = error

        return result

    def _create_llm_function(self, config: DatasetRunConfigSchema | dict | None):
        """Create a function that calls our API endpoint."""

        def llm_function(inputs):
            question = inputs.get("question", "")
            if not question:
                return {"answer": "No question provided"}

            try:
                import asyncio

                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                actual_output, success, error = loop.run_until_complete(
                    self._call_graph_api(question, config)
                )
                loop.close()

                if success:
                    return {"answer": actual_output}
                else:
                    return {"answer": f"Error: {error}"}
            except Exception as e:
                return {"answer": f"Error: {str(e)}"}

        return llm_function

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
