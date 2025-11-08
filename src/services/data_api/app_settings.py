import logging
import os

import httpx


class DataApiException(Exception):
    """Exception raised for errors in the Data API."""


class AppSettingsService:

    async def get_app_settings(self, app_id):
        """Get app settings from data ap[A-Z]"""

        base = os.environ.get("DATA_API_BASE_URL")
        path = os.environ.get("DATA_API_APP_SETTINGS_ROUTE_PATH").format(
            applicationId=app_id
        )
        url = base + path

        headers = {"Content-Type": "application/json", "Connection": "close"}

        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(url, headers=headers, timeout=30.0)
                if response.status_code == 200:
                    settings = {}
                    for setting in response.json():
                        settings[setting["key"]] = setting["value"]
                    return settings
                else:
                    error_msg = f"Error while getting app settings from data! Status code: {response.status_code}"
                    logging.getLogger("uvicorn").error(error_msg)
                    raise DataApiException(error_msg)
        except httpx.RequestError as ex:
            error_msg = f"App settings request error: {str(ex)}"
            logging.getLogger("uvicorn").error(error_msg)
            raise DataApiException(error_msg) from ex
        except Exception as ex:
            error_msg = f"App settings general error: {str(ex)}"
            logging.getLogger("uvicorn").error(error_msg)
            raise DataApiException(error_msg) from ex

    async def get_app_settings_by_key(self, key: str):
        """Get app settings from data ap[A-Z]"""

        base = os.environ.get("DATA_API_BASE_URL")
        path = os.environ.get("DATA_API_APP_SETTINGS_ROUTE_PATH")
        url = base + path

        headers = {"Content-Type": "application/json", "Connection": "close"}

        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(url, headers=headers, timeout=30.0)

                if response.status_code == 200:
                    for setting in response.json():
                        if setting["key"] == key:
                            return setting

                    error_msg = f"App setting with key '{key}' not found"
                    logging.getLogger("uvicorn").warning(error_msg)
                    return None
                else:
                    error_msg = f"Error while getting app setting '{key}' from data! Status code: {response.status_code}"
                    logging.getLogger("uvicorn").error(error_msg)
                    raise DataApiException(error_msg)
        except httpx.RequestError as ex:
            error_msg = f"Error while getting app setting '{key}' from data (request error): {str(ex)}"
            logging.getLogger("uvicorn").error(error_msg)
            raise DataApiException(error_msg) from ex
        except Exception as ex:
            error_msg = f"Error while getting app setting '{key}' from data (general error): {str(ex)}"
            logging.getLogger("uvicorn").error(error_msg)
            raise DataApiException(error_msg) from ex
