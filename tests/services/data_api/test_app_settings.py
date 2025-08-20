import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.services.data_api.app_settings import AppSettingsService, DataApiException


@pytest.mark.asyncio
def test_get_app_settings_success():
    service = AppSettingsService()
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = [
        {"key": "foo", "value": "bar"},
        {"key": "baz", "value": "qux"},
    ]
    with (
        patch("httpx.AsyncClient") as mock_client,
        patch(
            "os.environ.get",
            side_effect=lambda k: (
                "http://localhost"
                if k == "DATA_API_BASE_URL"
                else (
                    "/appsettings/{applicationId}"
                    if k == "DATA_API_APP_SETTINGS_ROUTE_PATH"
                    else None
                )
            ),
        ),
    ):
        mock_client.return_value.__aenter__.return_value.get = AsyncMock(
            return_value=mock_response
        )
        result = asyncio.run(service.get_app_settings(app_id=1))
        assert result == {"foo": "bar", "baz": "qux"}


@pytest.mark.asyncio
def test_get_app_settings_not_found():
    service = AppSettingsService()
    mock_response = MagicMock()
    mock_response.status_code = 404
    mock_response.json.return_value = []
    with (
        patch("httpx.AsyncClient") as mock_client,
        patch(
            "os.environ.get",
            side_effect=lambda k: (
                "http://localhost"
                if k == "DATA_API_BASE_URL"
                else (
                    "/appsettings/{applicationId}"
                    if k == "DATA_API_APP_SETTINGS_ROUTE_PATH"
                    else None
                )
            ),
        ),
    ):
        mock_client.return_value.__aenter__.return_value.get = AsyncMock(
            return_value=mock_response
        )
        with pytest.raises(DataApiException):
            asyncio.run(service.get_app_settings(app_id=1))


@pytest.mark.asyncio
def test_get_app_settings_by_key_success():
    service = AppSettingsService()
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = [
        {"key": "foo", "value": "bar"},
        {"key": "baz", "value": "qux"},
    ]
    with (
        patch("httpx.AsyncClient") as mock_client,
        patch(
            "os.environ.get",
            side_effect=lambda k: (
                "http://localhost"
                if k == "DATA_API_BASE_URL"
                else "/appsettings" if k == "DATA_API_APP_SETTINGS_ROUTE_PATH" else None
            ),
        ),
    ):
        mock_client.return_value.__aenter__.return_value.get = AsyncMock(
            return_value=mock_response
        )
        result = asyncio.run(service.get_app_settings_by_key("foo"))
        assert result == {"key": "foo", "value": "bar"}


@pytest.mark.asyncio
def test_get_app_settings_by_key_not_found():
    service = AppSettingsService()
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = [
        {"key": "baz", "value": "qux"},
    ]
    with (
        patch("httpx.AsyncClient") as mock_client,
        patch(
            "os.environ.get",
            side_effect=lambda k: (
                "http://localhost"
                if k == "DATA_API_BASE_URL"
                else "/appsettings" if k == "DATA_API_APP_SETTINGS_ROUTE_PATH" else None
            ),
        ),
    ):
        mock_client.return_value.__aenter__.return_value.get = AsyncMock(
            return_value=mock_response
        )
        result = asyncio.run(service.get_app_settings_by_key("foo"))
        assert result is None


@pytest.mark.asyncio
def test_get_app_settings_by_key_error():
    service = AppSettingsService()
    mock_response = MagicMock()
    mock_response.status_code = 500
    mock_response.json.return_value = []
    with (
        patch("httpx.AsyncClient") as mock_client,
        patch(
            "os.environ.get",
            side_effect=lambda k: (
                "http://localhost"
                if k == "DATA_API_BASE_URL"
                else "/appsettings" if k == "DATA_API_APP_SETTINGS_ROUTE_PATH" else None
            ),
        ),
    ):
        mock_client.return_value.__aenter__.return_value.get = AsyncMock(
            return_value=mock_response
        )
        with pytest.raises(DataApiException):
            asyncio.run(service.get_app_settings_by_key("foo"))
