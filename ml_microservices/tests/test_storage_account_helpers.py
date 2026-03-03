"""
Unit tests for shared/storage_account_helpers.py

Azure SDK calls are fully mocked so no real storage account is needed.
"""
import logging
import os
import tempfile
from unittest.mock import MagicMock, patch, call

import pytest

from shared.storage_account_helpers import (
    get_blob_service_client,
    list_blobs_in_prefix,
    download_blob_to_dir,
)


# ---------------------------------------------------------------------------
# Fixture
# ---------------------------------------------------------------------------

logger = logging.getLogger(__name__)

SAMPLE_URI = "wasbs://mycontainer@myaccount.blob.core.windows.net/path/to/data"


def _make_blob_service_mock(blob_names=None):
    """Return a minimal BlobServiceClient mock."""
    blob_service = MagicMock()
    blob_client = MagicMock()

    # Single-blob download
    blob_client.download_blob.return_value.readall.return_value = b"data"
    blob_service.get_blob_client.return_value = blob_client

    # Container listing
    container_client = MagicMock()
    blobs = []
    for name in (blob_names or []):
        b = MagicMock()
        b.name = name
        blobs.append(b)
    container_client.list_blobs.return_value = iter(blobs)
    blob_service.get_container_client.return_value = container_client

    return blob_service


# ---------------------------------------------------------------------------
# get_blob_service_client
# ---------------------------------------------------------------------------

class TestGetBlobServiceClient:
    def test_parses_account_name(self):
        with patch("shared.storage_account_helpers.BlobServiceClient") as mock_bsc:
            mock_bsc.from_connection_string.return_value = MagicMock()
            with patch.dict(os.environ,
                            {"AZURE_STORAGE_CONNECTION_STRING": "DefaultEndpointsProtocol=https;AccountName=myaccount;..."}):
                client, account_name, container, blob_path = get_blob_service_client(SAMPLE_URI)
        assert account_name == "myaccount"

    def test_parses_container_name(self):
        with patch("shared.storage_account_helpers.BlobServiceClient") as mock_bsc:
            mock_bsc.from_connection_string.return_value = MagicMock()
            with patch.dict(os.environ,
                            {"AZURE_STORAGE_CONNECTION_STRING": "conn_str"}):
                _, _, container, _ = get_blob_service_client(SAMPLE_URI)
        assert container == "mycontainer"

    def test_parses_blob_path(self):
        with patch("shared.storage_account_helpers.BlobServiceClient") as mock_bsc:
            mock_bsc.from_connection_string.return_value = MagicMock()
            with patch.dict(os.environ,
                            {"AZURE_STORAGE_CONNECTION_STRING": "conn_str"}):
                _, _, _, blob_path = get_blob_service_client(SAMPLE_URI)
        assert blob_path == "path/to/data"

    def test_uses_connection_string_when_env_set(self):
        with patch("shared.storage_account_helpers.BlobServiceClient") as mock_bsc:
            mock_bsc.from_connection_string.return_value = MagicMock()
            with patch.dict(os.environ,
                            {"AZURE_STORAGE_CONNECTION_STRING": "my_conn_str"}):
                get_blob_service_client(SAMPLE_URI)
        mock_bsc.from_connection_string.assert_called_once()

    def test_uses_default_credential_when_no_conn_string(self):
        with (
            patch("shared.storage_account_helpers.BlobServiceClient") as mock_bsc,
            patch("shared.storage_account_helpers.DefaultAzureCredential") as mock_cred,
            patch.dict(os.environ, {}, clear=True),
        ):
            mock_cred.return_value = MagicMock()
            mock_bsc.return_value = MagicMock()
            get_blob_service_client(SAMPLE_URI)
        mock_cred.assert_called_once()


# ---------------------------------------------------------------------------
# list_blobs_in_prefix
# ---------------------------------------------------------------------------

class TestListBlobsInPrefix:
    def test_returns_list_of_blob_names(self):
        blob_names = ["path/to/data/file1.json", "path/to/data/file2.json"]
        blob_service = _make_blob_service_mock(blob_names=blob_names)

        with patch("shared.storage_account_helpers.get_blob_service_client",
                   return_value=(blob_service, "myaccount", "mycontainer", "path/to/data")):
            result = list_blobs_in_prefix(SAMPLE_URI)

        assert result == blob_names

    def test_returns_empty_list_for_empty_prefix(self):
        blob_service = _make_blob_service_mock(blob_names=[])

        with patch("shared.storage_account_helpers.get_blob_service_client",
                   return_value=(blob_service, "myaccount", "mycontainer", "empty/prefix")):
            result = list_blobs_in_prefix(SAMPLE_URI)

        assert result == []

    def test_accepts_custom_logger(self):
        blob_service = _make_blob_service_mock(blob_names=["a.json"])
        custom_logger = logging.getLogger("custom")

        with patch("shared.storage_account_helpers.get_blob_service_client",
                   return_value=(blob_service, "myaccount", "mycontainer", "prefix")):
            result = list_blobs_in_prefix(SAMPLE_URI, logger=custom_logger)

        assert len(result) == 1


# ---------------------------------------------------------------------------
# download_blob_to_dir
# ---------------------------------------------------------------------------

class TestDownloadBlobToDir:
    def test_single_blob_is_downloaded(self, tmp_path):
        blob_service = _make_blob_service_mock()

        with patch("shared.storage_account_helpers.get_blob_service_client",
                   return_value=(blob_service, "myaccount", "mycontainer", "path/to/data/file.json")):
            download_blob_to_dir(SAMPLE_URI, str(tmp_path))

        # Single blob download should have been attempted
        blob_service.get_blob_client.assert_called()

    def test_directory_is_created(self, tmp_path):
        target_dir = str(tmp_path / "new_subdir")
        blob_service = _make_blob_service_mock()

        with patch("shared.storage_account_helpers.get_blob_service_client",
                   return_value=(blob_service, "myaccount", "mycontainer", "path/blob.json")):
            download_blob_to_dir(SAMPLE_URI, target_dir)

        assert os.path.isdir(target_dir)

    def test_prefix_downloads_multiple_blobs(self, tmp_path):
        blob_names = ["prefix/a.json", "prefix/b.json"]
        blob_service = _make_blob_service_mock(blob_names=blob_names)

        # Simulate that the single-blob path raises (it's a prefix, not a file)
        blob_service.get_blob_client.return_value.download_blob.side_effect = [
            Exception("BlobNotFound"),   # first call raises (not a single blob)
            MagicMock(readall=MagicMock(return_value=b"content")),  # second call for list
            MagicMock(readall=MagicMock(return_value=b"content")),  # third call for list
        ]

        with patch("shared.storage_account_helpers.get_blob_service_client",
                   return_value=(blob_service, "myaccount", "mycontainer", "prefix")):
            download_blob_to_dir(SAMPLE_URI, str(tmp_path))
