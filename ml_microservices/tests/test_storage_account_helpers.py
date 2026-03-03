"""
Unit tests for shared/storage_account_helpers.py

Tests cover:
  - get_blob_service_client  (URI parsing, authentication selection)
  - download_blob_to_dir     (single blob & prefix download paths)
  - list_blobs_in_prefix     (returns list of blob names)

All Azure SDK calls are mocked so no real Azure account is needed.
"""
import logging
import os
import tempfile
from unittest.mock import MagicMock, patch, PropertyMock

import pytest

from shared.storage_account_helpers import (
    get_blob_service_client,
    download_blob_to_dir,
    list_blobs_in_prefix,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_SAMPLE_URI = "abfss://mycontainer@mystorageaccount.blob.core.windows.net/data/raw"


def _make_mock_blob_service_client():
    client = MagicMock()
    return client


# ---------------------------------------------------------------------------
# get_blob_service_client
# ---------------------------------------------------------------------------

class TestGetBlobServiceClient:
    def test_parses_account_name(self):
        with patch("shared.storage_account_helpers.BlobServiceClient") as MockBSC:
            MockBSC.return_value = MagicMock()
            _, account_name, _, _ = get_blob_service_client(_SAMPLE_URI)
            assert account_name == "mystorageaccount"

    def test_parses_container(self):
        with patch("shared.storage_account_helpers.BlobServiceClient") as MockBSC:
            MockBSC.return_value = MagicMock()
            _, _, container, _ = get_blob_service_client(_SAMPLE_URI)
            assert container == "mycontainer"

    def test_parses_blob_path(self):
        with patch("shared.storage_account_helpers.BlobServiceClient") as MockBSC:
            MockBSC.return_value = MagicMock()
            _, _, _, blob_path = get_blob_service_client(_SAMPLE_URI)
            assert blob_path == "data/raw"

    def test_uses_connection_string_when_set(self, monkeypatch):
        monkeypatch.setenv("AZURE_STORAGE_CONNECTION_STRING", "DefaultEndpointsProtocol=https;...")
        with patch("shared.storage_account_helpers.BlobServiceClient") as MockBSC:
            MockBSC.from_connection_string.return_value = MagicMock()
            get_blob_service_client(_SAMPLE_URI)
            MockBSC.from_connection_string.assert_called_once()

    def test_uses_default_azure_credential_when_no_connection_string(self, monkeypatch):
        monkeypatch.delenv("AZURE_STORAGE_CONNECTION_STRING", raising=False)
        with patch("shared.storage_account_helpers.DefaultAzureCredential") as MockCred, \
             patch("shared.storage_account_helpers.BlobServiceClient") as MockBSC:
            MockCred.return_value = MagicMock()
            MockBSC.return_value = MagicMock()
            get_blob_service_client(_SAMPLE_URI)
            MockCred.assert_called_once()

    def test_raises_when_credential_fails(self, monkeypatch):
        monkeypatch.delenv("AZURE_STORAGE_CONNECTION_STRING", raising=False)
        with patch("shared.storage_account_helpers.DefaultAzureCredential", side_effect=Exception("auth error")), \
             patch("shared.storage_account_helpers.BlobServiceClient"):
            with pytest.raises(Exception, match="auth error"):
                get_blob_service_client(_SAMPLE_URI)

    def test_trailing_slash_stripped_from_blob_path(self):
        uri_with_slash = "abfss://mycontainer@mystorageaccount.blob.core.windows.net/data/raw/"
        with patch("shared.storage_account_helpers.BlobServiceClient") as MockBSC:
            MockBSC.return_value = MagicMock()
            _, _, _, blob_path = get_blob_service_client(uri_with_slash)
            assert not blob_path.endswith("/")

    def test_returns_four_tuple(self):
        with patch("shared.storage_account_helpers.BlobServiceClient") as MockBSC:
            MockBSC.return_value = MagicMock()
            result = get_blob_service_client(_SAMPLE_URI)
            assert len(result) == 4


# ---------------------------------------------------------------------------
# download_blob_to_dir
# ---------------------------------------------------------------------------

class TestDownloadBlobToDir:
    def _make_mock_blob_client(self, content: bytes = b"data"):
        mock = MagicMock()
        mock.download_blob.return_value.readall.return_value = content
        return mock

    def test_creates_download_directory(self):
        with tempfile.TemporaryDirectory() as d:
            target_dir = os.path.join(d, "downloads")
            mock_bsc = MagicMock()
            mock_blob_client = self._make_mock_blob_client()
            mock_bsc.get_blob_client.return_value = mock_blob_client
            mock_bsc.get_container_client.return_value = MagicMock()

            with patch("shared.storage_account_helpers.get_blob_service_client",
                       return_value=(mock_bsc, "acct", "cont", "blob.bin")):
                download_blob_to_dir(_SAMPLE_URI, target_dir)

            assert os.path.isdir(target_dir)

    def test_downloads_single_blob_file(self):
        with tempfile.TemporaryDirectory() as d:
            mock_bsc = MagicMock()
            mock_blob_client = self._make_mock_blob_client(b"hello")
            mock_bsc.get_blob_client.return_value = mock_blob_client
            mock_bsc.get_container_client.return_value = MagicMock()

            with patch("shared.storage_account_helpers.get_blob_service_client",
                       return_value=(mock_bsc, "acct", "cont", "myfile.bin")):
                download_blob_to_dir(_SAMPLE_URI, d)

            downloaded = os.path.join(d, "myfile.bin")
            assert os.path.exists(downloaded)
            with open(downloaded, "rb") as f:
                assert f.read() == b"hello"

    def test_downloads_multiple_blobs_from_prefix(self):
        with tempfile.TemporaryDirectory() as d:
            mock_bsc = MagicMock()

            # Simulate single-blob download raising an exception → falls through to prefix
            mock_bsc.get_blob_client.return_value.download_blob.side_effect = Exception("not found")

            blob_a = MagicMock()
            blob_a.name = "prefix/file_a.bin"
            blob_b = MagicMock()
            blob_b.name = "prefix/file_b.bin"
            mock_container = MagicMock()
            mock_container.list_blobs.return_value = [blob_a, blob_b]
            mock_bsc.get_container_client.return_value = mock_container

            # Each blob download should return bytes
            def side_effect_blob(container, blob):
                client = MagicMock()
                client.download_blob.return_value.readall.return_value = b"content"
                return client

            mock_bsc.get_blob_client.side_effect = side_effect_blob

            with patch("shared.storage_account_helpers.get_blob_service_client",
                       return_value=(mock_bsc, "acct", "cont", "prefix")):
                download_blob_to_dir(_SAMPLE_URI, d)

    def test_accepts_custom_logger(self):
        with tempfile.TemporaryDirectory() as d:
            logger = logging.getLogger("test_download")
            mock_bsc = MagicMock()
            mock_bsc.get_blob_client.return_value.download_blob.return_value.readall.return_value = b"ok"
            mock_bsc.get_container_client.return_value = MagicMock()

            with patch("shared.storage_account_helpers.get_blob_service_client",
                       return_value=(mock_bsc, "acct", "cont", "blob.bin")):
                download_blob_to_dir(_SAMPLE_URI, d, logger=logger)


# ---------------------------------------------------------------------------
# list_blobs_in_prefix
# ---------------------------------------------------------------------------

class TestListBlobsInPrefix:
    def test_returns_list_of_names(self):
        mock_bsc = MagicMock()
        blob_a = MagicMock()
        blob_a.name = "data/raw/a.json"
        blob_b = MagicMock()
        blob_b.name = "data/raw/b.json"
        mock_container = MagicMock()
        mock_container.list_blobs.return_value = [blob_a, blob_b]
        mock_bsc.get_container_client.return_value = mock_container

        with patch("shared.storage_account_helpers.get_blob_service_client",
                   return_value=(mock_bsc, "acct", "cont", "data/raw")):
            names = list_blobs_in_prefix(_SAMPLE_URI)

        assert names == ["data/raw/a.json", "data/raw/b.json"]

    def test_returns_empty_list_when_no_blobs(self):
        mock_bsc = MagicMock()
        mock_container = MagicMock()
        mock_container.list_blobs.return_value = []
        mock_bsc.get_container_client.return_value = mock_container

        with patch("shared.storage_account_helpers.get_blob_service_client",
                   return_value=(mock_bsc, "acct", "cont", "empty/prefix")):
            names = list_blobs_in_prefix(_SAMPLE_URI)

        assert names == []

    def test_accepts_custom_logger(self):
        logger = logging.getLogger("test_list")
        mock_bsc = MagicMock()
        mock_container = MagicMock()
        mock_container.list_blobs.return_value = []
        mock_bsc.get_container_client.return_value = mock_container

        with patch("shared.storage_account_helpers.get_blob_service_client",
                   return_value=(mock_bsc, "acct", "cont", "prefix")):
            names = list_blobs_in_prefix(_SAMPLE_URI, logger=logger)

        assert isinstance(names, list)
