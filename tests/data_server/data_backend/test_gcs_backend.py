"""Test Google Cloud Storage based Data Backend"""

import pytest
import pytest_asyncio
from gcsfs import GCSFileSystem
from gcsfs.retry import HttpError

from svalbard.data_server.data_backend.gcs_backend import GCSBackendConfig

from . import test_fs_backend as backend_tests


@pytest.mark.asyncio
async def test_init_backend_from_config(
    gcs_config: GCSBackendConfig, gcs_filesystem: GCSFileSystem
):
    """Test creating a fs backend from fs backend config"""
    gcs_config.init()


class TestGCSBackend(backend_tests.TestFSBackendWithStreaming):
    """Test class for testing the Google Cloud Storage backed Data Backend"""

    @pytest.fixture(name="backend")
    def fixture_gcs_backend(self, gcs_backend):
        """Fixture for creating fs backend for use in other tests"""
        yield gcs_backend


class TestGCSBackendFromConfig(backend_tests.FSBackendTests):
    """Test FSBackend created from config"""

    @pytest_asyncio.fixture(name="backend", scope="class")
    async def fixture_gcs_backend_config(
        self, gcs_config: GCSBackendConfig, gcs_filesystem: GCSFileSystem, test_bucket
    ):
        """Fixture for creating fs backend for use in other tests"""

        test_bucket = "aq_test"
        try:
            # ensure we're empty.
            try:
                gcs_filesystem.rm(test_bucket, recursive=True)
            except FileNotFoundError:
                pass
            try:
                gcs_filesystem.mkdir(test_bucket)
            except HttpError:
                pass
            yield gcs_config.init()
        finally:
            try:
                gcs_filesystem.rm(gcs_filesystem.find(test_bucket))
            except FileNotFoundError:
                pass
