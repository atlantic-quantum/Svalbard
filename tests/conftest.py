"""Common text fixtures and other test configurations for svalbard"""

import pytest

pytest_plugins = [
    "tests.data_server.fixtures",
    "tests.data_model.fixtures",
    "tests.data_router.fixtures",
    "tests.data_model.measurement.fixtures",
]
pytest.register_assert_rewrite("tests.data_server.utility")
