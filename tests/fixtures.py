import pytest
from svalbard.data_model.memory_models import SharedMemoryOut, __memory_reference__


@pytest.fixture(name="memory_fixture", autouse=True)
def fixture_memory(capsys):
    """Fixture that deletes all shared memory after each test"""
    yield
    # with capsys.disabled():
    #     print(f"memories after test:{len(__memory_reference__.keys())}")
    mems = list(__memory_reference__.keys())
    for key in mems:  # can't delete while iterating over dict as it changes size
        SharedMemoryOut.close(key)
    # with capsys.disabled():
    #     print(f"memories after cleanup:{len(__memory_reference__.keys())}")
