"""Test our Inter-Process Read Write Recursive Lock"""

import pytest

from svalbard.utility.shared_array import InterProcessRWRLock


def test_lock():
    """Basic tests for lock, lock already tested by fastener modules"""
    lock = InterProcessRWRLock("/tmp/test/lock")

    with pytest.raises(ValueError):
        lock.release_read_lock()

    with pytest.raises(ValueError):
        lock.release_read_lock()
