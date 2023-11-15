"""Test ipc pydantic models"""
import uuid

from svalbard.data_model.ipc import MeasurementHandle, SliceListModel, SliceModel


def test_measurement_handle():
    """Test that creating a measurment handle using
    new gives a handle that is a UUID and safe"""
    meas_handle = MeasurementHandle.new()
    assert isinstance(meas_handle.handle, uuid.UUID)
    assert meas_handle.handle.is_safe

    meas_handle2 = MeasurementHandle(handle=meas_handle.handle)
    assert meas_handle == meas_handle2

    # test creating handle from string rep of uuid
    meas_handle3 = MeasurementHandle(handle=str(meas_handle.handle))  # type: ignore
    assert meas_handle == meas_handle3
    assert isinstance(meas_handle3.handle, uuid.UUID)

    meas_handle4 = MeasurementHandle(**meas_handle.dict())
    assert meas_handle == meas_handle4

    meas_handle5 = MeasurementHandle(handle=uuid.UUID(str(meas_handle.handle)))
    assert meas_handle == meas_handle5


def test_slice_model():
    sm = SliceModel(start=0, stop=10)
    assert sm.to_slice() == slice(0, 10)
    sm2 = SliceModel(start=0, stop=10, step=2)
    assert sm2.to_slice() == slice(0, 10, 2)
    sm3 = SliceModel()
    assert sm3.to_slice() == slice(None)


def test_slice_model_from_slice():
    sm = SliceModel.from_slice(slice(10))
    assert sm.to_slice() == slice(10)
    assert (sm.start, sm.stop, sm.step) == (None, 10, None)
    sm2 = SliceModel.from_slice(slice(0, 10))
    assert sm2.to_slice() == slice(0, 10)
    assert (sm2.start, sm2.stop, sm2.step) == (0, 10, None)
    sm3 = SliceModel.from_slice(slice(0, 10, 2))
    assert sm3.to_slice() == slice(0, 10, 2)
    assert (sm3.start, sm3.stop, sm3.step) == (0, 10, 2)
    sm4 = SliceModel.from_slice(slice(None))
    assert sm4.to_slice() == slice(None)
    assert (sm4.start, sm4.stop, sm4.step) == (None, None, None)


def test_slice_list_model():
    slm = SliceListModel(
        slice_lists=[
            [SliceModel(start=0, stop=10)],
            [SliceModel(start=0, stop=10, step=2)],
        ]
    )
    assert slm.to_slice_lists() == [[slice(0, 10)], [slice(0, 10, 2)]]


def test_slice_list_model_from_slice_list():
    slm = SliceListModel.from_slice_lists(
        [[slice(10)], [slice(0, 10)], [slice(0, 10, 2)]]
    )
    assert slm.to_slice_lists() == [[slice(10)], [slice(0, 10)], [slice(0, 10, 2)]]
