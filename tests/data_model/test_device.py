from svalbard.data_model.device import AtlanticDeviceInfo, Device, DeviceInfo


def test_device_serialisation():
    dev = Device()
    l_dev = Device(**dev.model_dump())

    assert dev == l_dev


def test_device_info_serialisation():
    dev_info = DeviceInfo()
    l_dev_info = DeviceInfo(**dev_info.model_dump())

    assert dev_info == l_dev_info


def test_atlantic_device_info_serialisation():
    atl_dev_info = AtlanticDeviceInfo()
    l_atl_dev_info = AtlanticDeviceInfo(**atl_dev_info.model_dump())

    assert atl_dev_info == l_atl_dev_info
