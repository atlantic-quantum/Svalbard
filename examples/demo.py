"""
Demo for launching a data server, configuring it, saveing and loading data.
This assumes that the svalbard is already installed
in a conda environment called svalbard, docker desktop is set up and
the computer has access to Google Cloud Storage and you have .pem file with access
credentials for our mongoDB Atlas instance.

If you have a .pem file with access to our mongoDB Atlas instance you will have to
update the path to it in the data_server.json file. It is currently set to
run of Brandur's Macbook Pro

If you don't have access to GCS you can use the data_server_local.json config file 
instead of data_server.json
"""
import asyncio
import json

import httpx
import numpy as np
from svalbard.data_model.data_file import Data, DataFile, MeasurementHandle, MetaData
from svalbard.data_model.ipc import SavedPath
from svalbard.data_model.memory_models import SharedMemoryIn, SharedMemoryOut
from svalbard.data_server.frontend.frontend_v1 import FrontendV1Config
from svalbard.launch import remove_shm_from_resource_tracker

# 1
#
# Open up a terminal, activate svalbard and navigate to the svalbard folder,
# run docker-compose and navigate to the examples folder
#
#       conda activate svalbard
#       cd navigate/to/the/repository/svalbard/
#       docker-compose up -d
#       cd examples
#
# after launcing the docker containers you should be able to access
# a) a mongo db express database viewer at "http://127.0.0.1:8081/
# b) and a local fake gcs server at

# 2
#
# Launch the data server app by running:
#
#       python launch.py
#
# This should launch the data server app listening on port 5000 (of localhost/127.0.0.1)
# You should be able to navigate to "http://127.0.0.1:5000/docs" to see api
# documentation for the app.

# 3
#
# Open up another terminal activate the svalbard environment and navicate to
# svalbard/examples (see step 1 for details):
#
#       python demo.py

DATA_SERVER_URL = "http://127.0.0.1:5000"


def data_file():
    """Create a datafile with metadata and data with different shape"""
    shapes = [(10, 10)]  # , (3, 3, 3), (20, 20, 20, 10)]
    dtypes = [np.dtype("float"), np.dtype("float"), np.dtype("int"), np.dtype("bool")]
    mems_in = [
        SharedMemoryIn(dtype=dtype, shape=shape)
        for (shape, dtype) in zip(shapes, dtypes)
    ]
    mems_out = [SharedMemoryOut.from_memory_in(mem_in) for mem_in in mems_in]
    rng = np.random.default_rng()
    for mem_out in mems_out:
        mem_out.to_array()[:] = rng.standard_normal(size=mem_out.shape).astype(
            mem_out.dtype
        )
    datasets = [
        Data.DataSet(name=f"demo_data_name_{i}", memory=memory)
        for i, memory in enumerate(mems_out)
    ]
    handle = MeasurementHandle.new()
    metadata = MetaData(name="demo_metadata")
    data = Data(handle=handle, datasets=datasets)
    return DataFile(data=data, metadata=metadata)


async def async_save_load_part():
    print("# Saving datafile:")
    datafile = data_file()
    print(json.dumps(json.loads(datafile.json()), indent=4))
    print("\n\n\n")
    input("Press enter to save datafile")
    print("# Starting async client to communicate")
    print(f"async with httpx.AsyncClient(base_url='{DATA_SERVER_URL}') as client:")
    async with httpx.AsyncClient(base_url=DATA_SERVER_URL) as client:
        print("    response = await client.post('/data/save', content=datafile.json())")
        response = await client.post("/data/save", content=datafile.json())
        print("    path = SavedPath(**response.json()).path")
        path = SavedPath(**response.json()).path
        print(f"    Data saved in mongo db with ObjectID: {path}")
        print("\n\n\n")
        param_list = {
            "slice_list": "[[(0,2),(0,2)]]"
        }  # list of slices for each dimension
        input("    Press enter to load data back")
        print(f"    response = await client.get('/data/load/{path}'")
        response = await client.get(
            f"/data/load_partial/{path}", params=param_list, timeout=20
        )
        print("    l_datafile = DataFile(**response.json())")
        l_datafile = DataFile(**response.json())

        print(f"    # Loaded datafile using ObjectID: {path}")
        print(json.dumps(json.loads(l_datafile.json()), indent=4))

        print("\n\n\n")
        print("Note how the loaded datafields points to the data with")
        print(f"metadata.data_path: {l_datafile.metadata.data_path}")
        print("you should be able to find this 'folder' in the GCS bucket")
        print("\n\n\n")

        print("We can also see the saved and loaded data are the same")
        print(
            "e.g. look at the first dataset (acccess DataFile.data.datasets[0].memory.to_array()[:4]"
        )
        print(f"Saved:  {datafile.data.datasets[0].memory.to_array()[:4]}")  # type: ignore
        print(f"loaded: {l_datafile.data.datasets[0].memory.to_array()[:4]}")  # type: ignore
        print("\n\n\n")

        print(l_datafile.data.datasets[-1].memory.to_array().shape)  # type: ignore
        print(
            np.all(
                l_datafile.data.datasets[-1].memory.to_array()  # type: ignore
                == datafile.data.datasets[-1].memory.to_array()  # type: ignore
            )
        )


if __name__ == "__main__":
    remove_shm_from_resource_tracker()
    print("import json")
    print("import httpx")
    print("\n\n\n")
    with open("data_server_local.json") as f:
        cfg_json = json.load(f)
        print("# Loaded config from data_server.json as cfg_json:")
        print("cfg_json = json.load(f)")
        print(json.dumps(cfg_json, indent=4))

        data_server_cfg = FrontendV1Config(**cfg_json)
        print("# Created FrontendV1Config opject called data_server_cfg")
        print("data_server_cfg = FrontendV1Config(**cfg_json)")
        print("\n\n\n")

        input("Press enter to setup data server using loaded config...")
        print("# Setting up server with loaded config")
        print("# by using the '/data/setup' command to setup the data server")
        print(
            f"httpx.post('{DATA_SERVER_URL}/data/setup;, content=data_server_cfg.json())"
        )
        response = httpx.post(
            f"{DATA_SERVER_URL}/data/setup", content=data_server_cfg.json()
        )
        if response.status_code == 201:
            print("Setup successful")
        else:
            print("Something went wrong with the setup")
        print("\n\n\n")
        input("Press enter to continue on to saving DataFile")

    asyncio.run(async_save_load_part())

    input("Press enter to reset the data server...")
    print("# using the '/data/reset' command to reset server")
    print(f"httpx.put('{DATA_SERVER_URL}/data/reset")
    response = httpx.put(f"{DATA_SERVER_URL}/data/reset")
