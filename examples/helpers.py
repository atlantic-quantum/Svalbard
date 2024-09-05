import json
from pathlib import Path

from svalbard.data_server.metadata_backend.mongodb_backend import MongoDBConfig
from svalbard.utility.data_server_helper_functions import (
    get_many_documents,
    get_number_of_documents,
)

with Path(__file__).parent.joinpath("data_server_local.json").open("r") as f:
    config = MongoDBConfig(**json.load(f)["metadata_backend"])
    config.certificate = None

    print(get_number_of_documents(config))
    print(get_many_documents(config, {}, projection={"_id": 1, "name": 1})[0])
