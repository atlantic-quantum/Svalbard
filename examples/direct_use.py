import json

from svalbard.data_model.data_file import DataFile, MetaData
from svalbard.data_server.frontend.sync_frontend_v1 import SyncFrontendV1Config

config_path = "path/to/config.json"
data_file = DataFile(metadata=MetaData(name="example"))

# Create a frontend
with open(config_path, "r") as config_file:
    config_json = json.load(config_file)
    config = SyncFrontendV1Config(**config_json)
    frontend = config.init()

# Save data
data_path = frontend.save(data_file)

# Load data
loaded_data_file = frontend.load(data_path)  # type: ignore

docs: list[dict[str, str]] = [{"_id": "......"}]


loaded_data_file = frontend.load(str(docs[0]["_id"]))  # type: ignore
