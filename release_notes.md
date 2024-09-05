# Release Notes

## 0.2.0
---
 - BUGFIX: AQC-952: Units of InstrumentSettings can now update with new values when updating settings - fixing a bug where units provided by users at runtime were missing from datafile.
 - AQC-864: Adding fields to SharedLists for time information.
 - BUGFIX: AQC-940: Fix a file descriptor leak where shared memories were disallocated but their file descriptors remained open leading to "Too Many Files Open" errors
 - AQC-935: Added Integration tests to Jenkins pipeline (run AQ_Measurements tests on PRs)
 - AQC-923: Update data helpers notebook query examples with station names to help awoid getting data from multiple sations when performing queries.
 - AQC-928: Notebook "save_test.ipynb" moved from AQ_Measurements to svalbard
 - AQC-919: Support for creating local versions of datafile for better local testing
 - BUGFIX: AQC-906: SharedArrays now change to numpy arrays when operated on.
 - AQC-918: SharedCounter - keeps track of how many instances of shared memory out objects exist across different processes.
 - BUGFIX: AQC-901 - shape generation of log channel (null is all changed to null is nothing)
 - BUGFIX: json serialization
 - AQC-883: InstrumentSetting values can now be lists, support stepping through a list of strings and list of lists. Functions to create InstrumentModel instances from nested pydantic model classes and create pydantic model instances from InstrumentModel instances.
 

## 0.1.0
---
 - AQC-826: Multi-QA support
 - AQC-840: Better handling of values of StepItems originating from numpy arrays
 - AQC-854: Improved data server query notebook - get_data_group moved higher
 - AQC-856: Helper methods for getting multiple object ids from MongoDB
 - AQC-843: Changing hw_sweepable to hw_swept
 - AQC-831: Improved UX for data access
 - AQC-819: Measurement Executor Support
 - AQC-754: Process name of launched app now 'svalbard_server'
 - AQC-749: New AQ Compiler data model for use with measurement executor.
 - AQC-805: Make Drain model hashable
 - AQC-795: Sweepablity enumeration: for executor usage
 - AQC-801: introduce Drains for InstrumentSettings.
 - AQC-774: Added hw_sweepable flag to StepItems.
 - AQC-773: Metadata Added add_instrument_model method.
 - AQC-762: Added pre-commit hook 
 - AQC-743: Moved instrument identity method from aq_measurements to Metadata.
 - AQC-666: HOTFIX: LogChannel shape
 - AQC-735: Added metadata add_step method.
 - AQC-727: Added index for step items.
 - MEAS-241: save different shaped data
 - AQC-401: Upgrade to Pydantic 2.x
 - AQC-592: Additional helper function capability, now also except paths to config files and new function to get data
 - HOTFIX: support complex numbers with negative imaginary component.

## 0.0.8
---
 - AQC-263: Changed name from aq_data to svalbard, prepared for moving repository to open source.

## 0.0.7
---
 - Bumped python requirement to 3.10 and version to 0.0.7
 - AQC-579: Feature: Added methods for updating data and metadata after initialization, updating data support partial update.
 - AQC-577: Feature: Data Server app now attempts using config from .aq_configs to set itself up during startup
 - AQC-267: Feature: Make automatic commits to shared binaries include repository name.
 - AQC-489: Hotfix: Fix issues importing type hint from BSON
 - AQC-505: Feature: Cooldown field in metadata.
 - AQC-583: Feature: Added id string method for InstrumentModels, string value method for InstrumentSettings, and string range method for StepItems
 - AQC-648: Feature: Save Data Methods migrated from aq_measurements to svalbard
 - AQC-658: Feature: Added data_size field to MetaData object, meta data version version bumped to 0.1.3, pinned pytest_asyncio<0.23 
 - AQC-330: Feature: Methods for integration into design teams workflow.
 - AQC-409: Feature: Added label property to Channel data model.
 - AQC-428: Feature: Introduced BaseMetaData model and model type field to descriminate metadata models on. 
 - AQC-469: Feature: updated pyproject.toml and setup.cfg
 - AQC-467: Feature: helper method for creating datasets from CSV files.
 - AQC-678: Log shape for Measurements with no step items now a single point instead of no points

## 0.0.6
---
 - AQC-407: Changed favorites field in MetaData to Flags (enumeration), meta data version bumped to 0.1.2
 - AQC-432: Feature: Synchronous Frontend V1.
 - AQC-407: Feature: Added favorites field to MetaData.
 - AQC-403: Feature: Added function to create a Data object from a Measurement object.
 - AQC-404: Feature: Added helper functions to interface with the data server.
 - AQC-327: Feature: Added support for saving arb. files to the data router.
 - AQC-325: Feature: Added support for locations of arb. files to metadata model.
 - AQC-324: Feature: Added support for arb. files in data model.
 - AQC-359: Feature: Added Measurement data model.
 - AQC-323: Feature: Added support for saving/loading arb. files in data backend.
 - AQC-576: Bugfix: Added colorama as dependency

## 0.0.5
---
 - AQC-291: Feature: Zarr Metadata of FS Backend is now consolidated sacrificing slightly longer saveing for faster loading.
 - AQC-277: Fixed: Loading files on unix that where saved of windows.
 - AQC-278: Fixed: datetime generation for metadata.
 - AQC-253: Add functions to retrieve partial data loading. (Added SliceListModel and SliceModel types and load_partial functions in data router, frontend, and backend).

## 0.0.4
---
 - AQC-120: Defined Structure of DataFile (large changes to MetaData).
 - AQC-266: Improved validator of InstrumentSetting to handle initialization without type.

## 0.0.3
---
 - Added Release Notes.
 - Improved initialization of FrontentV1 to accuretly create FS or GCS data backend.
 - AQC-246 Added option to use X509 credentials to access metadata backend. Changed remote demo to use mongoDB Atlas instead of local Dockerized mongo server.

## 0.0.2
---
 - AQC-234 Added Close methods to web api.
 - AQC-232 Updated to work with windows, added rudimentary management of shared memories.

## 0.0.1
---
 - AQC-184 Test Building with python 3.11.
 - AQC-155 Data Server Sprint review demo.
 - AQC-154 Docker Compose file for CI/CD.
 - AQC-136 datasaver front end.
 - AQC-147 change jenkinsfile in svalbard to streamline testing.
 - AQC-139 design process of streaming data.
 - AQC-138 datasaver backend for data.
 - AQC-137 datasaver backend for metadata.