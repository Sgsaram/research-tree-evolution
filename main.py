import os
import shutil
import pathlib
import datetime
import dotenv
import matplotlib.pyplot as plt
import numpy as np
import PIL.Image
import eolearn.io
import eolearn.core
import sentinelhub as sh
import datetime

from typing import List

dotenv.load_dotenv(dotenv.find_dotenv())
BASE_DIR = os.getcwd()
OUTPUT_IMAGES_DIR = os.path.join(BASE_DIR, "output_images")
EOLEARN_CACHE_FOLDER = os.path.join(BASE_DIR, ".eolearn_cache")
SH_CLIENT_ID = os.environ.get("SH_CLIENT_ID", "")
SH_CLIENT_SECRET = os.environ.get("SH_CLIENT_SECRET", "")
EXAMPLE_COORDS = (
    47.4427,
    56.0485,
    47.5852,
    55.9740,
) # (Lng, Lat)
EXAMPLE_TIME_INTERVAL = (
    "2022-08-01",
    "2022-08-10",
)

def remove_folder_content(path: str):
    for filename in os.listdir(path):
        file_path = os.path.join(path, filename)
        if os.path.isfile(file_path) or os.path.islink(file_path):
            os.unlink(file_path)

def main():
    remove_folder_content(OUTPUT_IMAGES_DIR)
    config = sh.config.SHConfig(
        sh_client_id=SH_CLIENT_ID,
        sh_client_secret=SH_CLIENT_SECRET,
        use_defaults=True,
    )
    aoi_bbox = sh.geometry.BBox(
        bbox=EXAMPLE_COORDS,
        crs=sh.constants.CRS.WGS84,
    )
    input_task = eolearn.io.sentinelhub_process.SentinelHubInputTask(
        data_collection=sh.data_collections.DataCollection.SENTINEL2_L1C,
        bands=["B04", "B03", "B02"],
        bands_feature=(eolearn.core.constants.FeatureType.DATA, "L1C_data"),
        additional_data=[
            (eolearn.core.constants.FeatureType.MASK, "dataMask"),
        ],
        resolution=60,
        time_difference=datetime.timedelta(hours=12),
        config=config,
        max_threads=5,
        mosaicking_order=sh.constants.MosaickingOrder.LEAST_CC,
        cache_folder=EOLEARN_CACHE_FOLDER,
    )
    output_task = eolearn.core.eoworkflow_tasks.OutputTask("eopatch")
    workflow_nodes = eolearn.core.eonode.linearly_connect_tasks(
        input_task,
        output_task,
    )
    workflow = eolearn.core.eoworkflow.EOWorkflow(workflow_nodes)
    result = workflow.execute(
        {
            workflow_nodes[0]: {
                "bbox": aoi_bbox,
                "time_interval": EXAMPLE_TIME_INTERVAL,
            },
        },
    )
    eopatch: eolearn.core.eodata.EOPatch = result.outputs["eopatch"]
    time_data: List[datetime.datetime] = eopatch.timestamps.copy()
    mask_data = eopatch.mask["dataMask"].copy()
    tc_data = (eopatch.data["L1C_data"].copy() * 3.5 * 255).astype(np.uint8)
    print(tc_data.shape, mask_data.shape)
    print(time_data)
    for ind, time in enumerate(time_data):
        time_str = time.isoformat().replace(":", "-")
        save_image = PIL.Image.fromarray(tc_data[ind])
        save_image.save(
            os.path.join(
                OUTPUT_IMAGES_DIR,
                f"{time_str}.png",
            ),
        )


if __name__ == "__main__":
    main()
