import os
import shutil
import pathlib
import datetime
import dotenv
import matplotlib.pyplot as plt
import numpy as np
import PIL.Image
import PIL.GifImagePlugin
import eolearn.io
import eolearn.core
import sentinelhub as sh
import datetime
import math

from typing import Any, List

dotenv.load_dotenv(dotenv.find_dotenv())
BASE_DIR = os.getcwd()
OUTPUT_IMAGES_DIR = os.path.join(BASE_DIR, "output_images")
EOLEARN_CACHE_FOLDER = os.path.join(BASE_DIR, ".eolearn_cache")
SH_CLIENT_ID = os.environ.get("SH_CLIENT_ID", "")
SH_CLIENT_SECRET = os.environ.get("SH_CLIENT_SECRET", "")
EXAMPLE_COORDS = (
    90.22093,
    56.72387,
    90.28968,
    56.69019,
) # (Lng, Lat)
EXAMPLE_TIME_INTERVAL = (
    "2020-12-01",
    "2021-04-01",
)

def remove_folder_content(path: str):
    for filename in os.listdir(path):
        file_path = os.path.join(path, filename)
        if os.path.isfile(file_path) or os.path.islink(file_path):
            os.unlink(file_path)

def save_eodata():
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
        resolution=20,
        time_difference=datetime.timedelta(weeks=1),
        config=config,
        max_threads=5,
        mosaicking_order=sh.constants.MosaickingOrder.LEAST_CC,
        maxcc=1,
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

    class high_opt_nat_color:
        def __init__(self, data_col: str) -> None:
            self.data_col = data_col.lower()
        
        def __call__(self, a) -> Any:
            if self.data_col == "l1c":
                return min(255, max(0, math.cbrt(0.6 * a - 0.035)) * 255)
            elif self.data_col == "l2a":
                return min(255, max(0, math.cbrt(0.6 * a) * 255))
            return ValueError(f"Wrong data coolection {self.data_col}")
    
    v_process_func = np.vectorize(high_opt_nat_color("l1c"))
    v_min = np.vectorize(min)
    eopatch: eolearn.core.eodata.EOPatch = result.outputs["eopatch"]
    time_data: List[datetime.datetime] = eopatch.timestamps.copy()
    mask_data = eopatch.mask["dataMask"].copy()
    tc_data = v_min(eopatch.data["L1C_data"].copy() * 2.5 * 255, 255).astype(np.uint8)
    honc_data = v_process_func(eopatch.data["L1C_data"].copy()).astype(np.uint8)
    print(tc_data.shape, mask_data.shape)
    for ind, time in enumerate(time_data):
        time_str = time.isoformat().replace(":", "-")
        save_image = PIL.Image.fromarray(tc_data[ind])
        save_image.save(
            os.path.join(
                OUTPUT_IMAGES_DIR,
                f"{time_str}.png",
            ),
        )
        save_image = PIL.Image.fromarray(honc_data[ind])
        save_image.save(
            os.path.join(
                OUTPUT_IMAGES_DIR,
                f"honc_{time_str}.png",
            ),
        )

def main():
    image_object = PIL.Image.open(
        os.path.join(BASE_DIR, "assets/real_data_1/cropped-seq-3.gif"),
    )
    for frame in range(image_object.n_frames):
        image_object.seek(frame)
        image_object.save(
            os.path.join(
                BASE_DIR,
                f"assets/real_data_1/image_{frame}.png",
            )
        )


if __name__ == "__main__":
    main()
