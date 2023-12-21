import datetime
import math
import os
from typing import Any, List

import cv2 as cv
import dotenv
import eolearn.core
import eolearn.io
import numpy as np
import PIL.GifImagePlugin
import PIL.Image
import sentinelhub as sh
import utils

dotenv.load_dotenv(dotenv.find_dotenv())
BASE_DIR = os.getcwd()
SEQ_DIR = os.path.join(os.getcwd(), "assets/real_data_1")
SEQ_SIZE = 122
OUTPUT_IMAGES_DIR = os.path.join(BASE_DIR, "output_images")
EOLEARN_CACHE_FOLDER = os.path.join(BASE_DIR, ".eolearn_cache")
SH_CLIENT_ID = os.environ.get("SH_CLIENT_ID", "")
SH_CLIENT_SECRET = os.environ.get("SH_CLIENT_SECRET", "")
EXAMPLE_COORDS = (
    42.67107,
    59.59354,
    42.72840,
    59.54539,
)  # (Lng, Lat)
EXAMPLE_TIME_INTERVAL = (
    "2017-02-06",
    "2017-02-06",
)


def save_eodata():
    utils.remove_folder_content(OUTPUT_IMAGES_DIR)
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
        data_collection=sh.data_collections.DataCollection.SENTINEL2_L2A,
        bands=["B04", "B03", "B02"],
        bands_feature=(eolearn.core.constants.FeatureType.DATA, "L1C_data"),
        additional_data=[
            (eolearn.core.constants.FeatureType.MASK, "dataMask"),
        ],
        size=(128, 128),
        # resolution=20,
        time_difference=datetime.timedelta(hours=12),
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

    class HighOptNatColor:
        def __init__(self, data_col: str) -> None:
            self.data_col = data_col.lower()

        def __call__(self, a) -> Any:
            if self.data_col == "l1c":
                return min(255, max(0, math.cbrt(0.6 * a - 0.035)) * 255)
            elif self.data_col == "l2a":
                return min(255, max(0, math.cbrt(0.6 * a) * 255))
            return ValueError(f"Wrong data coolection {self.data_col}")

    v_process_func = np.vectorize(HighOptNatColor("l1c"))
    v_min = np.vectorize(min)
    eopatch: eolearn.core.eodata.EOPatch = result.outputs["eopatch"]
    time_data: List[datetime.datetime] = eopatch.timestamps.copy()
    mask_data = eopatch.mask["dataMask"].copy()
    tc_data = v_min(eopatch.data["L1C_data"].copy() * 2.5 * 255, 255).astype(np.uint8)
    # honc_data = v_process_func(eopatch.data["L1C_data"].copy()).astype(np.uint8)
    print(tc_data.shape, mask_data.shape)
    for ind, time in enumerate(time_data):
        time_str = time.isoformat().replace(":", "-")
        save_image = PIL.Image.fromarray(cv.cvtColor(tc_data[ind], cv.COLOR_RGB2GRAY), mode="L")
        save_image.save(
            os.path.join(
                OUTPUT_IMAGES_DIR,
                f"{time_str}.png",
            ),
        )


def process_kmeans():
    k = 2
    for frame in range(SEQ_SIZE):
        image = cv.imread(os.path.join(SEQ_DIR, f"image_{frame}.png"))
        assert image is not None
        proc_image = np.float32(image.reshape((-1, 3)))
        criteria = (
            cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER,
            30,
            0.5,
        )
        comp, labels, centers = cv.kmeans(
            proc_image,
            k,
            None,
            criteria,
            30,
            cv.KMEANS_RANDOM_CENTERS,
        )
        center = np.uint8(centers)
        res = center[labels.flatten()]
        res2 = res.reshape(image.shape)
        gray_final_image = cv.cvtColor(res2, cv.COLOR_BGR2GRAY)
        image_bin = PIL.Image.fromarray(gray_final_image, "L")
        image_bin.save(
            os.path.join(
                OUTPUT_IMAGES_DIR,
                f"image_{frame}.png",
            ),
        )


def main():
    save_eodata()
    


if __name__ == "__main__":
    main()
