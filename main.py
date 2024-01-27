import datetime
import math
import os
import typing

import cv2 as cv
import cv2.typing
import dotenv
import eolearn.core
import eolearn.io
import joblib
import matplotlib.pyplot as plt
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


class AddValidDataMaskTask(eolearn.core.eotask.EOTask):
    def execute(self, eopatch: eolearn.core.eodata.EOPatch)
        eopatch.mask["validData"] = eopatch.mask["dataMask"].astype(
            bool
        ) & ~eopatch.mask["CLM"].astype(bool)
        return eopatch


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
            (eolearn.core.constants.FeatureType.MASK, "CLM"),
        ],
        size=(128, 128),
        resolution=20,
        time_difference=datetime.timedelta(hours=12),
        config=config,
        max_threads=5,
        mosaicking_order=sh.constants.MosaickingOrder.LEAST_CC,
        maxcc=1,
        cache_folder=EOLEARN_CACHE_FOLDER,
    )
    output_task = eolearn.core.eoworkflow_tasks.OutputTask("eopatch")
    addition_valid_data_task = AddValidDataMaskTask()
    workflow_nodes = eolearn.core.eonode.linearly_connect_tasks(
        input_task,
        addition_valid_data_task,
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

        def __call__(self, a) -> typing.Any:
            if self.data_col == "l1c":
                return min(255, max(0, math.cbrt(0.6 * a - 0.035)) * 255)
            elif self.data_col == "l2a":
                return min(255, max(0, math.cbrt(0.6 * a) * 255))
            return ValueError(f"Wrong data coolection {self.data_col}")

    v_min = np.vectorize(min)
    eopatch: eolearn.core.eodata.EOPatch = typing.cast(eolearn.core.eodata.EOPatch, result.outputs["eopatch"])
    time_data: list[datetime.datetime] = eopatch.timestamps.copy()
    mask_data = eopatch.mask["dataMask"].copy()
    tc_data = v_min(eopatch.data["L1C_data"].copy() * 2.5 * 255, 255).astype(np.uint8)
    # honc_data = v_process_func(eopatch.data["L1C_data"].copy()).astype(np.uint8)
    print(tc_data.shape, mask_data.shape)
    for ind, time in enumerate(time_data):
        time_str = time.isoformat().replace(":", "-")
        save_image = PIL.Image.fromarray(
            cv.cvtColor(tc_data[ind], cv.COLOR_RGB2GRAY), mode="L"
        )
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
    data = joblib.load(os.path.join(SEQ_DIR, "data_save.sav"))

    def get_gray_scale(a):  # rgb
        return np.round(
            0.299 * a[0] + 0.587 * a[1] + 0.114 * a[2],
        )

    def gray_pixel_dist(a, b):
        return abs(get_gray_scale(a) - get_gray_scale(b))

    def process_pair(image, mask):
        k = 2
        d = 20
        blured_image = cv.medianBlur(image, 9)
        line_element = np.zeros(image.shape[:2], dtype=np.uint32)
        res_image = np.zeros(image.shape[:2], dtype=bool)
        image_line = []
        for x in range(image.shape[0]):
            for y in range(image.shape[1]):
                if mask[x, y][0]:
                    line_element[x, y] = len(image_line)
                    image_line.append(np.float32(blured_image[x, y]))
        image_line = np.array(image_line)
        if len(image_line) == 0:
            return res_image
        criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 20, 0.5)
        _, labels, centers = cv.kmeans(
            image_line,
            k,
            typing.cast(cv2.typing.MatLike, None),
            criteria,
            20,
            cv.KMEANS_RANDOM_CENTERS,
        )
        gray_centers = np.zeros(k, dtype=np.float32)
        for i in range(len(centers)):
            gray_centers[i] = get_gray_scale(np.uint8(centers[i]))
        truth_center = max(gray_centers)
        if abs(gray_centers[0] - gray_centers[1]) < d:
            return res_image
        line_res = gray_centers[labels.flatten()]
        for x in range(image.shape[0]):
            for y in range(image.shape[1]):
                if mask[x, y][0] and line_res[line_element[x, y]] == truth_center:
                    res_image[x, y] = True
        return res_image

    final_masks = []
    images_to_show = []
    for tc, mask, _ in data:
        cur_mask = utils.cluster_array_to_image(
            process_pair(tc, mask),
        )
        if len(final_masks) > 0:
            cur_mask |= final_masks[-1]
        final_masks.append(cur_mask)
        images_to_show.append(tc)
        images_to_show.append(final_masks[-1])

    utils.show_images(images_to_show, fig_size=(50, 50), row_size=18)
    plt.savefig(os.path.join(OUTPUT_IMAGES_DIR, "figure_0.png"))


if __name__ == "__main__":
    main()
