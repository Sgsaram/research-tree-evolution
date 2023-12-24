import datetime
import os

import cv2 as cv
import eolearn.core
import eolearn.io
import matplotlib.pyplot as plt
import numpy as np
import PIL.GifImagePlugin
import PIL.Image
import sentinelhub as sh


class CommunicationClient:
    class __AddValidDataMaskTask(eolearn.core.eotask.EOTask):
        def execute(self, eopatch: eolearn.core.eodata.EOPatch):
            eopatch.mask["validMask"] = eopatch.mask["dataMask"].astype(
                bool
            ) & ~eopatch.mask["CLM"].astype(bool)
            return eopatch

    __STR_TO_DATA_COLLECTION = {
        "sentinel2_l1c": sh.data_collections.DataCollection.SENTINEL2_L1C,
        "sentinel2_l2a": sh.data_collections.DataCollection.SENTINEL2_L2A,
    }

    def __init__(
        self,
        sh_client_id: str,
        sh_client_secret: str,
        cache_folder: str | None = None,
    ) -> None:
        self.config = sh.config.SHConfig(
            sh_client_id=sh_client_id,
            sh_client_secret=sh_client_secret,
            use_defaults=True,
        )
        self.cache_folder = cache_folder

    def get_data_otp(
        self,
        coords: tuple[float, float, float, float],
        time_interval: tuple[datetime.date, datetime.date],
        data_collection: str = "sentinel2_l1c",
        resolution: float | None = 20,
        size: tuple[int, int] | None = None,
        time_difference: datetime.timedelta = datetime.timedelta(hours=12),
    ) -> list[tuple[np.ndarray, np.ndarray, np.ndarray]]:
        """
        Returns array of true color images and
        valid masks (cloud coverage + data zones)
        in one time period data collection
        """
        aoi_bbox = sh.geometry.BBox(
            bbox=coords,
            crs=sh.constants.CRS.WGS84,
        )
        input_task = eolearn.io.sentinelhub_process.SentinelHubInputTask(
            data_collection=CommunicationClient.__STR_TO_DATA_COLLECTION[
                data_collection
            ],
            bands=["B04", "B03", "B02"],
            bands_feature=(eolearn.core.constants.FeatureType.DATA, "sentinel_data"),
            additional_data=[
                (eolearn.core.constants.FeatureType.MASK, "dataMask"),
                (eolearn.core.constants.FeatureType.MASK, "CLM"),
            ],
            size=size,
            resolution=resolution,
            time_difference=time_difference,
            config=self.config,
            max_threads=5,
            mosaicking_order=sh.constants.MosaickingOrder.LEAST_CC,
            cache_folder=self.cache_folder,
        )
        add_valid_data_task = CommunicationClient.__AddValidDataMaskTask()
        output_task = eolearn.core.eoworkflow_tasks.OutputTask("eopatch")
        input_node = eolearn.core.eonode.EONode(input_task)
        add_valid_data_node = eolearn.core.eonode.EONode(
            add_valid_data_task,
            inputs=[input_node],
        )
        output_node = eolearn.core.eonode.EONode(
            output_task,
            inputs=[add_valid_data_node],
        )
        workflow = eolearn.core.eoworkflow.EOWorkflow(
            [
                input_node,
                add_valid_data_node,
                output_node,
            ],
        )
        result = workflow.execute(
            {
                input_node: {
                    "bbox": aoi_bbox,
                    "time_interval": time_interval,
                },
            },
        )
        v_min = np.vectorize(min)
        eopatch: eolearn.core.eodata.EOPatch = result.outputs["eopatch"]
        return list(
            zip(
                v_min(eopatch.data["sentinel_data"] * 255 * 3.5, 255).astype(np.uint8),
                eopatch.mask["validMask"],
                np.array(eopatch.timestamps),
            )
        )

    def get_data_mtp(
        self,
        coords: tuple[float, float, float, float],
        time_intervals: list[tuple[datetime.date, datetime.date]],
        data_collection: str = "sentinel2_l1c",
        resolution: float | None = 20,
        size: tuple[int, int] | None = None,
        time_difference: datetime.timedelta = datetime.timedelta(hours=12),
    ) -> list[tuple[np.ndarray, np.ndarray, np.ndarray]]:
        res = []
        for timep in time_intervals:
            res.extend(
                self.get_data_otp(
                    coords,
                    timep,
                    data_collection,
                    resolution,
                    size,
                    time_difference,
                )
            )
        return res


def cluster_array_to_image(
    image: np.ndarray,
    colors: list[tuple] = [
        (0, 0, 0),
        (249, 65, 68),
        (144, 190, 109),
        (249, 199, 79),
    ],
) -> np.ndarray:
    cur_new_image = np.zeros(
        image.shape[:2] + (3,),
        dtype=np.uint8,
    )
    for x in range(image.shape[0]):
        for y in range(image.shape[1]):
            for k in range(3):
                if isinstance(image[x, y], list) or isinstance(image[x, y], np.ndarray):
                    cur_val = image[x, y][0]
                else:
                    cur_val = image[x, y]
                cur_new_image[x, y, k] = colors[cur_val][k]
    return cur_new_image


def cluster_arrays_to_images(
    images: np.ndarray,
    colors: list[tuple] = [
        (0, 0, 0),
        (249, 65, 68),
        (144, 190, 109),
        (249, 199, 79),
    ],
) -> np.ndarray:
    res = []
    for i in range(images.shape[0]):
        res.append(cluster_array_to_image(images[i], colors))
    return np.array(res)


def remove_folder_content(
    path: str,
) -> None:
    """
    EXAMPLE:
    remove_folder_content(OUTPUT_IMAGES_DIR)
    """
    for filename in os.listdir(path):
        file_path = os.path.join(path, filename)
        if os.path.isfile(file_path) or os.path.islink(file_path):
            os.unlink(file_path)


def show_images(
    images: np.ndarray,
    row_size: int = 6,
    fig_size: tuple[int, int] = (25, 30),
) -> None:
    """
    fig_size: (x, y), where x is width, and y - height
    EXAMPLE:
    show_folder_images(OUTPUT_IMAGES_DIR, SEQ_SIZE)
    """
    n_size = len(images)
    global_size = (n_size + row_size - 1) // row_size * row_size
    fig, table = plt.subplots(
        (n_size + row_size - 1) // row_size,
        row_size,
        squeeze=False,
    )
    fig.set_size_inches(fig_size)
    for frame in range(global_size):
        table[frame // row_size][frame % row_size].axis("off")
        if frame >= n_size:
            continue
        table[frame // row_size][frame % row_size].imshow(images[frame])


def process_image(
    image: np.ndarray,
) -> np.ndarray:
    k = 2
    blured_image = cv.medianBlur(image, 21)
    proc_image = np.float32(blured_image.reshape((-1, 3)))
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 20, 0.5)
    comp, labels, centers = cv.kmeans(
        proc_image,
        k,
        None,
        criteria,
        20,
        cv.KMEANS_RANDOM_CENTERS,
    )

    def get_gray_scale(r, g, b):
        return np.round(
            0.299 * r + 0.587 * g + 0.114 * b,
        )

    centers = np.uint8(centers)
    res = centers[labels.flatten()]
    return res.reshape(image.shape)


# def process_image(
#     image: np.ndarray,
# ) -> np.ndarray:
#     """
#     CHANGE THIS PART OF CODE
#     """
#     k = 3
#     min_dist_gray = 40
#     blurred_image = cv.medianBlur(image, 21)
#     proc_image = np.float32(blurred_image.reshape((-1, 3)))
#     criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 10, 0.5)
#     comp, labels, centers = cv.kmeans(
#         proc_image,
#         k,
#         None,
#         criteria,
#         10,
#         cv.KMEANS_RANDOM_CENTERS,
#     )
#     int_center = np.uint8(centers)
#     gray_center = np.zeros(2, dtype=np.uint8)
#     for i in range(len(int_center)):
#         gray_center[i] = np.round(
#             0.114 * int_center[i][0]
#             + 0.587 * int_center[i][1]
#             + 0.299 * int_center[i][2],
#         )
#     if abs(np.int16(gray_center[0]) - np.int16(gray_center[1])) < min_dist_gray:
#         return np.ones(image.shape[:2], dtype=bool)
#     else:
#         match_center = np.max(gray_center)
#         res = int_center[labels.flatten()]
#         res2 = res.reshape(image.shape)
#         gray_final_image = cv.cvtColor(res2, cv.COLOR_BGR2GRAY)
#         return gray_final_image == match_center


def process_images(
    images: np.ndarray,
) -> np.ndarray:
    res = []
    for i in range(images.shape[0]):
        res.append(process_image(images[i]))
    return np.array(res)


def split_gif(
    from_path: str,
    to_path: str,
) -> None:
    """
    EXAMPLE:
    split_gif(
        os.path.join(BASE_DIR, "assets/real_data_1/cropped-seq-3.gif"),
        os.path.join(BASE_DIR, "assets/real_data_1"),
    )
    """
    image_object = PIL.Image.open(from_path)
    for frame in range(image_object.n_frames):
        image_object.seek(frame)
        image_object.save(
            os.path.join(
                to_path,
                f"image_{frame}.png",
            )
        )
