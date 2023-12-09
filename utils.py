import os
from typing import Tuple

import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
import PIL.GifImagePlugin
import PIL.Image


def image_to_rgb_mask(
    image: np.ndarray, color: Tuple[int, int, int] = (255, 105, 97)
) -> np.ndarray:
    cur_new_image = np.zeros(
        image.shape + (3,),
        dtype=np.uint8,
    )
    for x in range(image.shape[0]):
        for y in range(image.shape[1]):
            if image[x, y]:
                for k in range(3):
                    cur_new_image[x, y, k] = color[k]
    return cur_new_image


def images_to_rgb_mask(
    images: np.ndarray, color: Tuple[int, int, int] = (255, 105, 97)
) -> np.ndarray:
    res = []
    for i in range(images.shape[0]):
        res.append(image_to_rgb_mask(images[i], color))
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
    fig_size: Tuple[int, int] = (25, 30),
) -> None:
    """
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
    """
    CHANGE THIS PART OF CODE
    """
    k = 2
    min_dist_gray = 40
    blurred_image = cv.medianBlur(image, 21)
    proc_image = np.float32(blurred_image.reshape((-1, 3)))
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.5)
    comp, labels, centers = cv.kmeans(
        proc_image,
        k,
        None,
        criteria,
        10,
        cv.KMEANS_RANDOM_CENTERS,
    )
    int_center = np.uint8(centers)
    gray_center = np.zeros(2, dtype=np.uint8)
    for i in range(len(int_center)):
        gray_center[i] = np.round(
            0.114 * int_center[i][0]
            + 0.587 * int_center[i][1]
            + 0.299 * int_center[i][2],
        )
    if abs(np.int16(gray_center[0]) - np.int16(gray_center[1])) < min_dist_gray:
        return np.ones(image.shape[:2], dtype=bool)
    else:
        match_center = np.max(gray_center)
        res = int_center[labels.flatten()]
        res2 = res.reshape(image.shape)
        gray_final_image = cv.cvtColor(res2, cv.COLOR_BGR2GRAY)
        return gray_final_image == match_center


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
