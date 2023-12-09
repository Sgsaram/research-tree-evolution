import os
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
import PIL.GifImagePlugin
import PIL.Image


def images_to_rgb_mask(
    images: np.ndarray, color: Tuple[int, int, int] = (255, 105, 97)
) -> np.ndarray:
    res = []
    n_shape = images.shape
    for i in range(n_shape[0]):
        cur_new_image = np.zeros(
            n_shape[1:] + (3,),
            dtype=np.uint8,
        )
        for x in range(n_shape[1]):
            for y in range(n_shape[2]):
                if images[i, x, y]:
                    for k in range(3):
                        cur_new_image[x, y, k] = color[k]
        res.append(cur_new_image)
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
    )
    fig.set_size_inches(fig_size)
    for frame in range(global_size):
        table[frame // row_size][frame % row_size].axis("off")
        if frame >= n_size:
            continue
        table[frame // row_size][frame % row_size].imshow(images[frame])


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
