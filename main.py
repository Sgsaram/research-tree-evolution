import os
import shutil
import pathlib
import datetime
import dotenv
import matplotlib.pyplot as plt
import numpy as np
import PIL.Image

from core import SentinelHubMaster

dotenv.load_dotenv(dotenv.find_dotenv())
BASE_DIR = os.getcwd()
OUTPUT_IMAGES_DIR = os.path.join(BASE_DIR, "output_images")
SH_CLIENT_ID = os.environ.get("SH_CLIENT_ID", "")
SH_CLIENT_SECRET = os.environ.get("SH_CLIENT_SECRET", "")
EXAMPLE_COORDS = (
    47.4427,
    56.0485,
    47.5852,
    55.9740,
) # (Lng, Lat)

def remove_folder_content(path: str):
    for filename in os.listdir(path):
        file_path = os.path.join(path, filename)
        if os.path.isfile(file_path) or os.path.islink(file_path):
            os.unlink(file_path)

def main():
    remove_folder_content(OUTPUT_IMAGES_DIR)
    time_intervals = [
        ("2022-08-01", "2022-08-30"),
        ("2022-09-01", "2022-09-30"),
        ("2022-10-01", "2022-10-30"),
    ]

    shmaster = SentinelHubMaster(
        sh_client_id=SH_CLIENT_ID,
        sh_client_secret=SH_CLIENT_SECRET,
    )
    print(SentinelHubMaster.get_pixel_size(
        coords=EXAMPLE_COORDS,
        resolution=40,
    ))
    tc_images = shmaster.get_box_tc_images(
        coords=EXAMPLE_COORDS,
        time_intervals=time_intervals,
        data_collection="sentinel2_l1c",
        resolution=40,
    )
    # for ind, image_array in enumerate(tc_images):
    #     save_image = PIL.Image.fromarray(image_array)
    #     save_image.save(
    #         os.path.join(OUTPUT_IMAGES_DIR, f"img_{ind}.png"),
    #     )

if __name__ == "__main__":
    main()
