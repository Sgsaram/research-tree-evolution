import os
import shutil
import pathlib
import datetime
import dotenv
import sentinelhub
import matplotlib.pyplot as plt
import numpy as np
import PIL.Image

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

def get_true_color_evalscript(factor = 2.5):
    return """
        //VERSION=3
        function setup() {{
            return {{
                input: [{{
                    bands: ["B02", "B03", "B04"]
                }}],
                output: {{
                    bands: 3
                }}
            }};
        }}
        function evaluatePixel(sample) {{
            return [sample.B04 * {0}, sample.B03 * {0}, sample.B02 * {0}];
        }}
    """.format(factor)

def remove_folder_content(path: str):
    for filename in os.listdir(path):
        file_path = os.path.join(path, filename)
        if os.path.isfile(file_path) or os.path.islink(file_path):
            os.unlink(file_path)

# def plot_image(
#     image: np.ndarray,
#     factor: float = 1.0,
#     clip_range: tuple[float, float] | None = None,
#     **kwargs,
# ) -> None:
#     """Utility function for plotting RGB images."""
#     _, ax = plt.subplots(nrows=1, ncols=1, figsize=(15, 15))
#     if clip_range is not None:
#         ax.imshow(np.clip(image * factor, *clip_range), **kwargs)
#     else:
#         ax.imshow(image * factor, **kwargs)
#     ax.set_xticks([])
#     ax.set_yticks([])

def main():
    remove_folder_content(OUTPUT_IMAGES_DIR)
    config = sentinelhub.SHConfig(
        sh_client_id=SH_CLIENT_ID,
        sh_client_secret=SH_CLIENT_SECRET,
        use_defaults=True,
    )
    if not config.sh_client_id or not config.sh_client_secret:
        print("Some credentials are not provided")
        return
    resolution=60
    n_bbox = sentinelhub.BBox(
        bbox=EXAMPLE_COORDS,
        crs=sentinelhub.CRS.WGS84,
    )
    n_size = sentinelhub.bbox_to_dimensions(
        n_bbox,
        resolution=resolution,
    )
    print("Size -> ", n_size)
    request_true_color = sentinelhub.SentinelHubRequest(
        evalscript=get_true_color_evalscript(factor=3.5),
        input_data=[
            sentinelhub.SentinelHubRequest.input_data(
                data_collection=sentinelhub.DataCollection.SENTINEL2_L1C,
                time_interval=("2023-07-11", "2023-07-14"),
            )
        ],
        responses=[
            sentinelhub.SentinelHubRequest.output_response(
                "default",
                sentinelhub.MimeType.PNG,
            ),
        ],
        bbox=n_bbox,
        size=n_size,
        config=config,
    )
    true_color_image_arrays = request_true_color.get_data()
    print("Length -> ", len(true_color_image_arrays))
    for ind, image_array in enumerate(true_color_image_arrays):
        save_image = PIL.Image.fromarray(image_array)
        save_image.save(
            os.path.join(OUTPUT_IMAGES_DIR, f"img_{ind}.png"),
        )

if __name__ == "__main__":
    main()
