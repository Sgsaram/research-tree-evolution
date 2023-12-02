import numpy as np
import sentinelhub

class SentinelHubMaster:
    to_dict = {}

    def __init__(
        self,
        sh_client_id,
        sh_client_secret,
    ) -> None:
        if not sh_client_id or not sh_client_secret:
            raise ValueError("Some credentials are not provided")
        self.config = sentinelhub.SHConfig(
            sh_client_id=sh_client_id,
            sh_client_secret=sh_client_secret,
            use_defaults=True,
        )
    
    def get_box_array_image(
        self,
        coords: tuple[float, float, float, float],
        time_interval: tuple[str, str],
        data_collection: str,
        resolution = 100, # "resolution" meters per pixel
    ) -> np.ndarray:
        n_bbox = sentinelhub.BBox(
            bbox=coords,
            crs=sentinelhub.CRS.WGS84,
        )
        n_size = sentinelhub.bbox_to_dimensions(
            n_bbox,
            resolution=resolution,
        )
        request_true_color = sentinelhub.SentinelHubRequest(
            evalscript=SentinelHubMaster.get_true_color_evalscript(
                factor=3.5
            ),
            input_data=[
                sentinelhub.SentinelHubRequest.input_data(
                    data_collection=sentinelhub.DataCollection.SENTINEL2_L1C,
                    time_interval=time_interval,
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
            config=self.config,
        )
        return request_true_color.get_data()[0]
    
    @staticmethod
    def get_true_color_evalscript(factor = 2.5) -> str:
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
                return [sample.B04 * {0},
                        sample.B03 * {0},
                        sample.B02 * {0}];
            }}
        """.format(factor)