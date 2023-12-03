import numpy as np
import sentinelhub as sh


class SentinelHubMaster:
    # possible data_collections
    TO_DATA_COLLECTION = {
        "SENTINEL2_L1C": sh.data_collections.DataCollection.SENTINEL2_L1C,
        "SENTINEL2_L2A": sh.data_collections.DataCollection.SENTINEL2_L2A,
        "SENTINEL1": sh.data_collections.DataCollection.SENTINEL1,
        "SENTINEL1_IW": sh.data_collections.DataCollection.SENTINEL1_IW,
        "SENTINEL1_IW_ASC": sh.data_collections.DataCollection.SENTINEL1_IW_ASC,
        "SENTINEL1_IW_DES": sh.data_collections.DataCollection.SENTINEL1_IW_DES,
        "SENTINEL1_EW": sh.data_collections.DataCollection.SENTINEL1_EW,
        "SENTINEL1_EW_ASC": sh.data_collections.DataCollection.SENTINEL1_EW_ASC,
        "SENTINEL1_EW_DES": sh.data_collections.DataCollection.SENTINEL1_EW_DES,
        "SENTINEL1_EW_SH": sh.data_collections.DataCollection.SENTINEL1_EW_SH,
        "SENTINEL1_EW_SH_ASC": sh.data_collections.DataCollection.SENTINEL1_EW_SH_ASC,
        "SENTINEL1_EW_SH_DES": sh.data_collections.DataCollection.SENTINEL1_EW_SH_DES,
        "DEM": sh.data_collections.DataCollection.DEM,
        "DEM_MAPZEN": sh.data_collections.DataCollection.DEM_MAPZEN,
        "DEM_COPERNICUS_30": sh.data_collections.DataCollection.DEM_COPERNICUS_30,
        "DEM_COPERNICUS_90": sh.data_collections.DataCollection.DEM_COPERNICUS_90,
        "MODIS": sh.data_collections.DataCollection.MODIS,
        "LANDSAT_MSS_L1": sh.data_collections.DataCollection.LANDSAT_MSS_L1,
        "LANDSAT_TM_L1": sh.data_collections.DataCollection.LANDSAT_TM_L1,
        "LANDSAT_TM_L2": sh.data_collections.DataCollection.LANDSAT_TM_L2,
        "LANDSAT_ETM_L1": sh.data_collections.DataCollection.LANDSAT_ETM_L1,
        "LANDSAT_ETM_L2": sh.data_collections.DataCollection.LANDSAT_ETM_L2,
        "LANDSAT_OT_L1": sh.data_collections.DataCollection.LANDSAT_OT_L1,
        "LANDSAT_OT_L2": sh.data_collections.DataCollection.LANDSAT_OT_L2,
        "SENTINEL5P": sh.data_collections.DataCollection.SENTINEL5P,
        "SENTINEL3_OLCI": sh.data_collections.DataCollection.SENTINEL3_OLCI,
        "SENTINEL3_SLSTR": sh.data_collections.DataCollection.SENTINEL3_SLSTR,
    }

    def __init__(
        self,
        sh_client_id: str,
        sh_client_secret: str,
    ) -> None:
        if not sh_client_id or not sh_client_secret:
            raise ValueError("Some credentials are not provided")
        self.config = sh.config.SHConfig(
            sh_client_id=sh_client_id,
            sh_client_secret=sh_client_secret,
            use_defaults=True,
        )
    
    def get_box_tc_images(
        self,
        coords: tuple[float, float, float, float],
        time_intervals: list[tuple[str, str]],
        data_collection: str,
        format: str = "png",
        factor: float = 2.5,
        resolution: float = 50,  # "resolution" meters per pixel
    ) -> list[np.ndarray]:
        n_bbox = sh.geometry.BBox(
            bbox=coords,
            crs=sh.constants.CRS.WGS84,
        )
        n_size = sh.geo_utils.bbox_to_dimensions(
            n_bbox,
            resolution=resolution,
        )
        requests = []
        for slot in time_intervals:
            request_true_color = sh.api.process.SentinelHubRequest(
                evalscript=SentinelHubMaster.get_true_color_evalscript(
                    factor=factor,
                ),
                input_data=[
                    sh.api.process.SentinelHubRequest.input_data(
                        data_collection=SentinelHubMaster.TO_DATA_COLLECTION[
                            data_collection.upper()
                        ],
                        time_interval=slot,
                        mosaicking_order=sh.constants.MosaickingOrder.LEAST_CC,
                    )
                ],
                responses=[
                    sh.api.process.SentinelHubRequest.output_response(
                        "default",
                        format,
                    ),
                ],
                bbox=n_bbox,
                size=n_size,
                config=self.config,
            )
            requests.append(request_true_color.download_list[0])
        return sh.download.sentinelhub_client.SentinelHubDownloadClient(
            config=self.config,
        ).download(requests, max_threads=5)

    def get_box_cloud_masks(
        self,
        coords: tuple[float, float, float, float],
        time_intervals: list[tuple[str, str]],
        data_collection: str,
        format: str = "png",
        resolution: float = 50,  # "resolution" meters per pixel
    ) -> list[np.ndarray]:
        n_bbox = sh.geometry.BBox(
            bbox=coords,
            crs=sh.constants.CRS.WGS84,
        )
        n_size = sh.geo_utils.bbox_to_dimensions(
            n_bbox,
            resolution=resolution,
        )
        requests = []
        for slot in time_intervals:
            request_true_color = sh.api.process.SentinelHubRequest(
                evalscript=SentinelHubMaster.get_cloud_mask_evalscript(),
                input_data=[
                    sh.api.process.SentinelHubRequest.input_data(
                        data_collection=SentinelHubMaster.TO_DATA_COLLECTION[
                            data_collection.upper()
                        ],
                        time_interval=slot,
                        mosaicking_order=sh.constants.MosaickingOrder.LEAST_CC,
                    )
                ],
                responses=[
                    sh.api.process.SentinelHubRequest.output_response(
                        "default",
                        format,
                    ),
                ],
                bbox=n_bbox,
                size=n_size,
                config=self.config,
            )
            requests.append(request_true_color.download_list[0])
        return sh.download.sentinelhub_client.SentinelHubDownloadClient(
            config=self.config,
        ).download(requests, max_threads=5)
    
    @staticmethod
    def get_pixel_size(
        coords: tuple[float, float, float, float],
        resolution: float = 50,
    ) -> tuple[int, int]:
        n_bbox = sh.geometry.BBox(
            bbox=coords,
            crs=sh.constants.CRS.WGS84,
        )
        return sh.geo_utils.bbox_to_dimensions(
            n_bbox,
            resolution=resolution
        )

    @staticmethod
    def get_true_color_evalscript(factor: float = 2.5) -> str:
        return """
            //VERSION=3
            function setup() {{
                return {{
                    input: [{{
                        bands: ["B02", "B03", "B04"]
                    }}],
                    output: {{
                        bands: 3,
                    }}
                }};
            }}
            function evaluatePixel(sample) {{
                return [sample.B04 * {0},
                        sample.B03 * {0},
                        sample.B02 * {0}];
            }}
        """.format(factor)
    
    @staticmethod
    def get_cloud_mask_evalscript() -> str:
        return """
            //VERSION=3
            function setup() {
                return {
                    input: [{
                        bands: ["CLM"]
                    }],
                    output: {
                        bands: 1
                    }
                }
            }

            function evaluatePixel(sample) {
                if (sample.CLM == 1) {
                    return [1]
                } else {
                    return [0]
                }
            }
        """