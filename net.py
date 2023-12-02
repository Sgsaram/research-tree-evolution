import requests
import io
import datetime
import oauthlib.oauth2
import requests_oauthlib
import PIL.Image


class SHSession:
    def __init__(
        self,
        client_id,
        client_secret,
    ) -> None:
        self.client_id = client_id
        self.client_secret = client_secret
        self.client = oauthlib.oauth2.BackendApplicationClient(
            client_id=self.client_id,
        )
        self.oauth = requests_oauthlib.OAuth2Session(
            client=self.client_id,
        )
        self.oauth.register_compliance_hook(
            "access_token_response",
            SHSession.sentinelhub_compliance_hook,
        )
        self.token = self.oauth.fetch_token(
            token_url="https://services.sentinel-hub.com/oauth/token",
            client_secret=self.client_secret,
            include_client_id=True,
        )
        self.access_token = self.token["access_token"]

    @staticmethod
    def sentinelhub_compliance_hook(response) -> str:
        response.raise_for_status()
        return response

    def expiration_date(self) -> float:
        return self.token["expires_at"]

    def get_box_image(
        self,
        point1,
        point2,
        format = "png",
        
    ) -> PIL.Image:
        # image = PIL.Image.open(io.BytesIO())