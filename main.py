import os
import requests
import dotenv
import oauthlib.oauth2
import requests_oauthlib
import PIL.Image
import io

dotenv.load_dotenv(dotenv.find_dotenv())

CLIENT_ID = os.environ.get("CLIENT_ID", "")
CLIENT_SECRET = os.environ.get("CLIENT_SECRET", "")


def sentinelhub_compliance_hook(response):
    response.raise_for_status()
    return response


def get_example_image(access_token):
    return requests.post(
        "https://services.sentinel-hub.com/api/v1/process",
        headers={"Authorization": f"Bearer {access_token}"},
        json={
            "input": {
                "bounds": {
                    "bbox": [
                        59.14260,
                        101.58525,
                        59.11345,
                        101.72894,
                    ]
                },
                "data": [
                    {
                        "type": "sentinel-2-l2a",
                    }
                ]
            },
            "evalscript": """
            //VERSION=3

            function setup() {
            return {
                input: ["B02", "B03", "B04"],
                output: {
                bands: 3
                }
            };
            }

            function evaluatePixel(
            sample,
            scenes,
            inputMetadata,
            customData,
            outputMetadata
            ) {
            return [2.5 * sample.B04, 2.5 * sample.B03, 2.5 * sample.B02];
            }
            """
        },
    )


def main():
    client = oauthlib.oauth2.BackendApplicationClient(client_id=CLIENT_ID)
    oauth = requests_oauthlib.OAuth2Session(client=client)

    oauth.register_compliance_hook(
        "access_token_response",
        sentinelhub_compliance_hook,
    )
    token = oauth.fetch_token(
        token_url="https://services.sentinel-hub.com/oauth/token",
        client_secret=CLIENT_SECRET,
        include_client_id=True,
    )
    access_token = token["access_token"]
    print(token)
    response = get_example_image(access_token)
    with open("output", "wb") as f:
        f.write(response.content)
    image = PIL.Image.open(io.BytesIO(response.content))
    image.show()


if __name__ == "__main__":
    main()
