from fastapi import FastAPI, UploadFile
import redis.asyncio as redis
from uuid import uuid4
import cv2
import numpy as np
from src.ai import AI
import pixelart_modules as pm
from numpy.typing import NDArray
from typing import cast
import src.utils.images as img_utils

app = FastAPI()
pool = redis.ConnectionPool(host="localhost", port=6379, db=0)
r = redis.Redis(connection_pool=pool)
ai = AI()


async def _get_redis(image_id):
    """Get an image from the server by its ID.
    This function retrieves the base64 encoded image string from Redis using the provided ID.

    Args:
        image_id (str): The ID of the image to retrieve.

    Returns:
        str: Base64 encoded image string.
    """
    data = await r.get(image_id)
    if data is None:
        return {"error": "Image not found"}
    return data


@app.get("/")
def read_root():
    return {"Hello": "World"}


@app.post("/v1/images/upload")
async def upload(upload_image: UploadFile):
    """Upload an image to the server.
    This function takes a base64 encoded image string, generates a unique ID for it,
    and stores it in Redis with a 1-hour expiration time.

    Args:
        base64_image (str): Base64 encoded image string.

    Returns:
        dict[str, str]: image_id
    """
    image = upload_image.file.read()
    cv_image = cv2.imdecode(np.frombuffer(image, np.uint8), cv2.IMREAD_COLOR)

    base64_image = img_utils.cv_to_base64(cv_image)

    id = str(uuid4())
    await r.set(id, base64_image, ex=60 * 5)  # Expire in 5 minutes
    return {"image_id": id}


@app.post("/v1/images/convert/kmeans")
async def kmeans(image_id: str, k: int = 8):
    """Sampling colors from an image using K-means clustering.

    Args:
        image_id (str): _description_.
        k (int, optional): _description_. Defaults to 8.

    Returns:
        dict[str, str]: {"cluster": [[int, int, int], ...]}
    """
    data = await _get_redis(image_id)
    img = img_utils.decode_base64(data)

    colors = ai.get_color(img, k, 1500)

    return {
        "cluster": np.array2string(np.array(colors), separator=",").replace("\n", "")
    }


@app.post("/v1/images/convert")
async def convert(image_id: str, palette: list[list[int]]):
    """Convert an image to a pixel art style color using a given palette.

    Args:
        image_id (str): _description_
        palette (list[list[int]]): _description_

    Returns:
        _type_: _description_
    """
    data = await _get_redis(image_id)
    img = img_utils.decode_base64(data)

    converted = cast(
        NDArray[np.uint64],
        pm.convert(img, np.array(palette, dtype=np.uint64)),  # type: ignore
    )

    b64_img = img_utils.cv_to_base64(converted)
    return {"image": f"data:image/png;base64,{b64_img}"}
