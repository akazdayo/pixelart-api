from fastapi import FastAPI, UploadFile
import redis.asyncio as redis
from uuid import uuid4
import base64
import cv2
import numpy as np
from src.ai import AI
import pixelart_modules as pm
from numpy.typing import NDArray
from typing import cast

app = FastAPI()
pool = redis.ConnectionPool(host="localhost", port=6379, db=0)
r = redis.Redis(connection_pool=pool)
ai = AI()


def decode_base64(b64_img):
    img_bytes = base64.b64decode(b64_img.decode("utf-8"))
    image_array = np.asarray(bytearray(img_bytes), dtype=np.uint8)
    image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
    return image


def resize_image(image):
    img_size = image.shape[0] * image.shape[1]
    # 画像をFull HDよりも小さくする
    ratio = (img_size / 2073600) ** 0.5
    new_height = int(image.shape[0] / ratio)
    new_width = int(image.shape[1] / ratio)
    result = cv2.resize(image, (new_width, new_height))
    return result


def cv_to_base64(img):
    _, encoded = cv2.imencode(".png", img)
    img_str = base64.b64encode(encoded).decode("ascii")

    return img_str


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

    base64_image = cv_to_base64(cv_image)

    id = str(uuid4())
    await r.set(id, base64_image, ex=60 * 60)  # Expire in 1 hour
    return {"image_id": id}


@app.post("/v1/images/convert/kmeans")
async def kmeans(image_id, k: int = 8):
    data = await _get_redis(image_id)
    img = decode_base64(data)

    colors = ai.get_color(img, k, 1500)

    return {
        "cluster": np.array2string(np.array(colors), separator=",").replace("\n", "")
    }


@app.post("/v1/images/convert")
async def convert(image_id: str, palette: list[list[int]]):
    data = await _get_redis(image_id)
    img = decode_base64(data)

    converted = cast(
        NDArray[np.uint64],
        pm.convert(img, np.array(palette, dtype=np.uint64)),  # type: ignore
    )

    b64_img = cv_to_base64(converted)
    return {"image": f"data:image/png;base64,{b64_img}"}


@app.get("/v1/images/get/{image_id}")
async def get_image(image_id: str):
    data = await _get_redis(image_id)
    return {"image": f"data:image/png;base64,{data}"}
