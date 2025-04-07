from fastapi import FastAPI, UploadFile
from fastapi.responses import FileResponse
import uvicorn
import redis.asyncio as redis
from uuid import uuid4
import base64
import cv2
import numpy as np
from ai import AI

app = FastAPI()
pool = redis.ConnectionPool(host="localhost", port=6379, db=0)
r = redis.Redis(connection_pool=pool)
ai = AI()


def decode_base64(b64_img):
    img_bytes = base64.b64decode(b64_img.decode("utf-8"))
    image_array = np.asarray(bytearray(img_bytes), dtype=np.uint8)
    image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
    return image


@app.get("/")
def read_root():
    return {"Hello": "World"}


@app.post("/v1/images/upload")
def upload(base64_image: str):
    """Upload an image to the server.
    This function takes a base64 encoded image string, generates a unique ID for it,
    and stores it in Redis with a 1-hour expiration time.

    Args:
        base64_image (str): Base64 encoded image string.

    Returns:
        dict[str, str]: image_id
    """
    id = str(uuid4())
    response = r.set(
        id, base64_image.replace("data:image/png;base64,", ""), ex=60 * 60
    )  # Expire in 1 hour
    if not response:
        return {"error": "Failed to upload image"}
    return {"image_id": id}


@app.post("/v1/images/convert/kmeans")
async def kmeans(image_id, k: int = 8):
    data = await r.get(image_id)
    if data is None:
        return {"error": "Image not found"}

    img = decode_base64(data)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGBA)
    colors = ai.get_color(img, k, 1500)

    return {"cluster": colors}


@app.get("/v1/images/convert")
def convert(image_id: str, palette: list[list[int]] = []):
    base64_image = r.get(image_id)
    if palette == []:  # KMeans Palette
        pass


@app.get("/v1/images/resize")
def resize(image_id: str, width: int, height: int):
    pass


@app.get("/v1/images/alpha")
def alpha(image_id):
    pass


@app.get("/v1/images/get/{image_id}")
def get_image(image_id: str):
    return FileResponse(f"images/{image_id}.tiff")


if __name__ == "__main__":
    uvicorn.run("app:app", host="127.0.0.1", port=8000, log_level="info")
