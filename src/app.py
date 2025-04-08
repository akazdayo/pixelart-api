from fastapi import FastAPI, UploadFile, Request
from fastapi.responses import JSONResponse
import redis.asyncio as redis
from uuid import uuid4
import cv2
import numpy as np
from src.ai import AI
import pixelart_modules as pm
from numpy.typing import NDArray
from typing import cast, Dict
import src.utils.images as img_utils
from src.exceptions import (
    PixelArtBaseException,
    ImageNotFoundError,
    ImageProcessingError,
    RedisConnectionError,
)
import logging

# ロガーの設定
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()
pool = redis.ConnectionPool(host="localhost", port=6379, db=0)
r = redis.Redis(connection_pool=pool)
ai = AI()


@app.exception_handler(PixelArtBaseException)
async def pixelart_exception_handler(
    request: Request, exc: PixelArtBaseException
) -> JSONResponse:
    """カスタム例外のハンドラー"""
    logger.error(f"Error occurred: {exc.detail}")
    return JSONResponse(status_code=exc.status_code, content=exc.detail)


async def _get_redis(image_id: str) -> bytes:
    """Get an image from the server by its ID.
    This function retrieves the base64 encoded image string from Redis using the provided ID.

    Args:
        image_id (str): The ID of the image to retrieve.

    Returns:
        bytes: Base64 encoded image data.

    Raises:
        ImageNotFoundError: If the image is not found in Redis.
        RedisConnectionError: If there is an error connecting to Redis.
    """
    try:
        data = await r.get(image_id)
        if data is None:
            raise ImageNotFoundError(image_id)
        return data
    except redis.ConnectionError as e:
        logger.error(f"Redis connection error: {str(e)}")
        raise RedisConnectionError()


@app.get("/")
def read_root():
    return {"Hello": "World"}


@app.post("/v1/images/upload")
async def upload(upload_image: UploadFile) -> Dict[str, str]:
    """Upload an image to the server.
    This function takes an uploaded image file, generates a unique ID for it,
    and stores it in Redis with a 5-minute expiration time.

    Args:
        upload_image (UploadFile): The uploaded image file.

    Returns:
        Dict[str, str]: Dictionary containing the image_id.

    Raises:
        ImageProcessingError: If there is an error processing the image.
        RedisConnectionError: If there is an error connecting to Redis.
    """
    try:
        image = await upload_image.read()
        cv_image = cv2.imdecode(np.frombuffer(image, np.uint8), cv2.IMREAD_COLOR)
        if cv_image is None:
            raise ImageProcessingError("Failed to decode image")

        base64_image = img_utils.cv_to_base64(cv_image)

        id = str(uuid4())
        try:
            await r.set(id, base64_image, ex=60 * 5)  # Expire in 5 minutes
        except redis.ConnectionError as e:
            logger.error(f"Redis connection error: {str(e)}")
            raise RedisConnectionError()

        return {"image_id": id}
    except Exception as e:
        logger.error(f"Error processing image: {str(e)}")
        raise ImageProcessingError("Failed to process image", {"error": str(e)})


@app.post("/v1/images/convert/kmeans")
async def kmeans(image_id: str, k: int = 8) -> Dict[str, str]:
    """Sampling colors from an image using K-means clustering.

    Args:
        image_id (str): The ID of the image to process.
        k (int, optional): Number of color clusters. Defaults to 8.

    Returns:
        Dict[str, str]: Dictionary containing the color clusters.

    Raises:
        ImageNotFoundError: If the image is not found.
        ImageProcessingError: If there is an error processing the image.
    """
    try:
        data = await _get_redis(image_id)
        img = img_utils.decode_base64(data)
        if img is None:
            raise ImageProcessingError("Failed to decode base64 image")

        colors = ai.get_color(img, k, 1500)
        return {
            "cluster": np.array2string(np.array(colors), separator=",").replace(
                "\n", ""
            )
        }
    except (ImageNotFoundError, RedisConnectionError):
        raise
    except Exception as e:
        logger.error(f"Error in kmeans processing: {str(e)}")
        raise ImageProcessingError(
            "Failed to process image with k-means", {"error": str(e)}
        )


@app.post("/v1/images/convert")
async def convert(image_id: str, palette: list[list[int]]) -> Dict[str, str]:
    """Convert an image to a pixel art style color using a given palette.

    Args:
        image_id (str): The ID of the image to convert.
        palette (list[list[int]]): The color palette to use for conversion.

    Returns:
        Dict[str, str]: Dictionary containing the converted image in base64 format.

    Raises:
        ImageNotFoundError: If the image is not found.
        ImageProcessingError: If there is an error processing the image.
    """
    try:
        data = await _get_redis(image_id)
        img = img_utils.decode_base64(data)
        if img is None:
            raise ImageProcessingError("Failed to decode base64 image")

        try:
            converted = cast(
                NDArray[np.uint64],
                pm.convert(img, np.array(palette, dtype=np.uint64)),  # type: ignore
            )
        except Exception as e:
            logger.error(f"Error in pixel art conversion: {str(e)}")
            raise ImageProcessingError(
                "Failed to convert image to pixel art", {"error": str(e)}
            )

        b64_img = img_utils.cv_to_base64(converted)
        return {"image": f"data:image/png;base64,{b64_img}"}
    except (ImageNotFoundError, RedisConnectionError):
        raise
    except Exception as e:
        logger.error(f"Error in convert processing: {str(e)}")
        raise ImageProcessingError(
            "Failed to process image conversion", {"error": str(e)}
        )
