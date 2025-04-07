from fastapi import FastAPI
from fastapi.responses import FileResponse
import uvicorn

app = FastAPI()


@app.get("/")
def read_root():
    return {"Hello": "World"}


@app.get("/v1/images/upload")
def upload(image: str):
    """Upload an image to the server.

    Args:
        image (str): Base64 encoded image string.
    """
    pass


@app.get("/v1/images/convert")
def convert(image_id: str, palette: str):
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
