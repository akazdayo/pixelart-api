import base64
import numpy as np
import cv2


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
