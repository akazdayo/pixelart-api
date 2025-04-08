from fastapi import HTTPException
from typing import Optional, Dict, Any


class PixelArtBaseException(HTTPException):
    """PixelArtアプリケーションの基本例外クラス"""

    def __init__(
        self,
        status_code: int,
        code: str,
        message: str,
        details: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(
            status_code=status_code,
            detail={"code": code, "message": message, "details": details or {}},
        )


class ImageNotFoundError(PixelArtBaseException):
    """画像が見つからない場合の例外"""

    def __init__(self, image_id: str):
        super().__init__(
            status_code=404,
            code="IMAGE_NOT_FOUND",
            message=f"Image with ID {image_id} not found",
            details={"image_id": image_id},
        )


class ImageProcessingError(PixelArtBaseException):
    """画像処理時のエラー"""

    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(
            status_code=400,
            code="IMAGE_PROCESSING_ERROR",
            message=message,
            details=details,
        )


class InvalidImageFormatError(PixelArtBaseException):
    """不正な画像フォーマットの場合のエラー"""

    def __init__(self, message: str = "Invalid image format"):
        super().__init__(status_code=400, code="INVALID_IMAGE_FORMAT", message=message)


class RedisConnectionError(PixelArtBaseException):
    """Redis接続エラー"""

    def __init__(self, message: str = "Failed to connect to Redis"):
        super().__init__(
            status_code=503, code="REDIS_CONNECTION_ERROR", message=message
        )
