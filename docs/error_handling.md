# エラーハンドリング改善計画

## 1. 主な改善点

### 1.1 カスタム例外の導入

- `ImageNotFoundError`: 画像が見つからない場合
- `ImageProcessingError`: 画像処理時のエラー
- `InvalidImageFormatError`: 不正な画像フォーマット
- `RedisConnectionError`: Redis接続エラー

### 1.2 一貫したエラーレスポンス形式

```json
{
    "error": {
        "code": "ERROR_CODE",
        "message": "詳細なエラーメッセージ",
        "details": {} // 追加情報（オプション）
    }
}
```

### 1.3 エラー時のログ記録

- エラーの発生時刻
- エラーの種類
- エラーメッセージ
- スタックトレース（必要な場合）
- リクエスト情報（エンドポイント、パラメータ）

## 2. 実装手順

1. カスタム例外クラスの実装
2. FastAPIのエラーハンドラーの実装
3. ログ記録機能の実装
4. 既存コードへのエラーハンドリングの適用

## 3. 具体的な実装例

### 3.1 カスタム例外

```python
from fastapi import HTTPException

class PixelArtBaseException(HTTPException):
    def __init__(self, status_code: int, code: str, message: str, details: dict = None):
        super().__init__(status_code=status_code, detail={
            "code": code,
            "message": message,
            "details": details or {}
        })

class ImageNotFoundError(PixelArtBaseException):
    def __init__(self, image_id: str):
        super().__init__(
            status_code=404,
            code="IMAGE_NOT_FOUND",
            message=f"Image with ID {image_id} not found",
            details={"image_id": image_id}
        )

class ImageProcessingError(PixelArtBaseException):
    def __init__(self, message: str, details: dict = None):
        super().__init__(
            status_code=400,
            code="IMAGE_PROCESSING_ERROR",
            message=message,
            details=details
        )
```

### 3.2 エラーハンドリング

```python
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
import logging

app = FastAPI()

@app.exception_handler(PixelArtBaseException)
async def pixelart_exception_handler(request: Request, exc: PixelArtBaseException):
    logging.error(f"Error occurred: {exc.detail['code']} - {exc.detail['message']}")
    return JSONResponse(
        status_code=exc.status_code,
        content={"error": exc.detail}
    )
```

## 4. 期待される効果

1. エラーの特定と対応が容易に
2. クライアントへの明確なエラー情報の提供
3. デバッグ作業の効率化
4. システムの安定性向上

## 5. 注意点

- 機密情報のログ漏洩防止
- 適切なエラーメッセージの選択
- パフォーマンスへの影響の最小化
