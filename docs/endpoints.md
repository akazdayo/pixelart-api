# APIエンドポイント詳細設計

## 1. 画像アップロード

**エンドポイント**: `POST /api/v1/images/upload`

**目的**: 変換対象の画像をサーバーにアップロードする

**リクエスト**:

```json
{
    "content_type": "multipart/form-data",
    "parameters": {
        "file": "画像ファイル（バイナリ）"
    }
}
```

**レスポンス**:

```json
{
    "status": "success",
    "data": {
        "image_id": "一意の画像ID",
        "original_filename": "元のファイル名",
        "content_type": "image/jpeg|png|etc",
        "size": "ファイルサイズ（バイト）"
    }
}
```

**バリデーション**:

- 許可される画像形式: JPEG, PNG, GIF
- 最大ファイルサイズ: 10MB
- 最小画像サイズ: 50x50px

## 2. パレットによる画像変換

**エンドポイント**: `POST /api/v1/images/convert`

**目的**: アップロードされた画像をドット絵風に変換する

**リクエスト**:

```json
{
  "image_id": "変換対象の画像ID",
  "palette": "使用するパレット名",
  "options": {
    "resize": true|false,
    "alpha_processing": true|false
  }
}
```

**レスポンス**:

```json
{
    "status": "success",
    "data": {
        "converted_image_id": "変換後の画像ID",
        "processing_time": "処理時間（ミリ秒）",
        "parameters": {
            "palette": "使用したパレット名",
            "original_size": {
                "width": "元の幅",
                "height": "元の高さ"
            },
            "converted_size": {
                "width": "変換後の幅",
                "height": "変換後の高さ"
            }
        }
    }
}
```

## 3. 画像リサイズ

**エンドポイント**: `POST /api/v1/images/resize`

**目的**: 画像を指定されたサイズにリサイズする

**リクエスト**:

```json
{
  "image_id": "リサイズ対象の画像ID",
  "options": {
    "target_size": {
      "width": "目標の幅",
      "height": "目標の高さ"
    },
    "maintain_aspect_ratio": true|false
  }
}
```

**レスポンス**:

```json
{
    "status": "success",
    "data": {
        "resized_image_id": "リサイズ後の画像ID",
        "original_size": {
            "width": "元の幅",
            "height": "元の高さ"
        },
        "new_size": {
            "width": "新しい幅",
            "height": "新しい高さ"
        }
    }
}
```

## 4. アルファチャンネル処理

**エンドポイント**: `POST /api/v1/images/alpha`

**目的**: 画像の透過処理を行う

**リクエスト**:

```json
{
    "image_id": "処理対象の画像ID",
    "operation": "delete_alpha|delete_transparent",
    "options": {
        "threshold": "透過度のしきい値（0-255）"
    }
}
```

**レスポンス**:

```json
{
    "status": "success",
    "data": {
        "processed_image_id": "処理後の画像ID",
        "operation": "実行された操作",
        "processing_time": "処理時間（ミリ秒）"
    }
}
```

## 5. パレット一覧取得

**エンドポイント**: `GET /api/v1/palettes`

**目的**: 利用可能なカラーパレットの一覧を取得する

**レスポンス**:

```json
{
    "status": "success",
    "data": {
        "palettes": [
            {
                "id": "パレットID",
                "name": "パレット名",
                "colors": [
                    {
                        "r": "赤成分（0-255）",
                        "g": "緑成分（0-255）",
                        "b": "青成分（0-255）"
                    }
                ],
                "preview_url": "パレットのプレビュー画像URL"
            }
        ]
    }
}
```

## 6. 処理済み画像の取得

**エンドポイント**: `GET /api/v1/images/{image_id}`

**目的**: アップロード済みまたは処理済みの画像を取得する

**URLパラメータ**:

- image_id: 取得したい画像のID

**クエリパラメータ**:

- format: 出力形式（jpeg|png|gif）
- download: ダウンロードモード（true|false）

**レスポンス**:

- Content-Type: image/jpeg|png|gif
- Content-Disposition: attachment（downloadがtrueの場合）

## エラーレスポンス（共通）

```json
{
    "status": "error",
    "error": {
        "code": "エラーコード",
        "message": "エラーメッセージ",
        "details": {
            "field": "エラーが発生したフィールド",
            "reason": "詳細な理由"
        }
    }
}
```

### 主なエラーコード

- `INVALID_REQUEST`: リクエストパラメータが不正
- `FILE_TOO_LARGE`: ファイルサイズが制限を超過
- `UNSUPPORTED_FORMAT`: サポートされていない画像形式
- `IMAGE_NOT_FOUND`: 指定されたIDの画像が見つからない
- `PALETTE_NOT_FOUND`: 指定されたパレットが見つからない
- `PROCESSING_ERROR`: 画像処理中のエラー
- `INTERNAL_ERROR`: サーバー内部エラー
