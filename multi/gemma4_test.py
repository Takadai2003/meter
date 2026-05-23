import argparse
import base64
import json
import mimetypes
import re
from pathlib import Path

import requests


DEFAULT_SERVER_URL = "http://127.0.0.1:8080/v1/chat/completions"
DEFAULT_MODEL = "gemma4"


PROMPT = """
あなたは計測器のデジタル表示を読む専用のOCRアシスタントです。

画像に写っている計測器の数値表示を読んでください。
7セグメント表示、小数点、マイナス記号に注意してください。

出力は必ず次のJSONだけにしてください。
Markdown、説明文、コードブロックは出力しないでください。

{
  "readable": true,
  "value": "25.3",
  "confidence": 0.95,
  "reason": "短い根拠"
}

読めない場合は次の形式にしてください。

{
  "readable": false,
  "value": null,
  "confidence": 0.0,
  "reason": "読めない理由"
}

条件:
- valueには数字、マイナス、小数点のみを入れてください。
- 単位は入れないでください。
- 読み取れない桁がある場合はreadableをfalseにしてください。
- confidenceは0.0から1.0の数値にしてください。
- reasonは1文で短くしてください。
"""


def image_to_data_uri(image_path: Path) -> str:
    mime_type, _ = mimetypes.guess_type(str(image_path))

    if mime_type is None:
        mime_type = "image/jpeg"

    with open(image_path, "rb") as f:
        encoded = base64.b64encode(f.read()).decode("utf-8")

    return f"data:{mime_type};base64,{encoded}"


def extract_json(text: str) -> dict:
    text = text.strip()

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    match = re.search(r"\{.*\}", text, re.DOTALL)

    if match is None:
        return {
            "readable": False,
            "value": None,
            "confidence": 0.0,
            "reason": "JSONを抽出できませんでした",
            "raw": text
        }

    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError:
        return {
            "readable": False,
            "value": None,
            "confidence": 0.0,
            "reason": "JSON解析に失敗しました",
            "raw": text
        }


def normalize_value(value):
    if value is None:
        return None

    value = str(value).strip()
    value = value.replace("−", "-")
    value = re.sub(r"[^0-9.\-]", "", value)

    if value == "":
        return None

    if value.count(".") > 1:
        return None

    if value.count("-") > 1:
        return None

    if "-" in value and not value.startswith("-"):
        return None

    return value


def normalize_result(result: dict) -> dict:
    readable = bool(result.get("readable", False))
    value = normalize_value(result.get("value", None))

    try:
        confidence = float(result.get("confidence", 0.0))
    except Exception:
        confidence = 0.0

    confidence = max(0.0, min(1.0, confidence))
    reason = str(result.get("reason", ""))

    if value is None:
        readable = False

    return {
        "readable": readable,
        "value": value,
        "confidence": confidence,
        "reason": reason
    }


def ask_llama_cpp(image_path: Path, server_url: str, model: str) -> tuple[dict, str]:
    data_uri = image_to_data_uri(image_path)

    payload = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": PROMPT
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": data_uri
                        }
                    }
                ]
            }
        ],
        "temperature": 0.0,
        "top_p": 0.1,
        "max_tokens": 512,
        "response_format": {
            "type": "json_object"
        }
    }

    response = requests.post(
        server_url,
        headers={"Content-Type": "application/json"},
        json=payload,
        timeout=300
    )

    response.raise_for_status()
    data = response.json()

    raw_text = data["choices"][0]["message"]["content"]
    parsed = extract_json(raw_text)
    normalized = normalize_result(parsed)

    return normalized, raw_text


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True)
    parser.add_argument("--server_url", default=DEFAULT_SERVER_URL)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    args = parser.parse_args()

    image_path = Path(args.image)

    if not image_path.exists():
        raise FileNotFoundError(image_path)

    result, raw_text = ask_llama_cpp(
        image_path=image_path,
        server_url=args.server_url,
        model=args.model
    )

    print("=== Parsed Result ===")
    print(json.dumps(result, ensure_ascii=False, indent=2))

    print("\n=== Raw Output ===")
    print(raw_text)


if __name__ == "__main__":
    main()