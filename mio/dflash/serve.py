# Copyright 2026 bstnxbt
# MIT License — see LICENSE file
# Based on DFlash (arXiv:2602.06036)

from __future__ import annotations

import argparse
import json
import time
import uuid
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Any, Optional

from mio.dflash.generate import (
    DRAFT_REGISTRY,
    decode_token,
    get_stop_token_ids,
    load_runtime_components,
)
from mio.dflash.runtime import (
    generate_baseline_once,
    generate_dflash_once,
    stream_baseline_generate,
    stream_dflash_generate,
)


@dataclass
class ServerState:
    model_ref: str
    draft_ref: Optional[str]
    use_chat_template: bool
    target_model: Any
    tokenizer: Any
    draft_model: Any


def _message_text(message: dict[str, Any]) -> str:
    content = message.get("content", "")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                parts.append(str(item.get("text", "")))
        return "".join(parts)
    return str(content)


def _messages_to_prompt(messages: list[dict[str, Any]]) -> str:
    parts: list[str] = []
    for message in messages:
        role = str(message.get("role", "user"))
        content = _message_text(message).strip()
        if content:
            parts.append(f"{role}: {content}")
    return "\n\n".join(parts)


def _build_prompt_request(
    *,
    tokenizer: Any,
    payload: dict[str, Any],
    use_chat_template: bool,
) -> tuple[str, Optional[list[int]]]:
    if "messages" in payload:
        messages = list(payload.get("messages") or [])
        if use_chat_template and hasattr(tokenizer, "apply_chat_template"):
            prompt_tokens = tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
            )
            return "", list(prompt_tokens)
        return _messages_to_prompt(messages), None
    return str(payload.get("prompt", "")), None


def _usage_from_summary(summary: dict[str, Any]) -> dict[str, int]:
    prompt_tokens = int(summary.get("prompt_token_count", 0))
    completion_tokens = int(summary.get("generation_tokens", 0))
    return {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": prompt_tokens + completion_tokens,
    }


def _make_chat_response(
    *,
    response_id: str,
    created: int,
    model_ref: str,
    text: str,
    summary: dict[str, Any],
) -> dict[str, Any]:
    return {
        "id": response_id,
        "object": "chat.completion",
        "created": created,
        "model": model_ref,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": text},
                "finish_reason": "stop",
            }
        ],
        "usage": _usage_from_summary(summary),
    }


def _sse_write(handler: BaseHTTPRequestHandler, payload: dict[str, Any]) -> None:
    handler.wfile.write(f"data: {json.dumps(payload)}\n\n".encode("utf-8"))
    handler.wfile.flush()


def _build_handler(state: ServerState):
    class Handler(BaseHTTPRequestHandler):
        server_version = "dflash-mlx/0.1.0"

        def _read_json(self) -> dict[str, Any]:
            content_length = int(self.headers.get("Content-Length", "0") or 0)
            raw = self.rfile.read(content_length) if content_length > 0 else b"{}"
            return json.loads(raw.decode("utf-8") or "{}")

        def _send_json(self, status: int, payload: dict[str, Any]) -> None:
            body = json.dumps(payload).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self) -> None:
            if self.path != "/v1/models":
                self._send_json(404, {"error": {"message": "Not found"}})
                return
            self._send_json(
                200,
                {
                    "object": "list",
                    "data": [
                        {
                            "id": state.model_ref,
                            "object": "model",
                            "owned_by": "bstnxbt",
                        }
                    ],
                },
            )

        def do_POST(self) -> None:
            if self.path != "/v1/chat/completions":
                self._send_json(404, {"error": {"message": "Not found"}})
                return

            payload = self._read_json()
            request_model = payload.get("model")
            if request_model and str(request_model) != state.model_ref:
                self._send_json(
                    400,
                    {"error": {"message": f"Loaded model is {state.model_ref}, got {request_model}"}},
                )
                return

            prompt, prompt_tokens_override = _build_prompt_request(
                tokenizer=state.tokenizer,
                payload=payload,
                use_chat_template=state.use_chat_template,
            )
            max_tokens = int(payload.get("max_tokens", 2048))
            stream = bool(payload.get("stream", False))
            stop_token_ids = get_stop_token_ids(state.tokenizer)
            response_id = f"chatcmpl-{uuid.uuid4().hex}"
            created = int(time.time())

            if stream:
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.send_header("Cache-Control", "no-cache")
                self.send_header("Connection", "keep-alive")
                self.end_headers()
                _sse_write(
                    self,
                    {
                        "id": response_id,
                        "object": "chat.completion.chunk",
                        "created": created,
                        "model": state.model_ref,
                        "choices": [{"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}],
                    },
                )
                if state.draft_model is None:
                    event_iter = stream_baseline_generate(
                        target_model=state.target_model,
                        tokenizer=state.tokenizer,
                        prompt=prompt,
                        max_new_tokens=max_tokens,
                        use_chat_template=False,
                        stop_token_ids=stop_token_ids,
                        prompt_tokens_override=prompt_tokens_override,
                    )
                else:
                    event_iter = stream_dflash_generate(
                        target_model=state.target_model,
                        tokenizer=state.tokenizer,
                        draft_model=state.draft_model,
                        prompt=prompt,
                        max_new_tokens=max_tokens,
                        use_chat_template=False,
                        stop_token_ids=stop_token_ids,
                        prompt_tokens_override=prompt_tokens_override,
                    )
                for event in event_iter:
                    if event.get("event") == "token":
                        text = decode_token(state.tokenizer, int(event["token_id"]))
                        _sse_write(
                            self,
                            {
                                "id": response_id,
                                "object": "chat.completion.chunk",
                                "created": created,
                                "model": state.model_ref,
                                "choices": [
                                    {"index": 0, "delta": {"content": text}, "finish_reason": None}
                                ],
                            },
                        )
                    elif event.get("event") == "summary":
                        _sse_write(
                            self,
                            {
                                "id": response_id,
                                "object": "chat.completion.chunk",
                                "created": created,
                                "model": state.model_ref,
                                "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
                            },
                        )
                self.wfile.write(b"data: [DONE]\n\n")
                self.wfile.flush()
                return

            if state.draft_model is None:
                summary = generate_baseline_once(
                    target_model=state.target_model,
                    tokenizer=state.tokenizer,
                    prompt=prompt,
                    max_new_tokens=max_tokens,
                    use_chat_template=False,
                    stop_token_ids=stop_token_ids,
                    prompt_tokens_override=prompt_tokens_override,
                )
            else:
                summary = generate_dflash_once(
                    target_model=state.target_model,
                    tokenizer=state.tokenizer,
                    draft_model=state.draft_model,
                    prompt=prompt,
                    max_new_tokens=max_tokens,
                    use_chat_template=False,
                    stop_token_ids=stop_token_ids,
                    prompt_tokens_override=prompt_tokens_override,
                )
            text = "".join(
                decode_token(state.tokenizer, int(token_id))
                for token_id in summary.get("generated_token_ids", [])
            )
            self._send_json(
                200,
                _make_chat_response(
                    response_id=response_id,
                    created=created,
                    model_ref=state.model_ref,
                    text=text,
                    summary=summary,
                ),
            )

        def log_message(self, format: str, *args: Any) -> None:
            return

    return Handler


def main() -> None:
    parser = argparse.ArgumentParser(description="OpenAI-compatible DFlash server.")
    parser.add_argument("--model", required=True, help="Target model reference.")
    parser.add_argument("--draft", default=None, help="Optional draft model override.")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--no-chat-template", action="store_true")
    args = parser.parse_args()

    target_model, tokenizer, draft_model, draft_ref = load_runtime_components(
        model_ref=args.model,
        draft_ref=args.draft,
    )
    state = ServerState(
        model_ref=args.model,
        draft_ref=draft_ref or DRAFT_REGISTRY.get(args.model),
        use_chat_template=not args.no_chat_template,
        target_model=target_model,
        tokenizer=tokenizer,
        draft_model=draft_model,
    )
    server = HTTPServer(("127.0.0.1", int(args.port)), _build_handler(state))
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
