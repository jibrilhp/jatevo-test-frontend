import os
import base64
import mimetypes
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from openai import OpenAI
import json

app = FastAPI(title="Jatevo AI Gateway", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

JATEVO_BASE_URL = "https://jatevo.id/api/open/v1/inference"
SUPPORTED_IMAGE_TYPES  = {"image/jpeg", "image/png", "image/gif", "image/webp"}
SUPPORTED_DOC_TYPES    = {"application/pdf", "text/plain", "text/markdown"}
SUPPORTED_VIDEO_TYPES  = {"video/mp4", "video/webm", "video/quicktime"}

ALL_MODELS = [
    {"id": "glm-4.7",      "name": "GLM 4.7",      "type": "text",   "description": "Fast & efficient"},
    {"id": "glm-4-plus",   "name": "GLM 4 Plus",   "type": "text",   "description": "Advanced reasoning"},
    {"id": "deepseek-v3",  "name": "DeepSeek V3",  "type": "text",   "description": "Deep reasoning & coding"},
    {"id": "qwen3.5-plus", "name": "Qwen 3.5 Plus","type": "vision", "description": "Multimodal vision"},
    {"id": "qwen-vl-plus", "name": "Qwen VL Plus", "type": "vision", "description": "Visual language"},
]


def get_client() -> OpenAI:
    """API key is read server-side from environment only — never from the client."""
    api_key = os.environ.get("JATEVO_API_KEY", "")
    if not api_key:
        raise HTTPException(
            status_code=500,
            detail="JATEVO_API_KEY is not configured on the server. Set it in your .env file."
        )
    return OpenAI(base_url=JATEVO_BASE_URL, api_key=api_key)


def build_messages(history: list, prompt: str, media_parts: list | None = None) -> list:
    messages = [{"role": m["role"], "content": m["content"]} for m in history]
    if media_parts:
        messages.append({"role": "user", "content": [{"type": "text", "text": prompt}] + media_parts})
    else:
        messages.append({"role": "user", "content": prompt})
    return messages


def encode_b64(data: bytes) -> str:
    return base64.b64encode(data).decode("utf-8")


def sse_stream(client, model, messages, max_tokens, temperature):
    def generate():
        try:
            resp = client.chat.completions.create(
                model=model, messages=messages,
                max_tokens=max_tokens, temperature=temperature, stream=True,
            )
            for chunk in resp:
                delta = chunk.choices[0].delta
                if delta.content:
                    yield f"data: {json.dumps({'content': delta.content, 'done': False})}\n\n"
            yield f"data: {json.dumps({'content': '', 'done': True})}\n\n"
        except Exception as e:
            yield f"data: {json.dumps({'error': str(e), 'done': True})}\n\n"
    return StreamingResponse(generate(), media_type="text/event-stream")


@app.get("/api/health")
async def health():
    return {"status": "ok", "api_key_configured": bool(os.environ.get("JATEVO_API_KEY"))}


@app.get("/api/models")
async def list_models():
    return {"models": ALL_MODELS}


@app.post("/api/chat")
async def chat_text(
    prompt:      str   = Form(...),
    model:       str   = Form("glm-4.7"),
    max_tokens:  int   = Form(1000),
    temperature: float = Form(0.7),
    history:     str   = Form("[]"),
):
    return sse_stream(get_client(), model, build_messages(json.loads(history), prompt), max_tokens, temperature)


@app.post("/api/chat/upload")
async def chat_with_upload(
    prompt:      str              = Form(...),
    model:       str              = Form("qwen3.5-plus"),
    max_tokens:  int              = Form(1000),
    temperature: float            = Form(0.7),
    history:     str              = Form("[]"),
    files:       list[UploadFile] = File(default=[]),
):
    media_parts = []
    for upload in files:
        raw  = await upload.read()
        mime = upload.content_type or mimetypes.guess_type(upload.filename or "")[0] or "application/octet-stream"
        if mime in SUPPORTED_IMAGE_TYPES:
            media_parts.append({"type": "image_url", "image_url": {"url": f"data:{mime};base64,{encode_b64(raw)}", "detail": "auto"}})
        elif mime in SUPPORTED_DOC_TYPES:
            media_parts.append({"type": "text", "text": f"\n\n[Document: {upload.filename}]\n{raw.decode('utf-8', errors='replace')}\n[End of Document]"})
        elif mime in SUPPORTED_VIDEO_TYPES:
            media_parts.append({"type": "video_url", "video_url": {"url": f"data:{mime};base64,{encode_b64(raw)}"}})
        else:
            raise HTTPException(status_code=415, detail=f"Unsupported type: {mime} ({upload.filename})")

    return sse_stream(get_client(), model, build_messages(json.loads(history), prompt, media_parts or None), max_tokens, temperature)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend:app", host="0.0.0.0", port=8000, reload=True)