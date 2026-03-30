import os
import json
from typing import Optional, Generator

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from interface.server import (
    query_zatca_knowledge,
    calculate_vat,
    run_crawler,
    check_for_updates,
)

app = FastAPI(title="ZATCA RAG API", version="1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class QueryRequest(BaseModel):
    query: str

class VATRequest(BaseModel):
    amount: float
    is_export: Optional[bool] = False

@app.get("/health")
def health_check():
    return {"status": "ok"}

@app.post("/query")
def query(request: QueryRequest):
    if not request.query.strip():
        raise HTTPException(status_code=400, detail="query is required")

    result = query_zatca_knowledge(request.query)
    return {"result": result}

@app.post("/query/stream")
def query_stream(request: QueryRequest):
    """Stream the LLM response token-by-token using Server-Sent Events."""
    if not request.query.strip():
        raise HTTPException(status_code=400, detail="query is required")

    from interface.server import get_query_service

    def event_generator() -> Generator[str, None, None]:
        service = get_query_service()
        if not service:
            yield "data: Error: Knowledge base not loaded.\n\n"
            return
        for chunk in service.ask_stream(request.query):
            # Escape newlines so SSE frame stays intact
            safe = chunk.replace("\n", "\\n")
            yield f"data: {safe}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"}
    )

@app.post("/vat")
def vat(request: VATRequest):
    if request.amount < 0:
        raise HTTPException(status_code=400, detail="amount must be non-negative")

    result = calculate_vat(request.amount, is_export=request.is_export)
    return {"result": result}

@app.post("/crawler")
def crawler():
    result = run_crawler()
    return {"result": result}

@app.post("/updates")
def updates():
    result = check_for_updates()
    return {"result": result}

if __name__ == "__main__":
    import uvicorn

    port = int(os.getenv("PORT", 8000))
    uvicorn.run("api:app", host="0.0.0.0", port=port, log_level="info")
