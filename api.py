import os
from typing import Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from interface.server import (
    query_zatca_knowledge,
    calculate_vat,
    run_crawler,
    check_for_updates,
)

app = FastAPI(title="ZATCA RAG API", version="1.0")

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
