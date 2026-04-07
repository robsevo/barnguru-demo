from fastapi import FastAPI

app = FastAPI(title="GRETZKY API", version="0.1.0")


@app.get("/health")
async def health() -> dict:
    return {"status": "ok", "version": "0.1.0"}
