from fastapi import FastAPI

app = FastAPI(
    title="NBA Player Longevity API",
    version="0.1.0",
)


@app.get("/health")
def healthcheck():
    """Simple endpoint to check if the API is running."""
    return {"status": "ok"}