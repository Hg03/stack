from fastapi import FastAPI
import uvicorn

# Create FastAPI instance
app = FastAPI(
    title="Stack API",
    description="A simple FastAPI application",
    version="1.0.0"
)

@app.get("/")
async def root():
    return {"message": "Hello, Stack!"}