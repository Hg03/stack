from fastapi import FastAPI
from stack.inference_pipeline.infer import Infer
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

@app.post("/single")
async def single_inference():
    inference_pipe = Infer()
    return {"pipe": inference_pipe.model_artifact_path}

if __name__ == "__main__":
    uvicorn.run(app=app)