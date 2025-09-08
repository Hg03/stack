from fastapi import FastAPI
from typing import Optional, Any
from stack.inference_pipeline.infer import Infer
from pydantic import BaseModel
import uvicorn

# Create FastAPI instance
app = FastAPI(
    title="Stack API",
    description="A simple FastAPI application",
    version="1.0.0"
)

# Define request model
class InferenceRequest(BaseModel):
    input_data: Any # Adjust this type based on what your Infer class expects
    # Add other parameters your inference pipeline might need
    # model_params: Optional[dict] = None

# Define response model
class InferenceResponse(BaseModel):
    result: Any
    status: str
    message: Optional[str] = None

@app.get("/")
async def root():
    return {"message": "Hello, Stack!"}

@app.post("/single", response_model=InferenceResponse)
async def single_inference(request: InferenceRequest):
    inference_pipe = Infer()
    
    return InferenceResponse(result="yeah", status="success", message="success")

if __name__ == "__main__":
    uvicorn.run(app=app, reload=True)