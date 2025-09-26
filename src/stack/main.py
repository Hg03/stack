from fastapi import FastAPI, File, UploadFile, HTTPException
from typing import Optional, Any
from stack.inference_pipeline.infer import Infer
from stack.inference_pipeline.infer_implementations import EmployeeData
from pydantic import Field
from pydantic import BaseModel
import uvicorn
import pandas as pd
import csv
import io

# Create FastAPI instance
app = FastAPI(
    title="Stack API",
    description="A simple FastAPI application",
    version="1.0.0"
)


# Define request model
class InferenceRequest(BaseModel):
    input_data: EmployeeData = Field(..., description="Employee data for stress level prediction")


# Define response model
class InferenceResponse(BaseModel):
    result: Any
    status: str
    message: Optional[str] = None

@app.get("/")
async def root():
    return {"message": "Hello, Stack!"}

@app.post("/inference", response_model=InferenceResponse)
async def single_inference(request: InferenceRequest):
    inference_pipe = Infer()
    # Access the employee data
    employee_data = request.input_data
    # Run inference
    result = inference_pipe.make_inference(payload=employee_data)
    return InferenceResponse(
        result=result, 
        status="success", 
        message="Inference completed successfully"
    )

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)