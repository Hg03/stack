from fastapi import FastAPI, File, UploadFile, HTTPException
from typing import Optional, Any, List
from stack.inference_pipeline.infer import Infer
from stack.inference_pipeline.infer_implementations import EmployeeData
from pydantic import Field, ValidationError
from pydantic import BaseModel
import uvicorn
import pandas as pd
import io
from typing import Dict

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

@app.post("/single", response_model=InferenceResponse)
async def single_inference(request: InferenceRequest):
    inference_pipe = Infer()
    # Access the employee data
    employee_data = request.input_data
    # Run inference
    result = inference_pipe.make_inference(payload=employee_data, infer_type="single")
    return InferenceResponse(
        result=result, 
        status="success", 
        message="Inference completed successfully"
    )

@app.post("/batch")
async def batch_inference(file: UploadFile = File(...)):
    # Validate file type
    if not file.filename.endswith('.csv'):
        raise HTTPException(status_code=400, detail="Only CSV files are supported")
    
    try:
        # Read the uploaded CSV file
        contents = await file.read()
        df = pd.read_csv(io.StringIO(contents.decode('utf-8')))
        
        # Initialize inference pipeline
        inference_pipe = Infer()
        
        # Store results
        results = []
        successful_predictions = 0
        failed_predictions = 0
        
        # Process each row in the dataframe
        for index, row in df.iterrows():
            try:
                # Convert row to dict and create EmployeeData instance
                row_dict = row.to_dict()
                
                # Create EmployeeData instance from the row
                employee_data = EmployeeData(**row_dict)
                
                # Make prediction
                result = inference_pipe.make_inference(payload=employee_data, infer_type="batch")
                
                # Store result with row index
                results.append({
                    "row_index": index,
                    "prediction": result,
                    "status": "success"
                })
                successful_predictions += 1
                
            except ValidationError as e:
                # Handle validation errors for individual rows
                results.append({
                    "row_index": index,
                    "error": str(e),
                    "status": "failed"
                })
                failed_predictions += 1
                
            except Exception as e:
                # Handle other errors for individual rows
                results.append({
                    "row_index": index,
                    "error": f"Prediction failed: {str(e)}",
                    "status": "failed"
                })
                failed_predictions += 1
        
        return InferenceResponse(
            result={
                "filename": file.filename,
                "content_type": file.content_type,
                "size": len(contents)
            }, 
            status="success", 
            message="File upload working successfully"
        )
        
    except pd.errors.EmptyDataError:
        raise HTTPException(status_code=400, detail="The uploaded CSV file is empty")
    
    except pd.errors.ParserError as e:
        raise HTTPException(status_code=400, detail=f"Error parsing CSV file: {str(e)}")
    
    except UnicodeDecodeError:
        raise HTTPException(status_code=400, detail="Error decoding file. Please ensure it's a valid UTF-8 encoded CSV file")
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred during batch inference: {str(e)}")

# Optional: Add an endpoint to get the expected CSV format
@app.get("/batch/format")
async def get_csv_format():
    """
    Returns the expected CSV format for batch inference
    """
    try:
        # Create a sample EmployeeData instance to show the expected fields
        sample_employee = EmployeeData.schema()
        required_fields = sample_employee.get('required', [])
        all_fields = sample_employee.get('properties', {})
        
        return {
            "message": "Expected CSV format for batch inference",
            "required_fields": required_fields,
            "all_fields": list(all_fields.keys()),
            "field_details": all_fields,
            "example_csv_headers": ",".join(all_fields.keys())
        }
    except Exception as e:
        return {
            "message": "Could not generate format information",
            "error": str(e),
            "note": "Please refer to the EmployeeData model documentation"
        }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)