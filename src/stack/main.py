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
    try:
        # Check if the uploaded file is a CSV
        if not file.filename.endswith('.csv'):
            raise HTTPException(status_code=400, detail="Only CSV files are supported")
        
        # Read the file content
        content = await file.read()
        
        # Convert bytes to string and create a StringIO object for CSV reading
        csv_content = content.decode('utf-8')
        csv_file = io.StringIO(csv_content)
        
        # Read CSV data directly into DataFrame
        df = pd.read_csv(csv_file)
        
        print("Original DataFrame columns:")
        print(list(df.columns))
        print("DataFrame head:")
        print(df.head())
        
        # Map CSV column names to EmployeeData field names
        column_mapping = {
            'Employee_Id': 'employee_id',
            'Avg_Working_Hours_Per_Day': 'avg_working_hours_per_day',
            'Work_From': 'work_from',
            'Work_Pressure': 'work_pressure',
            'Manager_Support': 'manager_support',
            'Sleeping_Habit': 'sleeping_habit',
            'Exercise_Habit': 'exercise_habit',
            'Job_Satisfaction': 'job_satisfaction',
            'Work_Life_Balance': 'work_life_balance',
            'Social_Person': 'social_person',
            'Lives_With_Family': 'lives_with_family',
            'Working_State': 'working_state',
            'Stress_Level': 'stress_level'  # In case it exists in CSV
        }
        
        # Rename columns to match EmployeeData fields
        df = df.rename(columns=column_mapping)
        
        # Remove any columns that aren't in EmployeeData (like Stress_Level if it exists)
        expected_fields = set(EmployeeData.__fields__.keys())
        df_fields = set(df.columns)
        extra_columns = df_fields - expected_fields
        
        if extra_columns:
            print(f"Removing extra columns: {extra_columns}")
            df = df.drop(columns=list(extra_columns))
        
        # Check if all required fields are present
        missing_fields = expected_fields - set(df.columns)
        if missing_fields:
            raise HTTPException(
                status_code=400,
                detail=f"Missing required columns: {', '.join(missing_fields)}. Available columns: {', '.join(df.columns)}"
            )
        
        print("After column mapping:")
        print(list(df.columns))
        
        # Convert each row to EmployeeData to ensure proper data types and validation
        processed_rows = []
        validation_errors = []
        
        for index, row in df.iterrows():
            try:
                # Convert row to dictionary and clean the data
                row_dict = row.to_dict()
                
                # Handle potential data cleaning
                for key, value in row_dict.items():
                    if pd.isna(value):
                        row_dict[key] = None
                    elif isinstance(value, str):
                        row_dict[key] = value.strip()  # Remove whitespace
                
                # Convert row to EmployeeData which will handle type conversion and validation
                employee_data = EmployeeData(**row_dict)
                processed_rows.append(employee_data)
                
            except Exception as e:
                error_msg = f"Row {index + 1}: {str(e)}"
                validation_errors.append(error_msg)
                print(f"Validation error for row {index + 1}: {e}")
        
        # If there are validation errors, return them
        if validation_errors:
            raise HTTPException(
                status_code=400, 
                detail=f"Data validation errors found:\n" + "\n".join(validation_errors[:5])  # Show first 5 errors
            )
        
        # Convert back to DataFrame with proper types for model inference
        # This ensures the data types match what your model expects
        processed_df = pd.DataFrame([row.dict() for row in processed_rows])
        
        # Convert column names to lowercase
        processed_df.columns = processed_df.columns.str.lower()
        
        print("Processed DataFrame dtypes:")
        print(processed_df.dtypes)
        print("Processed DataFrame head:")
        print(processed_df.head())
        
        # Now run batch inference on the processed DataFrame
        inference_pipe = Infer()
        result = inference_pipe.make_inference(payload=processed_df, infer_type="batch")
        
        return InferenceResponse(
            result={
                "predictions": result,
                "filename": file.filename,
                "content_type": file.content_type,
                "records_processed": len(processed_rows),
                "columns_found": list(df.columns)
            }, 
            status="success", 
            message="Batch inference completed successfully"
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing file: {str(e)}")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)