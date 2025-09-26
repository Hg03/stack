from pydantic import BaseModel, Field
from typing import Optional
from enum import Enum

class WorkFromOptions(str, Enum):
    HOME = "Home"
    OFFICE = "Office"
    HYBRID = "Hybrid"

class WorkLifeBalanceOptions(str, Enum):
    YES = "Yes"
    NO = "No"

class YesNoOptions(str, Enum):
    YES = "Yes"
    NO = "No"

class EmployeeData(BaseModel):
    employee_id: str = Field(..., description="Unique employee identifier")
    avg_working_hours_per_day: float = Field(..., ge=0, le=24, description="Average working hours per day")
    work_from: WorkFromOptions = Field(..., description="Work location preference")
    work_pressure: int = Field(..., ge=1, le=5, description="Work pressure level (1-5 scale)")
    manager_support: int = Field(..., ge=1, le=5, description="Manager support level (1-5 scale)")
    sleeping_habit: int = Field(..., ge=1, le=5, description="Sleeping habit quality (1-5 scale)")
    exercise_habit: int = Field(..., ge=1, le=5, description="Exercise habit frequency (1-5 scale)")
    job_satisfaction: int = Field(..., ge=1, le=5, description="Job satisfaction level (1-5 scale)")
    work_life_balance: WorkLifeBalanceOptions = Field(..., description="Work-life balance status")
    social_person: int = Field(..., ge=1, le=5, description="Social personality level (1-5 scale)")
    lives_with_family: YesNoOptions = Field(..., description="Whether lives with family")
    working_state: str = Field(..., description="State/region where working")

    class Config:
        use_enum_values = True  # This allows accepting string values that match enum values
        validate_assignment = True  # Validates on assignment too
        json_schema_extra = {
            "example": {
                "employee_id": "EMP0001",
                "avg_working_hours_per_day": 6.7,
                "work_from": "Home",
                "work_pressure": 3,
                "manager_support": 4,
                "sleeping_habit": 4,
                "exercise_habit": 2,
                "job_satisfaction": 5,
                "work_life_balance": "No",
                "social_person": 5,
                "lives_with_family": "Yes",
                "working_state": "Karnataka",
            }
        }