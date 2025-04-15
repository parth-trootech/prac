from datetime import datetime
from typing import Optional

from pydantic import BaseModel, EmailStr


# Pydantic model for User Registration (Signup)
class UserCreate(BaseModel):
    user_email: EmailStr
    user_password: str

    class Config:
        from_attributes = True


# Pydantic model for User Login
class UserLogin(BaseModel):
    user_email: EmailStr
    user_password: str

    class Config:
        from_attributes = True


# Response Model for Prediction Result
class PredictionResultResponse(BaseModel):
    prediction_id: int
    image_id: int
    predicted_digit: str
    confidence_score: Optional[float]
    prediction_time: datetime

    class Config:
        from_attributes = True


# Added PredictionRequest model here
class PredictionRequest(BaseModel):
    image_id: int
