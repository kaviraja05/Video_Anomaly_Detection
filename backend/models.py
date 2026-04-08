from pydantic import BaseModel, EmailStr
from typing import List, Optional

class RegisterRequest(BaseModel):
    name: str
    email: EmailStr
    password: str

class LoginRequest(BaseModel):
    email: EmailStr
    password: str

class TokenResponse(BaseModel):
    access_token: str
    token_type: str
    user_name: Optional[str] = None
    user_email: Optional[str] = None

class AnomalySegmentModel(BaseModel):
    start_frame: int
    end_frame: int
    timestamp_start: float
    timestamp_end: float
    confidence: float
    severity: str

class AnalysisResultRequest(BaseModel):
    video_name: str
    anomaly_score: float
    status: str
    segments: List[AnomalySegmentModel]
