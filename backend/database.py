import os
from datetime import datetime
from motor.motor_asyncio import AsyncIOMotorClient
from passlib.context import CryptContext

MONGO_URI = os.getenv("MONGO_URI", "mongodb+srv://kavitharraja84_db_user:pawgEd9BOl4P8hp6@cluster0.15bteww.mongodb.net/")
client = AsyncIOMotorClient(MONGO_URI)
db = client.video_anomaly_db

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

def get_password_hash(password):
    return pwd_context.hash(password)

def verify_password(plain_password, hashed_password):
    return pwd_context.verify(plain_password, hashed_password)

async def create_user(name: str, email: str, password: str):
    user = {
        "name": name,
        "email": email,
        "password_hash": get_password_hash(password),
        "created_at": datetime.utcnow()
    }
    result = await db.users.insert_one(user)
    user["_id"] = str(result.inserted_id)
    return user

async def get_user_by_email(email: str):
    user = await db.users.find_one({"email": email})
    if user:
        user["_id"] = str(user["_id"])
    return user

async def save_analysis_result(user_id: str, video_name: str, anomaly_score: float, status: str, segments: list):
    result = {
        "user_id": user_id,
        "video_name": video_name,
        "anomaly_score": anomaly_score,
        "status": status,
        "segments": segments,
        "created_at": datetime.utcnow()
    }
    res = await db.analysis_results.insert_one(result)
    result["_id"] = str(res.inserted_id)
    return result

async def get_user_results(user_id: str):
    cursor = db.analysis_results.find({"user_id": user_id}).sort("created_at", -1)
    results = []
    async for doc in cursor:
        doc["_id"] = str(doc["_id"])
        results.append(doc)
    return results
