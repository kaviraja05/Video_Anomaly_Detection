from fastapi import APIRouter, Depends
from backend.auth import get_current_user
from backend.database import get_user_results

router = APIRouter()

@router.get("/user-results")
async def get_history(current_user: dict = Depends(get_current_user)):
    user_id = current_user["_id"]
    results = await get_user_results(user_id)
    return {"history": results}
