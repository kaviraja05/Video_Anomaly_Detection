import asyncio
import sys
from pathlib import Path

project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from backend.database import get_user_by_email, create_user

async def main():
    try:
        print("Testing DB connection...")
        user = await get_user_by_email("test3@test.com")
        print("User lookup result:", user)
        if not user:
            print("Creating user...")
            new_user = await create_user("Test", "test3@test.com", "Password1!")
            print("Created user:", new_user)
    except Exception as e:
        import traceback
        traceback.print_exc()
        print("Error:", repr(e))

if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

asyncio.run(main())
