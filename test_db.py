import asyncio
from motor.motor_asyncio import AsyncIOMotorClient
client = AsyncIOMotorClient('mongodb+srv://kavitharraja84_db_user:pawgEd9BOl4P8hp6@cluster0.15bteww.mongodb.net/')
async def main():
    try:
        print(await client.server_info())
    except Exception as e:
        print(e)
asyncio.run(main())
