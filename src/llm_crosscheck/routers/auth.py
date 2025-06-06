from fastapi import APIRouter

router = APIRouter()


@router.get("/login")
async def login() -> dict[str, str]:
    return {"message": "Login endpoint - placeholder"}
