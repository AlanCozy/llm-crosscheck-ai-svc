from fastapi import APIRouter

router = APIRouter()


@router.get("/")
async def perform_crosscheck() -> dict[str, str]:
    return {"message": "Crosscheck placeholder"}
