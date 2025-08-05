# routes/recommend.py
from fastapi import APIRouter, HTTPException
from schemas import RecommendRequest, RecommendResponse, CategoryList
from services.recommend import get_recommendations, get_all_categories

router = APIRouter()

@router.post("/recommend", response_model=RecommendResponse, tags=["추천"])
async def recommend(req: RecommendRequest):
    try:
        return get_recommendations(req)
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except KeyError as ke:
        raise HTTPException(status_code=404, detail=str(ke))
    except Exception:
        raise HTTPException(status_code=500, detail="Internal Server Error")

@router.get("/categories", response_model=CategoryList, tags=["추천"])
async def categories():
    return {"categories": get_all_categories()}
