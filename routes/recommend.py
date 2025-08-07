# routes/recommend.py
from fastapi import APIRouter, HTTPException
from typing import List

from schemas import RecommendRequest, RecommendResponse
from services.recommend import get_recommendations, get_all_categories

router = APIRouter()

@router.get("/categories", response_model=List[str])
def categories():
    return get_all_categories()

@router.post("/", response_model=RecommendResponse)
def recommend(req: RecommendRequest):
    try:
        return get_recommendations(req)
    except KeyError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=500, detail=str(e))
