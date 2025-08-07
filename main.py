# main.py

from fastapi import FastAPI
from routes.recommend import router as recommend_router
from services.recommend import init_ai_system
from ai_recommender import AIFoodRecommendationSystem

app = FastAPI(
    title="Food Recommendation API",
    description="용기 기반 음식 추천 시스템",
    version="1.0"
)

# AI 시스템을 한 번만 초기화 & 주입
ai_system = AIFoodRecommendationSystem(csv_dir="data")
init_ai_system(ai_system)

# 라우터 등록
app.include_router(recommend_router)
