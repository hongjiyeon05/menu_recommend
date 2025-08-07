# main.py
from fastapi import FastAPI
import pandas as pd

from routes.recommend import router as recommend_router
from services.recommend import init_ai_system, AIFoodRecommendationSystem

app = FastAPI()

# 데이터 로드
menus_df       = pd.read_csv("data/final_menus_data.csv")
restaurants_df = pd.read_csv("data/restaurants.csv")

# AI 시스템 초기화 및 주입
ai_sys = AIFoodRecommendationSystem(menus_df, restaurants_df)
init_ai_system(ai_sys)

# 라우터 등록
app.include_router(recommend_router, prefix="/recommend", tags=["recommend"])
