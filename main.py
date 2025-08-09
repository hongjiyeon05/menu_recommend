# main.py (권장 리팩토링)
from fastapi import FastAPI
from contextlib import asynccontextmanager
from pathlib import Path
import pandas as pd

from routes.recommend import router as recommend_router
from services.recommender import init_ai_system, AIFoodRecommendationSystem

BASE = Path(__file__).parent

@asynccontextmanager
async def lifespan(app: FastAPI):
    menus_df       = pd.read_csv(BASE / "data/final_menus_data.csv")
    restaurants_df = pd.read_csv(BASE / "data/restaurants.csv")
    app.state.ai_sys = AIFoodRecommendationSystem(menus_df, restaurants_df)
    yield
    # (필요하면 종료 처리)

app = FastAPI(lifespan=lifespan)
app.include_router(recommend_router, prefix="/recommend", tags=["recommend"])
