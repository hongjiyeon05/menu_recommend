# main.py
from fastapi import FastAPI
from contextlib import asynccontextmanager
import pandas as pd, os
from routes.recommend import router as recommend_router
from services.recommender import AIFoodRecommendationSystem, init_ai_system

@asynccontextmanager
async def lifespan(app: FastAPI):
    data_dir = os.path.join(os.getcwd(), "data")
    menus_csv = os.path.join(data_dir, "final_menus_data.csv")
    rests_csv = os.path.join(data_dir, "restaurants.csv")

    # 파일 존재 확인 (Render 경로 기준)
    assert os.path.exists(menus_csv), f"NOT FOUND: {menus_csv}"
    assert os.path.exists(rests_csv), f"NOT FOUND: {rests_csv}"

    menus_df = pd.read_csv(menus_csv)
    rests_df = pd.read_csv(rests_csv)

    init_ai_system(AIFoodRecommendationSystem(menus_df, rests_df))
    yield

app = FastAPI(lifespan=lifespan)

# 라우터 연결
app.include_router(recommend_router, prefix="/recommend", tags=["recommend"])

# 헬스 체크(루트 404 피하기용)
@app.get("/health")
def health():
    return {"ok": True}
