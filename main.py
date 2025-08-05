from fastapi import FastAPI
from routes import recommend

app = FastAPI(
    title="Food Recommendation API",
    description="용기 기반 음식 추천 시스템",
    version="1.0"
)

app.include_router(recommend.router)

