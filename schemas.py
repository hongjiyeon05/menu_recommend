from pydantic import BaseModel, Field, conint, field_validator
from typing import List, Optional

class Container(BaseModel):
    width: conint(gt=0)  = Field(..., example=20, description="용기 가로 길이 (cm)")
    length: conint(gt=0) = Field(..., example=15, description="용기 세로 길이 (cm)")
    height: conint(gt=0) = Field(..., example=10, description="용기 높이 (cm)")

class RecommendRequest(BaseModel):
    container: Container = Field(..., description="사용자의 용기 사이즈 정보")
    categories: List[str]  = Field(..., example=["한식", "양식"], description="선택된 음식 카테고리 리스트")
    sort: Optional[str]    = Field("default", example="price_asc", description="정렬 기준")
    page: Optional[int]    = Field(1, example=1, description="페이지 번호")
    limit: Optional[int]   = Field(10, example=5, description="한 페이지당 결과 수")
    use_ai: Optional[bool] = Field(False, description="AI 추천 사용 여부")

class Recommendation(BaseModel):
    food_id: str         = Field(..., example="M203")
    food_name: str       = Field(..., example="김치볶음밥")
    restaurant_name: str = Field(..., example="맛있는집")
    price: int           = Field(..., example=8500)
    distance: float      = Field(..., example=1.2)
    container_fit: float = Field(..., example=0.92, description="용기에 대한 적합도")
    image_url: str       = Field("", example="https://example.com/image.jpg")  # 기본값 ""
    place_id: str        = Field("", example="ChIJN1t_tDeuEmsRUsoyG83frY4")     # 추가
    description: str     = Field(..., example="매콤한 김치볶음밥 설명")

    @field_validator("image_url", "place_id", mode="before")
    @classmethod
    def none_to_empty(cls, v):
        if v is None or str(v).strip().lower() in {"null", "none"}:
            return ""
        return str(v)

class RecommendResponse(BaseModel):
    page: int  = Field(..., example=1)
    limit: int = Field(..., example=5)
    total: int = Field(..., example=134)
    recommendations: List[Recommendation]

class CategoryList(BaseModel):
    categories: List[str] = Field(..., example=["한식", "중식", "일식", "양식", "기타"])
