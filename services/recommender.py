# services/recommend.py

import pandas as pd
from schemas import RecommendRequest, RecommendResponse, Recommendation

# ─────────────── AI 시스템 클래스 정의 ───────────────
class AIFoodRecommendationSystem:
    def __init__(self, menus_df: pd.DataFrame, restaurants_df: pd.DataFrame):
        # 모델 로드 혹은 초기화 로직
        self.menus       = menus_df
        self.restaurants = restaurants_df

    def get_ai_recommendations(
        self,
        *,
        user_width: float,
        user_length: float,
        user_height: float,
        category: str,
        top_k: int
    ) -> dict:
        # 카테고리별 메뉴 필터링
        filtered = self.menus[self.menus["category"] == category]

        # (예시) AI 점수 계산 로직 넣어주세요
        candidates = []
        for _, row in filtered.iterrows():
            candidates.append({
                "menu_id": row["menu_id"],
                "menu_name": row["menu_name"],
                "restaurant_name": row["restaurant_id"],
                "price": row["price"],
                "final_ai_score": 0.5,             # 예시
                "container_utilization": 50        # 예시 (%)
            })

        return {
            "status": "success",
            "recommendations": sorted(
                candidates,
                key=lambda x: x["final_ai_score"],
                reverse=True
            )[:top_k]
        }

# ────────────────────────────────────────────────────────────
# CSV 로드 및 기본 데이터 준비
# ────────────────────────────────────────────────────────────
menus_df       = pd.read_csv("data/final_menus_data.csv")
restaurants_df = pd.read_csv("data/restaurants.csv")
restaurants_df["distance"] = restaurants_df.index % 20 * 0.1 + 0.5
restaurants_df.rename(columns={"name": "restaurant_name"}, inplace=True)

# ────────────────────────────────────────────────────────────
# AI 시스템 주입 & 유틸 함수들
# ────────────────────────────────────────────────────────────
_ai_system: AIFoodRecommendationSystem = None

def init_ai_system(system: AIFoodRecommendationSystem):
    global _ai_system
    _ai_system = system

def calc_fit(w, l, h, cw, cl, ch):
    if w > cw or l > cl or h > ch:
        return 0.0
    vol_ratio = (w * l * h) / (cw * cl * ch)
    return round(min(vol_ratio, 1.0), 2)

def get_all_categories():
    return menus_df["category"].dropna().unique().tolist()

# ────────────────────────────────────────────────────────────
# 메인 추천 함수
# ────────────────────────────────────────────────────────────
def get_recommendations(req: RecommendRequest) -> RecommendResponse:
    valid = set(get_all_categories())
    for cat in req.categories:
        if cat not in valid:
            raise KeyError(f"지원하지 않는 카테고리: {cat}")

    # AI 모드
    if getattr(req, "use_ai", False):
        if _ai_system is None:
            raise ValueError("AI 시스템이 초기화되지 않았습니다.")
        candidates = []
        for cat in req.categories:
            ai_res = _ai_system.get_ai_recommendations(
                user_width=req.container.width,
                user_length=req.container.length,
                user_height=req.container.height,
                category=cat,
                top_k=req.limit
            )
            if ai_res.get("status") == "success":
                candidates.extend(ai_res["recommendations"])

        candidates.sort(key=lambda x: x["final_ai_score"], reverse=True)
        selected = candidates[:req.limit]

        recs = [
            Recommendation(
                food_id=str(item["menu_id"]),
                food_name=item["menu_name"],
                restaurant_name=item["restaurant_name"],
                price=item["price"],
                distance=0.0,
                container_fit=round(item["container_utilization"]/100, 2),
                image_url="",
                description=f"AI 점수: {item['final_ai_score']}"
            )
            for item in selected
        ]
        return RecommendResponse(page=1, limit=req.limit, total=len(recs), recommendations=recs)

    # ───────────── 볼륨 기반 추천 ─────────────
    filtered = menus_df[menus_df["category"].isin(req.categories)].copy()
    cw, cl, ch = req.container.width, req.container.length, req.container.height
    filtered["container_fit"] = filtered.apply(
        lambda r: calc_fit(r["width"], r["length"], r["height"], cw, cl, ch), axis=1
    )
    merged = filtered.merge(restaurants_df, on="restaurant_id", how="left")

    if req.sort == "distance":
        merged.sort_values("distance", inplace=True)
    elif req.sort == "price_asc":
        merged.sort_values("price", inplace=True)
    elif req.sort == "price_desc":
        merged.sort_values("price", ascending=False, inplace=True)
    else:
        merged.sort_values("container_fit", ascending=False, inplace=True)

    total = len(merged)
    start = (req.page - 1) * req.limit
    page_df = merged.iloc[start : start + req.limit]

    recs = [
        Recommendation(
            food_id=str(r["menu_id"]),
            food_name=r["menu_name"],
            restaurant_name=r["restaurant_name"],
            price=int(r["price"]),
            distance=round(float(r["distance"]), 2),
            container_fit=float(r["container_fit"]),
            image_url=str(r.get("image_url") or ""),
            description=str(r.get("notes") or "")
        )
        for _, r in page_df.iterrows()
    ]
    return RecommendResponse(page=req.page, limit=req.limit, total=total, recommendations=recs)
