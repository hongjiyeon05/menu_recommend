# services/recommend.py

import pandas as pd
from schemas import RecommendRequest, RecommendResponse, Recommendation
from ai_recommender import AIFoodRecommendationSystem

# ────────────────────────────────────────────────────────────
# 1) CSV 로드 및 기본 데이터 준비
# ────────────────────────────────────────────────────────────
menus_df = pd.read_csv("data/final_menus_data.csv")
restaurants_df = pd.read_csv("data/restaurants.csv")
restaurants_df["distance"] = restaurants_df.index % 20 * 0.1 + 0.5
restaurants_df.rename(columns={"name": "restaurant_name"}, inplace=True)

# ────────────────────────────────────────────────────────────
# 2) AI 시스템 주입용 변수 & 초기화 함수
# ────────────────────────────────────────────────────────────
_ai_system: AIFoodRecommendationSystem = None

def init_ai_system(system: AIFoodRecommendationSystem):
    """
    main.py에서 앱 시작 시 호출하여 AI 시스템을 주입합니다.
    """
    global _ai_system
    _ai_system = system

# ────────────────────────────────────────────────────────────
# 3) 유틸 함수들
# ────────────────────────────────────────────────────────────
def calc_fit(w, l, h, cw, cl, ch):
    if w > cw or l > cl or h > ch:
        return 0.0
    vol_ratio = (w * l * h) / (cw * cl * ch)
    return round(min(vol_ratio, 1.0), 2)

def get_all_categories():
    return menus_df["category"].dropna().unique().tolist()

# ────────────────────────────────────────────────────────────
# 4) 메인 추천 함수 (볼륨 기반 + AI 기반 통합)
# ────────────────────────────────────────────────────────────
def get_recommendations(req: RecommendRequest) -> RecommendResponse:
    # 카테고리 검증
    valid = set(get_all_categories())
    for cat in req.categories:
        if cat not in valid:
            raise KeyError(f"지원하지 않는 카테고리: {cat}")

    # AI 모드로 호출
    if getattr(req, "use_ai", False):
        if _ai_system is None:
            raise ValueError("AI 시스템이 초기화되지 않았습니다.")

        # 카테고리별 AI 추천을 모아서
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

        # 점수 순 정렬 후 상위 limit개
        candidates.sort(key=lambda x: x["final_ai_score"], reverse=True)
        selected = candidates[:req.limit]

        # Pydantic Recommendation 모델로 변환
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
        return RecommendResponse(
            page=1,
            limit=req.limit,
            total=len(recs),
            recommendations=recs
        )

    # ────────────────────────────────────────────────────────────
    # 기존 볼륨 기반 로직
    # ────────────────────────────────────────────────────────────
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
    return RecommendResponse(
        page=req.page,
        limit=req.limit,
        total=total,
        recommendations=recs
    )
