import pandas as pd
from schemas import RecommendRequest, RecommendResponse, Recommendation

# 실제 CSV 경로
menus_df = pd.read_csv("data/final_menus_data.csv")
restaurants_df = pd.read_csv("data/restaurants.csv")

# 거리 정보 없으므로 index 기준 가짜 거리 생성
restaurants_df["distance"] = restaurants_df.index % 20 * 0.1 + 0.5
restaurants_df.rename(columns={"name": "restaurant_name"}, inplace=True)

# 적합도 계산 함수
def calc_fit(w, l, h, cw, cl, ch):
    if w > cw or l > cl or h > ch:
        return 0.0
    vol_ratio = (w * l * h) / (cw * cl * ch)
    return round(min(vol_ratio, 1.0), 2)

def get_all_categories():
    return menus_df["category"].dropna().unique().tolist()

def get_recommendations(req: RecommendRequest) -> RecommendResponse:
    valid = set(get_all_categories())
    for cat in req.categories:
        if cat not in valid:
            raise KeyError(f"지원하지 않는 카테고리: {cat}")

    # 메뉴 필터링
    filtered = menus_df[menus_df["category"].isin(req.categories)].copy()

    # 적합도 계산
    c_w, c_l, c_h = req.container.width, req.container.length, req.container.height
    filtered["container_fit"] = filtered.apply(
        lambda row: calc_fit(row["width"], row["length"], row["height"], c_w, c_l, c_h),
        axis=1
    )

    # 식당 정보 병합
    merged = filtered.merge(restaurants_df, on="restaurant_id", how="left")

    # 정렬
    sort_key = req.sort
    if sort_key == "distance":
        merged.sort_values("distance", inplace=True)
    elif sort_key == "price_asc":
        merged.sort_values("price", inplace=True)
    elif sort_key == "price_desc":
        merged.sort_values("price", ascending=False, inplace=True)
    else:  # default or container_fit
        merged.sort_values("container_fit", ascending=False, inplace=True)

    # 페이징
    total = len(merged)
    start = (req.page - 1) * req.limit
    end = start + req.limit
    page_df = merged.iloc[start:end]

    # 응답 객체 생성
    recommendations = []
    for _, row in page_df.iterrows():
        recommendations.append(Recommendation(
            food_id=row["menu_id"],
            food_name=row["menu_name"],
            restaurant_name=row["restaurant_name"],
            price=int(row["price"]),
            distance=round(float(row["distance"]), 2),
            container_fit=row["container_fit"],
            image_url=str(row.get("image_url") or ""),
            description=str(row.get("notes") or "")
        ))

    return RecommendResponse(
        page=req.page,
        limit=req.limit,
        total=total,
        recommendations=recommendations
    )
