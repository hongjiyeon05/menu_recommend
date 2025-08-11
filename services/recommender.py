from __future__ import annotations

import warnings
from typing import Dict, List, Any, Optional

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer

from schemas import RecommendRequest, RecommendResponse, Recommendation

warnings.filterwarnings("ignore")

# 후보 컬럼 정의 (CSV 컬럼명이 달라도 흡수)
IMG_CANDIDATES = ["image_url", "img_url", "image", "imageLink", "thumbnail", "thumb_url"]
PLACE_CANDIDATES = ["place_id", "placeId", "google_place_id", "googlePlaceId"]
RID_CANDIDATES = ["restaurant_id", "restaurantId", "store_id", "storeId"]
RNAME_CANDIDATES = ["restaurant_name", "restaurantName", "name", "store_name"]
MID_CANDIDATES = ["menu_id", "menuId", "id"]
MNAME_CANDIDATES = ["menu_name", "menuName", "food_name", "foodName", "name"]

DEFAULT_VALUES = {"POPULARITY_SCORE": 5.0}

FLEXIBLE_FOODS: List[str] = [
    "비빔", "볶음", "덮밥", "샐러드", "쌈", "라면", "우동", "칼국수",
    "비건", "곱빼기", "주먹밥", "비지", "오믈렛", "비엔나", "소바"
]
RIGID_FOODS: List[str] = [
    "피자", "케이크", "도시락", "김밥", "햄버거", "만두", "호빵",
    "식빵", "타르트", "파이", "샌드위치"
]

def _first_col(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    for c in candidates:
        if c in df.columns:
            return c
    return None

def _coalesce_row(row: pd.Series, candidates: List[str]) -> str:
    for c in candidates:
        if c in row and pd.notna(row[c]) and str(row[c]).strip().lower() not in {"", "nan", "none", "null"}:
            return str(row[c])
    return ""

class AIFoodRecommendationSystem:
    def __init__(self, menus_df: pd.DataFrame, restaurants_df: pd.DataFrame):
        # 원본 보존
        self.menus_df = menus_df.copy()
        self.restaurants_df = restaurants_df.copy()

        # 레스토랑 DF 정리
        drop_cols = [c for c in self.restaurants_df.columns if str(c).startswith("Unnamed")]
        if drop_cols:
            self.restaurants_df = self.restaurants_df.drop(columns=drop_cols, errors="ignore")

        # 식당 ID / 이름 / place / 이미지 컬럼 정규화
        self.rid_col = _first_col(self.restaurants_df, RID_CANDIDATES) or "restaurant_id"
        self.rname_col = _first_col(self.restaurants_df, RNAME_CANDIDATES) or "restaurant_name"
        self.rplace_col = _first_col(self.restaurants_df, PLACE_CANDIDATES)  # 없을 수도 있음
        self.rimg_col = _first_col(self.restaurants_df, IMG_CANDIDATES)      # 없을 수도 있음

        if self.rid_col not in self.restaurants_df.columns:
            self.restaurants_df[self.rid_col] = ""
        if self.rname_col in self.restaurants_df.columns:
            self.restaurants_df[self.rname_col] = self.restaurants_df[self.rname_col].astype(str)
        self.restaurants_df[self.rid_col] = self.restaurants_df[self.rid_col].astype(str)
        if self.rplace_col and self.rplace_col in self.restaurants_df.columns:
            self.restaurants_df[self.rplace_col] = self.restaurants_df[self.rplace_col].fillna("").astype(str)
        if self.rimg_col and self.rimg_col in self.restaurants_df.columns:
            self.restaurants_df[self.rimg_col] = self.restaurants_df[self.rimg_col].fillna("").astype(str)

        # 빠른 조회용 맵
        self._rid2name = dict(zip(self.restaurants_df[self.rid_col], self.restaurants_df.get(self.rname_col, pd.Series(dtype=str))))
        self._rid2place = dict(zip(self.restaurants_df[self.rid_col], self.restaurants_df.get(self.rplace_col, pd.Series(dtype=str)))) if self.rplace_col else {}
        self._rid2img   = dict(zip(self.restaurants_df[self.rid_col], self.restaurants_df.get(self.rimg_col,   pd.Series(dtype=str)))) if self.rimg_col   else {}

        # 메뉴 DF 전처리
        self._preprocess_data()

        # ML 파이프라인
        self.tfidf_vectorizer = TfidfVectorizer(max_features=30, ngram_range=(1, 2), analyzer="char")
        self.scaler = StandardScaler()
        self.cluster_info: Dict[str, Dict[str, Any]] = {}
        self._build_ai_features()

    def _preprocess_data(self) -> None:
        # 메뉴측 핵심 컬럼 이름 파악
        self.mid_col = _first_col(self.menus_df, MID_CANDIDATES) or "menu_id"
        self.mname_col = _first_col(self.menus_df, MNAME_CANDIDATES) or "menu_name"
        self.mrid_col = _first_col(self.menus_df, RID_CANDIDATES) or "restaurant_id"
        self.mimg_col = _first_col(self.menus_df, IMG_CANDIDATES)  # 없을 수도

        # 타입 정리
        for col in ["price", "width", "length", "height", "popularity_score"]:
            if col in self.menus_df.columns:
                self.menus_df[col] = pd.to_numeric(self.menus_df[col], errors="coerce")

        if "popularity_score" in self.menus_df.columns:
            self.menus_df["popularity_score"] = self.menus_df["popularity_score"].fillna(DEFAULT_VALUES["POPULARITY_SCORE"])
        else:
            self.menus_df["popularity_score"] = DEFAULT_VALUES["POPULARITY_SCORE"]

        # 부피 계산
        if all(c in self.menus_df.columns for c in ["width", "length", "height"]):
            self.menus_df["volume"] = (
                self.menus_df["width"].fillna(0)
                * self.menus_df["length"].fillna(0)
                * self.menus_df["height"].fillna(0)
            )
        else:
            self.menus_df["volume"] = 0

        # 텍스트/ID/이미지 정리
        if self.mname_col not in self.menus_df.columns:
            self.menus_df[self.mname_col] = ""
        self.menus_df[self.mname_col] = self.menus_df[self.mname_col].astype(str)
        if self.mid_col not in self.menus_df.columns:
            self.menus_df[self.mid_col] = ""
        self.menus_df[self.mid_col] = self.menus_df[self.mid_col].astype(str)
        if self.mrid_col not in self.menus_df.columns:
            self.menus_df[self.mrid_col] = ""
        self.menus_df[self.mrid_col] = self.menus_df[self.mrid_col].astype(str)

        # 이미지 후보 컬럼 전부 빈값 캐스팅
        for alt in IMG_CANDIDATES:
            if alt in self.menus_df.columns:
                self.menus_df[alt] = self.menus_df[alt].fillna("").astype(str)

        # 유연/고정 분류
        self.menus_df["food_flexibility"] = self.menus_df[self.mname_col].apply(self._classify_food_flexibility)

    def _classify_food_flexibility(self, name: Any) -> str:
        if pd.isna(name):
            return "flexible"
        s = str(name).lower()
        for kw in RIGID_FOODS:
            if kw and kw.lower() in s:
                return "rigid"
        for kw in FLEXIBLE_FOODS:
            if kw and kw.lower() in s:
                return "flexible"
        return "flexible"

    def _build_ai_features(self) -> None:
        if "category" not in self.menus_df.columns:
            self.cluster_info = {}
            return
        categories = self.menus_df["category"].dropna().astype(str).unique().tolist()
        for category in categories:
            cat_df = self.menus_df[self.menus_df["category"].astype(str) == category].copy()
            if len(cat_df) < 2:
                self.cluster_info[category] = {
                    "clusters": {
                        0: {
                            "indices": cat_df.index.tolist(),
                            "avg_popularity": float(cat_df["popularity_score"].mean()) if "popularity_score" in cat_df.columns else DEFAULT_VALUES["POPULARITY_SCORE"],
                            "size": int(len(cat_df)),
                            "examples": cat_df[self.mname_col].head(2).tolist(),
                        }
                    },
                    "n_clusters": 1,
                }
                continue

            texts = cat_df[self.mname_col].fillna("").astype(str)
            text_vecs = self.tfidf_vectorizer.fit_transform(texts).toarray()

            numeric_cols = [c for c in ["price", "volume", "popularity_score"] if c in cat_df.columns]
            if numeric_cols:
                numeric_data = self.scaler.fit_transform(cat_df[numeric_cols].fillna(0))
                feats = np.hstack([text_vecs, numeric_data]) if text_vecs.size else numeric_data
            else:
                feats = text_vecs

            n = len(cat_df)
            k = min(max(2, n // 5), 4)
            km = KMeans(n_clusters=k, random_state=42, n_init=10)
            labels = km.fit_predict(feats)

            clusters: Dict[int, Dict[str, Any]] = {}
            for cid in range(k):
                idxs = cat_df.index[labels == cid].tolist()
                d = cat_df.loc[idxs]
                clusters[cid] = {
                    "indices": idxs,
                    "avg_popularity": float(d["popularity_score"].mean()) if "popularity_score" in d.columns and len(d) > 0 else DEFAULT_VALUES["POPULARITY_SCORE"],
                    "size": int(len(d)),
                    "examples": d[self.mname_col].head(2).tolist(),
                }
            self.cluster_info[category] = {"clusters": clusters, "n_clusters": k}

    def calculate_container_utilization(
        self, uw: float, ul: float, uh: float, mw: float, ml: float, mh: float, _menu_name: Any, flex: str
    ) -> float:
        try:
            uw, ul, uh = float(uw), float(ul), float(uh)
            mw, ml, mh = float(mw), float(ml), float(mh)
        except Exception:
            return 0.0
        uv = uw * ul * uh
        mv = mw * ml * mh
        if uv <= 0 or mv <= 0:
            return 0.0
        if mv > uv:
            return 0.0
        if str(flex).lower() == "flexible":
            return float(min(100.0, (mv / uv) * 110.0))
        if mw > uw or ml > ul or mh > uh:
            return 0.0
        return float((mv / uv) * 100.0)

    def calculate_ai_cluster_score(self, idx: int, category: str) -> float:
        info = self.cluster_info.get(str(category))
        if not info:
            return 50.0
        for _, data in info.get("clusters", {}).items():
            if idx in data.get("indices", []):
                avg = float(data.get("avg_popularity", DEFAULT_VALUES["POPULARITY_SCORE"]))
                size = int(data.get("size", 1))
                bonus = min(20.0, size * 5.0)
                return float(min(100.0, avg * 8.0 + bonus))
        return 50.0

    def get_ai_recommendations(self, user_width: float, user_length: float, user_height: float, category: str, top_k: int = 10) -> Dict[str, Any]:
        df = self.menus_df[self.menus_df["category"].astype(str) == str(category)]
        recs: List[Dict[str, Any]] = []
        for idx, m in df.iterrows():
            util = self.calculate_container_utilization(
                user_width, user_length, user_height,
                m.get("width", 0), m.get("length", 0), m.get("height", 0),
                m.get(self.mname_col, ""), m.get("food_flexibility", "flexible")
            )
            if util <= 0:
                continue

            cs = self.calculate_ai_cluster_score(idx, str(category))
            ps_src = m.get("popularity_score", DEFAULT_VALUES["POPULARITY_SCORE"])
            try:
                ps = float(ps_src)
            except Exception:
                ps = DEFAULT_VALUES["POPULARITY_SCORE"]
            ps = min(100.0, ps * 10.0)
            score = util * 0.5 + cs * 0.3 + ps * 0.2

            rid = str(m.get(self.mrid_col, ""))
            rname = self._rid2name.get(rid, "정보없음")
            rplace = self._rid2place.get(rid, "")
            rimg = self._rid2img.get(rid, "")

            recs.append({
                "menu_id": m.get(self.mid_col, idx),
                "menu_name": str(m.get(self.mname_col, "")),
                "restaurant_name": rname,
                "price": int(float(m.get("price", 0) or 0)) if pd.notna(m.get("price", 0)) else 0,
                "container_utilization": round(util, 1),
                "final_ai_score": round(float(score), 1),
                # 이미지는 메뉴측 우선 → 없으면 식당측
                "image_url": _coalesce_row(m, IMG_CANDIDATES) or rimg or "",
                "place_id": rplace or "",
            })
        recs.sort(key=lambda x: x["final_ai_score"], reverse=True)
        return {"status": "success", "recommendations": recs[: max(1, int(top_k))]}

# 전역 인스턴스 (main.py에서 init_ai_system으로 주입)
_ai_system: Optional[AIFoodRecommendationSystem] = None

def init_ai_system(system: AIFoodRecommendationSystem) -> None:
    global _ai_system
    _ai_system = system

def _ensure_ai() -> AIFoodRecommendationSystem:
    if _ai_system is None:
        raise ValueError("AI 시스템이 초기화되지 않았습니다.")
    return _ai_system

def calc_fit(mw: float, ml: float, mh: float, cw: float, cl: float, ch: float) -> float:
    try:
        mw, ml, mh = float(mw), float(ml), float(mh)
        cw, cl, ch = float(cw), float(cl), float(ch)
    except Exception:
        return 0.0
    if mw > cw or ml > cl or mh > ch:
        return 0.0
    uv = cw * cl * ch
    mv = mw * ml * mh
    if uv <= 0:
        return 0.0
    return round(min(mv / uv, 1.0), 2)

def get_all_categories() -> List[str]:
    ai = _ensure_ai()
    if "category" not in ai.menus_df.columns:
        return []
    return ai.menus_df["category"].dropna().astype(str).unique().tolist()

def get_recommendations(req: RecommendRequest) -> RecommendResponse:
    ai = _ensure_ai()
    menus = ai.menus_df

    # 카테고리 검증
    valid = set(menus.get("category", pd.Series(dtype=str)).dropna().astype(str).unique())
    for c in req.categories:
        if str(c) not in valid:
            raise KeyError(f"지원하지 않는 카테고리: {c}")

    # ── AI 모드 ───────────────────────────────────────────────
    if getattr(req, "use_ai", False):
        cand: List[Dict[str, Any]] = []
        for c in req.categories:
            res = ai.get_ai_recommendations(
                user_width=req.container.width,
                user_length=req.container.length,
                user_height=req.container.height,
                category=str(c),
                top_k=req.limit
            )
            if res.get("status") == "success":
                cand.extend(res["recommendations"])
        cand.sort(key=lambda x: x["final_ai_score"], reverse=True)
        sel = cand[: req.limit]

        recs = [
            Recommendation(
                food_id=str(i.get("menu_id", "")),
                food_name=str(i.get("menu_name", "")),
                restaurant_name=str(i.get("restaurant_name", "정보없음")),
                price=int(i.get("price", 0) or 0),
                distance=0.0,
                container_fit=round(float(i.get("container_utilization", 0.0)) / 100.0, 2),
                image_url=str(i.get("image_url", "") or ""),
                place_id=str(i.get("place_id", "") or ""),
                description=f"AI 점수: {i.get('final_ai_score', 0)}",
            )
            for i in sel
        ]
        return RecommendResponse(page=1, limit=req.limit, total=len(recs), recommendations=recs)

    # ── 기본(볼륨 기반) 경로 ──────────────────────────────────
    df = menus[menus["category"].astype(str).isin([str(c) for c in req.categories])].copy()
    cw, cl, ch = req.container.width, req.container.length, req.container.height

    df["container_fit"] = df.apply(
        lambda r: calc_fit(r.get("width", 0), r.get("length", 0), r.get("height", 0), cw, cl, ch),
        axis=1
    )

    # 식당 정보 조인 (이름, place, 이미지 후보 포함)
    cols = [c for c in {ai.rid_col, ai.rname_col, *(PLACE_CANDIDATES + IMG_CANDIDATES)} if c in ai.restaurants_df.columns]
    rests_min = ai.restaurants_df[cols].copy() if cols else ai.restaurants_df
    merged = df.merge(rests_min, left_on=ai.mrid_col, right_on=ai.rid_col, how="left")

    # 조인 후 이미지 coalesce (메뉴측 후보 → 식당측 후보)
    img_candidates = [c for c in IMG_CANDIDATES if c in merged.columns]
    if img_candidates:
        tmp = merged[img_candidates].astype(str).replace({"nan": "", "None": "", "null": ""})
        merged["__img_join"] = tmp.bfill(axis=1).iloc[:, 0].fillna("")
    else:
        merged["__img_join"] = ""

    # place coalesce
    place_candidates = [c for c in PLACE_CANDIDATES if c in merged.columns]
    if place_candidates:
        tmp = merged[place_candidates].astype(str).replace({"nan": "", "None": "", "null": ""})
        merged["__place_join"] = tmp.bfill(axis=1).iloc[:, 0].fillna("")
    else:
        merged["__place_join"] = ""

    # 정렬
    sort_key = str(getattr(req, "sort", "default") or "default")
    if sort_key == "distance" and "distance" in merged.columns:
        merged = merged.sort_values("distance", ascending=True)
    elif sort_key == "price_asc" and "price" in merged.columns:
        merged = merged.sort_values("price", ascending=True)
    elif sort_key == "price_desc" and "price" in merged.columns:
        merged = merged.sort_values("price", ascending=False)
    else:
        merged = merged.sort_values("container_fit", ascending=False)

    # 페이지네이션
    total = int(len(merged))
    start = max(0, (req.page - 1) * req.limit)
    page_df = merged.iloc[start : start + req.limit].copy()

    # 응답 변환
    recs = []
    for _, r in page_df.iterrows():
        # 가격/거리 안전 처리
        try: price_int = int(float(r.get("price", 0) or 0))
        except Exception: price_int = 0
        try: distance_f = float(r.get("distance", 0.0)) if pd.notna(r.get("distance", 0.0)) else 0.0
        except Exception: distance_f = 0.0

        # 이름/ID/이미지/플레이스
        food_id = str(r.get(ai.mid_col, r.get("menu_id", r.get("id", ""))))
        food_name = str(r.get(ai.mname_col, r.get("menu_name", r.get("name", ""))))
        rname = str(r.get(ai.rname_col, "정보없음"))
        image_url = str(r.get("__img_join") or "")
        place_id = str(r.get("__place_join") or "")

        recs.append(
            Recommendation(
                food_id=food_id,
                food_name=food_name,
                restaurant_name=rname,
                price=price_int,
                distance=distance_f,
                container_fit=float(r.get("container_fit", 0.0) or 0.0),
                image_url=image_url,
                place_id=place_id,
                description=str(r.get("notes") or ""),
            )
        )

    return RecommendResponse(page=req.page, limit=req.limit, total=total, recommendations=recs)
