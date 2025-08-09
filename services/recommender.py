# services/recommender.py

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

# ─────────────────────────────────────────────────────────────
# 기본값 / 키워드 설정 (모두 "문자열"이어야 함. Ellipsis(...) 쓰지 말 것!)
# 필요에 맞게 수정 가능
# ─────────────────────────────────────────────────────────────
DEFAULT_VALUES = {
    "POPULARITY_SCORE": 5.0
}

FLEXIBLE_FOODS: List[str] = [
    "비빔", "볶음", "덮밥", "샐러드", "쌈", "라면", "우동", "칼국수",
    "비건", "곱빼기", "주먹밥", "비지", "오믈렛", "비엔나", "소바"
]

RIGID_FOODS: List[str] = [
    "피자", "케이크", "도시락", "김밥", "햄버거", "만두", "호빵",
    "식빵", "타르트", "파이", "샌드위치"
]


# ─────────────────────────────────────────────────────────────
# AI 추천 시스템 본체
# ─────────────────────────────────────────────────────────────
class AIFoodRecommendationSystem:
    def __init__(self, menus_df: pd.DataFrame, restaurants_df: pd.DataFrame):
        # 원본 보존
        self.menus_df = menus_df.copy()
        self.restaurants_df = restaurants_df.copy()

        # 전처리
        self._preprocess_data()

        # ML 파이프라인 (카테고리별로 매번 fit하지만, 객체 재사용해도 무방)
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=30, ngram_range=(1, 2), analyzer="char"
        )
        self.scaler = StandardScaler()
        self.cluster_info: Dict[str, Dict[str, Any]] = {}

        # 카테고리별 클러스터 정보 구성
        self._build_ai_features()

    # ─────────────────────────────
    # 내부: 전처리
    # ─────────────────────────────
    def _preprocess_data(self) -> None:
        # 숫자 컬럼 안전 변환
        for col in ["price", "width", "length", "height", "popularity_score"]:
            if col in self.menus_df.columns:
                self.menus_df[col] = pd.to_numeric(self.menus_df[col], errors="coerce")

        # 결측치 기본값
        self.menus_df["popularity_score"] = self.menus_df["popularity_score"].fillna(
            DEFAULT_VALUES["POPULARITY_SCORE"]
        )

        # 부피
        if all(c in self.menus_df.columns for c in ["width", "length", "height"]):
            self.menus_df["volume"] = (
                self.menus_df["width"].fillna(0)
                * self.menus_df["length"].fillna(0)
                * self.menus_df["height"].fillna(0)
            )
        else:
            self.menus_df["volume"] = 0

        # 메뉴명 문자열화
        if "menu_name" in self.menus_df.columns:
            self.menus_df["menu_name"] = self.menus_df["menu_name"].astype(str)
        else:
            self.menus_df["menu_name"] = ""

        # 유연/고정 분류
        self.menus_df["food_flexibility"] = self.menus_df["menu_name"].apply(
            self._classify_food_flexibility
        )

    # ─────────────────────────────
    # 내부: 유연성 분류
    # ─────────────────────────────
    def _classify_food_flexibility(self, name: Any) -> str:
        if pd.isna(name):
            # 이름이 비어도 보수적으로 'flexible'
            return "flexible"

        s = str(name).lower()

        # 고정형 키워드가 먼저 매칭되면 'rigid'
        for kw in RIGID_FOODS:
            if isinstance(kw, str) and kw and kw.lower() in s:
                return "rigid"

        # 유연형 키워드 매칭
        for kw in FLEXIBLE_FOODS:
            if isinstance(kw, str) and kw and kw.lower() in s:
                return "flexible"

        # 기본은 flexible
        return "flexible"

    # ─────────────────────────────
    # 내부: 카테고리별 텍스트/수치 특징 → KMeans 클러스터
    # ─────────────────────────────
    def _build_ai_features(self) -> None:
        if "category" not in self.menus_df.columns:
            self.cluster_info = {}
            return

        categories = (
            self.menus_df["category"].dropna().astype(str).unique().tolist()
        )
        for category in categories:
            cat_df = self.menus_df[self.menus_df["category"].astype(str) == category].copy()

            # 표본이 1개면 클러스터 1개로 취급
            if len(cat_df) < 2:
                self.cluster_info[category] = {
                    "clusters": {
                        0: {
                            "indices": cat_df.index.tolist(),
                            "avg_popularity": float(cat_df["popularity_score"].mean())
                            if "popularity_score" in cat_df.columns
                            else DEFAULT_VALUES["POPULARITY_SCORE"],
                            "size": int(len(cat_df)),
                            "examples": cat_df["menu_name"].head(2).tolist(),
                        }
                    },
                    "n_clusters": 1,
                }
                continue

            # 텍스트 벡터화
            texts = cat_df["menu_name"].fillna("").astype(str)
            text_vecs = self.tfidf_vectorizer.fit_transform(texts).toarray()

            # 수치 특징
            numeric_cols = [c for c in ["price", "volume", "popularity_score"] if c in cat_df.columns]
            if numeric_cols:
                numeric_data = self.scaler.fit_transform(cat_df[numeric_cols].fillna(0))
                feats = np.hstack([text_vecs, numeric_data]) if text_vecs.size else numeric_data
            else:
                feats = text_vecs

            n = len(cat_df)
            k = min(max(2, n // 5), 4)  # 2~4 사이 적절히
            km = KMeans(n_clusters=k, random_state=42, n_init=10)
            labels = km.fit_predict(feats)

            clusters: Dict[int, Dict[str, Any]] = {}
            for cid in range(k):
                idxs = cat_df.index[labels == cid].tolist()
                d = cat_df.loc[idxs]
                clusters[cid] = {
                    "indices": idxs,
                    "avg_popularity": float(d["popularity_score"].mean())
                    if "popularity_score" in d.columns and len(d) > 0
                    else DEFAULT_VALUES["POPULARITY_SCORE"],
                    "size": int(len(d)),
                    "examples": d["menu_name"].head(2).tolist(),
                }

            self.cluster_info[category] = {"clusters": clusters, "n_clusters": k}

    # ─────────────────────────────
    # 내부: 컨테이너 적합도 (0~100)
    # ─────────────────────────────
    def calculate_container_utilization(
        self,
        uw: float, ul: float, uh: float,
        mw: float, ml: float, mh: float,
        _menu_name: Any,
        flex: str
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
            return float(min(100.0, (mv / uv) * 110.0))  # capped 100
        # rigid일 때 한 변이라도 큰 경우 0
        if mw > uw or ml > ul or mh > uh:
            return 0.0
        return float((mv / uv) * 100.0)

    # ─────────────────────────────
    # 내부: 클러스터 기반 점수 (0~100)
    # ─────────────────────────────
    def calculate_ai_cluster_score(self, idx: int, category: str) -> float:
        info = self.cluster_info.get(str(category))
        if not info:
            return 50.0
        clusters = info.get("clusters", {})
        for cid, data in clusters.items():
            if idx in data.get("indices", []):
                avg = float(data.get("avg_popularity", DEFAULT_VALUES["POPULARITY_SCORE"]))
                size = int(data.get("size", 1))
                bonus = min(20.0, size * 5.0)
                return float(min(100.0, avg * 8.0 + bonus))
        return 50.0

    # ─────────────────────────────
    # 공개: 카테고리 내 AI 추천
    # ─────────────────────────────
    def get_ai_recommendations(
        self,
        user_width: float,
        user_length: float,
        user_height: float,
        category: str,
        top_k: int = 10
    ) -> Dict[str, Any]:
        df = self.menus_df[self.menus_df["category"].astype(str) == str(category)]
        recs: List[Dict[str, Any]] = []

        for idx, m in df.iterrows():
            util = self.calculate_container_utilization(
                user_width, user_length, user_height,
                m.get("width", 0), m.get("length", 0), m.get("height", 0),
                m.get("menu_name", ""), m.get("food_flexibility", "flexible")
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

            rname = "정보없음"
            rid = m.get("restaurant_id", None)
            if rid is not None and "restaurant_id" in self.restaurants_df.columns:
                rest = self.restaurants_df[self.restaurants_df["restaurant_id"] == rid]
                if not rest.empty and "restaurant_name" in rest.columns:
                    rname = str(rest["restaurant_name"].iloc[0])

            price_val = m.get("price", 0)
            try:
                price_int = int(float(price_val))
            except Exception:
                price_int = 0

            recs.append({
                "menu_id": m.get("menu_id", idx),
                "menu_name": str(m.get("menu_name", "")),
                "restaurant_name": rname,
                "price": price_int,
                "container_utilization": round(util, 1),
                "final_ai_score": round(float(score), 1),
            })

        recs.sort(key=lambda x: x["final_ai_score"], reverse=True)
        return {"status": "success", "recommendations": recs[: max(1, int(top_k))]}


# ─────────────────────────────────────────────────────────────
# 전역 인스턴스 관리 (main.py에서 주입)
# ─────────────────────────────────────────────────────────────
_ai_system: Optional[AIFoodRecommendationSystem] = None

def init_ai_system(system: AIFoodRecommendationSystem) -> None:
    global _ai_system
    _ai_system = system

def _ensure_ai() -> AIFoodRecommendationSystem:
    if _ai_system is None:
        raise ValueError("AI 시스템이 초기화되지 않았습니다.")
    return _ai_system


# ─────────────────────────────────────────────────────────────
# 유틸/엔드포인트 헬퍼
# ─────────────────────────────────────────────────────────────
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
    rests = ai.restaurants_df

    # 카테고리 검증
    valid = set(menus.get("category", pd.Series(dtype=str)).dropna().astype(str).unique())
    for c in req.categories:
        if str(c) not in valid:
            raise KeyError(f"지원하지 않는 카테고리: {c}")

    # AI 모드
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
                food_id=str(i.get("menu_id")),
                food_name=str(i.get("menu_name", "")),
                restaurant_name=str(i.get("restaurant_name", "정보없음")),
                price=int(i.get("price", 0) or 0),
                distance=0.0,
                container_fit=round(float(i.get("container_utilization", 0.0)) / 100.0, 2),
                image_url="",
                description=f"AI 점수: {i.get('final_ai_score', 0)}",
            )
            for i in sel
        ]
        return RecommendResponse(page=1, limit=req.limit, total=len(recs), recommendations=recs)

    # 기본(볼륨 기반) 경로
    df = menus[menus["category"].astype(str).isin([str(c) for c in req.categories])].copy()
    cw, cl, ch = req.container.width, req.container.length, req.container.height

    # 적합도 계산
    df["container_fit"] = df.apply(
        lambda r: calc_fit(r.get("width", 0), r.get("length", 0), r.get("height", 0), cw, cl, ch),
        axis=1
    )

    # 레스토랑 조인
    merged = df.merge(rests, on="restaurant_id", how="left", suffixes=("", "_rest"))

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
        price_val = r.get("price", 0)
        try:
            price_int = int(float(price_val))
        except Exception:
            price_int = 0

        distance_val = r.get("distance", 0.0)
        try:
            distance_f = float(distance_val) if pd.notna(distance_val) else 0.0
        except Exception:
            distance_f = 0.0

        recs.append(
            Recommendation(
                food_id=str(r.get("menu_id", "")),
                food_name=str(r.get("menu_name", "")),
                restaurant_name=str(
                    r.get("restaurant_name", r.get("restaurant_name_rest", "정보없음"))
                ),
                price=price_int,
                distance=distance_f,
                container_fit=float(r.get("container_fit", 0.0) or 0.0),
                image_url=str(r.get("image_url") or ""),
                description=str(r.get("notes") or ""),
            )
        )

    return RecommendResponse(page=req.page, limit=req.limit, total=total, recommendations=recs)
