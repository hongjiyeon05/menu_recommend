import os
import pandas as pd
import numpy as np
import warnings
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer

from schemas import RecommendRequest, RecommendResponse, Recommendation

warnings.filterwarnings("ignore")

# =============================================================================
# 상수 정의
# =============================================================================
DEFAULT_VALUES = {
    'POPULARITY_SCORE': 5.0,
}

FLEXIBLE_FOODS = [
    '비빔밥', '볶음밥', '알밥', '잡채밥', '제육볶음', '매콤낙지비빔밥',
    '샐러드', '파스타샐러드', '리코타치즈샐러드', '스테이크샐러드', '타코샐러드',
    '죽', '곱도리탕', '닭곱새', '우도리탕', '닭도리탕',
    '떡볶이', '로제떡볶이', '마라떡볶이', '치즈떡볶이', '국물떡볶이', '짜장떡볶이',
    '라볶이', '고구마떡볶이', '차돌박이떡볶이',
    '마라탕', '마라샹궈', '꿔바로우',
    '스파게티', '파스타', '분짜', '쌀국수', '팟타이', '포', '마제소바',
    '짬뽕', '짜장면', '해물짬뽕',
    '카레',
    '오뎅', '탕수육', '닭발', '소테'
]

RIGID_FOODS = [
    '에그마요', '치아바타', '햄치즈에그', '잠봉뵈르', '바게트', '반미',
    '함박스테끼', '돈테끼', '치킨난반', '차슈동', '조밥', '초밥', '김밥',
    '피자', '라자냐', '냉면', '물냉면', '비빔냉면', '불냉면',
    '돈까스', '카츠', '가츠동', '특규동', '오야꼬동', '덮밥', '오므라이스',
    '밥버거', '탄두리', '치킨티카', '빈달루', '코르마', '마크니', '거라이',
    '팬치즈'
]


# ─────────────── AI 시스템 클래스 ───────────────
class AIFoodRecommendationSystem:
    def __init__(self, menus_df: pd.DataFrame, restaurants_df: pd.DataFrame):
        self.menus_df = menus_df.copy()
        self.restaurants_df = restaurants_df.copy()
        self._preprocess_data()

        # AI feature tools
        self.tfidf_vectorizer = TfidfVectorizer(max_features=30, ngram_range=(1, 2), analyzer='char')
        self.scaler = StandardScaler()
        self.cluster_info = {}
        self._build_ai_features()

    def _preprocess_data(self):
        # numeric conversion & fill
        for col in ['price', 'width', 'length', 'height', 'popularity_score']:
            if col in self.menus_df.columns:
                self.menus_df[col] = pd.to_numeric(self.menus_df[col], errors='coerce')
        self.menus_df['popularity_score'] = self.menus_df['popularity_score'].fillna(DEFAULT_VALUES['POPULARITY_SCORE'])
        self.menus_df['volume'] = (
            self.menus_df['width'] *
            self.menus_df['length'] *
            self.menus_df['height']
        )
        # 음식 유연성 분류
        self.menus_df['food_flexibility'] = self.menus_df['menu_name'].apply(self._classify_food_flexibility)

    def _classify_food_flexibility(self, name):
        if pd.isna(name):
            return 'flexible'
        s = name.lower()
        for kw in RIGID_FOODS:
            if kw in s:
                return 'rigid'
        for kw in FLEXIBLE_FOODS:
            if kw in s:
                return 'flexible'
        return 'flexible'

    def _build_ai_features(self):
        for category in self.menus_df['category'].unique():
            cat_df = self.menus_df[self.menus_df['category'] == category].copy()
            if len(cat_df) < 2:
                self.cluster_info[category] = {
                    'clusters': {
                        0: {
                            'indices': cat_df.index.tolist(),
                            'avg_popularity': cat_df['popularity_score'].mean(),
                            'size': len(cat_df),
                            'examples': cat_df['menu_name'].tolist()
                        }
                    },
                    'n_clusters': 1
                }
                continue

            # 텍스트 벡터화
            texts = cat_df['menu_name'].fillna('').astype(str)
            text_vecs = self.tfidf_vectorizer.fit_transform(texts).toarray()

            # numeric features
            feats = []
            num_cols = [c for c in ['price', 'volume', 'popularity_score'] if c in cat_df.columns]
            if num_cols:
                num_data = self.scaler.fit_transform(cat_df[num_cols].fillna(0))
                feats = np.hstack([text_vecs, num_data]) if text_vecs.size else num_data
            else:
                feats = text_vecs

            n = len(cat_df)
            k = min(max(2, n // 5), 4)
            km = KMeans(n_clusters=k, random_state=42, n_init=10)
            labels = km.fit_predict(feats)

            clusters = {}
            for cid in range(k):
                idxs = cat_df.index[labels == cid].tolist()
                d = cat_df.loc[idxs]
                clusters[cid] = {
                    'indices': idxs,
                    'avg_popularity': d['popularity_score'].mean(),
                    'size': len(d),
                    'examples': d['menu_name'].head(2).tolist()
                }

            self.cluster_info[category] = {
                'clusters': clusters,
                'n_clusters': k
            }

    def calculate_container_utilization(self, uw, ul, uh, mw, ml, mh, _, flex):
        uv = uw * ul * uh
        mv = mw * ml * mh
        if mv > uv:
            return 0
        if flex == 'flexible':
            return min(100, (mv / uv) * 110)
        else:
            if mw > uw or ml > ul or mh > uh:
                return 0
            return (mv / uv) * 100

    def calculate_ai_cluster_score(self, idx, cat):
        info = self.cluster_info.get(cat)
        if not info:
            return 50
        for cid, data in info['clusters'].items():
            if idx in data['indices']:
                avg = data['avg_popularity']
                bonus = min(20, data['size'] * 5)
                return min(100, avg * 8 + bonus)
        return 50

    def get_ai_recommendations(self, user_width, user_length, user_height, category, top_k=10):
        df = self.menus_df[self.menus_df['category'] == category]
        recs = []
        for idx, m in df.iterrows():
            util = self.calculate_container_utilization(
                user_width, user_length, user_height,
                m['width'], m['length'], m['height'],
                m['menu_name'], m['food_flexibility']
            )
            if util <= 0:
                continue
            cs = self.calculate_ai_cluster_score(idx, category)
            ps = min(100, m['popularity_score'] * 10)
            score = util * 0.5 + cs * 0.3 + ps * 0.2
            rest = self.restaurants_df[self.restaurants_df['restaurant_id'] == m['restaurant_id']]
            rname = rest['restaurant_name'].iloc[0] if not rest.empty else "정보없음"
            recs.append({
                'menu_id': m['menu_id'],
                'menu_name': m['menu_name'],
                'restaurant_name': rname,
                'price': int(m['price']),
                'container_utilization': round(util, 1),
                'final_ai_score': round(score, 1)
            })
        recs.sort(key=lambda x: x['final_ai_score'], reverse=True)
        return {'status':'success', 'recommendations': recs[:top_k]}


# ────────────────────────────────────────────────────────────
# CSV 로드 및 기본 데이터 준비
# ────────────────────────────────────────────────────────────
menus_df = pd.read_csv(os.path.join("data", "final_menus_data.csv"))
restaurants_df = pd.read_csv(os.path.join("data", "restaurants.csv"))
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
    return round(min((w*l*h)/(cw*cl*ch), 1.0), 2)

def get_all_categories():
    return menus_df["category"].dropna().unique().tolist()

# ────────────────────────────────────────────────────────────
# 메인 추천 함수
# ────────────────────────────────────────────────────────────
def get_recommendations(req: RecommendRequest) -> RecommendResponse:
    valid = set(get_all_categories())
    for c in req.categories:
        if c not in valid:
            raise KeyError(f"지원하지 않는 카테고리: {c}")

    # AI 모드
    if req.use_ai:
        if _ai_system is None:
            raise ValueError("AI 시스템이 초기화되지 않았습니다.")
        cand = []
        for c in req.categories:
            res = _ai_system.get_ai_recommendations(
                user_width=req.container.width,
                user_length=req.container.length,
                user_height=req.container.height,
                category=c,
                top_k=req.limit
            )
            if res.get("status") == "success":
                cand.extend(res["recommendations"])
        cand.sort(key=lambda x: x["final_ai_score"], reverse=True)
        sel = cand[:req.limit]
        recs = [
            Recommendation(
                food_id=str(i["menu_id"]),
                food_name=i["menu_name"],
                restaurant_name=i["restaurant_name"],
                price=i["price"],
                distance=0.0,
                container_fit=round(i["container_utilization"]/100, 2),
                image_url="",
                description=f"AI 점수: {i['final_ai_score']}"
            ) for i in sel
        ]
        return RecommendResponse(page=1, limit=req.limit, total=len(recs), recommendations=recs)

    # 볼륨 기반 추천
    df = menus_df[menus_df["category"].isin(req.categories)].copy()
    cw, cl, ch = req.container.width, req.container.length, req.container.height
    df["container_fit"] = df.apply(
        lambda r: calc_fit(r["width"], r["length"], r["height"], cw, cl, ch), axis=1
    )
    merged = df.merge(restaurants_df, on="restaurant_id", how="left")

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
    page_df = merged.iloc[start:start + req.limit]

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
        ) for _, r in page_df.iterrows()
    ]

    return RecommendResponse(page=req.page, limit=req.limit, total=total, recommendations=recs)
