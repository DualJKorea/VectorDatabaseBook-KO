import numpy as np
import faiss
from typing import List, Tuple


class SimilaritySearcher:
    def __init__(self, dimension: int, M: int = 16):
        """지정된 매개변수로 HNSW 인덱스 초기화."""
        self.dimension = dimension
        self.index = faiss.IndexHNSWFlat(dimension, M)
        # 구성 매개변수 설정
        self.index.hnsw.efConstruction = 40
        self.index.hnsw.efSearch = 16
        # 참조용 원본 항목 저장
        self.items = []

    def add_items(self, vectors: np.ndarray, items: List[str]):
        """벡터와 해당 항목을 인덱스에 추가."""
        assert vectors.shape[1] == self.dimension
        assert vectors.shape[0] == len(items)
        # FAISS 인덱스에 추가
        self.index.add(vectors.astype("float32"))
        # 원본 항목 저장
        self.items.extend(items)

    def search(self, query_vector: np.ndarray, k: int = 5) -> List[Tuple[str, float]]:
        """가장 유사한 k개 항목 검색."""
        # 쿼리 벡터의 올바른 형태 보장
        query_vector = query_vector.reshape(1, self.dimension)
        # 검색 수행
        distances, indices = self.index.search(query_vector.astype("float32"), k)
        # 항목과 거리 반환
        results = []
        for idx, dist in zip(indices[0], distances[0]):
            if idx != -1:  # 유효하지 않은 결과에 대해 FAISS가 -1 반환
                results.append((self.items[idx], dist))
        return results


# 사용 예
dimension = 128
searcher = SimilaritySearcher(dimension)

# 샘플 데이터 생성
num_items = 10000
vectors = np.random.random((num_items, dimension))
items = [f"item_{i}" for i in range(num_items)]

# 인덱스에 추가
searcher.add_items(vectors, items)

# 검색 수행
query = np.random.random(dimension)
results = searcher.search(query, k=5)

# 결과 출력
for item, distance in results:
    print(f"{item}: distance = {distance:.4f}")
