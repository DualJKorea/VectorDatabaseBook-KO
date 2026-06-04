import numpy as np
import faiss

# 샘플 데이터 생성
num_vectors = 10000
dimension = 128
vectors = np.random.random((num_vectors, dimension)).astype("float32")

# HNSW 인덱스 생성
M = 16  # 계층당 연결 수
ef_construction = 40  # 구축 중 고려할 후보 수
index = faiss.IndexHNSWFlat(dimension, M)
index.hnsw.efConstruction = ef_construction

# 인덱스에 벡터 추가
index.add(vectors)

# 검색 파라미터 설정
ef_search = 16  # 검색 중 고려할 후보 수
index.hnsw.efSearch = ef_search

# 검색 수행
k = 5  # 찾을 최근접 이웃 수
query = np.random.random((1, dimension)).astype("float32")
distances, indices = index.search(query, k)
