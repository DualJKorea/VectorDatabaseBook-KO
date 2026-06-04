import faiss
import numpy as np

# 샘플 데이터
d = 64  # 벡터의 차원
nb = 100000  # 데이터셋에 포함된 벡터 수
nq = 1000  # 쿼리 벡터 수

np.random.seed(1234)

# 임의의 32비트 부동소수점 합성 데이터 사용
xb = np.random.random((nb, d)).astype("float32")
xq = np.random.random((nq, d)).astype("float32")

# 인덱스 매개변수
nlist = 100  # 클러스터 수
k = 4  # 검색할 최근접 이웃 수

quantizer = faiss.IndexFlatL2(d)  # 클러스터링에 사용되는 인덱스
# ANN 인덱스도 사용 가능
index = faiss.IndexIVFFlat(quantizer, d, nlist)

# 인덱스 학습
index.train(xb)

# 인덱스에 벡터 추가
index.add(xb)

# 검색 매개변수
index.nprobe = 10  # 쿼리 시 검색할 클러스터 수
# 절충 관계: 값이 높을수록 더 정확하지만 속도는 느려짐

# 검색 수행
D, I = index.search(xq, k)  # D: 거리, I: 인덱스

# 결과 출력
print(I[:5])  # 처음 5개 쿼리에 대한 4개 최근접 이웃의 인덱스
print(D[:5])  # 처음 5개 쿼리에 대한 4개 최근접 이웃의 거리
