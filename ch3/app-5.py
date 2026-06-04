import faiss
import numpy as np

# 더미 데이터 생성
d = 128  # 차원 수
nb = 10000  # 벡터 수
xq = np.random.rand(10, d).astype("float32")  # 질의 벡터 10개
xb = np.random.rand(nb, d).astype("float32")  # 데이터베이스

# 1. 인덱스 선택(예: IVFPQ)
nlist = 100  # 데이터셋을 분할하는 데 사용되는 보로노이 셀(클러스터) 수
m = 8  # PQ를 위한 하위 벡터 수
nbits = 8  # 각 하위 벡터의 코드 인덱스를 표현하는 비트 수

quantizer = faiss.IndexFlatL2(d)  # IVF의 기본 인덱스
index = faiss.IndexIVFPQ(quantizer, d, nlist, m, nbits)

# 2. 데이터 샘플을 사용한 인덱스 학습
index.train(xb)

# 3. 인덱스에 벡터 추가
index.add(xb)

# 4. 최근접 이웃 검색
D, I = index.search(xq, 5)  # 최근접 이웃 5개 검색

# 결과 출력
print("Indices of the nearest neighbors:")
print(I)
print("Distances of the nearest neighbors")
print(D)
