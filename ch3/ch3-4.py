import numpy as np
import faiss
import time


def benchmark_indexes(dimension=128, num_vectors=100000, k=5):
    vectors = np.random.random((num_vectors, dimension)).astype("float32")
    query = np.random.random((1, dimension)).astype("float32")

    # 인덱스 생성
    flat_index = faiss.IndexFlatL2(dimension)
    ivf_index = faiss.IndexIVFFlat(
        faiss.IndexFlatL2(dimension), dimension, int(np.sqrt(num_vectors))
    )
    hnsw_index = faiss.IndexHNSWFlat(dimension, 16)

    # 벡터 학습 및 추가
    flat_index.add(vectors)
    ivf_index.train(vectors)
    ivf_index.add(vectors)
    hnsw_index.add(vectors)

    # 검색 벤치마크 수행
    results = {}

    for name, index in [("Flat", flat_index), ("IVF", ivf_index), ("HNSW", hnsw_index)]:
        start = time.time()

        for _ in range(100):
            index.search(query, k)

        results[name] = (time.time() - start) / 100

    return results


# if __name__ == "__main__":
#    results = benchmark_indexes()
#    print(results)
