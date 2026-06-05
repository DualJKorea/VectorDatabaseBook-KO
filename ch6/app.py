"""
6장: SQLite VSS와 Ollama로 RAG 시스템 구축
============================================================
벡터 검색을 위한 SQLite-VSS와 로컬 LLM 추론을 위한 Ollama를
결합한 로컬 비공개 RAG 시스템.

의존성: pip install sentence-transformers requests
추가 요구 사항: sqlite-vss 바이너리(vector0, vss0), 로컬에서 실행 중인 Ollama
"""

import sqlite3
import ollama
import requests
import json
import time
import hashlib
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Optional
import numpy as np


# =============================================================================
# 6.1 - 데이터베이스 기반
# =============================================================================

def setup_database(db_path='reddit_rag.db'):
    """VSS 확장을 사용하도록 SQLite를 설정하고 테이블 생성"""
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row # 행을 딕셔너리로 반환

    conn.enable_load_extension(True)
    try:
        conn.load_extension("./vector0")
        conn.load_extension("./vss0")
        print("VSS extension loaded successfully")
    except Exception as e:
        print(f"Error loading VSS: {e}")
        print("Download from: https://github.com/asg017/sqlite-vss/releases")
        raise

    cursor = conn.cursor()

    # posts 테이블 생성
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS posts (
            post_id TEXT PRIMARY KEY,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            title TEXT,
            content TEXT,
            subreddit TEXT,
            author TEXT,
            score INTEGER
        )
    """)

    # 벡터 지원을 포함한 청크 테이블 생성
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS content_chunks (
            chunk_id INTEGER PRIMARY KEY,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            post_id TEXT,
            chunk_index INTEGER,
            content TEXT,
            chunk_vector BLOB,  -- VSS용 BLOB으로 저장
            FOREIGN KEY (post_id) REFERENCES posts(post_id)
        )
    """)

    # 벡터 검색을 위한 VSS 가상 테이블 생성
    cursor.execute("""
        CREATE VIRTUAL TABLE IF NOT EXISTS chunk_vss USING vss0(
            chunk_vector(384)
        )
    """)

    # 키워드 검색을 위한 FTS5 생성
    cursor.execute("""
        CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
            content,
            content='content_chunks',
            content_rowid='chunk_id'
        )
    """)

    conn.commit()
    return conn


# =============================================================================
# 6.2 - 텍스트 처리 및 임베딩 생성
# =============================================================================

# 임베딩 모델의 전역 초기화
embedding_model = None

def get_embedding_model():
    """임베딩 모델을 가져오거나 초기화합니다.
    참고: 첫 호출 시 디스크에서 로드하므로 시간이 더 오래 걸립니다."""
    global embedding_model
    if embedding_model is None:
        embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        print(f"Loaded embedding model (dimension: "
            f"{embedding_model.get_sentence_embedding_dimension()})")
    return embedding_model


def chunk_text(text, chunk_size=200, overlap=50):
    """단어 수 기준의 단순 텍스트 청킹"""
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size - overlap):
        chunk = ' '.join(words[i:i + chunk_size])
        if chunk:
            chunks.append(chunk)
    return chunks


def store_post_with_chunks(conn, post_data):
    """임베딩을 포함하여 게시물과 해당 청크를 저장"""
    cursor = conn.cursor()

    # 게시물 저장
    cursor.execute("""
        INSERT OR REPLACE INTO posts (post_id, title, content, subreddit, author, score)
        VALUES (?, ?, ?, ?, ?, ?)
    """, (
        post_data['post_id'], post_data['title'], post_data['content'],
        post_data['subreddit'], post_data['author'], post_data.get('score', 0)
    ))

    # 제목과 본문을 결합하여 청크 생성
    full_text = f"{post_data['title']}\n\n{post_data['content']}"
    chunks = chunk_text(full_text)

    # 임베딩 모델 가져오기
    model = get_embedding_model()
    
    # 각 청크를 임베딩과 함께 저장
    for idx, chunk_content in enumerate(chunks):
        embedding = model.encode(chunk_content)
        
        # 청크 저장
        cursor.execute("""
            INSERT INTO content_chunks (post_id, chunk_index, content, chunk_vector)
            VALUES (?, ?, ?, ?)
        """, (post_data['post_id'], idx, chunk_content, embedding.tobytes()))

        chunk_id = cursor.lastrowid

        # VSS 인덱스에 추가
        vector_json = json.dumps(embedding.astype(float).tolist())
        cursor.execute("""
            INSERT INTO chunk_vss (rowid, chunk_vector)
            VALUES (?, ?)
        """, (chunk_id, vector_json))

        # FTS 인덱스에 추가
        cursor.execute("""
            INSERT INTO chunks_fts (rowid, content) VALUES (?, ?)
        """, (chunk_id, chunk_content))

    conn.commit()
    print(f"Stored post {post_data['post_id']} with {len(chunks)} chunks")


# =============================================================================
# 6.3 - 하이브리드 검색
# =============================================================================

def hybrid_search(conn, query_text, limit=5, semantic_weight=0.7):
    """벡터 검색과 키워드 검색을 결합한 하이브리드 검색 수행"""
    cursor = conn.cursor()
    model = get_embedding_model()

    # 질의 임베딩 생성
    query_embedding = model.encode(query_text)
    query_vector_json = json.dumps(query_embedding.astype(float).tolist())

    # --- 의미 기반 검색 ---
    try:
        cursor.execute("""
            SELECT rowid, distance
            FROM chunk_vss
            WHERE vss_search(chunk_vector, ?)
            LIMIT ?
        """, (query_vector_json, limit * 2))
        semantic_rows = cursor.fetchall()
    except Exception as e:
        print(f"Semantic search error: {e}")
        semantic_rows = []

    semantic_results = {}
    for row in semantic_rows:
        chunk_id = row[0]
        similarity = 1 - row[1]  # 거리를 유사도로 변환
        semantic_results[chunk_id] = similarity

    # --- 키워드 검색 ---
    keyword_results = {}
    try:
        safe_query = ''.join(
            ch if ch.isalnum() or ch.isspace() else ' '
            for ch in query_text
        ).strip()

        cursor.execute("""
            SELECT rowid, bm25(chunks_fts) as score
            FROM chunks_fts
            WHERE chunks_fts MATCH ?
            LIMIT ?
        """, (safe_query, limit * 2))

        keyword_rows = cursor.fetchall()
        # SQLite에서 BM25 점수는 음수이므로 반전
        keyword_results = {row[0]: -row[1] for row in keyword_rows}
    except sqlite3.OperationalError as e:
        print(f"Warning: Keyword search failed: {e}")

    # --- 키워드 점수 정규화 ---
    max_key = max(keyword_results.values()) if keyword_results else 1.0
    if max_key > 0:
        keyword_results = {k: v / max_key for k, v in keyword_results.items()}

    # --- 병합: semantic * 0.7 + keyword * 0.3 ---
    merged = {}
    keyword_weight = 1.0 - semantic_weight
    for cid, score in semantic_results.items():
        merged[cid] = score * semantic_weight
    for cid, score in keyword_results.items():
        merged[cid] = merged.get(cid, 0) + (score * keyword_weight)

    sorted_ids = sorted(merged.items(), key=lambda x: x[1], reverse=True)[:limit]

    # --- 전체 청크 데이터 조회 ---
    results = []
    for chunk_id, score in sorted_ids:
        cursor.execute("""
            SELECT chunk_id, post_id, content FROM content_chunks WHERE chunk_id = ?
        """, (chunk_id,))
        row = cursor.fetchone()
        if row:
            results.append({
                'chunk_id': row['chunk_id'],
                'post_id': row['post_id'],
                'content': row['content'],
                'score': score
            })

    return results


# =============================================================================
# 6.4 - Ollama를 사용한 LLM 통합
# =============================================================================

def call_ollama(prompt, model="llama3.1:8b", temperature=0.1):
    """간단한 Ollama API 호출"""
    url = "http://localhost:11434/api/generate"
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": temperature,
            "top_p": 0.9,
            "num_ctx": 4096
        }
    }
    try:
        response = requests.post(url, json=payload)
        response.raise_for_status()
        return response.json()['response']
    except Exception as e:
        print(f"Ollama error: {e}")
        return f"Error calling Ollama: {str(e)}"


def test_ollama():
    """Ollama가 실행 중인지 테스트"""
    try:
        response = requests.get("http://localhost:11434/api/tags")
        models = response.json()
        print("Available Ollama models:",
            [m['name'] for m in models.get('models', [])])
        return True
    except Exception:
        print("Ollama not running. Start with: ollama serve")
        return False


# =============================================================================
# 6.5 - RAG 파이프라인
# =============================================================================

def format_context(chunks, conn):
    """LLM을 위한 검색된 청크 포매팅"""
    cursor = conn.cursor()
    formatted_chunks = []

    for i, chunk in enumerate(chunks, 1):
        cursor.execute("""
            SELECT title, subreddit, author, score
            FROM posts WHERE post_id = ?
        """, (chunk['post_id'],))

        post = cursor.fetchone()
        if post:
            formatted_chunks.append(f"""
SOURCE {i}:
From r/{post['subreddit']} by u/{post['author']} (Score: {post['score']})
Title: {post['title']}

Content:
{chunk['content']}
""")

    return "\n---\n".join(formatted_chunks)


def answer_question(conn, question, num_chunks=5):
    """질문에 답변하기 위한 완전한 RAG 파이프라인"""
    print(f"\nQuestion: {question}")

    # 1단계: 관련 청크 검색
    start_time = time.time()
    chunks = hybrid_search(conn, question, limit=num_chunks)
    retrieval_time = (time.time() - start_time) * 1000

    if not chunks:
        return "No relevant information found in the database."

    print(f"Retrieved {len(chunks)} chunks in {retrieval_time:.1f}ms")

    # 2단계: 문맥 형식화
    context = format_context(chunks, conn)

    # 3단계: 프롬프트 생성
    prompt = f"""You are a helpful assistant. Answer the user's question using \
ONLY the retrieved information provided below.

Instructions:
1. If the information is not in the context, state that you do not know.
2. Do not use outside knowledge.
3. For every claim you make, cite the source number (e.g., [Source 1]).

RETRIEVED INFORMATION:
{context}

USER QUESTION:
{question}

ANSWER:
"""

    # 4단계: 답변 생성
    start_time = time.time()
    answer = call_ollama(prompt)
    generation_time = (time.time() - start_time) * 1000

    print(f"Generated answer in {generation_time:.1f}ms")
    print(f"Total time: {retrieval_time + generation_time:.1f}ms")

    return answer


# =============================================================================
# 6.6 - 샘플 데이터 및 데모
# =============================================================================

def load_sample_data(conn):
    """테스트용 Reddit 샘플 데이터 로드."""
    sample_posts = [
        {
            'post_id': 'post001',
            'title': 'ELI5: What is machine learning?',
            'content': """Machine learning is like teaching a computer to recognize \
patterns by showing it many examples. Instead of programming exact rules, you feed \
it data and it learns the patterns itself. For example, to teach it to recognize \
cats, you show it thousands of cat pictures until it learns what makes a cat a cat. \
It's used in spam filters, recommendation systems, and voice assistants.""",
            'subreddit': 'explainlikeimfive',
            'author': 'curious_user',
            'score': 245
        },
        {
            'post_id': 'post002',
            'title': 'Best Python libraries for beginners?',
            'content': """I recommend starting with: NumPy for numerical computing, \
Pandas for data manipulation, Matplotlib for basic plotting, Requests for HTTP \
requests, and Flask for simple web apps. These libraries cover most beginner needs \
and have excellent documentation. Don't try to learn them all at once - start with \
one based on your project needs.""",
            'subreddit': 'learnpython',
            'author': 'python_mentor',
            'score': 189
        },
        {
            'post_id': 'post003',
            'title': 'How do vector databases work?',
            'content': """Vector databases store data as high-dimensional vectors \
(lists of numbers) that represent the semantic meaning of the data. When you search, \
your query is converted to a vector and the database finds the most similar vectors \
using distance metrics like cosine similarity. This enables semantic search where you \
find content by meaning rather than exact keyword matches. Popular vector databases \
include Pinecone, Weaviate, and pgvector for PostgreSQL.""",
            'subreddit': 'programming',
            'author': 'db_expert',
            'score': 156
        }
    ]

    print("Loading sample data...")
    for post in sample_posts:
        store_post_with_chunks(conn, post)
    print(f"Loaded {len(sample_posts)} sample posts")


def main():
    """ RAG 시스템의 주요 시연"""
    print("=== Minimal RAG System with SQLite VSS and Ollama ===\n")

    # 1단계: 데이터베이스 설정
    print("1. Setting up database...")
    conn = setup_database()

    # 2단계: Ollama 확인
    print("\n2. Checking Ollama...")
    if not test_ollama():
        print("Please start Ollama first: ollama serve")
        print("Then pull a model: ollama pull llama3.1:8b")
        return

    # 3단계: 샘플 데이터 로드
    print("\n3. Loading sample data...")
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*) FROM posts")
    if cursor.fetchone()[0] == 0:
        load_sample_data(conn)
    else:
        print("Data already loaded")

    # 4단계: 데모 질문
    print("\n4. Demonstrating RAG system...")
    demo_questions = [
        "What is machine learning and how does it work?",
        "What Python libraries should a beginner learn?",
        "How do vector databases enable semantic search?"
    ]

    for question in demo_questions[:1]:
        answer = answer_question(conn, question)
        print(f"\nAnswer: {answer}\n")
        print("=" * 50)

    # 5단계: 대화형 모드
    print("\n5. Interactive Q&A (type 'quit' to exit)")
    print("-" * 50)

    while True:
        question = input("\nYour question: ").strip()
        if question.lower() in ['quit', 'exit', 'q']:
            break
        if not question:
            continue
        answer = answer_question(conn, question)
        print(f"\nAnswer: {answer}")

    conn.close()
    print("\nGoodbye!")


def quick_start():
    """개별 구성 요소 테스트를 위한 빠른 시작"""
    conn = setup_database()

    # 검색에 대한 빠른 테스트
    print("Testing hybrid search for 'machine learning'...")
    results = hybrid_search(conn, "machine learning", limit=3)

    for i, result in enumerate(results, 1):
        print(f"\n{i}. Score: {result['score']:.3f}")
        print(f"   Content: {result['content'][:150]}...")

    conn.close()


if __name__ == "__main__":
    # 메인 데모 실행
    main()

    # 또는 빠른 테스트 실행
    # quick_start()
