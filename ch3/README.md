# 제3장 FAISS를 활용한 유사도 검색

Reddit personal knowledge base with SQLite-VSS vector search.

## Prerequisites

N/A`

## Setup

```bash
python -m venv ch3_env
source ch3_env/bin/activate
pip install -r requirements.txt
```

## Configuration

N/A

## Run

```bash
# 예제 코드 실행
python app-1.py
python app-2.py
python app-3.py
python app-4.py
python app-5.py
```

## What it does

1. Fetches posts from Reddit via PRAW
2. Cleans/preprocesses text (markdown, URLs, reddit artifacts)
3. Generates embeddings with all-MiniLM-L6-v2
4. Stores in SQLite with VSS vector index
5. Performs semantic search with metadata filtering
6. Supports cross-subreddit analysis and similar-post discovery
