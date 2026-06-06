# 제4장 SQLite3를 이용한 의미 기반 검색

SQLite-VSS 벡터 검색을 활용해 Reddit 개인 지식 베이스를 구축합니다.

## 사전 준비 사항

1. **sqlite-vss 바이너리** — 다음 주소에서 다운로드합니다.
   - https://github.com/asg017/sqlite-vss/releases
   - vector0.so와 vss0.so 파일을 추출합니다.
     - Extract `vector0.so` and `vss0.so` (Linux), `.dylib` (macOS), or `.dll` (Windows)
   - Place them in the same directory as `app.py` (or set `EXTENSION_PATH`)
   - **Note**: sqlite-vss is not officially supported on Windows; use WSL2.

3. **Reddit API credentials** — https://www.reddit.com/prefs/apps
   - Create a "script" app
   - Note `client_id` and `client_secret`

## Setup

```bash
python -m venv ch4_env
source ch4_env/bin/activate
pip install -r requirements.txt
```

## Configuration

Edit `app.py` and set these near the bottom in `main()`:
- `REDDIT_CLIENT_ID`
- `REDDIT_CLIENT_SECRET`
- `REDDIT_USER_AGENT`
- `EXTENSION_PATH` (if binaries aren't in `.`)

## Run

```bash
# Verify sqlite-vss installation first
python -c "from app import verify_vss_installation; print(verify_vss_installation())"

# Run full pipeline
python app.py
```

## What it does

1. Fetches posts from Reddit via PRAW
2. Cleans/preprocesses text (markdown, URLs, reddit artifacts)
3. Generates embeddings with all-MiniLM-L6-v2
4. Stores in SQLite with VSS vector index
5. Performs semantic search with metadata filtering
6. Supports cross-subreddit analysis and similar-post discovery
