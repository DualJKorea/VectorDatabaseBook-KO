# 제8장 완전한 대화 검색 및 RAG 시스템 구축

대화 기록, FastAPI 서버, HTMX 프런트엔드, 그리고 로컬 LLM 추론을 위한 Ollama를 결합한 프로덕션 지향 RAG 시스템입니다.

## 사전 준비 사항

1. **pgvector** 확장이 설치된 **PostgreSQL 15+** 이상
2. **Ollama** — https://ollama.ai
3. Create database: `createdb conversation_rag`

## 설정

```bash
python -m venv ch8_env
source ch8_env/bin/activate
pip install -r requirements.txt

# 스키마 초기화
python app.py setup

# 샘플 문서 로드
python app.py load-samples

# Ollama 시작
ollama serve
ollama pull llama3.1:8b
```

## 실행

### 웹 서버 (FastAPI + HTMX)
```bash
python app.py serve        # 기본 포트 8000
python app.py serve 9000   # 사용자 지정 포트
```

그런 다음 브라우저에서 다음 주소로 접속합니다.
```Plain text
http://localhost:8000
```

### CLI 모드
```bash
python app.py
```

## API 엔드포인트

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET    | /api/health | 상태 확인(Ollama 상태 포함) |
| POST   | /api/sessions | 대화 세션 생성 |
| GET    | /api/sessions | 세션 목록 조회 |
| GET    | /api/sessions/{id}/messages | 대화 기록 조회 |
| POST   | /api/ask?session_id=X&question=Y | 질문하기(RAG) |
| POST   | /api/documents | 문서 수집 |
| GET    | /api/search?q=X | 문서 청크 검색 |
| GET    | /api/search/conversations?q=X | 과거 대화 검색 |

## 아키텍처

1. **문서 처리** 문서 → 청크 분할 → 임베딩 생성 → pgvector 저장(HNSW 인덱스)
2. **검색**: 의미 기반 검색과 ts_rank 기반 키워드 검색을 결합한 하이브리드 검색
3. **대화**: 세션 기반으로 관리하며, 세션 간 검색을 위해 모든 메시지를 임베딩합니다.
4. **생성**: 대화 기록과 검색된 청크를 함께 사용해 Ollama로 답변을 생성합니다.
5. **프런트엔드**: HTMX를 활용해 서버에서 렌더링한 HTML 조각을 동적으로 갱신합니다.
6. **커넥션 풀**: 동시 요청 처리를 위해 ThreadedConnectionPool을 사용합니다.
