# 제7장 PostgreSQL과 pgvector로 과학 논문 RAG 시스템 구축하기

과학 논문 질의응답을 위한 RAG 시스템입니다.
청크로 분할한 논문에 대해 하이브리드 검색(의미 기반 검색 + 키워드 검색)을 수행하고, Ollama를 사용해 로컬 환경에서 답변을 생성합니다.

## 사전 준비 사항

1. **pgvector** 확장이 설치된 **PostgreSQL 15+** 이상
2. **Ollama** — https://ollama.ai
3. 5장에서 만든 데이터베이스 또는 새로 생성한 데이터베이스

## 설정

```bash
python -m venv ch7_env
source ch7_env/bin/activate
pip install -r requirements.txt

# 데이터베이스 스키마 초기화
python app.py setup

# 테스트용 샘플 논문 로드
python app.py load-samples

# Ollama 시작
ollama serve
ollama pull llama3.1:8b
```

## 구성

환경 변수를 설정하거나 app.py의 DB_CONFIG / OLLAMA_* 값을 수정합니다:
- `DB_HOST`, `DB_PORT`, `DB_NAME`, `DB_USER`, `DB_PASSWORD`
- `OLLAMA_URL` (기본값: http://localhost:11434)
- `OLLAMA_MODEL` (기본값: llama3.1:8b)

## 실행

```bash
python app.py
```

## 아키텍처

1. **수집**: 논문 텍스트 → 섹션을 고려한 청크 분할 → 임베딩 생성 → pgvector 저장
2. **검색**: HNSW 기반 코사인 유사도 70%와 ts_rank 기반 키워드 검색 30%를 결합한 하이브리드 검색
3. **생성**: 출처 인용을 포함한 근거 기반 프롬프트를 사용해 Ollama로 답변 생성
4. **분석**: 검색 품질 분석을 위해 질의 로그와 임베딩을 함께 기록
