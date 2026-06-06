# 제5장 PostgreSQL pgvector로 ArXiv 논문 검색 시스템 구축하기

**참고**: 이 장은 아키텍처 스캐폴드입니다. 대부분의 메서드는 스텁(pass)으로 남겨져 있습니다. 전체 구현은 책 머리말에서 언급한 동반 GitHub 저장소에서 확인할 수 있습니다.

SQL 스키마와 _upsert_paper 메서드는 완전히 구현되어 있습니다.

## 사전 준비 사항

1. **pgvector** 확장이 설치된 **PostgreSQL 15+** 이상 
2. 데이터베이스 생성: createdb arxiv_papers

## 설정

```bash
python -m venv ch5_env
source ch5_env/bin/activate
pip install -r requirements.txt

# 스키마 초기화
python app.py setup
# 또는
psql -d arxiv_papers -f schema.sql
```

## 구성

환경 변수를 설정하거나 app.py의 DB_CONFIG를 수정합니다.:
- `DB_HOST`, `DB_PORT`, `DB_NAME`, `DB_USER`, `DB_PASSWORD`

## 구현 완료 항목과 스텁 항목

**구현 완료**: SQL 스키마, _upsert_paper, EmbeddingGenerator 싱글턴, 모든 데이터클래스
**스텁**: ArxivClient 메서드, PDFDownloader, PDFExtractor, TextChunker, 검색 메서드, CLI

## Docker

이 장에는 pgvector/pgvector:pg15를 사용하는 Docker Compose 구성도 포함되어 있습니다. 자세한 내용은 책 본문을 참조하시기 바랍니다..
