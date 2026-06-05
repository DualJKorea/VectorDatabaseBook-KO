"""
5장: PostgreSQL pgvector를 활용한 ArXiv 논문 검색 시스템 구축
=========================================================================
클래스/메서드 정의로 구성한 아키텍처 스캐폴드.

도서 주석: "이 장 전체는 클래스와 메서드 정의를 통해 아키텍처를 개괄합니다.
이를 구현한 코드는 머리말에서 언급한 함께 제공되는 GitHub 리포지토리에 있습니다."

`pass`로 표시된 메서드는 스텁(STUB)입니다. 직접 구현하거나
함께 제공되는 리포지토리에서 가져와 사용하십시오. SQL 스키마와 일부 핵심 메서드
(_upsert_paper, 싱글턴 EmbeddingGenerator)는 완전히 제공됩니다.

의존성: pip install arxiv PyMuPDF psycopg2-binary sentence-transformers
              requests numpy tqdm python-dotenv
"""

import time
import logging
import os
import hashlib
import re
import sys
from pathlib import Path
from typing import List, Dict, Optional, Generator, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum
from urllib.parse import quote

import arxiv
import requests
import numpy as np
import psycopg2
from psycopg2.extras import RealDictCursor, execute_batch
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)


# =============================================================================
# 3절 — 환경 및 구성
# =============================================================================

DB_CONFIG = {
    'host': os.getenv('DB_HOST', 'localhost'),
    'port': os.getenv('DB_PORT', '5432'),
    'database': os.getenv('DB_NAME', 'arxiv_papers'),
    'user': os.getenv('DB_USER', 'postgres'),
    'password': os.getenv('DB_PASSWORD', 'Welcome123'),
}


# =============================================================================
# 4절 — 데이터베이스 스키마(전체 SQL)
# =============================================================================

SCHEMA_SQL = """
-- 필요한 확장 기능 활성화
CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS pg_trgm;
CREATE EXTENSION IF NOT EXISTS btree_gin;

-- 기본 논문 테이블
CREATE TABLE IF NOT EXISTS papers (
    id SERIAL PRIMARY KEY,
    arxiv_id VARCHAR(50) UNIQUE NOT NULL,
    title TEXT NOT NULL,
    abstract TEXT,
    authors TEXT[],
    categories TEXT[],
    primary_category VARCHAR(50),
    published_date DATE,
    updated_date DATE,
    pdf_url TEXT,
    comment TEXT,
    journal_ref TEXT,
    doi VARCHAR(100),
    pdf_downloaded BOOLEAN DEFAULT FALSE,
    pdf_processed BOOLEAN DEFAULT FALSE,
    embedding_generated BOOLEAN DEFAULT FALSE,
    processing_error TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_papers_published_date ON papers (published_date DESC);
CREATE INDEX IF NOT EXISTS idx_papers_categories ON papers USING GIN (categories);
CREATE INDEX IF NOT EXISTS idx_papers_authors ON papers USING GIN (authors);

-- 임베딩을 포함하는 청크 테이블
CREATE TABLE IF NOT EXISTS paper_chunks (
    id SERIAL PRIMARY KEY,
    paper_id INTEGER REFERENCES papers(id) ON DELETE CASCADE,
    chunk_index INTEGER NOT NULL,
    chunk_text TEXT NOT NULL,
    chunk_tokens INTEGER,
    embedding vector(384),
    section_name VARCHAR(255),
    page_number INTEGER,
    char_start INTEGER,
    char_end INTEGER,
    has_math BOOLEAN DEFAULT FALSE,
    has_code BOOLEAN DEFAULT FALSE,
    has_references BOOLEAN DEFAULT FALSE,
    language VARCHAR(10) DEFAULT 'en',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE (paper_id, chunk_index)
);

CREATE INDEX IF NOT EXISTS idx_chunks_embedding ON paper_chunks
USING hnsw (embedding vector_cosine_ops) WITH (m = 16, ef_construction = 64);

CREATE INDEX IF NOT EXISTS idx_chunks_paper_id ON paper_chunks (paper_id);
CREATE INDEX IF NOT EXISTS idx_chunks_text_trgm ON paper_chunks USING GIN (chunk_text gin_trgm_ops);

-- 저자 테이블
CREATE TABLE IF NOT EXISTS authors (
    id SERIAL PRIMARY KEY,
    name TEXT NOT NULL,
    normalized_name TEXT,
    affiliation TEXT,
    orcid VARCHAR(50),
    email VARCHAR(255),
    UNIQUE (normalized_name)
);

CREATE INDEX IF NOT EXISTS idx_authors_name_trgm ON authors USING GIN (name gin_trgm_ops);

-- 논문-저자 연결 테이블
CREATE TABLE IF NOT EXISTS paper_authors (
    paper_id INTEGER REFERENCES papers(id) ON DELETE CASCADE,
    author_id INTEGER REFERENCES authors(id) ON DELETE CASCADE,
    author_position INTEGER,
    is_corresponding BOOLEAN DEFAULT FALSE,
    PRIMARY KEY (paper_id, author_id)
);

CREATE INDEX IF NOT EXISTS idx_paper_authors_author_id ON paper_authors (author_id);

-- 범주
CREATE TABLE IF NOT EXISTS categories (
    code VARCHAR(20) PRIMARY KEY,
    name TEXT NOT NULL,
    description TEXT,
    parent_category VARCHAR(20)
);

-- 검색 이력
CREATE TABLE IF NOT EXISTS search_history (
    id SERIAL PRIMARY KEY,
    query_text TEXT NOT NULL,
    query_embedding vector(384),
    result_count INTEGER,
    execution_time_ms INTEGER,
    filters JSONB,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_search_history_created_at ON search_history (created_at DESC);

-- 처리 큐
CREATE TABLE IF NOT EXISTS processing_queue (
    id SERIAL PRIMARY KEY,
    paper_id INTEGER REFERENCES papers(id) ON DELETE CASCADE,
    operation VARCHAR(50) NOT NULL,
    status VARCHAR(20) DEFAULT 'pending',
    priority INTEGER DEFAULT 0,
    retry_count INTEGER DEFAULT 0,
    error_message TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    started_at TIMESTAMP,
    completed_at TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_queue_status_priority ON processing_queue (status, priority DESC);

-- 자동 갱신 트리거
CREATE OR REPLACE FUNCTION update_updated_at()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

DROP TRIGGER IF EXISTS update_papers_updated_at ON papers;
CREATE TRIGGER update_papers_updated_at
    BEFORE UPDATE ON papers
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at();
"""


def setup_database():
    """데이터베이스 스키마 생성."""
    conn = psycopg2.connect(**DB_CONFIG)
    conn.autocommit = True
    cursor = conn.cursor()
    cursor.execute(SCHEMA_SQL)
    cursor.close()
    conn.close()
    print("Database schema created.")


# =============================================================================
# 5절 — ArXiv 클라이언트(스텁)
# =============================================================================

@dataclass
class ArxivPaper:
    """ArXiv 논문의 구조화된 표현."""
    arxiv_id: str
    title: str
    abstract: str
    authors: List[str]
    categories: List[str]
    primary_category: str
    published_date: datetime
    updated_date: datetime
    pdf_url: str
    comment: Optional[str] = None
    journal_ref: Optional[str] = None
    doi: Optional[str] = None


class ArxivClient:
    """속도 제한과 오류 처리를 갖춘 견고한 ArXiv API 클라이언트."""

    def __init__(self, rate_limit_seconds: float = 3.0,
                max_results_per_query: int = 100):
        self.rate_limit_seconds = rate_limit_seconds
        self.max_results_per_query = max_results_per_query
        self.last_request_time = 0.0

    def _rate_limit(self):
        """API 호출 사이의 속도 제한 적용."""
        elapsed = time.time() - self.last_request_time
        if elapsed < self.rate_limit_seconds:
            time.sleep(self.rate_limit_seconds - elapsed)
        self.last_request_time = time.time()

    def search_papers(self, query: str, max_results: int = 100,
                    sort_by=arxiv.SortCriterion.SubmittedDate,
                    sort_order=arxiv.SortOrder.Descending
                    ) -> Generator[ArxivPaper, None, None]:
        """ArXiv 검색. 스텁 — 직접 구현하거나 함께 제공되는 리포지토리에서 가져오기."""
        pass

    def fetch_by_ids(self, arxiv_ids: List[str]) -> List[ArxivPaper]:
        """ArXiv ID로 특정 논문 가져오기. 스텁."""
        pass

    def search_recent_papers(self, categories: List[str],
                            days_back: int = 7) -> Generator[ArxivPaper, None, None]:
        """범주별 최신 논문 가져오기. 스텁."""
        pass


# =============================================================================
# 5절 — PDF 다운로더(스텁)
# =============================================================================

class PDFDownloader:
    """재시도 로직과 정리 기능을 갖춘 PDF 다운로드 관리."""

    def __init__(self, storage_path: str = "./data/pdfs",
                max_retries: int = 3, timeout: int = 30):
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)
        self.max_retries = max_retries
        self.timeout = timeout

    def _get_pdf_path(self, arxiv_id: str, published_date: datetime) -> Path:
        """체계화된 저장 경로 생성. 스텁."""
        pass

    def download_pdf(self, pdf_url: str, arxiv_id: str,
                    published_date: datetime,
                    force: bool = False) -> Tuple[bool, Optional[Path], Optional[str]]:
        """재시도 로직을 사용한 PDF 다운로드. 스텁."""
        pass

    def _validate_pdf(self, pdf_path: Path) -> bool:
        """파일이 유효한 PDF인지 확인. 스텁."""
        pass


# =============================================================================
# 5절 — _upsert_paper를 포함한 논문 프로세서(부분 구현)
# =============================================================================

class PaperProcessor:
    """전체 논문 처리 파이프라인 조율."""

    def __init__(self, db_config: dict, arxiv_client: ArxivClient,
                pdf_downloader: PDFDownloader, max_workers: int = 4):
        self.db_config = db_config
        self.arxiv_client = arxiv_client
        self.pdf_downloader = pdf_downloader
        self.max_workers = max_workers

    def process_papers_batch(self, query: str, max_papers: int = 100,
                            skip_existing: bool = True) -> dict:
        """검색 쿼리에서 논문 배치 처리. 스텁."""
        pass

    def _upsert_paper(self, cursor, paper: ArxivPaper) -> int:
        """논문 메타데이터 삽입 또는 갱신 후 논문 ID 반환. 구현 완료."""
        cursor.execute("""
            INSERT INTO papers (arxiv_id, title, abstract, authors, categories,
                                primary_category, published_date, updated_date,
                                pdf_url, comment, journal_ref, doi)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (arxiv_id) DO UPDATE SET title = EXCLUDED.title
            RETURNING id
        """, (paper.arxiv_id, paper.title, paper.abstract,
            paper.authors, paper.categories, paper.primary_category,
            paper.published_date, paper.updated_date,
            paper.pdf_url, paper.comment, paper.journal_ref, paper.doi))

        paper_db_id = cursor.fetchone()['id']

        # 관계형 저자 테이블 동기화
        for author_name in paper.authors:
            cursor.execute("""
                INSERT INTO authors (name, normalized_name)
                VALUES (%s, %s)
                ON CONFLICT (normalized_name) DO NOTHING
                RETURNING id
            """, (author_name, author_name))

            res = cursor.fetchone()
            if res:
                author_id = res['id']
            else:
                cursor.execute("SELECT id FROM authors WHERE normalized_name = %s",
                            (author_name,))
                author_id = cursor.fetchone()['id']

            cursor.execute("""
                INSERT INTO paper_authors (paper_id, author_id)
                VALUES (%s, %s) ON CONFLICT DO NOTHING
            """, (paper_db_id, author_id))

        return paper_db_id

    def _process_queue(self, conn, stats: dict):
        """큐 항목 처리. 스텁."""
        pass


# =============================================================================
# 6절 — PDF 추출(스텁)
# =============================================================================

@dataclass
class ExtractedPage:
    page_num: int
    text: str
    blocks: List[Dict]
    has_columns: bool
    has_math: bool
    has_tables: bool
    confidence: float


class PDFExtractor:
    """학술 논문을 위한 고급 PDF 텍스트 추출."""

    def __init__(self):
        pass

    def extract_paper_text(self, pdf_path: str) -> Dict[str, any]:
        """PDF에서 전체 텍스트 추출. 스텁."""
        pass

    def _extract_page(self, page, page_num: int) -> ExtractedPage:
        pass

    def _detect_columns(self, blocks: Dict) -> bool:
        pass

    def _clean_extracted_text(self, text: str) -> str:
        pass


class TextChunker:
    """학술 논문을 위한 지능형 텍스트 청킹."""

    def __init__(self, target_chunk_size: int = 768, min_chunk_size: int = 256,
                max_chunk_size: int = 1024, overlap_size: int = 128):
        self.target_chunk_size = target_chunk_size
        self.min_chunk_size = min_chunk_size
        self.max_chunk_size = max_chunk_size
        self.overlap_size = overlap_size

    def chunk_paper(self, text: str, sections: List[Dict],
                    preserve_sections: bool = True) -> List[Dict]:
        """논문 텍스트의 지능형 청킹. 스텁."""
        pass

    def _chunk_text(self, text: str,
                    section_name: Optional[str] = None) -> List[Dict]:
        pass


# =============================================================================
# 7절 — 임베딩 생성기(구현 완료: 싱글턴)
# =============================================================================

class EmbeddingGenerator:
    """싱글턴 패턴을 활용한 효율적인 임베딩 생성."""

    _instance = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super(EmbeddingGenerator, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self, model_name: str = 'all-MiniLM-L6-v2'):
        if self._initialized:
            return
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)
        self.dimension = self.model.get_sentence_embedding_dimension()
        self._initialized = True
        print(f"Loaded {model_name} (dim={self.dimension})")

    def generate_embeddings(self, texts: List[str],
                            show_progress: bool = True) -> np.ndarray:
        """텍스트 목록에 대한 임베딩 생성. 스텁."""
        pass

    def _preprocess_text(self, text: str) -> str:
        """임베딩 전 텍스트 전처리. 스텁."""
        pass

    def generate_query_embedding(self, query: str) -> np.ndarray:
        """검색 쿼리에 대한 임베딩 생성. 스텁."""
        pass


class EmbeddingPipeline:
    """임베딩 생성 및 저장을 위한 전체 파이프라인."""

    def __init__(self, db_config: dict, embedding_generator: EmbeddingGenerator,
                batch_size: int = 100):
        self.db_config = db_config
        self.embedding_generator = embedding_generator
        self.batch_size = batch_size

    def process_paper(self, paper_id: int, chunks: List[Dict]) -> Dict[str, any]:
        """단일 논문의 모든 청크 처리. 스텁."""
        pass

    def _store_chunks_with_embeddings(self, paper_id: int, chunks: List[Dict],
                                    embeddings: np.ndarray):
        """청크와 임베딩을 데이터베이스에 저장. 스텁."""
        pass

    def process_pending_papers(self, limit: int = 10) -> Dict[str, any]:
        """임베딩 생성이 필요한 논문 처리. 스텁."""
        pass


# =============================================================================
# 8절 — 검색 엔진(스텁)
# =============================================================================

class SearchMode(Enum):
    VECTOR = "vector"
    HYBRID = "hybrid"
    KEYWORD = "keyword"


@dataclass
class SearchResult:
    paper_id: int
    arxiv_id: str
    title: str
    abstract: str
    authors: List[str]
    score: float
    matched_chunks: List[Dict]
    published_date: str
    categories: List[str]


class PaperSearchEngine:
    """학술 논문을 위한 고급 검색 엔진."""

    def __init__(self, db_config: dict, embedding_generator: EmbeddingGenerator):
        self.db_config = db_config
        self.embedding_generator = embedding_generator

    def search(self, query: str, mode: SearchMode = SearchMode.HYBRID,
            limit: int = 10, filters: Optional[Dict] = None) -> List[SearchResult]:
        """기본 검색 인터페이스. 스텁."""
        pass

    def _vector_search(self, query: str, limit: int,
                    filters: Optional[Dict]) -> List[SearchResult]:
        pass

    def _hybrid_search(self, query: str, limit: int,
                    filters: Optional[Dict]) -> List[SearchResult]:
        pass

    def _keyword_search(self, query: str, limit: int,
                    filters: Optional[Dict]) -> List[SearchResult]:
        pass

    def find_similar_papers(self, paper_id: int,
                        limit: int = 10) -> List[SearchResult]:
        pass


# =============================================================================
# 9절 — CLI(스텁)
# =============================================================================

def cli_main():
    """CLI 진입점. 스텁 — click 기반 CLI는 함께 제공되는 리포지토리 참조."""
    print("ArXiv Paper Search System")
    print("Run setup_database() to initialize schema.")
    print("See companion GitHub repo for full CLI implementation.")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "setup":
        setup_database()
    else:
        cli_main()
