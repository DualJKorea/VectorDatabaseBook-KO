"""
4장: SQLite3를 활용한 의미 기반 검색
========================================
SQLite-VSS 벡터 검색을 사용하는 로컬 Reddit CSV 지식 베이스.

이 버전은 PRAW로 Reddit에 연결하는 대신, 로컬 Kaggle Reddit 댓글 CSV를
수집 대상으로 사용함.

의존성:
    pip install sentence-transformers numpy

sqlite-vss 바이너리(vector0, vss0)도 필요함 - README.md 참고.
"""

from __future__ import annotations

import csv
import hashlib
import json
import re
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Generator, Iterable, List, Optional, Sequence, Union

import numpy as np

# =============================================================================
# 4.2.2 - VSS 설치 확인
# =============================================================================

def verify_vss_installation(extension_path: str = ".") -> bool:
    """
    sqlite-vss 확장이 정상적으로 로드되고 작동하는지 확인.

    Args:
        extension_path: vss0 및 vector0 확장이 들어 있는 디렉터리.

    Returns:
        검증에 성공하면 True, 그렇지 않으면 False.
    """
    conn = sqlite3.connect(":memory:")
    conn.enable_load_extension(True)

    try:
        # 필요한 두 확장 기능을 순서대로 로드
        conn.load_extension(f"{extension_path}/vector0")
        conn.load_extension(f"{extension_path}/vss0")

        # 보안을 위해 확장 기능 로딩 비활성화
        conn.enable_load_extension(False)

        # 벡터 생성과 L2 거리 계산 테스트
        # 직교 단위 벡터의 경우 L2 거리 = sqrt(2) ≈ 1.414
        result = conn.execute("""
            SELECT vss_version(),
                    vss_distance_l2(
                        vector_from_json('[1.0, 0.0, 0.0]'),
                        vector_from_json('[0.0, 1.0, 0.0]')
                    )
        """).fetchone()

        version, distance = result
        expected_distance = 1.414  # 직교 단위 벡터에 대한 sqrt(2) 값.

        print(f"sqlite-vss version: {version}")
        print(f"L2 distance test: {distance:.3f} (expected ~{expected_distance:.3f})")
        return abs(distance - expected_distance) < 0.01

    except Exception as exc:
        print(f"Verification failed: {exc}")
        return False

    finally:
        conn.close()

# =============================================================================
# 4.2.3 - 운영용 PRAGMA 설정
# =============================================================================

def configure_connection(conn: sqlite3.Connection) -> None:
    """성능과 정확성을 위해 권장되는 PRAGMA 설정 적용."""
    # 동시 읽기 성능 향상
    conn.execute("PRAGMA journal_mode=WAL;")
    # 더 빠른 쓰기와 WAL 사용 시 충돌 안전성 유지
    conn.execute("PRAGMA synchronous=NORMAL;")
    # 임시 테이블의 메모리 저장
    conn.execute("PRAGMA temp_store=MEMORY;")
    # 외래 키 제약 조건 적용
    conn.execute("PRAGMA foreign_keys=ON;")

# =============================================================================
# 4.3.2 - 데이터베이스 생성 및 스키마
# =============================================================================

def create_database(db_path: str, extension_path: str = ".") -> sqlite3.Connection:
    """
    적절한 스키마를 갖춘 로컬 Reddit 지식 베이스 데이터베이스 생성.

    Args:
        db_path: SQLite 데이터베이스 파일 경로.
        extension_path: vss0 및 vector0 확장이 들어 있는 디렉터리.

    Returns:
        확장이 로드된 데이터베이스 연결.
    """
    conn = sqlite3.connect(db_path)
    conn.enable_load_extension(True)
    conn.load_extension(f"{extension_path}/vector0")
    conn.load_extension(f"{extension_path}/vss0")
    conn.enable_load_extension(False)

    configure_connection(conn)

    # 콘텐츠와 메타데이터를 포함하는 기본 posts 테이블
    # rowid의 별칭인 INTEGER PRIMARY KEY 사용,
    # VSS 인덱스 조인을 위한 안정적인 식별자 제공
    conn.execute("""
        CREATE TABLE IF NOT EXISTS posts (
            id INTEGER PRIMARY KEY,
            post_id TEXT UNIQUE NOT NULL,
            title TEXT NOT NULL,
            selftext TEXT,
            url TEXT,
            subreddit TEXT NOT NULL,
            author TEXT,
            score INTEGER DEFAULT 0,
            num_comments INTEGER DEFAULT 0,
            created_utc INTEGER NOT NULL,
            embedding BLOB,
            embedding_model TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)

    # 일반적인 필터 작업을 위한 인덱스
    conn.execute("CREATE INDEX IF NOT EXISTS idx_posts_subreddit ON posts(subreddit)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_posts_created ON posts(created_utc)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_posts_score ON posts(score)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_posts_post_id ON posts(post_id)")

    conn.commit()
    return conn

def create_vector_index(conn: sqlite3.Connection, dimension: int) -> None:
    """
    벡터 유사도 검색을 위한 VSS 가상 테이블 생성.

    Args:
        conn: vss 확장이 로드된 데이터베이스 연결.
        dimension: 임베딩 벡터 차원. 예: all-MiniLM-L6-v2의 경우 384.
    """

    # 차원 변경 시 필요한 기존 인덱스 삭제
    conn.execute("DROP TABLE IF EXISTS posts_vss")

    # Flat(정확) 인덱스를 사용하는 VSS 가상 테이블 생성
    # posts 테이블과의 조인을 위해 rowid를 키로 벡터 저장
    conn.execute(f"""
        CREATE VIRTUAL TABLE posts_vss USING vss0(
            embedding({dimension})
        )
    """)

    conn.commit()
    print(f"Created vector index with dimension {dimension}")

# =============================================================================
# 4.4.2 - 로컬 Reddit CSV 소스
# =============================================================================

class LocalRedditDatasetSource:
    """
    로컬 CSV 파일에서 Reddit 형식의 레코드를 스트리밍.

    Kaggle 댓글 파일은 일반적으로 게시물이 아니라 댓글을 저장함. 이 클래스는
    댓글 행을 의미 기반 검색 파이프라인의 나머지 부분에서 사용하는 표준 필드로
    매핑함.
        post_id, title, selftext, url, subreddit, author, score,
        num_comments, created_utc, record_type
    컬럼명 변형을 의도적으로 허용하므로 Kaggle 댓글 CSV뿐만 아니라 유사한
    내보내기 파일에서도 동작 가능함.
    """

    # 여러 CSV 내보내기 형식에서 본문 컬럼명이 다를 수 있으므로 후보 컬럼명을 정의.
    BODY_COLUMNS = ("body", "comment", "comment_body", "text", "selftext", "content")
    TITLE_COLUMNS = ("title", "post_title", "submission_title")
    ID_COLUMNS = ("id", "comment_id", "commentid", "post_id", "submission_id")
    URL_COLUMNS = ("permalink", "url", "link", "comment_url")
    SUBREDDIT_COLUMNS = (
        "subreddit.name",
        "subreddit",
        "subreddit_name",
        "subreddit.display_name",
    )
    AUTHOR_COLUMNS = ("author", "username", "user", "user_name")
    SCORE_COLUMNS = ("score", "ups", "upvotes")
    NUM_COMMENTS_COLUMNS = ("num_comments", "comments", "comment_count")
    CREATED_COLUMNS = ("created_utc", "created", "created_at", "date", "timestamp")
    TYPE_COLUMNS = ("type", "kind", "record_type")

    def __init__(self, csv_path: Union[str, Path], encoding: str = "utf-8"):
        # 문자열 경로와 Path 객체를 모두 받을 수 있도록 Path로 정규화.
        self.csv_path = Path(csv_path)
        self.encoding = encoding

        # 파일이 없으면 이후 단계에서 모호한 오류가 발생하지 않도록 즉시 실패 처리.
        if not self.csv_path.exists():
            raise FileNotFoundError(f"CSV file not found: {self.csv_path}")

    def iter_records(
        self,
        subreddits: Optional[Iterable[str]] = None,
        sort: str = "csv",
        limit: Optional[int] = None,
        time_filter: str = "all",
    ) -> Generator[Dict[str, Any], None, None]:
        """
        CSV에서 표준 Reddit 레코드를 생성.

        Args:
            subreddits: 포함할 subreddit 이름 목록. 지정하지 않을 수 있으며 대소문자를 구분하지 않음.
            sort: csv, none, top, score, hot, rising, new, old 중 하나.
            limit: 필터링/정렬 후 생성할 최대 행 수.
            time_filter: all, day, week, month, year. 로컬 과거 데이터셋에서는
                상대 시간 필터가 현재 날짜가 아니라 필터링된 CSV의 최신 행을 기준으로 계산됨.
        """
        sort = sort.lower()

        # Reddit API의 대표 정렬 옵션과 CSV 원본 순서 유지 옵션을 함께 지원.
        valid_sorts = {"csv", "none", "top", "score", "hot", "rising", "new", "old"}
        if sort not in valid_sorts:
            raise ValueError(f"Unknown local CSV sort method: {sort}")

        subreddit_filter = None
        if subreddits:
            subreddit_filter = {s.lower().removeprefix("r/") for s in subreddits}

        # 정렬이나 시간 필터가 필요하면 전체 레코드를 메모리에 적재한 뒤 처리.
        must_materialize = sort not in {"csv", "none"} or time_filter != "all"

        if must_materialize:
            # 먼저 CSV에서 필터링 가능한 레코드를 모두 읽어 목록으로 변환.
            records = list(self._iter_filtered_records(subreddit_filter))
            # 로컬 데이터셋 기준 상대 시간 필터를 적용.
            records = self._apply_time_filter(records, time_filter)
            # 점수나 생성 시각 기준으로 정렬.
            records = self._sort_records(records, sort)
            # limit은 필터링과 정렬이 끝난 뒤 적용해야 사용자가 기대하는 상위 N개가 됨.
            if limit is not None:
                records = records[:limit]
            # 이미 만들어진 목록을 제너레이터처럼 하나씩 반환.
            yield from records
            return

        # CSV 원래 순서를 유지하는 경우에는 메모리 사용을 줄이기 위해 스트리밍 처리.
        yielded = 0
        for record in self._iter_filtered_records(subreddit_filter):
            yield record
            yielded += 1
            if limit is not None and yielded >= limit:
                break

    def get_subreddit_posts(
        self,
        subreddit_name: str,
        sort: str = "top",
        limit: int = 100,
        time_filter: str = "all",
    ) -> Generator[Dict[str, Any], None, None]:
        """
        이전 수집 파이프라인 메서드명과의 호환성을 위한 래퍼.
        네트워크 요청을 수행하는 대신, 하나의 subreddit에 해당하는 로컬 CSV
        레코드를 반환함.
        """
        yield from self.iter_records(
            subreddits=[subreddit_name],
            sort=sort,
            limit=limit,
            time_filter=time_filter,
        )

    def _iter_filtered_records(
        self,
        subreddit_filter: Optional[set[str]],
    ) -> Generator[Dict[str, Any], None, None]:
        # CSV 행을 순회하면서 표준 레코드로 변환하고 subreddit 필터를 적용.
        with self.csv_path.open("r", encoding=self.encoding, newline="") as file_obj:
            reader = csv.DictReader(file_obj)

            # start=2는 실제 CSV 파일에서 데이터가 2번째 줄부터 시작하기 때문.
            # 오류 추적이나 안정적 ID 생성 시 원본 행 번호를 활용할 수 있음.
            for row_number, row in enumerate(reader, start=2):
                record = self._row_to_record(row, row_number=row_number)

                # 본문과 제목이 모두 없는 행처럼 검색에 쓸 수 없는 행은 건너뜀.
                if not record:
                    continue
                
                # subreddit 필터가 지정된 경우, 해당 subreddit에 속하지 않는 레코드는 제외.
                if (
                    subreddit_filter
                    and record["subreddit"].lower() not in subreddit_filter
                ):
                    continue

                yield record

    def _row_to_record(
        self, row: Dict[str, Any], row_number: int
    ) -> Optional[Dict[str, Any]]:
        # 컬럼명 비교를 쉽게 하기 위해 CSV 컬럼명을 소문자 기준으로 정규화.
        normalized = {
            str(key).strip().lower(): (value if value is not None else "")
            for key, value in row.items()
            if key is not None
        }

        body = self._first_value(normalized, self.BODY_COLUMNS)
        title = self._first_value(normalized, self.TITLE_COLUMNS)
        record_type = (
            self._first_value(normalized, self.TYPE_COLUMNS, default="comment")
            or "comment"
        )

        # 본문과 제목이 모두 없으면 검색할 텍스트가 없으므로 제외.
        if not body and not title:
            return None

        # 각 표준 필드에 들어갈 원본 값을 후보 컬럼 목록에서 추출.
        raw_id = self._first_value(normalized, self.ID_COLUMNS)
        permalink = self._first_value(normalized, self.URL_COLUMNS)
        created_utc = self._parse_timestamp(
            self._first_value(normalized, self.CREATED_COLUMNS)
        )
        subreddit = (
            self._first_value(normalized, self.SUBREDDIT_COLUMNS, default="unknown")
            or "unknown"
        )
        author = (
            self._first_value(normalized, self.AUTHOR_COLUMNS, default="[unknown]")
            or "[unknown]"
        )
        score = self._parse_int(
            self._first_value(normalized, self.SCORE_COLUMNS), default=0
        )
        num_comments = self._parse_int(
            self._first_value(normalized, self.NUM_COMMENTS_COLUMNS), default=0
        )

        # 원본 ID가 없을 경우 중복 방지를 위해 안정적인 해시 기반 ID 생성.
        if not raw_id:
            raw_id = self._stable_id(
                body=body, permalink=permalink, row_number=row_number
            )

        # 댓글 데이터에는 제목이 없을 수 있으므로 본문 앞부분을 표시용 제목으로 사용.
        if not title:
            title = self._make_comment_title(body, subreddit)

        # 이후 저장소와 검색 엔진이 기대하는 표준 dict 구조로 반환.
        # 필드 이름은 기존 posts 테이블 스키마 및 ingestion pipeline과 맞춰져 있음.
        return {
            "post_id": str(raw_id),
            "title": title,
            "selftext": body,
            "url": permalink,
            "subreddit": subreddit,
            "author": author,
            "score": score,
            "num_comments": num_comments,
            "created_utc": created_utc,
            "is_self": True,
            "record_type": record_type,
        }

    @staticmethod
    def _first_value(
        row: Dict[str, Any], columns: Sequence[str], default: str = ""
    ) -> str:
        # 후보 컬럼 목록에서 처음 발견되는 비어 있지 않은 값을 반환.
        for column in columns:
            value = row.get(column)
            if value not in (None, ""):
                return str(value).strip()
        return default

    @staticmethod
    def _parse_int(value: Any, default: int = 0) -> int:
        # 숫자 문자열과 실수형 문자열을 정수로 안전하게 변환.
        if value in (None, ""):
            return default
        try:
            return int(float(str(value).strip()))
        except (TypeError, ValueError):
            return default

    @classmethod
    def _parse_timestamp(cls, value: Any) -> int:
        # created_utc처럼 초 단위 epoch 값이 들어오는 경우를 우선 처리.
        if value in (None, ""):
            return 0

        text = str(value).strip()
        try:
            return int(float(text))
        except ValueError:
            pass

        # 2022-03-01T12:00:00Z와 같은 일반적인 ISO 타임스탬프 형식 처리.
        # Z는 UTC를 의미하므로 Python의 fromisoformat()이 이해할 수 있는 +00:00으로 변환.
        try:
            parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
            if parsed.tzinfo is None:
                parsed = parsed.replace(tzinfo=timezone.utc)
            return int(parsed.timestamp())
        except ValueError:
            return 0

    @staticmethod
    def _make_comment_title(body: str, subreddit: str, max_chars: int = 100) -> str:
        # 댓글 본문을 한 줄로 축약해 검색 결과에 표시할 제목 생성.
        compact = " ".join(body.split())
        if len(compact) > max_chars:
            compact = compact[: max_chars - 1].rstrip() + "..."
        return compact or f"Comment in r/{subreddit}"

    @staticmethod
    def _stable_id(body: str, permalink: str, row_number: int) -> str:
        # permalink, 본문, 행 번호를 조합해 재실행해도 동일한 ID가 나오도록 생성.
        seed = f"{permalink}\n{body}\n{row_number}".encode("utf-8", errors="ignore")
        return hashlib.sha1(seed).hexdigest()[:16]

    @staticmethod
    def _sort_records(records: List[Dict[str, Any]], sort: str) -> List[Dict[str, Any]]:
        # Reddit API의 정렬 옵션과 유사하게 로컬 레코드를 정렬.
        if sort in {"csv", "none"}:
            return records
        if sort in {"top", "score", "hot", "rising"}:
            return sorted(records, key=lambda item: item.get("score", 0), reverse=True)
        if sort == "new":
            return sorted(
                records, key=lambda item: item.get("created_utc", 0), reverse=True
            )
        if sort == "old":
            return sorted(records, key=lambda item: item.get("created_utc", 0))
        raise ValueError(f"Unknown local CSV sort method: {sort}")

    @staticmethod
    def _apply_time_filter(
        records: List[Dict[str, Any]], time_filter: str
    ) -> List[Dict[str, Any]]:
        # 로컬 데이터셋의 최신 시점을 기준으로 상대 기간 필터 적용.
        time_filter = time_filter.lower()
        if time_filter == "all":
            return records

        # Reddit API의 time_filter 옵션과 유사한 기간을 초 단위로 정의.
        seconds_by_filter = {
            "hour": 60 * 60,
            "day": 60 * 60 * 24,
            "week": 60 * 60 * 24 * 7,
            "month": 60 * 60 * 24 * 31,
            "year": 60 * 60 * 24 * 365,
        }
        if time_filter not in seconds_by_filter:
            raise ValueError(f"Unknown time_filter: {time_filter}")

        # 필터링 대상 레코드 중 가장 최신 timestamp를 기준 시각으로 사용.
        newest = max((record.get("created_utc", 0) for record in records), default=0)
        if newest <= 0:
            return records

        # 기준 시각에서 지정 기간을 뺀 값을 임계값으로 삼아 최근 레코드만 남김.
        threshold = newest - seconds_by_filter[time_filter]
        return [
            record for record in records if record.get("created_utc", 0) >= threshold
        ]

# =============================================================================
# 4.5.1 - 텍스트 전처리
# =============================================================================

class TextPreprocessor:
    """임베딩을 위해 Reddit 텍스트 콘텐츠를 정제하고 정규화."""

    def __init__(self):
        # 효율성을 위해 정규식 패턴을 한 번만 컴파일
        self.url_pattern = re.compile(r'https?://[^\s<>"{}|\\^`$begin:math:display$$end:math:display$]+')
        self.reddit_link_pattern = re.compile(r"(?<!\w)/r/\w+|(?<!\w)/u/\w+")
        self.markdown_link_pattern = re.compile(r"\[([^\]]+)\]\([^)]+\)")
        self.whitespace_pattern = re.compile(r"\s+")

    def clean_markdown(self, text: str) -> str:
        """
        마크다운 서식 제거 또는 단순화.
        정규식을 사용하는 가벼운 최선형 클리너임. 완전한 마크다운
        처리를 위해서는 적절한 파서를 사용해야 하지만, 이는 이 장의
        범위를 넘어서는 복잡성을 추가함.
        Args:
            text: 가공하지 않은 마크다운 텍스트
        Returns:
            정리된 텍스트
        """
        # 마크다운 링크를 링크 텍스트만 남기도록 대체
        text = self.markdown_link_pattern.sub(r"\1", text)
        text = re.sub(r"\*\*([^*]+)\*\*", r"\1", text)  # 굵게 표시.
        text = re.sub(r"\*([^*]+)\*", r"\1", text)  # 기울임꼴.
        text = re.sub(r"~~([^~]+)~~", r"\1", text)  # 취소선.
        text = re.sub(r"`([^`]+)`", r"\1", text)  # 인라인 코드.
        text = re.sub(r"^#+\s*", "", text, flags=re.MULTILINE)  # 제목.
        text = re.sub(r"^[>\-*]\s*", "", text, flags=re.MULTILINE)  # 인용문/목록.
        return text

    def handle_urls(self, text: str) -> str:
        """
        URL을 자리표시자 토큰으로 대체.
        링크의 존재 자체가 의미상 중요한 경우가 많기 때문에 URL을 완전히
        제거하지 않고 [URL]을 사용함(예: "여기에 튜토리얼이 있음"과
        단순 텍스트의 차이). 이 자리표시자는 길고 고유한 URL이 임베딩을
        지배하지 않도록 하면서 이러한 신호를 보존함.
        """
        return self.url_pattern.sub("[URL]", text)

    def clean_reddit_artifacts(self, text: str) -> str:
        # Reddit 특유의 삭제 표시, 사용자/서브레딧 링크, HTML 엔티티 정리.
        text = re.sub(r"\[removed\]|\[deleted\]", "", text)
        # 서브레딧 및 사용자 참조 제거(임베딩에 노이즈를 추가하기 때문)
        text = self.reddit_link_pattern.sub("", text)
        # HTML 엔티티 디코딩
        text = text.replace("&amp;", "&")
        text = text.replace("&lt;", "<")
        text = text.replace("&gt;", ">")
        text = text.replace("&nbsp;", " ")
        return text

    def normalize_whitespace(self, text: str) -> str:
        # 여러 공백과 줄바꿈을 하나의 공백으로 정규화.
        text = self.whitespace_pattern.sub(" ", text)
        return text.strip()

    def process(self, text: Optional[str]) -> str:
        """
        전체 전처리 파이프라인 적용.
        Args:
            text: 가공하지 않은 텍스트 콘텐츠
        Returns:
        정리되고 정규화된 텍스트
        """
        if not text:
            return ""
        
        text = self.clean_markdown(text)
        text = self.handle_urls(text)
        text = self.clean_reddit_artifacts(text)
        text = self.normalize_whitespace(text)
        return text

class ContentPreparer:
    """임베딩 생성을 위해 로컬 Reddit 콘텐츠 준비."""

    def __init__(self):
        self.preprocessor = TextPreprocessor()

    def prepare_post(self, post: dict) -> str:
        # 제목과 본문을 정제한 뒤 임베딩에 사용할 대표 텍스트 구성.
        title = self.preprocessor.process(post.get("title", ""))
        selftext = self.preprocessor.process(post.get("selftext", ""))

        # Kaggle 댓글 파일은 본문 텍스트를 주요 콘텐츠로 사용함. 생성된 제목은
        # 표시용 미리보기에 불과하므로 같은 내용을 두 번 임베딩하지 않도록 처리.
        if post.get("record_type") == "comment":
            return selftext or title

        # 제목과 본문 결합, 강조를 위해 제목을 먼저 배치
        if selftext:
            return f"{title}. {selftext}" if title else selftext
        return title

    def is_quality_content(
        self, text: str, min_length: int = 20, min_words: int = 3
    ) -> bool:
        """
        콘텐츠가 품질 기준을 충족하는지 확인.
        Args:
            text: 전처리된 텍스트
            min_length: 최소 문자 수
            min_words: 최소 단어 수
        Returns:
        콘텐츠가 품질 검사를 통과하면 True
        """
        if len(text) < min_length:
            return False
        if len(text.split()) < min_words:
            return False
        if text in ["[URL]", "", " "]:
            return False
        return True

# =============================================================================
# 4.6.1 - 임베딩 생성기
# =============================================================================

class EmbeddingGenerator:
    """SentenceTransformers 모델을 사용한 임베딩 생성."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """
        임베딩 생성기 초기화.
        재현성을 위해 SentenceTransformers 모델 버전
        (가능하다면 transformers/torch 버전도 함께)을 고정하여,
        서로 다른 설치 환경에서도 임베딩이 안정적으로 유지되도록 함.
        Args:
            model_name: SentenceTransformers 모델 식별자
        """
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise ImportError(
                "sentence-transformers is required for embedding generation. "
                "Install it with: pip install sentence-transformers"
            ) from exc

        self.model_name = model_name
        self.model = SentenceTransformer(model_name)
        self.dimension = self.model.get_sentence_embedding_dimension()
        # 사용 가능한 최적의 디바이스 결정
        self.device = self._select_device()
        self.model.to(self.device)
        print(f"Loaded {model_name} (dim={self.dimension}) on {self.device}")

    def _select_device(self) -> str:
        # 사용 가능한 하드웨어 가속 장치를 선택하고, 없으면 CPU 사용.
        import torch

        if torch.cuda.is_available():
            return "cuda"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    def encode(
        self,
        texts: Union[str, List[str]],
        batch_size: int = 32,
        show_progress: bool = False,
    ) -> np.ndarray:
        """
        텍스트에 대한 임베딩 생성.
        Args:
            texts: 단일 텍스트 또는 텍스트 목록
            batch_size: 인코딩에 사용할 배치 크기
            show_progress: 진행률 표시줄 표시 여부
        Returns:
        임베딩의 NumPy 배열, 형태는 (n_texts, dimension)
        """
        if isinstance(texts, str):
            texts = [texts]
        return self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=show_progress,
            convert_to_numpy=True,
        )

    @staticmethod
    def serialize_embedding(embedding: np.ndarray) -> bytes:
        """
        SQLite 저장을 위해 NumPy 임베딩을 바이트로 변환.
        Args:
            embedding: 1차원 NumPy 배열
        Returns:
        바이트 표현
        """
        return embedding.astype(np.float32).tobytes()

    @staticmethod
    def deserialize_embedding(blob: bytes, dimension: int) -> np.ndarray:
        """
    def deserialize_embedding(blob: bytes, dimension: int) -> np.ndarray:
    바이트를 다시 NumPy 임베딩으로 변환.
    Args:
        blob: SQLite에서 가져온 바이트
        dimension: 예상 벡터 차원
    Returns:
    NumPy 배열
    """
        return np.frombuffer(blob, dtype=np.float32).reshape(dimension)

# =============================================================================
# 4.6.2 - 지식 베이스 저장소
# =============================================================================

class KnowledgeBaseStorage:
    """임베딩이 포함된 로컬 Reddit 콘텐츠의 저장 및 조회 처리."""

    def __init__(
        self, db_path: str, extension_path: str = ".", embedding_dim: int = 384
    ):
        """
        저장소 연결 초기화.
        Args:
            db_path: SQLite 데이터베이스 경로
            extension_path: vss 확장 기능이 포함된 디렉터리
            embedding_dim: 임베딩 벡터 차원
        """
        self.db_path = db_path
        self.extension_path = extension_path
        self.embedding_dim = embedding_dim
        self.conn = self._connect()

    def _connect(self) -> sqlite3.Connection:
        # SQLite 데이터베이스에 연결하고 VSS 확장 및 운영용 PRAGMA 설정 적용.
        conn = sqlite3.connect(self.db_path)
        conn.enable_load_extension(True)
        conn.load_extension(f"{self.extension_path}/vector0")
        conn.load_extension(f"{self.extension_path}/vss0")
        conn.enable_load_extension(False)
        configure_connection(conn)
        conn.row_factory = sqlite3.Row
        return conn

    def insert_post(
        self, post: Dict[str, Any], embedding: Optional[np.ndarray], model_name: str
    ) -> int:
        """
        게시물과 해당 임베딩 INSERT 또는 갱신.
        충돌 발생 시 rowid를 보존하기 위해 UPSERT를 사용하며, 이는
        VSS 인덱스의 일관성을 유지하는 데 필수적임.
        Args:
            post: 게시물 데이터 딕셔너리
            embedding: 임베딩 벡터(또는 건너뛰기 위한 None)
            model_name: 사용된 임베딩 모델 이름
        Returns:
        INSERT 또는 갱신된 게시물의 행 ID
        """
        embedding_blob = None
        if embedding is not None:
            embedding_blob = EmbeddingGenerator.serialize_embedding(embedding)

        try:
            cursor = self.conn.execute(
                """
                INSERT INTO posts
                (post_id, title, selftext, url, subreddit, author,
                    score, num_comments, created_utc, embedding, embedding_model)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(post_id) DO UPDATE SET
                    title = excluded.title, selftext = excluded.selftext,
                    url = excluded.url, subreddit = excluded.subreddit,
                    author = excluded.author, score = excluded.score,
                    num_comments = excluded.num_comments,
                    created_utc = excluded.created_utc,
                    embedding = excluded.embedding,
                    embedding_model = excluded.embedding_model
                RETURNING id
            """,
                (
                    post["post_id"],
                    post["title"],
                    post.get("selftext", ""),
                    post.get("url", ""),
                    post["subreddit"],
                    post.get("author", "[unknown]"),
                    post.get("score", 0),
                    post.get("num_comments", 0),
                    post["created_utc"],
                    embedding_blob,
                    model_name,
                ),
            )
            row_id = cursor.fetchone()[0]
        except sqlite3.OperationalError:
            # RETURNING을 지원하지 않는 이전 SQLite 버전을 위한 대체 경로.
            self.conn.execute(
                """
                INSERT INTO posts
                (post_id, title, selftext, url, subreddit, author,
                    score, num_comments, created_utc, embedding, embedding_model)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(post_id) DO UPDATE SET
                    title = excluded.title, selftext = excluded.selftext,
                    url = excluded.url, subreddit = excluded.subreddit,
                    author = excluded.author, score = excluded.score,
                    num_comments = excluded.num_comments,
                    created_utc = excluded.created_utc,
                    embedding = excluded.embedding,
                    embedding_model = excluded.embedding_model
            """,
                (
                    post["post_id"],
                    post["title"],
                    post.get("selftext", ""),
                    post.get("url", ""),
                    post["subreddit"],
                    post.get("author", "[unknown]"),
                    post.get("score", 0),
                    post.get("num_comments", 0),
                    post["created_utc"],
                    embedding_blob,
                    model_name,
                ),
            )
            row_id = self.conn.execute(
                "SELECT id FROM posts WHERE post_id = ?", (post["post_id"],)
            ).fetchone()[0]

        self.conn.commit()
        return row_id

    def insert_posts_batch(
        self, posts: List[Dict[str, Any]], embeddings: np.ndarray, model_name: str
    ) -> int:
        """
        여러 게시물과 임베딩의 효율적 INSERT.
        Args:
            posts: 게시물 딕셔너리 목록
            embeddings: 임베딩 배열, 형태는 (n_posts, dim)
            model_name: 임베딩 모델 이름
        Returns:
        INSERT된 게시물 수
        """
        data = []
        for post, embedding in zip(posts, embeddings):
            embedding_blob = EmbeddingGenerator.serialize_embedding(embedding)
            data.append(
                (
                    post["post_id"],
                    post["title"],
                    post.get("selftext", ""),
                    post.get("url", ""),
                    post["subreddit"],
                    post.get("author", "[unknown]"),
                    post.get("score", 0),
                    post.get("num_comments", 0),
                    post["created_utc"],
                    embedding_blob,
                    model_name,
                )
            )

        self.conn.executemany(
            """
            INSERT INTO posts
            (post_id, title, selftext, url, subreddit, author,
                score, num_comments, created_utc, embedding, embedding_model)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(post_id) DO UPDATE SET
                title = excluded.title, selftext = excluded.selftext,
                url = excluded.url, subreddit = excluded.subreddit,
                author = excluded.author, score = excluded.score,
                num_comments = excluded.num_comments,
                created_utc = excluded.created_utc,
                embedding = excluded.embedding,
                embedding_model = excluded.embedding_model
        """,
            data,
        )

        self.conn.commit()
        return len(data)

    def get_post_count(self) -> int:
        """데이터베이스에 저장된 전체 게시물 수 반환."""
        result = self.conn.execute("SELECT COUNT(*) FROM posts").fetchone()
        return result[0]

    def close(self) -> None:
        """데이터베이스 연결 종료."""
        self.conn.close()

# =============================================================================
# 4.6.3 - 수집 파이프라인
# =============================================================================

class RedditIngestionPipeline:
    """로컬 Reddit CSV 콘텐츠 수집을 위한 엔드투엔드 파이프라인."""

    def __init__(
        self,
        data_source: LocalRedditDatasetSource,
        storage: KnowledgeBaseStorage,
        embedding_generator: EmbeddingGenerator,
    ):
        self.data_source = data_source
        self.storage = storage
        self.embedder = embedding_generator
        self.preparer = ContentPreparer()

    def ingest_dataset(
        self,
        subreddits: Optional[List[str]] = None,
        sort: str = "top",
        limit: Optional[int] = 100,
        time_filter: str = "all",
        batch_size: int = 32,
    ) -> dict:
        """
        서브레딧에서 게시물 수집.
        Args:
            subreddit_name: 게시물을 가져올 서브레딧
            sort: 정렬 방식
            limit: 최대 게시물 수
            time_filter: 'top' 정렬에 사용할 시간 필터
            batch_size: 임베딩 배치 크기
        Returns:
        통계 딕셔너리
        """
        stats = {"fetched": 0, "processed": 0, "skipped": 0}
        posts_batch = []
        texts_batch = []

        for post in self.data_source.iter_records(
            subreddits=subreddits,
            sort=sort,
            limit=limit,
            time_filter=time_filter,
        ):
            stats["fetched"] += 1
            # 임베딩을 위한 텍스트 준비
            text = self.preparer.prepare_post(post)
            # 품질 필터
            if not self.preparer.is_quality_content(text):
                stats["skipped"] += 1
                continue

            posts_batch.append(post)
            texts_batch.append(text)
            # 배치 단위 처리
            if len(posts_batch) >= batch_size:
                self._process_batch(posts_batch, texts_batch)
                stats["processed"] += len(posts_batch)
                posts_batch = []
                texts_batch = []
        # 남은 항목 처리
        if posts_batch:
            self._process_batch(posts_batch, texts_batch)
            stats["processed"] += len(posts_batch)

        return stats

    def ingest_subreddit(
        self,
        subreddit_name: str,
        sort: str = "top",
        limit: int = 100,
        time_filter: str = "all",
        batch_size: int = 32,
    ) -> dict:
        """로컬 CSV에서 하나의 subreddit을 수집하기 위한 호환성 래퍼."""
        return self.ingest_dataset(
            subreddits=[subreddit_name],
            sort=sort,
            limit=limit,
            time_filter=time_filter,
            batch_size=batch_size,
        )

    def _process_batch(self, posts: List[Dict[str, Any]], texts: List[str]) -> None:
        # 배치 단위로 텍스트 임베딩을 생성하고 저장소에 기록.
        embeddings = self.embedder.encode(texts)
        self.storage.insert_posts_batch(posts, embeddings, self.embedder.model_name)

# =============================================================================
# 4.7.2 - 벡터 인덱스 관리자
# =============================================================================

class VectorIndexManager:
    """VSS 벡터 인덱스 관리."""

    def __init__(self, storage: KnowledgeBaseStorage):
        self.storage = storage
        self.conn = storage.conn
        self.dimension = storage.embedding_dim

    def create_index(self, factory_string: str = "Flat") -> None:
        """
        벡터 검색 인덱스 생성 또는 재생성.
        참고: 기본 factory_string은 버전에 따라 달라질 수 있음
        Args:
            factory_string: FAISS 인덱스 유형. 선택지는 다음과 같음:
                - "Flat": 정확 검색(기본값, 10만 개 미만 벡터에 가장 적합)
                - "IVF100,Flat": 100개 클러스터를 사용하는 근사 검색
                - "HNSW16": 그래프 기반 근사 검색
        """
        # 기존 인덱스 삭제
        self.conn.execute("DROP TABLE IF EXISTS posts_vss")

        # 새 VSS 가상 테이블 생성
        if factory_string == "Flat":
            self.conn.execute(f"""
                CREATE VIRTUAL TABLE posts_vss USING vss0(
                    embedding({self.dimension})
                )
            """)
        else:
            self.conn.execute(f"""
                CREATE VIRTUAL TABLE posts_vss USING vss0(
                    embedding({self.dimension}) factory="{factory_string}"
                )
            """)

        self.conn.commit()
        print(f"Created vector index: {factory_string}, dimension={self.dimension}")

    def populate_index(self, batch_size: int = 500) -> int:
        """
        기존 게시물 임베딩으로 인덱스 채우기.
        참고: 이 구현은 성능보다 명확성을 우선함.
        성능을 높이려면 다음 방법을 사용할 수 있음:
        - 더 큰 배치와 함께 executemany 사용
        - 사용 가능한 경우 바이너리 벡터 INSERT API 사용
        - 전체 작업을 단일 트랜잭션으로 감싸기
        Args:
            batch_size: 한 번에 처리할 행 수
        Returns:
        인덱싱된 벡터 수
        """
        # 임베딩이 있는 게시물 수 계산
        total = self.conn.execute(
            "SELECT COUNT(*) FROM posts WHERE embedding IS NOT NULL"
        ).fetchone()[0]

        if total == 0:
            print("No embeddings to index")
            return 0

        indexed = 0
        offset = 0

        while offset < total:
            # 임베딩이 있는 게시물 배치 조회
            rows = self.conn.execute(
                """
                SELECT id, embedding FROM posts
                WHERE embedding IS NOT NULL
                ORDER BY id LIMIT ? OFFSET ?
            """,
                (batch_size, offset),
            ).fetchall()

            if not rows:
                break

            # 배치를 VSS 인덱스에 추가
            # 조인을 위해 id(rowid의 별칭)를 사용
            insert_data = []
            for row in rows:
                row_id = row["id"]
                embedding = EmbeddingGenerator.deserialize_embedding(
                    row["embedding"], self.dimension
                )
                vector_json = json.dumps(embedding.tolist())
                insert_data.append((row_id, vector_json))

            self.conn.executemany(
                """
                INSERT INTO posts_vss (rowid, embedding)
                VALUES (?, vector_from_json(?))
            """,
                insert_data,
            )

            self.conn.commit()
            indexed += len(rows)
            offset += batch_size

            if indexed % 1000 == 0 or indexed == total:
                print(f"Indexed {indexed}/{total} vectors")

        return indexed

    def rebuild_index(self, factory_string: str = "Flat") -> int:
        # 벡터 인덱스를 새로 만들고 전체 임베딩을 다시 적재.
        self.create_index(factory_string)
        return self.populate_index()

# =============================================================================
# 4.8.1 - 검색 결과 컨테이너
# =============================================================================

@dataclass
class SearchResult:
    """단일 검색 결과를 담는 컨테이너."""

    rowid: int
    post_id: str
    title: str
    selftext: str
    subreddit: str
    author: str
    score: int
    num_comments: int
    created_utc: int
    distance: float

    @property
    def similarity_score(self) -> float:
        """
        표시용으로 L2 거리를 유사도 점수로 변환.
        이는 수학적으로 엄밀한 유사도 척도가 아니라 표시를 위한 휴리스틱임.
        거리가 낮을수록 유사도는 높음. 이 공식은 거리를 0~1 범위로 매핑하며,
        1에 가까울수록 가장 유사함.
        """
        return 1.0 / (1.0 + self.distance)

    @property
    def created_date(self) -> str:
        """생성 타임스탬프를 날짜 문자열로 형식화."""
        return datetime.fromtimestamp(self.created_utc, tz=timezone.utc).strftime(
            "%Y-%m-%d"
        )

# =============================================================================
# 4.8.2 - 의미 기반 검색 엔진
# =============================================================================

class SemanticSearchEngine:
    """로컬 Reddit 지식 베이스에 대한 의미 기반 검색."""

    def __init__(
        self,
        storage: KnowledgeBaseStorage,
        embedding_generator: EmbeddingGenerator,
        candidate_multiplier: int = 10,
    ):
        self.storage = storage
        self.conn = storage.conn
        self.embedder = embedding_generator
        self.preparer = ContentPreparer()
        self.candidate_multiplier = candidate_multiplier

    def search(
        self,
        query: str,
        limit: int = 10,
        subreddits: Optional[List[str]] = None,
        min_score: Optional[int] = None,
        after_date: Optional[datetime] = None,
        before_date: Optional[datetime] = None,
    ) -> List[SearchResult]:
        """
        질의와 의미적으로 유사한 게시물 검색.
        sqlite-vss가 반환하는 거리는 원시 L2 거리임. 그 스케일은
        임베딩 모델과 콘텐츠에 따라 달라짐. 정규화된 표시 값이 필요한 경우
        similarity_score 속성 사용.
        Args:
            query: 자연어 검색 질의
            limit: 반환할 최대 결과 수
            subreddits: 특정 서브레딧으로 필터링
            min_score: 최소 게시물 점수
            after_date: 이 날짜 이후의 게시물만 포함
            before_date: 이 날짜 이전의 게시물만 포함
        Returns:
        유사도 순서로 정렬된 SearchResult 객체 목록(가장 가까운 항목 우선)
        """
        # 질의를 정제하고 임베딩으로 변환.
        clean_query = self.preparer.preprocessor.process(query)
        query_embedding = self.embedder.encode(clean_query)[0]
        query_vector_json = json.dumps(query_embedding.tolist())

        # 후보를 초과 조회한 뒤 SQL에서 메타데이터 필터를 적용함
        # 많은 sqlite-vss 버전에서 필터가 최근접 이웃 검색으로 푸시다운되지 않기 때문에
        # 이 방식은 결과 안정성을 높여 줌
        k = max(limit * self.candidate_multiplier, limit)

        sql = """
            WITH candidates AS (
                SELECT rowid, distance
                FROM posts_vss
                WHERE vss_search(embedding, vector_from_json(?))
                ORDER BY distance ASC
                LIMIT ?
            )
            SELECT
                p.id, p.post_id, p.title, p.selftext, p.subreddit,
                p.author, p.score, p.num_comments, p.created_utc,
                c.distance
            FROM candidates c
            INNER JOIN posts p ON p.id = c.rowid
            WHERE 1=1
        """
        params: List[Any] = [query_vector_json, k]

        # 후보 결과에 선택적 메타데이터 필터 적용.
        if subreddits:
            placeholders = ",".join("?" * len(subreddits))
            sql += f" AND p.subreddit IN ({placeholders})"
            params.extend(subreddits)

        if min_score is not None:
            sql += " AND p.score >= ?"
            params.append(min_score)

        if after_date is not None:
            sql += " AND p.created_utc >= ?"
            params.append(int(after_date.timestamp()))

        if before_date is not None:
            sql += " AND p.created_utc <= ?"
            params.append(int(before_date.timestamp()))

        sql += " ORDER BY c.distance ASC LIMIT ?"
        params.append(limit)
        
        # 실행 및 결과 파싱
        rows = self.conn.execute(sql, params).fetchall()

        return [
            SearchResult(
                rowid=row["id"],
                post_id=row["post_id"],
                title=row["title"],
                selftext=row["selftext"],
                subreddit=row["subreddit"],
                author=row["author"],
                score=row["score"],
                num_comments=row["num_comments"],
                created_utc=row["created_utc"],
                distance=row["distance"],
            )
            for row in rows
        ]

    def search_with_facets(self, query: str, limit: int = 50) -> dict:
        """
        패싯별 결과 분해를 포함한 검색.
        Args:
            query: 검색 질의
            limit: 최대 결과 수
        Returns:
        결과와 패싯 개수를 포함하는 딕셔너리
        """
        results = self.search(query, limit=limit)

        # 패싯 계산
        subreddit_counts = {}
        score_ranges = {"low": 0, "medium": 0, "high": 0}

        for result in results:
            # 서브레딧 패싯
            subreddit_counts[result.subreddit] = (
                subreddit_counts.get(result.subreddit, 0) + 1
            )

            # 점수 범위 패싯
            if result.score < 10:
                score_ranges["low"] += 1
            elif result.score < 100:
                score_ranges["medium"] += 1
            else:
                score_ranges["high"] += 1

        return {
            "results": results,
            "facets": {
                "subreddits": dict(
                    sorted(
                        subreddit_counts.items(), key=lambda item: item[1], reverse=True
                    )
                ),
                "score_ranges": score_ranges,
            },
            "total": len(results),
        }

    def find_similar_posts(
        self,
        post_id: str,
        limit: int = 10,
        exclude_same_subreddit: bool = False,
    ) -> List[SearchResult]:
        """
        주어진 게시물과 유사한 게시물 찾기.
        Args:
            post_id: 기준 게시물의 ID
            limit: 최대 결과 수
            exclude_same_subreddit: 같은 서브레딧의 게시물을 제외할지 여부
        Returns:
        유사한 게시물 목록
        """
        # 기준 게시글의 임베딩 가져오기
        row = self.conn.execute(
            "SELECT id, embedding, subreddit FROM posts WHERE post_id = ?",
            (post_id,),
        ).fetchone()

        if not row or not row["embedding"]:
            return []

        embedding = EmbeddingGenerator.deserialize_embedding(
            row["embedding"], self.embedder.dimension
        )
        vector_json = json.dumps(embedding.tolist())
        source_subreddit = row["subreddit"]
        source_id = row["id"]

        # 유사한 게시글 검색
        sql = """
            SELECT
                p.id, p.post_id, p.title, p.selftext, p.subreddit,
                p.author, p.score, p.num_comments, p.created_utc,
                vss.distance
            FROM posts_vss vss
            INNER JOIN posts p ON p.id = vss.rowid
            WHERE vss_search(vss.embedding, vector_from_json(?))
            AND p.id != ?
        """
        params: List[Any] = [vector_json, source_id]

        if exclude_same_subreddit:
            sql += " AND p.subreddit != ?"
            params.append(source_subreddit)

        sql += " ORDER BY vss.distance ASC LIMIT ?"
        params.append(limit)

        rows = self.conn.execute(sql, params).fetchall()

        return [
            SearchResult(
                rowid=row["id"],
                post_id=row["post_id"],
                title=row["title"],
                selftext=row["selftext"],
                subreddit=row["subreddit"],
                author=row["author"],
                score=row["score"],
                num_comments=row["num_comments"],
                created_utc=row["created_utc"],
                distance=row["distance"],
            )
            for row in rows
        ]

    def cross_subreddit_analysis(
        self, query: str, limit_per_subreddit: int = 5
    ) -> dict:
        """
        특정 주제가 여러 서브레딧에서 어떻게 논의되는지 분석.
        Args:
            query: 분석할 주제
            limit_per_subreddit: 서브레딧당 최대 결과 수
        Returns:
        서브레딧을 관련 게시물에 매핑한 딕셔너리
        """
        # 광범위한 결과 가져오기
        results = self.search(query, limit=100)

        # 서브레딧별 그룹화
        by_subreddit: Dict[str, List[SearchResult]] = {}
        for result in results:
            if result.subreddit not in by_subreddit:
                by_subreddit[result.subreddit] = []
            if len(by_subreddit[result.subreddit]) < limit_per_subreddit:
                by_subreddit[result.subreddit].append(result)

        # 전체 관련도 기준으로 서브레딧 정렬(유사도 점수 합계)
        subreddit_relevance = {
            subreddit: sum(result.similarity_score for result in posts)
            for subreddit, posts in by_subreddit.items()
        }
        sorted_subreddits = sorted(
            subreddit_relevance.keys(),
            key=lambda subreddit: subreddit_relevance[subreddit],
            reverse=True,
        )

        return {
            "subreddits": sorted_subreddits,
            "posts_by_subreddit": {
                subreddit: by_subreddit[subreddit] for subreddit in sorted_subreddits
            },
            "relevance_scores": subreddit_relevance,
        }

# =============================================================================
# 4.9 - 전체 구성 요소 조립
# =============================================================================

def main() -> None:
    """로컬 Reddit CSV 콘텐츠를 수집하고 의미 기반 검색을 수행하는 전체 예제."""

    # 구성 설정.
    DB_PATH = "reddit_knowledge.db"
    EXTENSION_PATH = "."  # vss0 및 vector0이 들어 있는 디렉터리.
    MODEL_NAME = "all-MiniLM-L6-v2"
    CSV_PATH = "/Users/Shared/ai-demo/vectordb/the-reddit-dataset-dataset-comments.csv"

    # CSV의 모든 subreddit을 수집하려면 None으로 설정. 참조한 Kaggle 파일은
    # 일반적으로 r/datasets 전용이므로 ["datasets"]도 안전한 설정.
    SUBREDDITS: Optional[List[str]] = None

    # 모든 행을 임베딩하려면 시간이 걸릴 수 있음. 값을 늘리거나 None으로 설정하면 전체 수집.
    INGEST_LIMIT: Optional[int] = 2000
    BATCH_SIZE = 64

    # 구성 요소 초기화
    print("Initializing components...")

    if not Path(CSV_PATH).exists():
        raise FileNotFoundError(f"Local Reddit CSV not found: {CSV_PATH}")

    # 데이터베이스 및 스키마 생성
    conn = create_database(DB_PATH, EXTENSION_PATH)
    conn.close()
    
    # 임베딩 생성기 초기화
    embedder = EmbeddingGenerator(MODEL_NAME)

    # 스토리지 초기화
    storage = KnowledgeBaseStorage(
        DB_PATH, EXTENSION_PATH, embedding_dim=embedder.dimension
    )
    data_source = LocalRedditDatasetSource(CSV_PATH)

    # 수집 파이프라인 생성
    pipeline = RedditIngestionPipeline(data_source, storage, embedder)
    
    # 여러 서브레딧에서 콘텐츠 수집
    print("\nIngesting content from local CSV...")
    stats = pipeline.ingest_dataset(
        subreddits=SUBREDDITS,
        sort="top",
        limit=INGEST_LIMIT,
        time_filter="all",
        batch_size=BATCH_SIZE,
    )
    print(
        f"  Fetched: {stats['fetched']}, "
        f"Processed: {stats['processed']}, "
        f"Skipped: {stats['skipped']}"
    )

    print(f"\nTotal records in database: {storage.get_post_count()}")

    # 저장된 임베딩을 VSS 인덱스에 적재해 의미 기반 검색 준비.
    print("\nBuilding vector index...")
    index_manager = VectorIndexManager(storage)
    index_manager.create_index()
    indexed = index_manager.populate_index()
    print(f"Indexed {indexed} vectors")

    # 검색 엔진 초기화
    search_engine = SemanticSearchEngine(storage, embedder)

    # 검색 예제
    print("\n" + "=" * 60)
    print("SEMANTIC SEARCH EXAMPLES")
    print("=" * 60)
    
    # 기본 의미 기반 검색
    query = "deep learning for edge devices"
    print(f"\nQuery: '{query}'")
    print("-" * 40)

    results = search_engine.search(query, limit=5)
    for index, result in enumerate(results, 1):
        print(f"{index}. [r/{result.subreddit}] {result.title[:80]}...")
        print(
            f"   Score: {result.score} | Similarity: {result.similarity_score:.3f} "
            f"| Distance: {result.distance:.3f}"
        )

    # 필터를 적용한 검색
    query = "public data sources for research"
    print(f"\nQuery: '{query}' (filtered to r/datasets, score >= 1)")
    print("-" * 40)

    results = search_engine.search(query, limit=5, subreddits=["datasets"], min_score=1)
    for index, result in enumerate(results, 1):
        print(f"{index}. {result.title[:80]}...")
        print(f"   Score: {result.score} | Date: {result.created_date}")

    # 서브레딧 간 분석
    query = "python best practices"
    print(f"\nCross-subreddit analysis: '{query}'")
    print("-" * 40)

    analysis = search_engine.cross_subreddit_analysis(query)
    for subreddit in analysis["subreddits"][:3]:
        relevance = analysis["relevance_scores"][subreddit]
        print(f"\nr/{subreddit} (relevance: {relevance:.2f}):")
        for result in analysis["posts_by_subreddit"][subreddit][:2]:
            print(f"  - {result.title[:70]}...")

    # 정리
    storage.close()
    print("\nDone!")

if __name__ == "__main__":
    main()
