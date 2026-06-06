# 제4장 SQLite3를 이용한 의미 기반 검색

SQLite-VSS 벡터 검색을 활용해 Reddit 개인 지식 베이스를 구축합니다.

## 사전 준비 사항

1. **sqlite-vss 바이너리** — 다음 주소에서 다운로드합니다.
   - 다음 주소에서 다운로드합니다.
     - https://github.com/asg017/sqlite-vss/releases
   - vector0.so와 vss0.so 파일을 추출합니다. 운영체제별 확장자 형식은 다음과 같습니다.
     - Linux: .so
     - macOS: .dylib
     - Windows: .dll
   - 추출한 파일은 app.py와 같은 디렉터리에 두거나, EXTENSION_PATH를 설정합니다.
   - **Note**: sqlite-vss는 Windows를 공식적으로 지원하지 않습니다. Windows에서는 WSL2를 사용하시기 바랍니다.

2. **Kaggle Reddit 댓글 데이터셋**
   — "the-reddit-dataset-dataset-comments.csv" 파일은 app.py와 같은 디렉터리에 둡니다.


## 설정

```bash
python -m venv ch4_env
source ch4_env/bin/activate
pip install -r requirements.txt
```

## 구성

해당 없음

## Run

```bash
# 먼저 sqlite-vss가 올바르게 설치되었는지 확인합니다.
python -c "from app import verify_vss_installation; print(verify_vss_installation())"

# 전체 파이프라인을 실행합니다.
python app.py
```

## 수행 작업

1. Kaggle Reddit 댓글 데이터셋을 통해 게시물을 가져옵니다.
2. 마크다운, URL, Reddit 관련 불필요한 내용 등을 정리하고 텍스트를 전처리합니다.
3. all-MiniLM-L6-v2 모델로 임베딩을 생성합니다.
4. SQLite에 데이터를 저장하고 VSS 벡터 인덱스를 구성합니다.
5. 메타데이터 필터링과 함께 의미 기반 검색을 수행합니다.
6. 여러 서브레딧을 아우르는 분석과 유사 게시물 탐색을 지원합니다.
