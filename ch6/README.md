# 제6장 SQLite VSS와 Ollama로 검색 증강 생성 시스템 구축하기

SQLite-VSS 벡터 검색과 Ollama LLM 추론을 결합한 로컬·프라이빗 RAG 시스템입니다.

## 사전 준비 사항

1. **sqlite-vss 바이너리** — https://github.com/asg017/sqlite-vss/releases
   - vector0.so와 vss0.so 파일을 app.py와 같은 디렉터리에 둡니다.
   - Windows는 공식적으로 지원되지 않으므로 WSL2 사용을 권장합니다.

2. **Ollama** — https://ollama.ai
   ```bash
   ollama serve          # Ollama 서버를 시작합니다.
   ollama pull llama3.1:8b  ollama pull llama3.1:8b
   ```

## 설정

```bash
python -m venv ch6_env
source ch6_env/bin/activate
pip install -r requirements.txt
```

## 실행

```bash
python app.py
```

샘플 데이터를 불러온 뒤, 데모 질문을 RAG 파이프라인으로 실행하고 이어서 대화형 질의응답 모드로 진입합니다.

## 아키텍처

1. 벡터 저장에는 SQLite와 VSS 확장을 사용합니다. 임베딩은 all-MiniLM-L6-v2 모델의 384차원 벡터를 사용합니다.
2. 키워드 검색에는 FTS5를 사용하며, BM25 점수 산정 방식을 적용합니다.
3. 하이브리드 검색은 의미 기반 검색 70%, 키워드 검색 30%의 비율로 결합합니다. 두 검색 결과는 정규화한 뒤 병합합니다.
4. 답변 생성에는 Ollama의 llama3.1:8b 모델을 사용하며, 낮은 temperature 값인 0.1을 적용합니다.
