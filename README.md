# Transformer Q/A Chatbot (학습용 프로젝트)

이 프로젝트는 **Transformer 구조를 직접 구현**해서  
간단한 Q/A 데이터셋을 학습시키고 챗봇처럼 질의응답을 해보는 실습용 예제입니다.

---

## 🚀 실행 방법

### 1. 환경 세팅
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

2. 학습
python train_qabot_words.py
→ 학습이 끝나면 qabot_words.pt 모델 파일이 생성됩니다.
3. 대화
python chat_words.py
💬 실행 예시
(venv) lyune@MacBook-Pro transformer_qabot % python chat_words.py
너: 트랜스포머가 뭐야?
봇: q : 트랜스포머가 뭐야 ? a : 어텐션을 이용해 문맥을 한꺼번에 보는 딥러닝 모델이야 . q : 트랜스포머의 장점은 ? a : 병렬 학습이 가능하고 긴 의존 관계를 잘 학습할 수 있어 . q : 트랜스포머의 단점은
너: GPT는 뭐야?
봇: q : gpt 는 뭐야 ? a : 트랜스포머 디코더 기반의 언어 생성 모델이야 .
📂 파일 설명
data.txt : Q/A 학습 데이터
my_tokenizer.py : 단어 단위 토큰화 함수
train_qabot_words.py : Transformer 학습 스크립트
chat_words.py : 학습된 모델을 불러와 질의응답 실행
requirements.txt : 필요한 파이썬 라이브러리
README.md : 프로젝트 소개 문서
.gitignore : 업로드 제외 규칙 (venv, .env, *.pt 등)
⚠️ 주의사항
이 프로젝트는 학습/실습용입니다.
데이터셋이 작아서 GPT 같은 성능은 기대하기 어렵습니다.
