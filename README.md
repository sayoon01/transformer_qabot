
# Transformer Q/A Chatbot (학습용)

직접 구현한 **Transformer 블록**으로 작은 Q/A 데이터셋을 학습시키고, 간단한 질의응답을 수행하는 예제입니다.  
목표: “작게 만들어서 동작 원리를 이해”하는 것.

---

## 📂 디렉토리 구조 

transformer_qabot

├── data.txt # Q/A 학습 데이터
├── my_tokenizer.py # 단어 단위 토크나이저(표준 tokenize 충돌 방지)
├── train_qabot_words.py # 단어 단위 Q/A 학습 스크립트(Transformer)
├── chat_words.py # 학습된 모델로 질의응답 실행
├── requirements.txt # 의존성 (torch, numpy)
├── README.md # 이 문서
├── .gitignore # venv, .env, *.pt 등 제외
├── qabot_words.pt # (학습 후 생성) 단어 단위 모델 가중치
├── train_qabot.py # (옵션) 문자 단위 학습 스크립트
├── chat.py # (옵션) 문자 단위 생성 실행
└── pycache/ , venv/ # (자동 생성 / 업로드 제외 대상)

> 참고: `qabot_words.pt`는 학습이 끝나면 생성됩니다.

---
🚀 실행
1) 학습
python train_qabot_words.py
종료 시 qabot_words.pt 저장
2) 대화
python chat_words.py
입력: 평문 질문
내부 입력 형식: "q: {질문} a:"로 프롬프트 구성 후 생성
출력 길이 조절: chat_words.py의 max_new_tokens 값(기본 30)을 80~100 등으로 늘리면 더 김.

📈 학습 로그 예시
실행 중 콘솔 출력(예시):
step 0 loss 3.9079
step 50 loss 0.8625
step 100 loss 0.3361
step 150 loss 0.1027
step 200 loss 0.0647
step 250 loss 0.0462
step 300 loss 0.0489
step 350 loss 0.0426
step 400 loss 0.0465
step 450 loss 0.0540
손실이 빠르게 떨어지며, 작은 데이터셋에서는 거의 “암기”에 가까운 패턴 학습이 일어남.
과적합/진동이 보이면 dropout을 0.1→0.2로 높이거나 max_iters를 조절.

💬 실행 예시
너: 트랜스포머가 뭐야?
봇: q : 트랜스포머가 뭐야 ? a : 어텐션을 이용해 문맥을 한꺼번에 보는 딥러닝 모델이야 . q : 트랜스포머의 장점은 ? a : 병렬 학습이 가능하고 긴 의존 관계를 잘 학습할 수 있어 . q : 트랜스포머의 단점은
뒤에 다른 Q/A까지 이어지는 이유:
데이터가 매우 작아 패턴을 통째로 생성하는 경향.
max_new_tokens가 짧아 중간에 잘릴 수 있음.
개선 팁:
max_new_tokens를 80~100으로 늘림.
data.txt에서 Q:/A: 접두어를 제거해 “답변만” 나오도록 유도.
Q/A 쌍을 더 추가해서 패턴 일반화.




예시 입력:
너: 트랜스포머가 뭐야?
너: GPT는 뭐야?
종료: exit / quit 또는 Ctrl+C / Ctrl+D

## 실행 화면
![실행 화면](docs/images/transformer_demo.png)
'''
      
