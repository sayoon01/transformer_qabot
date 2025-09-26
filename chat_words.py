# chat_words.py
import re
import torch
from train_qabot_words import TransformerLM, encode, decode, stoi, itos, vocab, device

STOP_SEQ = ["q", ":"]
END_PUNCT = {".", "?", "!"}

model = TransformerLM(len(vocab)).to(device)
model.load_state_dict(torch.load("qabot_words.pt", map_location=device))
model.eval()

def ask(question: str):
    start = "q: " + question + " a:"
    x = torch.tensor([encode(start)], dtype=torch.long, device=device)

    block_size = 16  
    max_new_tokens = 80
    min_tokens = 10  # 너무 일찍 끊기지 않도록 최소 생성 길이

    with torch.no_grad():
        last_end_punct_at = None
        for t in range(max_new_tokens):
            logits, _ = model(x[:, -block_size:])
            # 확률 샘플링 유지(매번 다른 결과 OK)
            probs = torch.softmax(logits[:, -1, :], dim=-1)
            next_id = torch.multinomial(probs, 1)
            x = torch.cat([x, next_id], dim=1)

            # ----- 중단 규칙 -----
            gen_ids = x[0].tolist()
            gen_tokens = [itos[i] for i in gen_ids]

            # ① 'q :' 패턴 나오면 멈춤 (다음 질문으로 넘어가려는 신호)
            if len(gen_tokens) >= 2 and gen_tokens[-2:] == STOP_SEQ and t >= min_tokens:
                # 'q :' 직전까지의 답변만 남김
                gen_tokens = gen_tokens[:-2]
                x = torch.tensor([[stoi[w] for w in gen_tokens]], dtype=torch.long, device=device)
                break

            # ② 문장 완결 멈춤: 마침표/물음표/느낌표 뒤 1~2토큰 생성 후 정지
            if gen_tokens[-1] in END_PUNCT:
                last_end_punct_at = len(gen_tokens) - 1
            if last_end_punct_at is not None and len(gen_tokens) - last_end_punct_at >= 2 and t >= min_tokens:
                break

    text = decode(x[0].tolist())
    # 프롬프트(starts with "q: ... a:") 제거
    text = re.sub(r"^\s*q\s*:\s*.*?\s*a\s*:\s*", "", text, flags=re.IGNORECASE)
    # 혹시 남은 q: 제거
    text = re.split(r"\bq\s*:\s*", text, flags=re.IGNORECASE)[0]
    return text.strip()

if __name__ == "__main__":
    while True:
        try:
            q = input("너: ")
        except (EOFError, KeyboardInterrupt):
            print("\n종료")
            break
        if q.strip().lower() in ["quit", "exit"]:
            break
        print("봇:", ask(q))

