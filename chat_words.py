import torch
from train_qabot_words import TransformerLM, encode, decode, stoi, vocab, device

# 모델 불러오기
model = TransformerLM(len(vocab)).to(device)
model.load_state_dict(torch.load("qabot_words.pt", map_location=device))
model.eval()

def ask(question: str):
    start = "q: " + question.lower() + " a:"
    x = torch.tensor([encode(start)], dtype=torch.long, device=device)
    out = model.generate(x, max_new_tokens=30)[0].tolist()
    return decode(out)

while True:
    q = input("너: ")
    if q.strip().lower() in ["quit","exit"]: break
    print("봇:", ask(q))
