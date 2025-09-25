import torch
from train_qabot import TransformerLM, stoi, itos, encode, decode, device

# 모델 로드
model = TransformerLM(len(stoi)).to(device)
model.load_state_dict(torch.load("qabot.pt", map_location=device))
model.eval()

def ask(question: str):
    start = "Q: " + question + "\nA:"
    x = torch.tensor([encode(start)], dtype=torch.long, device=device)
    out = model.generate(x, max_new_tokens=100)[0].tolist()
    return decode(out)

while True:
    q = input("너: ")
    if q.strip().lower() in ["quit","exit"]: break
    print("봇:", ask(q))
