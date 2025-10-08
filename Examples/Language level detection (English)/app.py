from LexiDecay import LexiDecayModel

data1 = open('A1.txt', encoding='utf-8').read()
data2 = open('C1.txt', encoding='utf-8').read()


model = LexiDecayModel()
model.add_category("A1", data1)
model.add_category("C1", data2)



print('Artificial intelligence model based on LexiDecay for language level detection (A1 / C1 only) *Naturally, the results of artificial intelligence may not be completely accurate and should not be relied upon.')

input_text = str(input("Enter English Text: "))


out = model.classify(
    input_text,
    decay=0.4,
    use_idf=True,
    auto_common_reduce=True,
    common_decay=0.8,
    min_common_mult=0.08
)



print("Top category:", out["top"])
print("All probabilities:")
for k, v in out["probs"].items():
    print(f"  {k}: {v:.4f}")




    #model.save_model("lexidecay_model.pkl")
    #m2 = LexiDecayModel.load_model("lexidecay_model.pkl")

    # m2.add_category("side3", "some new texts")
    # m2.save_model("lexidecay_model_v2.pkl")