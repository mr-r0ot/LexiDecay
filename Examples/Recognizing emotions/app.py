from LexiDecay import LexiDecayModel

ANGER = open('ANGER.txt', encoding='utf-8').read()
FEAR = open('FEAR.txt', encoding='utf-8').read()
JOY = open('JOY.txt', encoding='utf-8').read()
LOVE = open('LOVE.txt', encoding='utf-8').read()
SADNESS = open('SADNESS.txt', encoding='utf-8').read()


model = LexiDecayModel()
model.add_category("ANGER", ANGER)
model.add_category("FEAR", FEAR)
model.add_category("JOY", JOY)
model.add_category("LOVE", LOVE)
model.add_category("SADNESS", SADNESS)



print('Artificial intelligence model based on LexiDecay for Recognizing emotions *Naturally, the results of artificial intelligence may not be completely accurate and should not be relied upon.')

input_text = str(input("Enter English Text: "))


out = model.classify(
    input_text,
    decay=0.4,
    use_idf=True,
    auto_common_reduce=True,
    common_decay=0.1,
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