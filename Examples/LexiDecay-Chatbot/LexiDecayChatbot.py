from LexiDecay import LexiDecayModel


# open data file
data = open('data.txt', encoding='utf-8').read()



# Create a LexiDecay
model = LexiDecayModel()

# set one token
f_pro='hi!, my name is LexiDecay Chatbot, Im fast! and i coded by Mohammad Taha Gorji :)'
model.add_category(f_pro,f_pro)


# Set new tokens by user
try:
    D = str(input('Do you want set system prompt?(Y/n): '))
    if 'y' in D.lower():
        sysp=str(input("Enter system prompt(split with ,): "))
        for p in sysp.split(','):
            model.add_category(p,p)
            print(f'add {p}')
except:
    print("Ok. we have problem!")




# Convert txt to token
n=0
for line in data.split('.'):
    model.add_category(line, line) # add tokens to model
    n+=1

model.save_model("lexidecay_model_chatbot.pkl")
print(f"=====load {n} token=====\n\n")



try:
    creativity = float(input("Enter creativity (from 0.0 to 1.0): "))
except:
    print('Ok. we have problem!, Model Use 0.9 creativity(decay)')
    creativity = 0.9




while True:
    user_prompt = str(input("  [#] User Prompt: ")) # get user prompt

    out = model.classify(
        user_prompt,
        decay=creativity,
        use_idf=True,
        auto_common_reduce=True,
        common_decay=0.6,
        min_common_mult=0.08
    )
    print("\n ====Model output====")
    print("Model Top token:\n", out["top"])
    print(" ====================\n")




#print("All probabilities:")
#for k, v in out["probs"].items():
#    input(f"  {k}: {v:.4f}")

# m2.load_model("lexidecay_model_chatbot.pkl")