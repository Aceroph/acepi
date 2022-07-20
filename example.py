from acepi import DeepLearning

# load intents
data = DeepLearning.intents("intents.json")

# model vocabulary
words, labels, training, output = DeepLearning.vocabulary(data, "data.pickle")

# model object
model = DeepLearning.brain(training, output)

# test model accuracy
print(DeepLearning.accuracy(model, training, output))

# chatbot
while True:
    inp = input("You : ")
    if inp == "quit":
        break
    
    answer = DeepLearning.chat(inp, model, labels, words, data)
    print("Bot :", answer)