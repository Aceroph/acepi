from acepi import DeepLearning as DL

intents = DL.intents("intents.json")

words, labels, training, output = DL.vocabulary(intents, "data.pickle")

model = DL.brain(training, output)

while True:
    inp = input("You : ")

    if inp.lower() == "quit":
        break
    elif inp.lower() == "acc" or inp.lower() == "accuracy":
        score = DL.accuracy(model, training, output)
        print("Accuracy of", score)
    elif inp.lower() == "train":
        DL.train(model, training, output)
    else:
        answer = DL.chat(inp, model, labels, words, intents)
        print("Bot :", answer)
