import os
import nltk
from nltk.stem.lancaster import LancasterStemmer
stemmer = LancasterStemmer()
nltk.download('punkt')

import numpy
import tflearn
import tensorflow

import random
import json
import pickle

class DeepLearning:
    def __init__(self):
        pass

    def intents(filepath: str):
        with open(filepath) as file:
            data = json.load(file)
            return data

    def vocabulary(datajson, picklefilepath: str):
        try:
            with open(picklefilepath, "rb") as f:
                words, labels, training, output = pickle.load(f)
        except:
            words = []
            labels = []
            docs_x = []
            docs_y = []

            for intent in datajson["intents"]:
                for pattern in intent["patterns"]:
                    wrds = nltk.word_tokenize(pattern)
                    words.extend(wrds)
                    docs_x.append(wrds)
                    docs_y.append(intent["tag"])
                    
                if intent["tag"] not in labels:
                    labels.append(intent["tag"])

            words = [stemmer.stem(w.lower()) for w in words if w != "?"]
            words = sorted(list(set(words)))

            labels = sorted(labels)

            training = []
            output = []

            out_empty = [0 for _ in range(len(labels))]

            for x, doc in enumerate(docs_x):
                bag = []
                
                wrds = [stemmer.stem(w) for w in doc]
                
                for w in words:
                    if w in wrds:
                        bag.append(1)
                    else:
                        bag.append(0)
                        
                output_row = out_empty[:]
                output_row[labels.index(docs_y[x])] = 1
                
                training.append(bag)
                output.append(output_row)

            training = numpy.array(training)
            output = numpy.array(output)
            
            with open(picklefilepath, "wb") as f:
                pickle.dump((words, labels, training, output), f)
        
        return words, labels, training, output

    def brain(x, y):
        tensorflow.compat.v1.reset_default_graph()

        net = tflearn.input_data(shape=[None, len(x[0])])
        net = tflearn.fully_connected(net, 8)
        net = tflearn.fully_connected(net, 8)
        net = tflearn.fully_connected(net, len(y[0]), activation="softmax")
        net = tflearn.regression(net)

        model = tflearn.DNN(net)

        if os.path.exists("Model.tflearn.index") or os.path.exists("Model.tflearn.meta"):
            model.load("Model.tflearn")
        else:
            model.fit(x, y, n_epoch=1000, batch_size=8, show_metric=True)
            model.save("Model.tflearn")
        
        return model
    
    def train(model: tflearn.DNN, x, y):
        model.fit(x, y, n_epoch=1000, batch_size=8, show_metric=True)
        model.save("Model.tflearn")
    
    def chat(input: str, model: tflearn.DNN, labels, words, datajson):
        bag = [0 for _ in range(len(words))]
        
        s_words = nltk.word_tokenize(input)
        s_words = [stemmer.stem(word.lower()) for word in s_words]
        
        for se in s_words:
            for i, w in enumerate(words):
                if w == se:
                    bag[i] = 1

        results = model.predict([numpy.array(bag)])
        results_index = numpy.argmax(results)
        tag = labels[results_index]
        
        for tg in datajson["intents"]:
            if tg['tag'] == tag:
                responses = tg['responses']
        
        return random.choice(responses)

    def accuracy(model: tflearn.DNN, test_x, test_y):
        score = model.evaluate(test_x, test_y)
        return '%0.4f%%' % (score[0] * 100)
