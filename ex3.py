
import sys
import numpy as np
from math import log
import math
from datetime import datetime
import multiprocessing
import pickle

# import matplotlib.pyplot as plt

GROUPS = 9
K = 10
EPSILON = 0.001
LAMBDA = 0.06
WORDS_THRESHOLD = 3
STOP_EPSILON = 10  # This is nothing in logaritmic scale

startTime = datetime.now()


def group(lst, n):
    for i in range(0, len(lst), n):
        val = lst[i:i + n]
        if len(val) == n:
            yield tuple(val)


def passedTime():
    return str(datetime.now() - startTime)


def calcPi(i, documents, words, P):
    Pi = {}
    den = sum(map(lambda (wt, t, d, l): wt[i] * l, documents)) + len(words) * LAMBDA
    for word in words:  # k = word
        num = sum(map(lambda (wt, t, d, l): wt[i] * (d[word] if word in d else 0), documents)) + LAMBDA
        Pi[word] = float(num) / den
    P[i] = Pi


def saveGraphs(epochsData):
    files = {
        "likelihood": 0,
        "perplexity": 1
    }

    # for fileName, i in files.items():
    #     plt.figure(i)  # different every time
    #     plt.plot(range(len(epochsData)), [a[i] for a in epochsData])
    #     plt.xlabel('Epochs')
    #     plt.ylabel(fileName)
    #     plt.savefig(fileName + '.png')


class EM:
    def __init__(self, fileName, topicsFile, modelFile):
        self.modelFile = modelFile
        self.alpha = self.P = None
        self.scores = []

        self.documents, self.wordsCounter = self.readFile(fileName)
        print "Parsed documents", len(self.documents), passedTime()
        print "Filtered Vocab Size", len(self.wordsCounter)
        print "----------"
        self.topics = filter(lambda line: line != "", map(lambda l: l.strip(), open(topicsFile).read().split("\n")))
        print self.topics

    def save(self):
        print "Starting save", passedTime()
        model = [self.alpha, self.P, self.documents, self.wordsCounter, self.scores]
        pickle.dump(model, open(self.modelFile, 'wb'))
        saveGraphs(self.scores)
        print "Finished save", passedTime()

    def load(self):
        self.alpha, self.P, self.documents, self.wordsCounter, self.scores = pickle.load(open(self.modelFile, 'rb'))
        print "Loaded from model"
        self.save()
        print self.scores

    def genMatrix(self):
        matrix = [{topic: 0 for topic in self.topics} for i in range(GROUPS)]
        counter = np.zeros(GROUPS)

        for i in range(GROUPS):
            for (wt, t, d, l) in self.documents:
                if np.argmax(wt) == i:
                    counter[i] += 1
                    for topic in t:
                        matrix[i][topic] += 1
                        # matrix[i][t[0]] += 1
        lines = [(str(i) + "," + (",".join(map(lambda (t, c): str(c), row.items())) + "," + str(counter[i])))
                 for i, row in enumerate(matrix)]
        lines.insert(0, "," + ",".join(self.topics))

        fh = open("matrix.csv", "w")
        fh.write("\n".join(lines))
        fh.close()

    def run(self):
        while True:
            if len(self.scores) > 3 and self.scores[-1][0] - self.scores[-3][0] < STOP_EPSILON:
                print "Reached a stasis point, at least 3 epochs"
                break
            self.mStep()
            lh = self.likelihood()
            mp = self.perplexity()
            self.scores.append((lh, mp))

            print "Likelihood\t\t\t\t", lh
            print "Mean Perplexity\t\t\t", mp

            self.eStep()
            self.save()

    def likelihood(self):
        lnL = 0.0
        for (wt, t, d, l) in self.documents:
            z = [(log(self.alpha[i]) + sum(map(lambda (word, count): count * log(self.P[i][word]), d.items())))
                 for i in range(GROUPS)]
            m = max(z)
            allowedIndexes = filter(lambda j: z[j] - m >= -K, range(GROUPS))
            lnL += m + sum(map(lambda i: math.exp(z[i] - m), allowedIndexes))
        return lnL

    def perplexity(self):
        perplexity = 0
        for (wt, t, d, l) in self.documents:
            probability = 0
            c = np.argmax(wt)
            for word, count in d.items():
                pX = (self.P[c][word] * l + LAMBDA) / (l + len(self.wordsCounter) * LAMBDA)
                probability += log(pX) * count
            perplexity += math.exp(probability / -l)

        return perplexity / len(self.documents)

    def eStep(self):
        print "Starting eStep", passedTime()
        for (wt, t, d, l) in self.documents:
            z = [(log(self.alpha[i]) + sum(map(lambda (word, count): count * log(self.P[i][word]), d.items())))
                 for i in range(GROUPS)]
            m = max(z)

            allowedIndexes = filter(lambda j: z[j] - m >= -K, range(GROUPS))
            denominator = sum(map(lambda j: math.exp(z[j] - m), allowedIndexes))

            for i in range(GROUPS):
                wt[i] = 0
            for i in allowedIndexes:
                wt[i] = math.exp(z[i] - m) / denominator

        print "Finished eStep", passedTime()

    def mStep(self):
        print "Starting mStep", passedTime()
        # Distribution of documents
        alpha = [((1.0 / len(self.documents)) * sum(map(lambda (wt, t, d, l): wt[i], self.documents)))
                 for i in range(GROUPS)]
        alpha = [a if a > 0 else EPSILON for a in alpha]  # Don't get to 0
        alphaSum = sum(alpha)
        alpha = [a / alphaSum for a in alpha]  # Make alpha a probability

        manager = multiprocessing.Manager()
        P = manager.dict()

        # Multithread calculating Pi, as it takes a while
        words = self.wordsCounter.keys()
        processes = [multiprocessing.Process(target=calcPi, args=(i, self.documents, words, P))
                     for i in range(GROUPS)]
        for process in processes:
            process.start()
        for process in processes:
            process.join()

        self.alpha = alpha
        self.P = dict(P)

        print "Finished mStep", passedTime()

    def readFile(self, fileName):
        fileData = filter(lambda line: line != "", map(lambda l: l.strip(), open(fileName).read().split("\n")))
        wordsCounter = {}
        documents = []
        for (tags, document) in group(fileData, 2):
            tags = tags.lstrip("<").rstrip(">").split()[1:]
            ordinalIndex = int(tags.pop(0))
            document = document.split()

            docVector = {}
            for word in document:
                if word not in docVector:
                    docVector[word] = 0
                docVector[word] += 1

            for word, count in docVector.items():
                if word not in wordsCounter:
                    wordsCounter[word] = 0
                wordsCounter[word] += count

            association = np.zeros(GROUPS)
            association[ordinalIndex % GROUPS] = 1
            documents.append((association, tags, docVector, len(document)))

        # Apply words filter
        for word, count in wordsCounter.items():
            if count <= WORDS_THRESHOLD:
                del wordsCounter[word]

        for (wt, tags, docVector, l) in documents:
            for word in docVector.keys():
                if word not in wordsCounter:
                    del docVector[word]

        return documents, wordsCounter


if __name__ == "__main__":
    developFile = sys.argv[1]  # "../dataset/develop.txt"
    topicsFile = sys.argv[2]  # "../dataset/topics.txt"
    modelFile = "model"
    em = EM(developFile, topicsFile, modelFile)
    try:
        em.load()
    except:
        print "Couldn't load from file"

    em.run()

    em.genMatrix()