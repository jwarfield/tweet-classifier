import nltk
import numpy as np
import csv
import sys
import sklearn
from sklearn.ensemble import RandomForestClassifier
from nltk.tokenize import TweetTokenizer
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import re

#[writeData preds filename] takes in a 1-D array of prediction labels
#either -1 or 1 and a filename string
#and writes a .csv file in format as follows:
#Notes: no spaces, all data are integers, Label column in the preds
#in order, ID is the number of the sample
#   ID,Label
#   0,1
#   1,-1
#   etc.
def writeData(preds, filename):
    f = open(filename,'w')
    writer = csv.writer(f, delimiter=',')
    count = 0
    writer.writerow(['ID','Label'])
    for p in range(len(preds)):
        writer.writerow([count,int(preds[p])])
        count += 1


#myToken takes a string and tokenizes the words into substrings
#we deem valuable
def myToken(text):
    tkn = TweetTokenizer(preserve_case=False, reduce_len=True,
    strip_handles=False)
    text = re.sub(r"https:\/\/[^\s]*","https", text)
    text = re.sub(r"@[^\s]*","@", text)
    text = re.sub(r"#[^\s]*","#", text)
    r = tkn.tokenize(text)
    return r

#[bagWords lst] returns data features of a list of tokenized twitter
#text
def bagWords(s):
    vocab = ['badly','crazy','weak','spent','talking','strong','joke',
    'senate','dumb','dead','brexit','ago','treated','temperament','guns','funny',
    'divided','correct', 'best', 'highest', 'most', 'left', 'media', 'huge',
    'believe','incredible','fake', 'tremendous', 'lot',
    #above is Trump stuff, below is Staffer words
    '@', 'https','#']
    vectorizer = TfidfVectorizer(tokenizer = myToken, vocabulary = vocab)
    xTr = vectorizer.fit_transform(s)
    return xTr

#[getData filename] takes in a csv filename string
#and returns training data and training labels
def getData(filenameIn, isTrainData):
    labels = []
    texts = []
    firstLine = True
    f = open(filenameIn, 'r')
    reader = csv.reader(f)
    for line in reader:
        if firstLine:
            firstLine = False
        else:
            #texts is list of strings
            texts.append(line[1])
            #labels is list of 1 or 0
            if isTrainData:
                labels.append(int(line[17]))
    f.close()

    #yTr is defined
    yTr = np.array(labels, dtype=int)
    #xTr is defined
    xTr = bagWords(texts)
    return xTr, yTr

def getPreds(xTe, clf):
    preds = []
    n,d = xTe.shape
    for i in range(n):
        preds.append(clf.predict(xTe[i,:]))
    return preds

def main(argv):
    if len(argv) > 0 and argv[0] == 'validate':
        trainData = 'doubletrain.csv'
        testData = 'validation.csv'
        outFile = 'outputValidation.csv'
    else:
        trainData = 'train.csv'
        testData = 'test.csv'
        outFile = 'outputTest.csv'

    xTr, yTr = getData(trainData, True)
    clf = RandomForestClassifier(n_estimators = 10)
    clf = clf.fit(xTr, yTr)
    xTe, _ = getData(testData, False)
    preds = getPreds(xTe,clf)
    #writes out data
    writeData(preds,outFile)


if __name__ == "__main__":
   main(sys.argv[1:])
