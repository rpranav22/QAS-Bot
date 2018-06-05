import spacy
import en_core_web_sm
from time import time

from model_classification import classifyQuestion
from docScorer import rankDocs
from featureExtraction import extractFeatures
from findAnswers import possibleAnswers


def answerQues(doc, nlp):

    startTime = time()
    qclass = classifyQuestion(doc)
    print ("\nQclass: ", qclass)
    endTime = time()
    totalTime = endTime - startTime
    print ("Classification Total Time Taken: ", totalTime)


    startTime = time()
    keywords = extractFeatures(doc)
    print ("\nKeywords: ", keywords)
    endTime = time()
    totalTime = endTime - startTime
    print ("Feature exraction Total Time Taken: ", totalTime)

    startTime = time()
    docRank = rankDocs(keywords)
    print ("\nDocRank: ", docRank[:3])
    endTime = time()
    totalTime = endTime - startTime
    print ("Doc scoring Total Time Taken: ", totalTime)

    startTime = time()
    answers = possibleAnswers(keywords, docRank, nlp)
    endTime = time()
    totalTime = endTime - startTime
    print ("\nFinding Answers Total Time Taken: ", totalTime)

    return answers


def Main():
    startTime = time()
    nlp = en_core_web_sm.load()
    endTime = time()
    totalTime = endTime - startTime
    print ("NLP Loading Total Time Taken: ", totalTime)

    question = input("\nInput your question: ")
    print ("Question: ", question)
    doc = nlp(u'' + question)

    startTime = time()
    answer = answerQues(doc, nlp)
    endTime = time()
    totalTime = endTime - startTime
    print ("\n\nOverall Total Time Taken: ", totalTime)
    print ("Answer: ", answer[0])


if __name__ == '__main__':
    Main()
