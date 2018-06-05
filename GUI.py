from tkinter import Tk, Label, Button, Entry
import spacy
import en_core_web_sm
from time import time

from model_classification import classifyQuestion
from docScorer import rankDocs
from featureExtraction import extractFeatures
from findAnswers import possibleAnswers


class QASystem:
    def __init__(self, master):
        self.master = master
        master.title("Question-Answering System")
        master.minsize(width=1000, height=500)
        master.maxsize(width=1000,height=500)

        self.topic = Label(master, text= "Documents: \n 1.Ashoka University \n 2.India \n 3.Cristiano Ronaldo \n 4.Cushman and Wakefield", justify= "left",font=("Times New Roman", 20, "bold", ))
        self.topic.pack()

        self.label = Label(master, text="Enter Your Question:", font=("Times New Roman", 20, "bold", ))
        self.label.pack()
        self.label1 = Label(master, text=" ", font=("Times New Roman", 20, "bold",))
        self.label1.pack()
        self.entry = Entry(master)
        self.entry.pack()

        def res():
            question = self.entry.get()

            startTime = time()
            answer = answerQues(question)
            endTime = time()

            totalTime = endTime - startTime
            print ("Total Time Taken: ", totalTime)
            print ("Answer: ", answer[0])
            self.w1.config(text= "Total Time Taken: " + str(totalTime) + " seconds. \n" + "Answer: "+ answer[0])

        def load():

            self.label1.config(text="Loading...")
            self.label1.update_idletasks()
            res()
            print ("end sleep")
            self.label1.config(text=" ")

        self.b = Button(master, text='Submit', command=load, relief = 'raised')
        self.b.pack()
        self.w1 = Label(master, text=" ", font=("Times New Roman", 20), justify='left',wraplength= 1000)
        self.w1.pack()


def answerQues(question):
    nlp = en_core_web_sm.load()
    doc = nlp(u'' + question)

    qclass = classifyQuestion(doc)
    keywords = extractFeatures(doc)
    docRank = rankDocs(keywords)
    answers = possibleAnswers(keywords, docRank, nlp)
    return answers

def Main():
    root = Tk()
    gui = QASystem(root)
    root.mainloop()


if __name__ == '__main__':
    Main()
