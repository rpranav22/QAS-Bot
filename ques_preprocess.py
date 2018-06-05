
import csv
import en_core_web_md


"""
WH Bi-gram
Root word = Part of Speech
Bi-gram = Part of Speech
"""


def ProcessQuestion(question, qclass, nlp):
    doc = nlp(u'' + question)
    sent_list = list(doc.sents)
    sent = sent_list[0]
    wh_bigram = []
    root_token = ""
    wh_pos = ""
    wh_nbor_pos = ""
    wh_word = ""

    for token in sent:
        if token.tag_ == "WDT" or token.tag_ == "WP" or token.tag_ == "WP$" or token.tag_ == "WRB":
            wh_pos = token.tag_
            wh_word = token.text
            wh_bigram.append(token.text)
            wh_bigram.append(str(doc[token.i + 1]))
            wh_nbor_pos = doc[token.i + 1].tag_
        if token.dep_ == "ROOT":
            root_token = token.tag_

        # print(wh_pos, wh_nbor_pos)
        # print(wh_bigram)
        # print(root_token)

    with open('qclassifier_test_extra.csv', 'a', newline='') as csv_fp:
        csvWrite = csv.writer(csv_fp, delimiter=',')
        csvWrite.writerow([ question, str.lower(wh_word), str.lower(" ".join(wh_bigram)), wh_pos, wh_nbor_pos, root_token, qclass])
        csv_fp.close()


def readInFile(fp, nlp):
    # question = "How did serfdom develop in and then leave Russia ?"
    for line in fp:
        list_line = line.split(" ")
        qclass_list = list_line[0].split(":")
        question = " ".join(list_line[1:len(list_line)])
        question = question.strip("\n")
        qclass = qclass_list[0]
        # print(qclass, question)
        ProcessQuestion(question, qclass, nlp)




def cleanCSVData():

    with open('qclassifier_test_extra.csv', 'w', newline='') as csv_fp:
        csvWrite = csv.writer(csv_fp, delimiter=',')
        csvWrite.writerow(['Question', 'WH', 'WH-Bigram', 'WH-POS', 'WH-NBOR-POS', 'Root-POS', 'Class'])

        csv_fp.close()


def Main():
    cleanCSVData()
    nlp = en_core_web_sm.load()

    with open('Docs/qc/test.txt', 'r', encoding='latin-1') as fp:
        readInFile(fp, nlp)
        fp.close()
        print("CSV Data Trained...")


if __name__ == '__main__':
    Main()
