from sklearn import datasets
from sklearn.multiclass import OneVsRestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import LinearSVC
from sklearn.externals import joblib
from scipy.sparse import csr_matrix
import pandas
import spacy
import en_core_web_sm
from time import time


def preProcess(dta):
    return pandas.get_dummies(dta)


def transform_data_matrix(X_train, X_predict):
    X_train_columns = list(X_train.columns)
    X_predict_columns = list(X_predict.columns)

    X_trans_columns = list(set(X_train_columns + X_predict_columns))
    # print(X_trans_columns, len(X_trans_columns))

    trans_data_train = {}

    for col in X_trans_columns:
        if col not in X_train:
            trans_data_train[col] = [0 for i in range(len(X_train.index))]
        else:
            trans_data_train[col] = list(X_train[col])

    XT_train = pandas.DataFrame(trans_data_train)
    XT_train = csr_matrix(XT_train)
    # getDataInfo(XT_train)

    trans_data_predict = {}

    for col in X_trans_columns:
        if col not in X_predict:
            trans_data_predict[col] = 0
        else:
            trans_data_predict[col] = list(X_predict[col])  # KeyError

    XT_predict = pandas.DataFrame(trans_data_predict)
    XT_predict = csr_matrix(XT_predict)
    # getDataInfo(XT_predict)

    return XT_train, XT_predict

# print results of training model
def SVM(X_train, y, X_predict):
    lin_clf = LinearSVC()
    lin_clf.fit(X_train, y)
    print("Model score: {0}".format(lin_clf.score(X_train, y)))
    # joblib.dump(lin_clf, 'lsvc.pkl')
    prediction = lin_clf.predict(X_predict)
    return prediction


def quesPredictionData(doc):
    sent_list = list(doc.sents)
    sent = sent_list[0]
    wh_bi_gram = []
    root_token = ""
    wh_pos = ""
    wh_nbor_pos = ""
    wh_word = ""
    for token in sent:
        if token.tag_ == "WDT" or token.tag_ == "WP" or token.tag_ == "WP$" or token.tag_ == "WRB":
            wh_pos = token.tag_
            wh_word = token.text
            wh_bi_gram.append(token.text)
            wh_bi_gram.append(str(doc[token.i + 1]))
            wh_nbor_pos = doc[token.i + 1].tag_
        if token.dep_ == "ROOT":
            root_token = token.tag_
    qdata_frame = [{'WH':wh_word, 'WH-POS':wh_pos, 'WH-NBOR-POS':wh_nbor_pos, 'Root-POS':root_token}]
    # qdata_list = [wh_word, wh_pos, wh_nbor_pos, root_token]
    # dta = pandas.DataFrame(qdata_list, columns=column_list)
    dta = pandas.DataFrame(qdata_frame)
    return dta


def classifyQuestion(doc):

    dta = pandas.read_csv('qclassifier_trainer_extra.csv', sep=',')

    y = dta.pop('Class')
    # dta.pop('WH')
    # dta.pop('#Question')
    # dta.pop('WH-Bigram')

    X_train = preProcess(dta)

    question_data = quesPredictionData(doc)
    X_predict = preProcess(question_data)

    X_train, X_predict = transform_data_matrix(X_train, X_predict)

    return str(SVM(X_train, y, X_predict))


def Main():
    start = time()
    # nlp = spacy.load("en_core_web_md")
    nlp = en_core_web_sm.load()



    # question = 'How many pounds in kgs?'

    question = input("Question to be classified: ")
    doc = nlp(u'' + question)

    # clf = joblib.load('lsvc.pkl')
    print('')
    print(question)
    print(classifyQuestion(doc))

    end = time()
    print("Total time :", end - start)
    print('')


if __name__ == '__main__':
    Main()
