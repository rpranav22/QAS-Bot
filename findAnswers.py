import gensim
from collections import Counter, OrderedDict
import spacy
import nltk
from nltk.tag import StanfordNERTagger
from nltk.tokenize import word_tokenize



def query2vec(query, dictionary):

    # print("Searching: ", query, dictionary)
    corpus = dictionary.doc2bow(query)
    # print (corpus)

    return corpus


def doc2vec(documents):
    with open('Docs/stop_list.txt', 'r', newline='') as stp_fp:
        stop_list = (stp_fp.read()).lower().split("\n")
    texts = [[word for word in doc.lemma_.split() if word not in stop_list]for doc in documents]

    frequency = Counter()
    for sent in texts:
        for token in sent:
            frequency[token] += 1

    dictionary = gensim.corpora.Dictionary(texts)

    corpus = [dictionary.doc2bow(snipp) for snipp in texts]

    return corpus, dictionary


def transform_vec(corpus, query_corpus):
    lsidf = gensim.models.LsiModel(corpus)

    corpus_lsidf = lsidf[corpus]
    query_lsidf = lsidf[query_corpus]

    return corpus_lsidf, query_lsidf


def similarity(corpus_lsidf, query_lsidf):
    index = gensim.similarities.SparseMatrixSimilarity(corpus_lsidf, num_features=100000)

    simi = index[query_lsidf]

    simi_sorted = sorted(enumerate(simi), key=lambda item: -item[1])
    # print("Rank:")
    # pprint(simi_sorted)
    return simi_sorted


def combine(sub_keys, keywords_splits, lb, mb, ub):
    whitespace = ' '
    while mb != ub:
        keywords_splits.append(whitespace.join(sub_keys[lb: mb]))
        keywords_splits.append(whitespace.join(sub_keys[mb: ub]))
        mb += 1
    del sub_keys[0]
    if len(sub_keys) > 2:
        combine(sub_keys, keywords_splits, 0, 1, len(sub_keys))


def keywords_splitter(keywords, keywords_splits):

    for key in keywords:
        sub_keys = key.split()

        if len(sub_keys) > 2:
            combine(sub_keys, keywords_splits, 0, 1, len(sub_keys))


def pre_query(keywords):


    keywords = [keywords[feat].lower() for feat in range(0, len(keywords))]
    whitespace = ' '
    keywords_splits = whitespace.join(keywords).split()

    keywords_splitter(keywords, keywords_splits)
    keywords_splits = list(set(keywords_splits + keywords))

    return keywords_splits


def retrieveDocs(rankedDocs):
    fp = open('Docs/context.txt', encoding='utf-8')
    documents = fp.readlines()

    processed_documents = ""


    processed_documents += documents[rankedDocs[0]]

    return processed_documents

def extractNER(sentence):
    model = '/Users/pranavr/anaconda/lib/python3.5/site-packages/stanford-ner-2017-06-09/classifiers/english.muc.7class.distsim.crf.ser.gz'
    jar = '/Users/pranavr/anaconda/lib/python3.5/site-packages/stanford-ner-2017-06-09/stanford-ner.jar'

    st = StanfordNERTagger(model, jar, encoding='utf-8')

    tokenized_text = word_tokenize(sentence)
    classified_text = st.tag(tokenized_text)
    taggedWords = []
    for tup in classified_text:
        if (tup[1]!= 'O'):
            taggedWords.append(tup)

    return taggedWords


def possibleAnswers(keywords_query, rankedDocs, en_nlp):

    # print(keywords_query)
    keywords_query = pre_query(keywords_query)
    print(rankedDocs)
    document = retrieveDocs(rankedDocs[0])
    print("Documents: ",document)
    en_doc = en_nlp(u'' + document)

    sentences = list(en_doc.sents)

    corpus, dictionary = doc2vec(sentences)

    query_corpus = query2vec(keywords_query, dictionary)
    # print (query_corpus)

    corpus_lsidf, query_lsidf = transform_vec(corpus, query_corpus)

    simi_sorted = similarity(corpus_lsidf, query_lsidf)

    if len(simi_sorted) > 5:
        simi_sorted = simi_sorted[0:5]

    for sent in simi_sorted:
        sent_id = sent[0]
        # namedEntities = extractNER(str(sentences[sent_id]))
        # print (sent)

    candidate_ans = []
    for sent in simi_sorted:
        # print (sent)
        sent_id = sent[0]
        candidate_ans.append(str(sentences[sent_id]))

    return candidate_ans

def Main():
    nlp = spacy.load("en_core_web_sm")
    rankedDocs = [(0, 0.60192931), (1, 0.0), (2, 0.0)]
    keywords = ['Ashoka University', 'be']
    qclass = ['LOC']

    ans = possibleAnswers(keywords, rankedDocs, qclass, nlp)
    print ("Answer: ", ans[0])


# if __name__ == '__main__':
#     Main()
