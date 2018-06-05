import gensim
import re
from collections import Counter, OrderedDict
from pprint import pprint


def query2vec(query, dictionary):
    # print ('going into query2vec', query, dictionary)
    corpus = dictionary.doc2bow(query)  # HERE LIES THE ERROR THAT NEEDS TO BE LOOKED AT
    # print("Q:")
    # print(corpus)

    return corpus


def doc2vec(documents):
    # print ('going into doc2vec')
    with open('Docs/stop_list.txt',
              'r', newline='') as stp_fp:
        stop_list = (stp_fp.read()).lower().split("\n")
    texts = [[word for word in doc.lower().split() if word not in stop_list] for doc in documents]
    frequency = Counter()
    for sent in texts:
        # print (sent)
        for token in sent:
            frequency[token] += 1

    # texts = [[token for token in snipp if frequency[token] > 1]for snipp in texts]
    # print(texts)

    dictionary = gensim.corpora.Dictionary(texts)
    # print(dictionary)
    # print(dictionary.token2id)

    corpus = [dictionary.doc2bow(snipp) for snipp in texts]
    # print("C:")
    # print(corpus)

    return corpus, dictionary


def transform_vec(corpus, query_corpus):
    tfidf = gensim.models.TfidfModel(corpus)
    # print (corpus)
    corpus_tfidf = tfidf[corpus]
    # print ("qcorp: ", query_corpus)
    query_tfidf = tfidf[query_corpus]

    # for doc in corpus_tfidf:
    #     print("C:", doc)
    # for doc in query_tfidf:
    #     print("Q:", doc)

    return corpus_tfidf, query_tfidf


def similarity(corpus_tfidf, query_tfidf):
    index = gensim.similarities.SparseMatrixSimilarity(corpus_tfidf, num_features=10000)

    simi = index[query_tfidf]

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
    # print("Keywords: ", keywords)
    keywords = [keywords[feat].lower() for feat in range(0, len(keywords))]
    whitespace = ' '
    keywords_splits = whitespace.join(keywords).split()

    keywords_splitter(keywords, keywords_splits)
    keywords_splits = list(set(keywords_splits + keywords))

    return keywords_splits


def scoreDocs(documents, keywords):
    keywords = pre_query(keywords)
    # print (keywords)
    corpus, dictionary = doc2vec(documents)
    query_corpus = query2vec(keywords, dictionary)
    # print ("qcorp is: ", query_corpus)
    # print ("corp is: ", corpus)
    corpus_tfidf, query_tfidf = transform_vec(corpus, query_corpus)
    # print (corpus_tfidf)
    # print (query_tfidf)

    simi_sorted = similarity(corpus_tfidf, query_tfidf)
    # print ('simi: ', simi_sorted)

    if len(simi_sorted) > 3:
        return simi_sorted[0:3]
    else:
        return simi_sorted


def rankDocs(keywords):
    fp = open('Docs/context.txt', encoding='utf-8')
    documents = fp.readlines()

    ranked_docs = scoreDocs(documents, keywords)

    # pprint(ranked_docs)

    return ranked_docs


def Main():
    features = ['Ashoka University', 'be']

    docRank = rankDocs(features)
    print(docRank)


if __name__ == '__main__':
    Main()
