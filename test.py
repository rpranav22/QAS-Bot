import nltk

from nltk.tag import StanfordNERTagger
from nltk.tokenize import word_tokenize


with open('sample.txt', 'r') as f:
    sample = f.read()





model = '/Users/pranavr/anaconda/lib/python3.5/site-packages/stanford-ner-2017-06-09/classifiers/english.muc.7class.distsim.crf.ser.gz'
jar = '/Users/pranavr/anaconda/lib/python3.5/site-packages/stanford-ner-2017-06-09/stanford-ner.jar'

st = StanfordNERTagger(model, jar, encoding='utf-8')

# text = 'Ashoka University is a private research university with a focus on liberal arts, located in Sonipat, Haryana, India.'
#
# tokenized_text = word_tokenize(text)
# classified_text = st.tag(tokenized_text)

sentences = nltk.sent_tokenize(sample)
tokenized_sentences = [nltk.word_tokenize(sentence) for sentence in sentences]
tagged_sentences = [st.tag(sentence) for sentence in tokenized_sentences]

for sentence in tagged_sentences:
    print (sentence)

