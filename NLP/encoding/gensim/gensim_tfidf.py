# Corpus
raw_corpus = ["Human machine interface for lab abc computer applications",
             "A survey of user opinion of computer system response time",
             "The EPS user interface management system",
             "System and human system engineering testing of EPS",
             "Relation of user perceived response time to error measurement",
             "The generation of random binary unordered trees",
             "The intersection graph of paths in trees",
             "Graph minors IV Widths of trees and well quasi ordering",
             "Graph minors A survey"]

print("raw_corpus")
print(raw_corpus)

# Create a set of frequent words
stoplist = set('for a of the and to in'.split(' '))
# Lowercase each document, split it by white space and filter out stopwords
texts = [[word for word in document.lower().split() if word not in stoplist]
         for document in raw_corpus]

# Count word frequencies
from collections import defaultdict
frequency = defaultdict(int)
for text in texts:
    for token in text:
        frequency[token] += 1

# Only keep words that appear more than once
processed_corpus = [[token for token in text if frequency[token] > 1] for text in texts]
print("processed_corpus")
for corpus in processed_corpus:
    print(corpus)


from gensim import corpora

dictionary = corpora.Dictionary(processed_corpus)
print("dictionary")
print(dictionary)
print(dictionary.token2id)


# Vector

bow_corpus = [dictionary.doc2bow(text) for text in processed_corpus]
print("bow_corpus")
print(bow_corpus)

# Model

from gensim import models
# train the model
tfidf = models.TfidfModel(bow_corpus)
# transform the "system minors" string
print("system minors")
print(tfidf[dictionary.doc2bow("system minors".lower().split())])
