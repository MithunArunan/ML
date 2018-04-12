from gensim.similarities import WmdSimilarity
import gensim

model = gensim.models.Word2Vec.load("word2vec_1.pkl")

wmd_corpus = ["Computer is now working", "server outage"]

# Create a set of frequent words
stoplist = set('for a of the and to in'.split(' '))
# Lowercase each document, split it by white space and filter out stopwords
wmd_texts = [[word for word in document.lower().split() if word not in stoplist]
         for document in wmd_corpus]
print("WMD corpuses")
print(wmd_texts)

query = "computer is working"
print("query")
print(query)

wmd_instance = WmdSimilarity(wmd_texts, model, num_best=3)
print("WMD similarity")
print(wmd_instance[query.split(" ")])
