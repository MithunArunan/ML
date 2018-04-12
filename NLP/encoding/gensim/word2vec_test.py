import gensim

model = gensim.models.Word2Vec.load("word2vec_1.pkl")
print(model.wv['computer'])
