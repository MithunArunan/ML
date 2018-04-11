# Reference: http://scikit-learn.org/stable/tutorial/text_analytics/working_with_text_data.html#working-with-text-data
# Imports data set http://qwone.com/~jason/20Newsgroups/
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.naive_bayes import MultinomialNB

news_categories = ['alt.atheism', 'soc.religion.christian', 'comp.graphics', 'sci.med']

# Loads the dataset for specific categories only
news_train_dataset = fetch_20newsgroups(subset='train',categories=news_categories, shuffle=True, random_state=42)
print("\n\nLoading 20NewsGroups training dataset .......")

print("sample size * first sample size")
print(str(len(news_train_dataset.data))+"*"+str(len(news_train_dataset.data[0]))+"\n")
print(news_train_dataset.data[0])
print(news_train_dataset.target[0])
print(news_train_dataset.target_names[0])

# Vectorizing the words - tf-idf term frequencies times inverse document frequencies
# high-dimensional sparse datasets - scipy.sparse matrices to save lot of memory
count_vect = CountVectorizer()
X_train_count = count_vect.fit_transform(news_train_dataset.data)

#print("CountVectorizer shape: "+str(X_train_count.shape))
tfid_vect = TfidfTransformer()
X_train_tfidf = tfid_vect.fit_transform(X_train_count)

# Training the classifier
classifier = MultinomialNB().fit(X_train_tfidf, news_train_dataset.target)

# Prediction
# Documents from only these catefories -
test_documents = ["scikit-learn is love, it's my god", "I'm down with fever", "TPU machines are faster than GPU/CPU"]
X_test_count = count_vect.transform(test_documents)
X_test_tfidf = tfid_vect.transform(X_test_count)
predictions = classifier.predict(X_test_tfidf)

for doc, prediction in zip(test_documents, predictions):
    print('%r => %s' % (doc, news_train_dataset.target_names[prediction]))
