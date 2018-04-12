# Initialize and train a Word2Vec model
import gensim, logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

# Word2Vec vocabulary
raw_vocabulary = [
                "Can I order a Pizza",
                "Do you have deep dish pizzas available",
                "Do you server gluten-free pizza",
                "How much does a slice of pizza cost",
                "I feel like eating some pizza",
                "I want to order pizza for lunch",
                "I'd like to order a Pie please",
                "Let's order a cheese pizza",
                "Order Pizza",
                "What types of pizzas can I order",
                "What's on the menu today",
                "Would love a large Pepperoni please",
                "Would you happen to have thin crust options on your Pizzas",
                "Can i cancel my order",
                "Cancel my order",
                "Cancel my Pizza please",
                "Don't need my Pie anymore",
                "How do I cancel my order",
                "I don't want my Pizza anymore",
                "I really don't want the Pizza anymore",
                "I'd like to cancel my order please",
                "Its been more than 20 mts Please cancel my order and issue a refund to my card",
                "Need to cancel my order",
                "Please cancel my pizza order",
                "Please don't deliver my Pizza"
                "What is the price of pizza"
                "What much does the pizza cost"
                "What is the cost of pizza"
                "how should i pay for pizza"
             ]

# Create a set of frequent words
stoplist = set('for a of the and to in'.split(' '))
# Lowercase each document, split it by white space and filter out stopwords
vocabulary = [[word for word in document.lower().split() if word not in stoplist]
         for document in raw_vocabulary]

# train word2vec
model = gensim.models.Word2Vec(vocabulary, size=100, window=5, min_count=2, workers=4)
model.save("pizzabot.pkl")
