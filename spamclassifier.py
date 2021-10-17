# importing the Dataset
import pandas as pd
from scipy.sparse.construct import random

data = pd.read_csv('SMSSpamCollection', sep = '\t', names = ["label", "message"])

# Visualizing the data 
data.head()

# Data Cleaning and preprocessing
import re
import nltk
nltk.download('stopwords')

from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

# applying stemming
ps = PorterStemmer()

# list to store processed messages
corpus = []

for i in range(0, len(data)):
    review = re.sub('[^a-zA-Z]', ' ', data['message'][i])
    review = review.lower()
    review = review.split()

    review = [ps.stem(word) for word in review if word not in stopwords.words('english')]
    review = ' '.join(review)
    corpus.append(review)

# creating a Bag of Words model
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features = 2500)
x = cv.fit_transform(corpus).toarray()

# categorical encoding for label column
y = pd.get_dummies(data['label'])
y = y.iloc[:,1].values

# Train Test Split
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.3, random_state = 10)

# Training model by Naive Bayes Classifier
from sklearn.naive_bayes import MultinomialNB
spam_detect_model = MultinomialNB().fit(x_train, y_train)

# making predictions on test data
y_pred = spam_detect_model.predict(x_test)

# Evaluating the model
from sklearn.metrics import confusion_matrix, accuracy_score
print(confusion_matrix(y_test, y_pred))
print(accuracy_score(y_test, y_pred))