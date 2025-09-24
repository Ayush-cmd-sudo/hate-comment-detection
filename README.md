# hate-comment-detection
ai ml project
import pandas as pd
import numpy as np
import nltk
import re
import string
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
nltk.download('stopwords')
nltk.download('punkt')
df = pd.read_csv('twitter.csv')
print(df.head(10))
df['labels'] = df['class'].map({0: 'Hate Speech', 1: 'Offensive Language', 2: 'Normal'})
print(df.head(10))
#splitting the columns
df = df[['tweet', 'labels']]
print(df.head())
stemmer = PorterStemmer()
stopwords = stopwords.words('english')

def clean(text):
    text = str(text).lower()
    text = re.sub('\[.*?\]', '', text)
    text = re.sub('https?://\S+|www\.\S+', '', text)
    text = re.sub('<.*?>+', '', text)
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub('\n', '', text)
    text = re.sub('\w*\d\w*', '', text)
    text = [word for word in text.split(' ') if word not in stopwords]
    text = ' '.join(text)
    text = [stemmer.stem(word) for word in text.split(' ')]
    text = ' '.join(text)
    return text

df['tweet'] = df['tweet'].apply(clean)

X = np.array(df['tweet'])
y = np.array(df['labels'])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25, random_state=42)
cv = CountVectorizer()
X_train = cv.fit_transform(X_train)
X_val = cv.transform(X_val)
X_test = cv.transform(X_test)
clf = DecisionTreeClassifier()
clf.fit(X_train, y_train)

from sklearn.metrics import classification_report
y_pred = clf.predict(X_val)
print("Validation Report:")
print(classification_report(y_val, y_pred, target_names=['Hate Speech', 'Offensive Language', 'Normal']))

# Evaluate on test set
y_pred = clf.predict(X_test)
print("Test Report:")
print(classification_report(y_test, y_pred, target_names=['Hate Speech', 'Offensive Language', 'Normal']))


# Load the text sample
#sample = 'Nigger' # test with for instance: kill, dog, idiot, hello
sample = 'idiot' # test with for instance: kill, dog, idiot, hello
sample = 'non sense' # test with for instance: kill, dog, idiot, hello
sample = 'bastard' # test with for instance: kill, dog, idiot, hello
sample = 'bitch' # test with for instance: kill, dog, idiot, hello


sample_processed = clean(sample)

# Vectorize the preprocessed sample text
sample_vector = cv.transform([sample_processed])

# Predict the label for the sample text
sample_prediction = clf.predict(sample_vector)

print("Prediction for sample text '{}': {}".format(sample, sample_prediction[0]))







