import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
import pickle

# 1. Load data
df = pd.read_csv("spam.csv", encoding="latin-1")
df = df.iloc[:, :2]
df.columns = ['label', 'message']

# 2. Map labels
df['label'] = df['label'].str.strip().map({'ham': 0, 'spam': 1})

# 3. Define X and y  <-- THIS WAS MISSING
X = df['message']
y = df['label']

# 4. Vectorize
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(X)  # line 15 - now works

# 5. Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 6. Train model
model = MultinomialNB()
model.fit(X_train, y_train)

# 7. Save model and vectorizer
pickle.dump(model, open("model.pkl", "wb"))
pickle.dump(vectorizer, open("vectorizer.pkl", "wb"))

print("Model trained and saved!")