import pickle

model = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

msg = input("Enter message: ")

msg_vector = vectorizer.transform([msg])
prediction = model.predict(msg_vector)

if prediction[0] == 1:
    print("🚨 Spam")
else:
    print("✅ Not Spam")