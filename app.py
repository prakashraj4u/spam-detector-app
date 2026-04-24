import os
from flask import Flask, render_template, request
import pickle
import pytesseract
from PIL import Image

app = Flask(__name__)

# ✅ Point to Tesseract install location
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

model = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

def predict_spam(message):
    data = vectorizer.transform([message])
    result = model.predict(data)[0]
    return "🚨 SPAM" if result == 1 else "✅ Not Spam"

@app.route('/')
def index():
    return render_template('index.html', prediction=None, extracted_text=None)

@app.route('/predict', methods=['POST'])
def predict():
    extracted_text = None

    # ✅ Handle image upload with OCR
    file = request.files.get('image')
    if file and file.filename != '':
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(filepath)

        # Extract text from image
        image = Image.open(filepath)
        extracted_text = pytesseract.image_to_string(image).strip()

        if extracted_text:
            label = predict_spam(extracted_text)
            return render_template('index.html',
                                   prediction=label,
                                   extracted_text=extracted_text)
        else:
            return render_template('index.html',
                                   prediction="⚠️ No text found in image!",
                                   extracted_text=None)

    # ✅ Handle text input
    message = request.form.get('message', '').strip()
    if message:
        label = predict_spam(message)
        return render_template('index.html', prediction=label, extracted_text=None)

    return render_template('index.html',
                           prediction="⚠️ Please enter a message or upload an image!",
                           extracted_text=None)

if __name__ == '__main__':
    app.run(debug=True)