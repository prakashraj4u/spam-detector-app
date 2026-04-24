import os
import requests
from flask import Flask, render_template, request
import pickle
import pytesseract
from PIL import Image
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)

pytesseract.pytesseract.tesseract_cmd = '/usr/bin/tesseract'

UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

model = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

def predict_spam(message):
    data = vectorizer.transform([message])
    result = model.predict(data)[0]
    return "🚨 SPAM" if result == 1 else "✅ Not Spam"

def check_url_safety(url):
    api_url = f"https://safebrowsing.googleapis.com/v4/threatMatches:find?key={GOOGLE_API_KEY}"
    payload = {
        "client": {"clientId": "spam-detector", "clientVersion": "1.0"},
        "threatInfo": {
            "threatTypes": ["MALWARE", "SOCIAL_ENGINEERING", "UNWANTED_SOFTWARE", "POTENTIALLY_HARMFUL_APPLICATION"],
            "platformTypes": ["ANY_PLATFORM"],
            "threatEntryTypes": ["URL"],
            "threatEntries": [{"url": url}]
        }
    }
    try:
        response = requests.post(api_url, json=payload, timeout=5)
        result = response.json()
        if result.get("matches"):
            threat = result["matches"][0]["threatType"]
            return "DANGEROUS", f"🚨 DANGEROUS LINK! Threat: {threat}"
        else:
            return "SAFE", "✅ Link appears safe!"
    except Exception as e:
        return "ERROR", f"⚠️ Could not check link: {str(e)}"

@app.route('/')
def index():
    return render_template('index.html', prediction=None, extracted_text=None, url_result=None)

@app.route('/predict', methods=['POST'])
def predict():
    extracted_text = None
    url_result = None

    # Handle URL check
    url = request.form.get('url', '').strip()
    if url:
        status, message = check_url_safety(url)
        return render_template('index.html',
                               prediction=None,
                               extracted_text=None,
                               url_result={"status": status, "message": message})

    # Handle image upload with OCR
    file = request.files.get('image')
    if file and file.filename != '':
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(filepath)
        image = Image.open(filepath)
        extracted_text = pytesseract.image_to_string(image).strip()
        if extracted_text:
            label = predict_spam(extracted_text)
            return render_template('index.html',
                                   prediction=label,
                                   extracted_text=extracted_text,
                                   url_result=None)
        else:
            return render_template('index.html',
                                   prediction="⚠️ No text found in image!",
                                   extracted_text=None,
                                   url_result=None)

    # Handle text message
    message = request.form.get('message', '').strip()
    if message:
        label = predict_spam(message)
        return render_template('index.html',
                               prediction=label,
                               extracted_text=None,
                               url_result=None)

    return render_template('index.html',
                           prediction="⚠️ Please enter a message, URL or upload an image!",
                           extracted_text=None,
                           url_result=None)

if __name__ == '__main__':
    app.run(debug=True)