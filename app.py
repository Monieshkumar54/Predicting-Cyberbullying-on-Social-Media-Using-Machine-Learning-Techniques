from flask import Flask, render_template, request
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer

app = Flask(__name__)

# Load stopwords
with open("stopwords.txt", "r") as file:
    stopwords = file.read().splitlines()

# Load the vectorizer and model
try:
    vectorizer = pickle.load(open("tfidfvectoizer.pkl", "rb"))
    model = pickle.load(open("LinearSVCTuned.pkl", 'rb'))
except Exception as e:
    print(f"Error loading model or vectorizer: {e}")
    vectorizer = None
    model = None

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    if request.method == 'POST':
        user_input = request.form['text']
        if vectorizer and model:
            try:
                # Use transform instead of fit_transform
                transformed_input = vectorizer.transform([user_input])
                prediction = model.predict(transformed_input)[0]
            except Exception as e:
                print(f"Error during prediction: {e}")
                prediction = "Error in prediction"
        else:
            prediction = "Model or Vectorizer not loaded properly"
    
    return render_template('index.html', prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True)