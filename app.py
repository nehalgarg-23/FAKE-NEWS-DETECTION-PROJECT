from flask import Flask, render_template, request
import re
import nltk
import pickle
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Check and download necessary NLTK data if not present
try:
    stopwords.words('english')
except:
    nltk.download('stopwords')

try:
    nltk.data.find('corpora/wordnet')
except:
    nltk.download('wordnet')

# Initialize Flask app
app = Flask(__name__, template_folder='./templates', static_folder='./static')

# Load pre-trained machine learning model and vectorizer
try:
    loaded_model = pickle.load(open("model.pkl", 'rb'))
    vector = pickle.load(open("vector.pkl", 'rb'))
except FileNotFoundError:
    print("Error: model.pkl or vector.pkl not found.")
    exit()

# Initialize NLP tools
lemmatizer = WordNetLemmatizer()
stpwrds = set(stopwords.words('english'))

# Function for text preprocessing and fake news prediction
def fake_news_det(news):
    # Clean and preprocess the input news text
    review = re.sub(r'[^a-zA-Z\s]', '', news).lower()
    review = nltk.word_tokenize(review)
    corpus = [lemmatizer.lemmatize(word) for word in review if word not in stpwrds]
    
    # Vectorize the preprocessed text and make a prediction
    vectorized_input_data = vector.transform([' '.join(corpus)])
    prediction = loaded_model.predict(vectorized_input_data)
    
    return prediction

# Define routes for Flask app
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        message = request.form['news']
        pred = fake_news_det(message)
        
        # Display appropriate prediction message
        if pred[0] == 1:
            result = "Prediction: Fake News 📰"
        else:
            result = "Prediction: Real News 📰"
        
        return render_template("prediction.html", prediction_text=result)
    
    return render_template('prediction.html', prediction_text="Enter a news headline to predict.")

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/contact', methods=['GET', 'POST'])
def contact():
    if request.method == 'POST':
        name = request.form['name']
        email = request.form['email']
        message = request.form['message']
        # Process/store contact data (you can implement an actual email service if needed)
        return render_template('contact.html', success=True)
    
    return render_template('contact.html')

# Run the Flask app
if __name__ == '__main__':
    app.run(debug=True)
