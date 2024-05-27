from flask import Flask, request, render_template
from sklearn.feature_extraction.text import TfidfVectorizer
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import nltk
import joblib
import re
from nltk.corpus import stopwords

# Load the individual models and the TF-IDF vectorizer
logistic_model = joblib.load('logistic_model.pkl')
nb_model = joblib.load('nb_model.pkl')
tfidf_vectorizer = joblib.load('tfidf_vectorizer.pkl')

nltk.download('punkt')

nltk.download('wordnet')
# Download NLTK stopwords
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

app = Flask(__name__)

def preprocess_text(text):
    # Remove special characters and digits
    text = re.sub(r'[^a-zA-Z\s]', '', str(text))
    # Convert to lowercase
    text = text.lower()
    # Tokenization
    tokens = nltk.word_tokenize(text)
    # Removing stopwords
    filtered_tokens = [word for word in tokens if word not in stop_words]
    # Lemmatization
    lemmatizer = nltk.WordNetLemmatizer()
    lemmatized_tokens = [lemmatizer.lemmatize(word) for word in filtered_tokens]
    # Join tokens back into text
    processed_text = ' '.join(lemmatized_tokens)
    return processed_text

@app.route('/')
def my_form():
    return render_template('form.html')

@app.route('/', methods=['POST'])
def my_form_post():
    text1 = request.form['text1']

    # Preprocess the input text
    processed_text = preprocess_text(text1)

    # Use TF-IDF vectorizer to transform preprocessed text into a feature vector
    feature_vector = tfidf_vectorizer.transform([processed_text])

    # Predict sentiment using individual models
    logistic_pred = logistic_model.predict(feature_vector)[0]
    nb_pred = nb_model.predict(feature_vector)[0]

    # Voting - Simple Majority Voting
    pred_counts = {'positive': 0, 'neutral': 0, 'negative': 0}
    pred_counts[logistic_pred] += 1
    pred_counts[nb_pred] += 1

    # Get the sentiment with the maximum count
    combined_prediction = max(pred_counts, key=pred_counts.get)

   
    sa = SentimentIntensityAnalyzer()
    score = sa.polarity_scores(processed_text)
    compound = round((1 + score['compound']) / 2, 2)
    if compound >= 0.65:
        sentiment_label = "Positive"
    elif compound <= 0.4:
        sentiment_label = "Negative"
    else:
        sentiment_label = "Neutral"

    # Determine the final sentiment based on the hybrid approach
    sentiment_label = combined_prediction

    return render_template('form.html',final=sentiment_label,combined=combined_prediction, sentiment_label=sentiment_label, text1=text1, text2=score['pos'], 
       text5=score['neg'], text4=compound,text3=score['neu'])

if __name__ == "__main__":
    app.run(debug=True, host="127.0.0.1", port=5002, threaded=True)
