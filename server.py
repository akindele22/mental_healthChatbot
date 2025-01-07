from flask import Flask, render_template, request
import nltk
import pickle
import numpy as np

# Download required NLTK data
nltk.download('vader_lexicon')
from nltk.stem import WordNetLemmatizer
from nltk.sentiment.vader import SentimentIntensityAnalyzer  # For sentiment analysis
from keras.models import load_model
import json
import random
from datetime import datetime
import csv
from transformers import pipeline  # For emotion detection

# Initialize components
lemmatizer = WordNetLemmatizer()
sia = SentimentIntensityAnalyzer()  # Sentiment intensity analyzer
emotion_classifier = pipeline('sentiment-analysis', model='j-hartmann/emotion-english-distilroberta-base')  # Emotion classifier

# Initialize the Flask app
app = Flask(__name__)

# Load model and data
model = load_model('model.h5')
intents = json.loads(open('intents.json').read())
words = pickle.load(open('texts.pkl', 'rb'))
classes = pickle.load(open('labels.pkl', 'rb'))

# Metrics to track user interactions
metrics = {
    'session_start': None,
    'conversation_length': 0,
    'total_responses': 0,
    'total_interaction_time': 0,
    'responses': [],
    'mental_health_issues_detected': {
        'anxiety': 0,
        'depression': 0,
        'stress': 0,
        'overall': False
    }
}

# List of red flag words/phrases for mental health concerns
red_flag_words = ["hopeless", "worthless", "suicidal", "tired of life", "give up"]

# Start session timer
metrics['session_start'] = datetime.now()

# Save metrics to CSV
def save_metrics_to_csv():
    with open('chatbot_metrics.csv', 'w', newline='') as csvfile:
        fieldnames = ['input', 'response', 'timestamp', 'mental_health_issue']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for interaction in metrics['responses']:
            writer.writerow(interaction)

# Clean up and process user input
def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    return sentence_words

# Create a bag of words
def bow(sentence, words, show_details=True):
    sentence_words = clean_up_sentence(sentence)
    bag = [0] * len(words)
    for s in sentence_words:
        for i, w in enumerate(words):
            if w == s:
                bag[i] = 1
                if show_details:
                    print("found in bag: %s" % w)
    return np.array(bag)

# Predict the intent
def predict_class(sentence, model):
    p = bow(sentence, words, show_details=False)
    res = model.predict(np.array([p]))[0]
    ERROR_THRESHOLD = 0.25
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = [{"intent": classes[r[0]], "probability": str(r[1])} for r in results]
    return return_list

# Sentiment Analysis function
def get_sentiment_score(msg):
    sentiment = sia.polarity_scores(msg)
    return sentiment

# Emotion Detection function
def get_emotion(msg):
    emotions = emotion_classifier(msg)
    return emotions[0]

# Red flag checker
def check_red_flags(msg):
    for word in red_flag_words:
        if word in msg.lower():
            return True
    return False

# Get response based on intent and classify mental health issue
def getResponse(ints, intents_json):
    if not ints:  # If no intents are predicted, return a default message
        return "I'm sorry, I couldn't understand that. Could you try rephrasing?"

    tag = ints[0]['intent']
    list_of_intents = intents_json['intent']
    result = ""
    for i in list_of_intents:
        if i['tag'] == tag:
            result = random.choice(i['responses'])

            # Track specific mental health issues
            if tag in ['anxiety', 'depression', 'stress']:
                metrics['mental_health_issues_detected'][tag] += 1
                metrics['mental_health_issues_detected']['overall'] = True
            break

    return result

# Track metrics and provide chatbot response
def chatbot_response(msg):
    start_time = datetime.now()  # Track response time
    ints = predict_class(msg, model)
    res = getResponse(ints, intents)

    end_time = datetime.now()  # End of response time
    
    # Track sentiment and emotion
    sentiment = get_sentiment_score(msg)
    emotion = get_emotion(msg)

    # Check for red flags
    red_flag_detected = check_red_flags(msg)

    # Track conversation metrics
    interaction_time = (end_time - start_time).total_seconds()
    metrics['conversation_length'] += 1
    metrics['total_responses'] += 1
    metrics['total_interaction_time'] += interaction_time

    # Log interaction
    metrics['responses'].append({
        'input': msg,
        'response': res,
        'timestamp': datetime.now(),
        'sentiment': sentiment,
        'emotion': emotion,
        'mental_health_issue': metrics['mental_health_issues_detected']['overall'] or red_flag_detected
    })

    return res

# Generate mental health evaluation report
def generate_mental_health_report():
    report = f"Mental Health Evaluation:\n"
    report += f"Conversation Length: {metrics['conversation_length']} responses\n"
    report += f"Total Interaction Time: {metrics['total_interaction_time']} seconds\n"
    if metrics['mental_health_issues_detected']['overall']:
        report += f"Mental Health Issues Detected:\n"
        for issue, count in metrics['mental_health_issues_detected'].items():
            if count > 0 and issue != 'overall':
                report += f" - {issue.capitalize()}: {count} instances\n"
    else:
        report += "No significant mental health concerns detected.\n"
    return report

# Home route to render the chatbot UI
@app.route("/chat")
def home():
    return render_template("chatbot.html")

# Route to handle the chatbot response
@app.route("/get")
def get_bot_response():
    userText = request.args.get('msg')
    bot_response = chatbot_response(userText)

    return bot_response

# Route to end the session and save metrics
@app.route("/end_session")
def end_session():
    save_metrics_to_csv()
    report = generate_mental_health_report()
    return f"Session ended. Metrics saved.\n\n{report}"

if __name__ == "__main__":
    app.run(debug=True, port=8000)
