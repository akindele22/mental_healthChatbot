from flask import Flask, render_template, request
import nltk
import pickle
import numpy as np

import nltk
nltk.download('popular')
from nltk.stem import WordNetLemmatizer

# Initialize the lemmatizer
lemmatizer = WordNetLemmatizer()

from keras.models import load_model
import json
import random
from datetime import datetime
import csv

# Initialize the Flask app
app = Flask(__name__)

# Load model and data
model = load_model('model.h5')
intents = json.loads(open('intents.json').read())
words = pickle.load(open('texts.pkl','rb'))
classes = pickle.load(open('labels.pkl','rb'))

# Metrics to track user interactions
metrics = {
    'session_start': None,
    'conversation_length': 0,
    'total_responses': 0,
    'total_interaction_time': 0,
    'responses': [],
    'mental_health_issue_detected': False
}

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
    bag = [0]*len(words)
    for s in sentence_words:
        for i,w in enumerate(words):
            if w == s: 
                bag[i] = 1
                if show_details:
                    print ("found in bag: %s" % w)
    return(np.array(bag))

# Predict the intent
def predict_class(sentence, model):
    p = bow(sentence, words, show_details=False)
    res = model.predict(np.array([p]))[0]
    ERROR_THRESHOLD = 0.25
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = [{"intent": classes[r[0]], "probability": str(r[1])} for r in results]
    return return_list

# Get response based on intent and classify mental health issue
def getResponse(ints, intents_json):
    if not ints:  # If no intents are predicted, return a default message
        return "I'm sorry, I couldn't understand that. Could you try rephrasing?"

    tag = ints[0]['intent']
    list_of_intents = intents_json['intent']  # Corrected to 'intents'
    for i in list_of_intents:
        if i['tag'] == tag:
            result = random.choice(i['responses'])

            # Detect mental health issue based on intent
            if tag == "mental_health_issue":  # Assuming an intent like "mental_health_issue"
                metrics['mental_health_issue_detected'] = True  # Mark as detected

            break
    return result

# Track metrics and provide chatbot response
def chatbot_response(msg):
    start_time = datetime.now()  # Track response time
    ints = predict_class(msg, model)
    res = getResponse(ints, intents)
    
    end_time = datetime.now()  # End of response time
    
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
        'mental_health_issue': metrics['mental_health_issue_detected']
    })
    
    return res

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
    return "Session ended. Metrics saved."

if __name__ == "__main__":
    app.run(debug=True, port = 8000)
