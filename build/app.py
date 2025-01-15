pip install nltk


import nltk
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('omw-1.4')

from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()
import json
import pickle
import numpy as np


from keras.models import load_model
model=load_model("/content/chatbot_model.h5")
import json
import random
intents = json.loads(open('/content/intents.json',encoding="utf8").read())
words = pickle.load(open('/content/words.pkl','rb'))
classes = pickle.load(open('/content/classes.pkl','rb'))

!pip install flask flask-ngrok pyngrok
from pyngrok import ngrok
ngrok.set_auth_token("2rclX8Z0vuQIZJJiSLaF0p1RTrE_5LNgbygjqKDdUieq1udch")


def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    return sentence_words



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

def predict_class(sentence, model):
    
    p = bow(sentence, words,show_details=False)
    res = model.predict(np.array([p]))[0]
    ERROR_THRESHOLD = 0.25
    results = [[i,r] for i,r in enumerate(res) if r>ERROR_THRESHOLD]
    
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append({"intent": classes[r[0]], "probability": str(r[1])})
    return return_list

 def getResponse(ints, intents_json):
    tag = ints[0]['intent']
    list_of_intents = intents_json['intents']
    for i in list_of_intents:
        if(i['tag']== tag):
            result = random.choice(i['responses'])
            break
    return result


def chatbot_response(msg):
    ints = predict_class(msg, model)
    res = getResponse(ints, intents)
    return res

pip install flask_cors

from flask import Flask, jsonify
from pyngrok import ngrok
from flask_cors import CORS
import json
import os


ngrok.set_auth_token("2rclX8Z0vuQIZJJiSLaF0p1RTrE_5LNgbygjqKDdUieq1udch")

app = Flask(__name__)
CORS(app)


responses_file = '/content/intents.json'


def load_responses():
    if not os.path.exists(responses_file):
        print(f"Error: The file {responses_file} was not found.")
        return None

    try:
        with open(responses_file, 'r', encoding='utf-8') as file:
            data = json.load(file)
        return data
    except json.JSONDecodeError as e:
        print(f"Error: The JSON file could not be decoded properly. {str(e)}")
        return None
    except Exception as e:
        print(f"Error loading responses: {str(e)}")
        return None


def find_best_match(user_message):
    responses = load_responses()
    if not responses:
        return "Sorry, I'm having trouble accessing my knowledge base."

    user_message = user_message.lower()
    best_match = None
    highest_score = 0

    for intent in responses['intents']:
        for pattern in intent['patterns']:
            if pattern.lower() in user_message:
                score = len(pattern) / len(user_message)
                if score > highest_score:
                    highest_score = score
                    best_match = intent

    if best_match and highest_score > 0.5:
        return best_match['responses'][0]  

    
    for intent in responses['intents']:
        if intent['tag'] == 'noanswer':
            return intent['responses'][0]

    return "I'm not sure how to respond to that."


def decrypt(msg):
    return msg.replace("+", " ")

@app.route("/", methods=['GET', 'POST'])
def hello():
    return jsonify({"key": "home page value"})

@app.route('/query/<sentence>')
def query_chatbot(sentence):
    dec_msg = decrypt(sentence)
    response = find_best_match(dec_msg)
    return jsonify({"top": {"res": response}})

if __name__ == '__main__':
    
    public_url = ngrok.connect(5000)
    print(f" * ngrok public URL: {public_url}")

    
    app.run(port=5000, debug=False, threaded=True)