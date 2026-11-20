import json
import pickle
import nltk
import random
from nltk.stem import WordNetLemmatizer

nltk.download('punkt', quiet=True)
nltk.download('wordnet', quiet=True)

lemmatizer = WordNetLemmatizer()

# 加载你的Fatih数据集
intents = {
    "intents": [
        {
            "tag": "name",
            "patterns": [
                "What is my name?",
                "Who am I?",
                "What's my name?",
                "Can you tell me my name?",
                "Do you know what I am called?"
            ],
            "responses": [
                "Your name is Fatih."
            ]
        },
        {
            "tag": "birthday",
            "patterns": [
                "When is my birthday?",
                "When was I born?",
                "Tell me my birthday.",
                "What day is my birthday?",
                "Do you know my date of birth?"
            ],
            "responses": [
                "Your birthday is 11/12."
            ]
        },
        {
            "tag": "height",
            "patterns": [
                "What is my height?",
                "How tall am I?",
                "How tall am I in cm?",
                "Can you tell me how tall I am?",
                "What is my height in centimeters?"
            ],
            "responses": [
                "Your height is 175 cm."
            ]
        },
        {
            "tag": "weight",
            "patterns": [
                "What is my weight?",
                "How heavy am I?",
                "How much do I weigh?",
                "Tell me my weight.",
                "What is my weight in kg?"
            ],
            "responses": [
                "Your weight is 60 kg."
            ]
        },
        {
            "tag": "shoe_size",
            "patterns": [
                "What is my shoe size?",
                "What size shoes do I wear?",
                "What shoe size am I?",
                "Tell me my shoe size.",
                "What EU shoe size do I have?"
            ],
            "responses": [
                "Your shoe size is EU 42."
            ]
        },
        {
            "tag": "favorite_subject",
            "patterns": [
                "What is my favorite subject?",
                "What subject do I like most?",
                "Which school subject do I love?",
                "Tell me my favourite subject.",
                "What is my best subject?"
            ],
            "responses": [
                "Your favorite subject is Mathematics."
            ]
        },
        {
            "tag": "hobby",
            "patterns": [
                "What is my hobby?",
                "What sport do I play?",
                "What do I like to do for fun?",
                "Tell me my hobby.",
                "What activity do I enjoy?"
            ],
            "responses": [
                "Your hobby is playing snow skiing."
            ]
        },
        {
            "tag": "future_goal",
            "patterns": [
                "What is my future goal?",
                "What is my dream job?",
                "What job do I want in the future?",
                "Tell me my career goal.",
                "Where do I want to work later?"
            ],
            "responses": [
                "Your future goal is to work at SpaceX."
            ]
        },
        {
            "tag": "greeting",
            "patterns": [
                "Hello",
                "Hi",
                "Hey",
                "Good morning",
                "Good afternoon"
            ],
            "responses": [
                "Hello! I am your personal AI. Ask me anything about you."
            ]
        },
        {
            "tag": "thanks",
            "patterns": [
                "Thank you",
                "Thanks",
                "Thank you very much"
            ],
            "responses": [
                "You are welcome!"
            ]
        },
        {
            "tag": "goodbye",
            "patterns": [
                "Bye",
                "Goodbye",
                "See you"
            ],
            "responses": [
                "Goodbye! Come back to chat again."
            ]
        },
        {
            "tag": "how_are_you",
            "patterns": [
                "How are you?",
                "How are you doing?"
            ],
            "responses": [
                "I'm an AI, I don't have feelings, but I am ready to help you!"
            ]
        },
        {
            "tag": "help",
            "patterns": [
                "What can you do?",
                "What can you tell me?",
                "Help me"
            ],
            "responses": [
                "I know your personal information. You can ask about my name, birthday, height, weight, shoe size, favorite subject, hobby and future goal."
            ]
        }
    ]
}

words = []
classes = []
documents = []
ignore_chars = ["?", ".", ",", "!"]

for intent in intents["intents"]:
    tag = intent["tag"]
    for pattern in intent["patterns"]:
        token_list = nltk.word_tokenize(pattern)
        token_list = [lemmatizer.lemmatize(w.lower()) for w in token_list if w not in ignore_chars]
        words.extend(token_list)
        documents.append((token_list, tag))
        if tag not in classes:
            classes.append(tag)

words = sorted(list(set(words)))
classes = sorted(list(set(classes)))

def bag_of_words(sentence, word_list):
    tokens = nltk.word_tokenize(sentence)
    tokens = [lemmatizer.lemmatize(w.lower()) for w in tokens]
    bag = [0] * len(word_list)
    for token in tokens:
        if token in word_list:
            idx = word_list.index(token)
            bag[idx] = 1
    return bag

def predict(user_input):
    bow = bag_of_words(user_input, words)
    best_score = 0
    best_tag = None
    for token_list, tag in documents:
        target_bow = bag_of_words(" ".join(token_list), words)
        score = sum([bow[i] & target_bow[i] for i in range(len(bow))])
        if score > best_score:
            best_score = score
            best_tag = tag
    if best_tag is None:
        return {"response":"Sorry, I don't understand that question.", "intent":"unknown"}
    for intent in intents["intents"]:
        if intent["tag"] == best_tag:
            return {"response": random.choice(intent["responses"]), "intent": best_tag}
    return {"response":"Sorry, I don't understand that question.", "intent":"unknown"}

# 保存模型数据
with open("model_data.pkl", "wb") as f:
    pickle.dump({"words":words, "classes":classes, "documents":documents, "intents":intents}, f)
