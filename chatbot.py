import random
import json
import pickle
import numpy as np
import nltk
import os
from nltk.stem import WordNetLemmatizer
from keras.models import load_model
import logging

# Define la ruta base
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FolderPathNltk = os.path.join(BASE_DIR, 'nltk_data')

# Configuración de NLTK
nltk.data.path.append(FolderPathNltk)

# Configuración de logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Inicializar el lematizador
lemmatizer = WordNetLemmatizer()

# Cargar recursos
def load_resources():
    try:
        intents = json.loads(open('intents.json', encoding='utf-8').read())
        words = pickle.load(open('words.pkl', 'rb'))
        classes = pickle.load(open('classes.pkl', 'rb'))
        model = load_model('chatbot_model.h5')
        #print(words)
        return intents, words, classes, model
    except Exception as e:
        logging.error(f"Error al cargar recursos: {e}")
        exit()

# Limpiar y tokenizar la oración
def clean_up_sentence(sentence):
    sentence = sentence.lower()  # Convertir a minúsculas
    sentence = ''.join(char for char in sentence if char.isalnum() or char.isspace())  # Eliminar caracteres especiales
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word) for word in sentence_words]
    #print(f"Palabras tokenizadas y lematizadas: {sentence_words}")
    return sentence_words

# Convertir la oración en una bolsa de palabras
def bag_of_words(sentence, words):
    """Convierte una oración en una bolsa de palabras."""
    sentence_words = clean_up_sentence(sentence)
    word_set = set(sentence_words)
    bag = [1 if word in word_set else 0 for word in words]
    #logging.info(f"Bag of words: {bag}")
    return np.array(bag)

# Predecir la clase de la oración
def predict_class(sentence, words, classes, model, threshold=0.5):
    """Predice la clase de una oración."""
    bow = bag_of_words(sentence, words)
    #print(f"BOW: {bow}")
    if not np.any(bow):  # Si la bolsa de palabras está vacía
        return None

    res = model.predict(np.array([bow]))[0]
    #logging.info(f"Predicción bruta: {res}")

    max_index = np.argmax(res)
    if res[max_index] < threshold:
        return None

    predicted_class = classes[max_index]
    #logging.info(f"Clase predicha: {predicted_class}")
    return predicted_class

# Obtener una respuesta aleatoria
def get_response(tag, intents):
    """Obtiene una respuesta aleatoria en función del tag."""
    if tag is None:
        suggestions = ["¿Podrías decirlo de otra manera?",
                        "No estoy seguro de haber entendido. ¿Puedes aclararlo?",
                        "Lo siento, no entendí eso."]
        return random.choice(suggestions)

    for intent in intents['intents']:
        if intent["tag"] == tag:
            response = random.choice(intent['responses'])
            #logging.info(f"Respuesta seleccionada: {response}")
            return response

    return "No tengo una respuesta para eso."

