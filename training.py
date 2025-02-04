import random
import json
import pickle
import numpy as np
import unicodedata
import nltk
import os
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords, wordnet
from keras.models import Sequential
from keras.layers import Dense, Dropout, BatchNormalization
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.regularizers import l2
import re

# Define la ruta base
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FolderPathNltk = os.path.join(BASE_DIR, 'nltk_data')

# Configuración de NLTK
nltk.data.path.append(FolderPathNltk)
nltk.download('punkt_tab', download_dir=FolderPathNltk)
nltk.download('punkt', download_dir=FolderPathNltk)
nltk.download('wordnet', download_dir=FolderPathNltk)
nltk.download('omw-1.4', download_dir=FolderPathNltk)
nltk.download('stopwords', download_dir=FolderPathNltk)

# Inicializar lematizador y stopwords
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('spanish'))

# Función para normalizar texto
def normalize_text(text):
    text = text.lower()
    text = unicodedata.normalize('NFKD', text).encode('ASCII', 'ignore').decode('ASCII')
    # Eliminar puntuación usar import re
    text = re.sub(r'[^\w\s]', '', text)
    return text.strip()

# Función para aumentar patrones con sinónimos
def augment_pattern(pattern):
    words = nltk.word_tokenize(pattern)
    augmented_patterns = []
    for word in words:
        synonyms = wordnet.synsets(word)
        if synonyms:
            synonym = synonyms[0].lemmas()[0].name()
            new_pattern = pattern.replace(word, synonym)
            augmented_patterns.append(new_pattern)
    return augmented_patterns


# Cargar intenciones desde el archivo JSON
intents = json.loads(open('intents.json', encoding='utf-8').read())

# Variables para almacenar palabras, clases y documentos
words = []
classes = []
documents = []
ignore_letters = ['?', '!', '¿', '.', ',', '-', '_', '(', ')', ':', ';', '…', '“', '”']

# Clasifica los patrones y las categorías
for intent in intents['intents']:
    for pattern in intent['patterns']:
        # Normalizar y tokenizar el patrón
        pattern = normalize_text(pattern)
        augmented_patterns = augment_pattern(pattern)

        # Incluir el patrón original y sus variaciones
        all_patterns = [pattern] + augmented_patterns

        for pat in all_patterns:
            word_list = nltk.word_tokenize(pat)
            # Filtrar palabras irrelevantes
            filtered_words = [lemmatizer.lemmatize(word) for word in word_list if word not in ignore_letters and word not in stop_words and len(word) > 2]
            words.extend(filtered_words)
            documents.append((filtered_words, intent["tag"]))

        if intent["tag"] not in classes:
            classes.append(intent["tag"])

# Eliminar duplicados y ordenar palabras y clases
words = sorted(set(words))
classes = sorted(set(classes))

# Guardar palabras y clases en archivos pickle
pickle.dump(words, open('words.pkl', 'wb'))
pickle.dump(classes, open('classes.pkl', 'wb'))

# Codificación de las palabras presentes en cada categoría
training = []
output_empty = [0] * len(classes)

for document in documents:
    bag = []
    word_patterns = document[0]
    for word in words:
        bag.append(1) if word in word_patterns else bag.append(0)
    output_row = list(output_empty)
    output_row[classes.index(document[1])] = 1
    training.append([bag, output_row])

# Mezclar los datos
random.shuffle(training)
print(len(training))

# Separar entradas y salidas
train_x = np.array([x[0] for x in training])
train_y = np.array([x[1] for x in training])

# Crear la red neuronal con arquitectura optimizada
model = Sequential()
model.add(Dense(512, input_shape=(len(train_x[0]),), activation='relu', name="inp_layer"))
model.add(Dropout(0.5))
model.add(Dense(256, activation='relu', kernel_regularizer=l2(0.01)))
model.add(BatchNormalization())
model.add(Dropout(0.4))
model.add(Dense(128, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(len(train_y[0]), activation='softmax'))

# Compilación del modelo con optimizador Adam
model.compile(loss='categorical_crossentropy', optimizer=Adam(learning_rate=0.001), metrics=['accuracy'])

# Early stopping para evitar sobreajuste
early_stopping = EarlyStopping(monitor='val_loss', patience=10)

# ModelCheckpoint para guardar el mejor modelo
checkpoint = ModelCheckpoint(
    'best_chatbot_model.h5',
    monitor='val_accuracy',
    save_best_only=True,
    mode='max',
    verbose=1
)

# Entrenamiento del modelo
history = model.fit(
    train_x, train_y,
    epochs=200,
    batch_size=16,
    validation_split=0.1,
    callbacks=[early_stopping, checkpoint],
    verbose=1
)

# Guardar el modelo entrenado
model.save("chatbot_model.h5")