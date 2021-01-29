"""
Autorzy: Przemysław Scharmach, Michał Zaremba

Skorzystano ze zdjęć pobranych z Google Grafika przy pomocy wtyczki
Fatkun Batch do przeglądarki chrome.

Referencje: Świetny tutorial użytkownika Aladdin Persson na youtube:
https://www.youtube.com/watch?v=q7ZuZ8ZOErE

Opis problemu:
Naszym zadaniem jest nauczenie sieci neuronowej klasyfikacji danych.


Instrukcja przygotowania środowiska (dla systemów operacyjnych Windows)
W wierszu poleceń należy wpisać kolejno: 
    
    1. curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py
    2. python get-pip.py
    3. pip install tensorflow
    4. folder "Rick and Morty" umiescic w tym samym miejscu, w którym znajduje 
    się plik KlasyfikatorRickaIMortiego.py (nie wrzucać pliku py do tego folderu)

    W przypadku posiadania pip (instalator pakietów) pozycje 3. można
    wpisać również w terminalu IDE.
"""

import os

#Umozliwia pozbycie się logów o błędach z alokacją pamięci i tym podobnych.
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"


import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.image import ImageDataGenerator


#Standaryzacja wielkosci zdjęć oraz batch_size
"""
Batch_size określa liczbę próbek, które będą propagowane w sieci. 
Na przykład, powiedzmy, że mamy 1050 próbek uczących i chcemy ustawić
batch_size równą 100. Algorytm pobiera pierwsze 100 próbek (od 1 do 100) 
z uczącego zestawu danych i trenuje sieć. Następnie pobiera kolejne 
100 próbek (od 101 do 200) i ponownie trenuje sieć. Możemy kontynuować tę 
procedurę, dopóki nie rozpropagujemy wszystkich próbek w sieci.
"""

img_height = 28
img_width = 28
batch_size = 2

# Tworzenie modelu
"""Definujemy model. Model Sequential jest odpowiedni dla zwykłego stosu 
warstw, w którym każda warstwa ma dokładnie jeden tensor wejściowy i jeden 
tensor wyjściowy.


Warstwa Conv2D tworzy jądro splotu, które jest splecione z danymi wejściowymi 
warstwy, aby utworzyć tensor wyników. Jeśli use_bias ma wartość True, wektor 
odchylenia jest tworzony i dodawany do wyników. Wreszcie, jeśli activation nie
jest None, jest stosowany również do wyjść. Używając tej warstwy jako pierwszej
warstwy w modelu, należy podać argument słowa kluczowego input_shape.

Padding to specjalna forma maskowania, w której zamaskowane kroki znajdują się
na początku lub na początku sekwencji. Wypełnienie wynika z potrzeby 
zakodowania danych sekwencji w ciągłe paczki: aby wszystkie sekwencje w 
partii pasowały do ​​określonej standardowej długości, konieczne jest 
dopełnienie lub obcięcie niektórych sekwencji. 

Parametr „SAME” próbuje równomiernie wypełnić lewą i prawą stronę, ale jeśli 
liczba dodawanych kolumn jest nieparzysta, doda dodatkową kolumnę po prawej 
stronie.


Normalizacja wsadowa (BatchNormalization) stosuje transformację, która 
utrzymuje średni wynik na poziomie bliskim 0, a odchylenie standardowe wyjścia 
bliskie 1.

Model Flatten spłaszcza tensor wejściowy, zachowując oś wsadu.

Dense to zwykła, gęsto połączona warstwa NN. implementuje operację: wyjście =
aktywacja (kropka (wejście, jądro) + odchylenie)
gdzie aktywacja jest elementarną funkcją aktywacji przekazaną jako argument 
aktywacji, jądro to macierz wag utworzona przez warstwę, a odchylenie to 
utworzony wektor odchylenia przez warstwę.

MaxPooling2D próbkuje w dół reprezentację wejściową, pobierając maksymalną 
wartość dla każdego wymiaru wzdłuż osi.

RELU to funkcja aktywacji która daje 0, jeśli dane wejściowe są negatywne i same dane 
wejściowe, jeśli dane wejściowe są równe 0 lub pozytywne. 
"""


model = keras.Sequential([
    layers.Conv2D(16, 3, padding="same"),
    layers.Conv2D(32, 3, padding="same"),
    layers.MaxPooling2D(),
    layers.Flatten(input_shape=(28, 28)), 
    layers.Dense(256, activation='relu'), 
    layers.Dense(10) 
])

#Użycie ImageDataGenerator
"""
ImageDataGenerator to prawdziwa perełka. 
Pozwala na ulepszanie obrazów w czasie rzeczywistym, gdy model nadal trenuje. 
Można zastosować dowolne losowe transformacje do każdego obrazu uczącego, gdy
jest on przekazywany do modelu. To nie tylko sprawi, że twój model będzie 
solidny, ale także zaoszczędzi pamięć.

Zdjęcia są skalowane, powiększane, obracane - a wszystko to po to, aby mieć
jak najlepszy model i największą szansę poprawnej predykcji.
"""

datagen = ImageDataGenerator(
    rescale=1.0 / 255,
    rotation_range=5,
    zoom_range=(0.95, 0.95),
    horizontal_flip=False,
    vertical_flip=False,
    data_format="channels_last",
    validation_split=0.0,
    dtype=tf.float32,
)

train_generator = datagen.flow_from_directory(
    "Rick and Morty",
    target_size=(img_height, img_width),
    batch_size=batch_size,
    color_mode="grayscale",
    class_mode="sparse",
    shuffle=True,
    subset="training",
    seed=123,
)


#Kompilowanie
model.compile(
    optimizer=keras.optimizers.Adam(),
    loss=[keras.losses.SparseCategoricalCrossentropy(from_logits=True),],
    metrics=["accuracy"],
)

# Dopasowanie modelu (z krokami co epokę)
"""
Dzięki parametrowi verbose = 2 wyswietlamy linie w konsoli co epoke.
Ustawiamy 10 epok, co oznacza że sieć przetworzy cały zestaw treningowy
dziesięciokrotnie. Ustawiamy 25 kroków na epokę.
"""
model.fit(
    train_generator,
    epochs=10,
    steps_per_epoch=25,
    verbose=2,

)
# Wynik
"""
Ewaluacja. Wyswietlamy procentową skutecznosc.
"""
scores = model.evaluate(train_generator, verbose=0)
print("Accuracy: %.2f%%" % (scores[1]*100))
