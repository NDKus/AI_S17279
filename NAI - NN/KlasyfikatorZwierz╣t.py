"""
Autorzy: Przemysław Scharmach, Michał Zaremba
Skorzystano z zestawu danych cifar10.

Opis problemu:
Naszym zadaniem jest nauczenie sieci neuronowej klasyfikacji danych.


Instrukcja przygotowania środowiska (dla systemów operacyjnych Windows)
W wierszu poleceń należy wpisać kolejno: 
    
    1. curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py
    2. python get-pip.py
    3. pip install numpy
    4. pip install tensorflow

    
    W przypadku posiadania pip (instalator pakietów) pozycje 3. oraz 4. można
    wpisać również w terminalu IDE.
"""


import numpy
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, BatchNormalization, Activation
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.constraints import maxnorm
from keras.utils import np_utils

# Czynnik losowosci
"""
Dla zwiększenia powtarzalnosci ustawiamy konkretny seed losowy, dzięki temu 
rozstrzał między wynikami będzie mniejszy.
"""

seed = 21

from keras.datasets import cifar10

# Wczytanie danych
"""
W tej sekcji wczytujemy nasze dane do - wykorzystujemy zbiór cifar10, 
który udostępnia tensorflow
"""
     
(X_train, y_train), (X_test, y_test) = cifar10.load_data()

# Normalizacja
"""
Normalizujemy dane dzieląc je przez 255
"""
    
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train = X_train / 255.0
X_test = X_test / 255.0

# Wyjscia
"""
Okreslamy nasze wyjscia - dane testowe i dane trenigowe
"""

y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)
class_num = y_test.shape[1]

# Tworzenie modelu
"""Definujemy model. Model Sequential jest odpowiedni dla zwykłego stosu 
warstw, w którym każda warstwa ma dokładnie jeden tensor wejściowy i jeden 
tensor wyjściowy.
"""

model = Sequential()

"""
RELU to funkcja aktywacji która daje 0, jeśli dane wejściowe są negatywne i
same dane wejściowe, jeśli dane wejściowe są równe 0 lub pozytywne. 

Wrstwa Conv2D tworzy jądro splotu, które jest splecione z danymi wejściowymi 
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

Model Dropout losowo ustawia jednostki wejściowe na 0 z częstotliwością rate 
na każdym kroku podczas treningu, co pomaga zapobiegać nadmiernemu dopasowaniu. 
Wejścia nie ustawione na 0 są skalowane w górę o 1 / (1 - stopa) tak, że suma 
wszystkich wejść pozostaje niezmieniona.

Normalizacja wsadowa (BatchNormalization) stosuje transformację, która 
utrzymuje średni wynik na poziomie bliskim 0, a odchylenie standardowe wyjścia 
bliskie 1.

Model Flatten spłaszcza tensor wejściowy, zachowując oś wsadu.

Softmax przekształca rzeczywisty wektor w wektor prawdopodobieństw jakościowych.

Dense to zwykła, gęsto połączona warstwa NN. implementuje operację: wyjście = 
aktywacja (kropka (wejście, jądro) + odchylenie)
gdzie aktywacja jest elementarną funkcją aktywacji przekazaną jako argument 
aktywacji, jądro to macierz wag utworzona przez warstwę, a odchylenie to 
utworzony wektor odchylenia przez warstwę.
"""

model.add(Conv2D(32, (3, 3), input_shape=X_train.shape[1:], padding='same'))
model.add(Activation('relu'))
model.add(Conv2D(32, (3, 3), input_shape=(3, 32, 32), activation='relu', padding='same'))
model.add(Dropout(0.2))
model.add(BatchNormalization())

model.add(Conv2D(64, (3, 3), padding='same'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.2))
model.add(BatchNormalization())

model.add(Conv2D(64, (3, 3), padding='same'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.2))
model.add(BatchNormalization())
    
model.add(Conv2D(128, (3, 3), padding='same'))
model.add(Activation('relu'))
model.add(Dropout(0.2))
model.add(BatchNormalization())

model.add(Flatten())
model.add(Dropout(0.2))

model.add(Dense(256, kernel_constraint=maxnorm(3)))
model.add(Activation('relu'))
model.add(Dropout(0.2))
model.add(BatchNormalization())
    
model.add(Dense(128, kernel_constraint=maxnorm(3)))
model.add(Activation('relu'))
model.add(Dropout(0.2))
model.add(BatchNormalization())

model.add(Dense(class_num))
model.add(Activation('softmax'))

"""
Epoch to w machine learningu cały proces przetwarzania przez algorytm nauki 
zestawu treningowego.

Jako optimizer wskazujemy algorytm "adam", który wydaje się być najbardziej
popularny.
"""

epochs = 25
optimizer = 'adam'

# Kompilowanie
"""
Batch_size określa liczbę próbek, które będą propagowane w sieci. 
Na przykład, powiedzmy, że mamy 1050 próbek uczących i chcemy ustawić
batch_size równą 100. Algorytm pobiera pierwsze 100 próbek (od 1 do 100) 
z uczącego zestawu danych i trenuje sieć. Następnie pobiera kolejne 
100 próbek (od 101 do 200) i ponownie trenuje sieć. Możemy kontynuować tę 
procedurę, dopóki nie rozpropagujemy wszystkich próbek w sieci.
"""

model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

print(model.summary())

numpy.random.seed(seed)
model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=epochs, batch_size=64)


# Wynik
"""
Ewaluacja. Wyswietlamy procentową skutecznosc.
"""

scores = model.evaluate(X_test, y_test, verbose=0)
print("Accuracy: %.2f%%" % (scores[1]*100))