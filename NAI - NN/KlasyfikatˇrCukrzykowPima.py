"""
Autorzy: Przemysław Scharmach, Michał Zaremba
Skorzystano z zestawu danych Pima Indiands Diabetes Dataset do pobrania tutaj:
https://machinelearningmastery.com/standard-machine-learning-datasets/

Opis problemu:
Naszym zadaniem jest nauczenie sieci neuronowej klasyfikacji danych.


Instrukcja przygotowania środowiska (dla systemów operacyjnych Windows)
W wierszu poleceń należy wpisać kolejno: 
    
    1. curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py
    2. python get-pip.py
    3. pip install numpy
    4. pip install tensorflow
    5. plik pima-indians-diabetes.csv należy umiescic w tym samym folderze co
    plik skryptowy
    
    W przypadku posiadania pip (instalator pakietów) pozycje 3. oraz 4. i 5 można
    wpisać również w terminalu IDE.
"""

from keras.models import Sequential
from keras.layers import Dense
import numpy as np

# Czynnik losowosci
"""
Dla zwiększenia powtarzalnosci ustawiamy konkretny seed losowy, dzięki temu 
rozstrzał między wynikami będzie mniejszy.
"""

np.random.seed(7)

# Wczytanie pliku
"""
W tej sekcji wczytujemy nasz plik CSV do zmiennej dataset.
"""

dataset = np.loadtxt("pima-indians-diabetes.csv", delimiter=",")

# Przypisanie atrybutów do zmiennych 
"""
Pod zmienną X wskazujemy nasze podstawowe 
atrybuty, natomiast pod zmienną Y kryje się atrybut przypisania do klasy.
"""

X = dataset[:,0:8]
Y = dataset[:,8]

# Tworzenie modelu
"""Definujemy model. Model Sequential jest odpowiedni dla zwykłego stosu 
warstw, w którym każda warstwa ma dokładnie jeden tensor wejściowy i jeden 
tensor wyjściowy.

RELU to funkcja aktywacji która daje 0, jeśli dane wejściowe są negatywne i same dane 
wejściowe, jeśli dane wejściowe są równe 0 lub pozytywne. 

Funkcja sigmoida zawsze zwraca wartość z przedziału od 0 do 1 zaleznie od tego
czy ma do czynienia z niską wartoscia (<5) czy dużą (>5)

binary_crossentropy oblicza stratę krzyżową entropii między prawdziwymi 
etykietami a przewidywanymi etykietami.

Jako optimizer wskazujemy algorytm "adam", który wydaje się być najbardziej
popularny.

Dense to zwykła, gęsto połączona warstwa NN. implementuje operację: wyjście 
= aktywacja (kropka (wejście, jądro) + odchylenie)
gdzie aktywacja jest elementarną funkcją aktywacji przekazaną jako argument 
aktywacji, jądro to macierz wag utworzona przez warstwę, a odchylenie to 
utworzony wektor odchylenia przez warstwę.

"""

model = Sequential()
model.add(Dense(12, input_dim=8, kernel_initializer='uniform', activation='relu'))
model.add(Dense(8, kernel_initializer='uniform', activation='relu'))
model.add(Dense(1, kernel_initializer='uniform', activation='sigmoid'))

# Kompilowanie
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Dopasowanie
"""
Epoch to w machine learningu cały proces przetwarzania przez algorytm nauki 
zestawu treningowego.

validation_split umożliwia automatyczne zarezerwowanie części danych 
szkoleniowych do walidacji. Wartość argumentu reprezentuje ułamek danych, 
które mają być zarezerwowane do walidacji, więc powinna być ustawiona na 
liczbę większą niż 0 i mniejszą niż 1. Na przykład validation_split=0.33
oznacza „użyj 33% danych do walidacji”,.

batch_size określa liczbę próbek, które będą propagowane w sieci. 
Na przykład, powiedzmy, że mamy 1050 próbek uczących i chcemy ustawić
batch_size równą 100. Algorytm pobiera pierwsze 100 próbek (od 1 do 100) 
z uczącego zestawu danych i trenuje sieć. Następnie pobiera kolejne 
100 próbek (od 101 do 200) i ponownie trenuje sieć. Możemy kontynuować tę 
procedurę, dopóki nie rozpropagujemy wszystkich próbek w sieci.
"""

model.fit(X, Y, validation_split=0.33, epochs=150, batch_size=10)

# Wynik
"""
Wyswietlamy procentową skutecznosc
"""

scores = model.evaluate(X, Y)
print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
