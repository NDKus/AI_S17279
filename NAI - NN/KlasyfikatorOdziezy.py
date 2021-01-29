"""
Autorzy: Przemysław Scharmach, Michał Zaremba
Skorzystano z zestawu danych Fashion MNIST, do którego dostęp mamy bezpośrednio z tensorflow.

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
#%%
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

#Wczytanie danych
"""Zbiór danych Fashion MNIST bierzemy bezpośrednio z biblioteki Tensoflow"""
fashion_mnist = tf.keras.datasets.fashion_mnist

"""Przypisujemy dane do zbiorów treningowych i testowych"""
(images_train, labels_train), (images_test, labels_test) = fashion_mnist.load_data()

"""W tym przypadku obrazy mają wymiary 28x28. Każdy obraz jest przypisany do jednej etykiety.
Tworzymy zbiór z nazwami etykiet w celu użycia ich później podczas drukowania wyników"""
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 
                'Dress', 'Coat', 'Sandal', 'Shirt', 
                'Sneaker', 'Bag', 'Ankle boot']

# Normalizacja
"""
Normalizujemy dane dzieląc je przez 255
"""
images_train, images_test = images_train / 255.0, images_test / 255.0

# Tworzenie Modelu
"""Pierwsza warstwa modelu przekształca format obrazów z tablicy dwuwymiarowej(28 x 28) do tablicy jednowymiarowej(28 * 28 = 784) = rozpakowywanie 
rzędów pikseli na obezrku i ustawianiu ich w jednej lini
Druga warstwa modelu określa ilość węzłów(neuronów)
Trzecia warstwa to tablica logitów o długości 10"""
model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(10)
])

# Kompilacja modelu
model.compile(optimizer = 'adam',
            loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits = True), 
            metrics=['accuracy'])

# Trenowanie modelu
"""Pierwszym krokiem trenowania modelu jest wprwadzenie danych szkoleniowych do modelu, 
następnie uczymy model kojarzyć obrazy i etykiety w 12 epokach."""

model.fit(images_train, labels_train, epochs=12)

"""Sprawdzamy czy wytrenowany model zgadza się z testowym i wyświetlamy trafność"""
test_loss, test_acc = model.evaluate(images_test,  labels_test, verbose=2)
print('Trafność:', test_acc)

probability_model = tf.keras.Sequential([model,
                tf.keras.layers.Softmax()
])

# Prognozowanie
predictions = probability_model.predict(images_test)
"""Prognozowanie to w naszym wypadku tablica 10 liczb, które reprezentują pewność dobrania odpowiedniej etykiety do obrazu,
Wyciągając z tych danych największą wartość ufności dla etykiet, pokaże, która etykieta według modelu jest najbardziej odpowiednia dla obrazu"""
predictions[420]  
np.argmax(predictions[420]) 

# Wykres 
def plot_image(i, predictions_array, true_label, img):
    '''Tworzy wykres z obrazkiem przewidywanego przedmiotu, nadaje kolor nazwie w zależności od tego, czy model dobrze przewidział przedmiot'''
    true_label, img = true_label[i], img[i]
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])
    plt.imshow(img, cmap=plt.cm.binary)
    #sprawdzanie poprawności
    predicted_label = np.argmax(predictions_array)
    if predicted_label == true_label:
        color = 'teal'
    else:
        color = 'red'
    #etykieta
    plt.xlabel("{} {:2.0f}% ({})".format(class_names[predicted_label],
                                100*np.max(predictions_array),
                                class_names[true_label]),
                                color=color)

def plot_value_array(i, predictions_array, true_label):
    '''Tworzy wykres słupkowy przedstawiający wartości ufności dla przewidzianych etykiet'''
    true_label = true_label[i]
    plt.grid(False)
    plt.xticks(range(10))
    plt.yticks([])
    thisplot = plt.bar(range(10), predictions_array, color="#777777")
    plt.ylim([0, 1])
    predicted_label = np.argmax(predictions_array)
    thisplot[predicted_label].set_color('red')
    thisplot[true_label].set_color('teal')

"""Tworzenie wykresu 6x4 z pseudo przypadkowymi przedmiotami z zestawu testowego"""
import random

num_rows = 6 
num_cols = 4 
num_images = num_rows*num_cols
plt.figure(figsize=(2*2*num_cols, 2*num_rows))
for i in range(num_images):
    random_item = random.randrange(10000)
    plt.subplot(num_rows, 2*num_cols, 2*i+1)
    plot_image(random_item, predictions[random_item], labels_test, images_test)
    plt.subplot(num_rows, 2*num_cols, 2*i+2)
    plot_value_array(random_item, predictions[random_item], labels_test)
plt.tight_layout()
plt.show()