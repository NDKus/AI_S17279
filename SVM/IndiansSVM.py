"""
Autorzy: Przemysław Scharmach, Michał Zaremba
Skorzystano z zestawu danych Pima Indiands Diabetes Dataset do pobrania tutaj:
https://machinelearningmastery.com/standard-machine-learning-datasets/

Opis problemu:
Naszym zadaniem jest nauczenie SVM (Support Vector Machine) klasyfikacji danych.
SVM to model uczenia nadzorowanego w machine learningu, który umożliwia
kategoryzację poprzez separowanie w celu utworzenia granicy decyzyjnej. Dzięki
danym testowym możemy nauczyć SVM samemu decydować (z bardzo wysoką skutecznoscią)
jak sklasyfikować kolejne rekordy.


Instrukcja przygotowania środowiska (dla systemów operacyjnych Windows)
W wierszu poleceń należy wpisać kolejno: 
    
    1. curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py
    2. python get-pip.py
    3. pip install pandas
    4. pip install -U scikit-learn
    5. pip install seaborn
    
    W przypadku posiadania pip (instalator pakietów) pozycje 3. oraz 4. i 5 można
    wpisać również w terminalu IDE.
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

import seaborn as sns

# Wczytanie pliku CSV do zmiennej i nadanie etykiet
"""
W tej sekcji definiujemy zmienne które będziemy stosować oraz nadajemny etykiety
do kolumn z naszymi danymi z pliku csv. Pod zmienną X wskazujemy nasze podstawowe 
atrybuty, natomiast pod zmienną Y kryje się atrybut przypisania do klasy.

Nasze atrybuty to:
- Liczba przypadków ciąży
- Stężenie glukozy w osoczu przez 2 godziny w doustnym teście tolerancji glukozy
- Rozkurczowe ciśnienie krwi (mm Hg)
- Grubość fałdu skórnego tricepsa (mm)
- 2-godzinna insulina w surowicy (μU / ml)
- Wskaźnik masy ciała (masa w kg / (wzrost wm) ^ 2)
- Funkcja rodowodu cukrzycy
- Wiek (lata)
- Atrybut klasy
"""

cols = ["Times pregnant", "glucose", "blood pr", "skinfold thickness", "serum insulin", "BMI", "Diabetes pedigree function", "Age", "class variable"]
df = pd.read_csv('pima-indians-diabetes.csv', sep=',', header=None, names=cols)

X = df.iloc[:, 0:8]
y = df.iloc[:, 8]

# "Rysowanie" heatmapy
"""
Korzystając z seaborna tworzymy reprezentację graficzną z wykorzystaniem naszych
atrytutów podstawowych ze zmiennej X. Dzięki włączeniu funcji "annot" wyswietlimy
również współczynnik korelacji.
"""

sns.heatmap(X.corr(), annot = True)

# Specyfikacja danych treningowych
"""
Musimy ustawić random_state, jego wartosc nie jest tak istotna jak jego obecnosc,
ponieważ gdybysmy nie wpisali dla niego wartosci, to za każdym razem przy kompilacji
kodu byłaby generowana losowo i dane treningowe jak i testowe miałyby 
za każdym razem inne wartosci. Dzięki wpisaniu okrelonej wartosci wyniki bedą 
tożsame po każdym uruchomieniu. Random state dzieli co prawda losowo ale z ustalonym
"twistem", dzięki któremu kolejnoć jest taka sama. 

Wielkosc próbki testowej ustawiamy jako 20% danych.

Dokonujemy również standaryzacji zbioru danych: mogą one działać źle, jeśli
poszczególne cechy nie wyglądają mniej więcej jak standardowe dane o rozkładzie 
normalnym (np. Gaussa z zerową średnią i jednostkową wariancją). Standaryzuje 
ona funkcje, usuwając średnią i skalując do wariancji jednostkowej.

Funkcja fit w każdej transformacji sklearna po prostu oblicza parametry
 (np. średnią i wariancę jednostkową w przypadku StandardScaler) i zapisuje je
 jako stan obiektu wewnętrznego. Następnie można wywołać jego metodę transform, 
 aby zastosować transformację do dowolnego określonego zestawu przykładów, ale 
 zamiast tego można zastosować też fit_transform, która łączy te dwa kroki i jest 
 stosowana do początkowego dopasowania parametrów w biorze uczącym x jednocześnie 
 zwaracając przekształcone x. Najpierw obiekt wywołuje wewnętrznie fit(), a 
 następnie (transform) na tych samych danych.
 
 Najbardziej podstawowym sposobem korzystania z SVC jest jądro liniowe, 
 co oznacza, że ​​granicą decyzyjną jest linia prosta 
 (lub hiperpłaszczyzna w wyższych wymiarach). C jest klasyfiaktorem kary, okresla
 tolerancyjnosc. Zwyczajowo im większa wartosc C, tym mniejsza odpornosc na blędną
 klasyfikację.
 
 Przy pomocy funkcji predict dokonujemy przewidywań aby zestawić je później z
 danymi testowymi.

"""

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=4231, test_size=0.28)
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)

classifier = SVC(random_state=0, C=1, kernel='linear')
classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test)

# Wynik
"""Wyswietlamy procentową skutecznosc wykorzystując do tego funkcję accuracy_score
oraz nasze próbki testowe jak i przewidywane wyniki.
"""

print(accuracy_score(y_test, y_pred))


