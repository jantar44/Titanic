
# coding: utf-8

# <h1>Tytanic</h1>

# RMS Titanic – brytyjski transatlantyk typu Olympic, angielskiego towarzystwa okrętowego White Star Line (formalnie, od 1902 r., pod kontrolą amerykańskiego holdingu Juniusa Pierponta Morgana – International Mercantile Marine).
# 
# W nocy z 14 na 15 kwietnia 1912 roku, podczas swojego dziewiczego rejsu na trasie Southampton-Cherbourg-Queenstown-Nowy Jork, otarł się o górę lodową i zatonął.
# 
# Góra lodowa rozpruła kadłub statku na długości 90 metrów – długość rozdarcia poszycia kadłuba wynosiła 90 m, ale na podstawie wykonanych ultrasonografem badań rozmiaru zniszczeń stwierdzono jednoznacznie, że była to seria pęknięć, których łączna powierzchnia wynosiła nieco ponad 1 metr kwadratowy (1,18), czyli była równa powierzchni ciała dorosłego człowieka

# <b>Jest to mój pierwszy bardziej rozbudowany program.</b> Pisząc go rozpoczynam przygodę z "poważniejszym programowaniem". Wiele obliczeń można zrobić w szybszy, ładniejszy sposób, jednak próbowałem testować różne podejścia (w szczególności w rozdziale 2). 

# <h4>Etapy</h4>
# <i>
# 1. Import bibliotek i danych
# 2. Analiza, czyszczenie i  wizualizacja danych
# 3. Przygotowanie danych do modelu
# 4. Stworzenie modelu
# 5. Analiza poprawnosci rozwiązania i wizualizacja
# 6. Analiza metody
# 7. Zgłoszenie rozwiązania na kaggle
# </i>

# <b>1. Import bibliotek i danych</b>

# In[1]:


import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt #Plotting library

get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


train = pd.DataFrame(pd.read_csv('train.csv'))
test = pd.DataFrame(pd.read_csv('test.csv'))
test_w = pd.DataFrame(pd.read_csv('test.csv'))


# In[3]:


test.set_index('PassengerId', inplace=True)
train.set_index('PassengerId', inplace=True)


# <b>2. Analiza, czyszczenie i  wizualizacja danych</b>

# <i>
# 2.1. Płeć <br>
# 2.2. Wiek<br>
# 2.3. Ilośc rodzeństwa/partnerów<br>
# 2.4. Ilość rodziców/dzieci<br>
# 2.5. Klasa<br>
# 2.6. Cena biletu<br>
# 2.7. Port początkowy<br>
# </i>

# <b>2.1. Płeć</b>

# In[5]:


nan_płeć = train["Sex"][train["Sex"]=="NaN"].count()
print("Ilość pustych danych:",nan_płeć)


# <i>2 sposób</i>

# In[6]:


print("Ilość pustych danych:", train['Sex'][train['Sex'].isnull()].count())


# In[7]:


m_płeć = train['Sex'][train['Sex']=='male'].count()
f_płeć = train['Sex'][train['Sex']=='female'].count()
print('Ilość mężczyzn:',m_płeć,'\nIlość kobiet:',f_płeć)


# In[121]:


mężczyźni = train["Survived"][train["Sex"]=="male"].value_counts(normalize=True).sort_index()
kobiety = train["Survived"][train["Sex"]=="female"].value_counts(normalize=True).sort_index()
print("Mężczyźni:\n", mężczyźni, '\n')
print("Kobiety:\n", kobiety)


# Posiadamy pełne dane nt. płci. Procentowo uratowało się znacznie więcej kobiet niż mężczyzn.

# 2.2. Wiek

# In[9]:


nan_wiek = train["Age"].isnull().sum()
print("Ilość pustych danych: ", nan_wiek)


# In[10]:


wiek_5 = train['Survived'][train["Age"].between(0,5)].value_counts(normalize=True).sort_index()

wiek_10 = train['Survived'][train["Age"].between(5,10)].value_counts(normalize=True).sort_index()

wiek_20 = train['Survived'][train["Age"].between(11,20)].value_counts(normalize=True).sort_index()

wiek_30 = train['Survived'][train["Age"].between(21,30)].value_counts(normalize=True).sort_index()

wiek_40 = train['Survived'][train["Age"].between(31,40)].value_counts(normalize=True).sort_index()

wiek_50 = train['Survived'][train["Age"].between(41,50)].value_counts(normalize=True).sort_index()

wiek_60 = train['Survived'][train["Age"].between(51,60)].value_counts(normalize=True).sort_index()

wiek_70 = train['Survived'][train["Age"].between(61,70)].value_counts(normalize=True).sort_index()

wiek_80 = train['Survived'][train["Age"].between(71,80)].value_counts(normalize=True).sort_index()

wiek = [5,10,20,30,40,50,60,70,80]

ilość_s = [wiek_5[1],wiek_10[1],wiek_20[1],wiek_30[1],wiek_40[1],wiek_50[1],wiek_60[1],wiek_70[1],wiek_80[1]]


# In[21]:


plt.plot(wiek, ilość_s, 'bo', ls = '-', label = 'Survived')
plt.title('Survival rate(Age)')
z = np.polyfit(wiek, ilość_s,1)
p = np.poly1d(z)
plt.plot(wiek,p(wiek),"r--")
print("y=%.6fx+%.6f"%(z[0],z[1]))


# Istnieje 177 rekordów z brakiem danych na temat wieku.

# 2.3. Rodzeństwo

# In[22]:


nan_sib = train["SibSp"].isnull().sum()
print("Ilość pustych danych: ", nan_sib)


# In[32]:


sib_0 = train["Survived"][train["SibSp"]==0].value_counts(normalize=True).sort_index()

sib_1 = train["Survived"][train["SibSp"]==1].value_counts(normalize=True).sort_index()

sib_2 = train["Survived"][train["SibSp"]==2].value_counts(normalize=True).sort_index()

sib_3 = train["Survived"][train["SibSp"]==3].value_counts(normalize=True).sort_index()

sib_4 = train["Survived"][train["SibSp"]==4].value_counts(normalize=True).sort_index()

sib_5 = train["Survived"][train["SibSp"]==5].value_counts(normalize=True).sort_index()

sib_6 = train["Survived"][train["SibSp"]==6].value_counts(normalize=True).sort_index()

sib_7 = train["Survived"][train["SibSp"]==7].value_counts(normalize=True).sort_index()

sib_8 = train["Survived"][train["SibSp"]==8].value_counts(normalize=True).sort_index()

sib = [0,1,2,3,4,5,8]
ilość_sib_s = [sib_0[1],sib_1[1],sib_2[1],sib_3[1],sib_4[1],sib_5[0],sib_8[0]]


# In[33]:


print("Ilość przypadków: ")
for n in range (0,9):
    print(n,train["Survived"][train["SibSp"]==n].count())


# Ze względu na małą ilość przypadków przypadki o sib > 5 są mało wiarygodne (oraz nie ukazane na wykresie)

# In[34]:


plt.plot(sib[0:5], ilość_sib_s[0:5], 'bo', ls = '-', label = 'Survived')
plt.title('Survival rate(No. siblings)')
z = np.polyfit(sib[0:5], ilość_sib_s[0:5],1)
p = np.poly1d(z)
plt.plot(sib[0:5],p(sib[0:5]),"r--")
print("y=%.6fx+%.6f"%(z[0],z[1]))


# 2.4. Ilość rodziców/dzieci na pokładzie

# In[35]:


nan_par = train["Parch"].isnull().sum()
print("Ilość pustych danych: ", nan_par)


# In[36]:


par = [0,0,0,0,0,0,0] 
for n in range(0,7):
    par[n] = train["Survived"][train["Parch"]==n]


# In[37]:


for n in range (0,7):
    print(n,par[n].count())


# Odrzucane są wartości parch = 6

# In[38]:


parch = [] 
for n in range(0,6):
    parch.append(1-train["Survived"][train["Parch"]==n].value_counts(normalize=True).sort_index(ascending = True)[0])


# In[39]:


train[["Parch", "Survived"]].groupby(['Parch'], as_index=False).mean().sort_values(by='Parch', ascending=True)


# In[40]:


for n in range(0,6):
    if n == 4:
        continue
    print(n, parch[n])


# In[41]:


a =[0,1,2,3,4,5]
plt.plot(a,parch, 'bo', ls = '-', label = 'Survived')
plt.title('Survival rate(No. siblings)')
z = np.polyfit(a, parch,1)
p = np.poly1d(z)
plt.plot(a,p(a),"r--")
print("y=%.6fx+%.6f"%(z[0],z[1]))


# Widać nieznaczny wpływ widoczny dla osób posiadających 3 rodziców/dzieci - można odrzucić.

# In[42]:


train['rozmiar_rodziny'] = train['Parch'] + train['SibSp']


# In[43]:


train[['rozmiar_rodziny','Survived']].groupby('rozmiar_rodziny', as_index = False).mean().sort_values(by = 'Survived', ascending = False)


# Największe szanse przeżycia miały rodziny 3 osobowe.

# 2.5. Klasa

# In[44]:


klasa_1 = train["Survived"][train["Pclass"]==1].value_counts(normalize=True).sort_index()

klasa_2 = train["Survived"][train["Pclass"]==2].value_counts(normalize=True).sort_index()

klasa_3 = train["Survived"][train["Pclass"]==3].value_counts(normalize=True).sort_index()

klasa = [1,2,3]
klasa_s = [klasa_1[1],klasa_2[1],klasa_3[1]]


# In[48]:


print("Ilość przypadków: ")
for n in range (1,4):
    print(n,train["Survived"][train["Pclass"]==n].count())


# In[49]:


plt.plot(klasa, klasa_s, 'bo', ls = '-', label = 'Survived')
plt.title('Survival rate(Klasa)')


# Widać znaczny wpływ klasy na przeżywalność.

# 2.6. Koszt biletu

# In[65]:


fare_5 = train['Survived'][train["Fare"].between(0,5)].value_counts(normalize=True).sort_index()

fare_10 = train['Survived'][train["Fare"].between(5,10)].value_counts(normalize=True).sort_index()

fare_20 = train['Survived'][train["Fare"].between(10,20)].value_counts(normalize=True).sort_index()

fare_30 = train['Survived'][train["Fare"].between(20,30)].value_counts(normalize=True).sort_index()

fare_40 = train['Survived'][train["Fare"].between(30,40)].value_counts(normalize=True).sort_index()

fare_45 = train['Survived'][train["Fare"].between(40,45)].value_counts(normalize=True).sort_index()

fare_50 = train['Survived'][train["Fare"].between(45,50)].value_counts(normalize=True).sort_index()

fare_60 = train['Survived'][train["Fare"].between(50,60)].value_counts(normalize=True).sort_index()

fare_70 = train['Survived'][train["Fare"].between(60,70)].value_counts(normalize=True).sort_index()

fare_80 = train['Survived'][train["Fare"].between(70,80)].value_counts(normalize=True).sort_index()

fare_90 = train['Survived'][train["Fare"].between(80,90)].value_counts(normalize=True).sort_index()

fare_100 = train['Survived'][train["Fare"].between(90,100)].value_counts(normalize=True).sort_index()

fare_150 = train['Survived'][train["Fare"].between(100,150)].value_counts(normalize=True).sort_index()

fare_313 = train['Survived'][train["Fare"].between(150,313)].value_counts(normalize=True).sort_index()

fare = [5,10,20,30,40,45,50,60,70,80,90,100,150,313]

fare_s = [fare_5[1], fare_10[1], fare_20[1], fare_30[1], fare_40[1], fare_45[1], fare_50[1], fare_60[1], fare_70[1], fare_80[1], fare_90[1], fare_100[1], fare_150[1], fare_313[1]]


# In[66]:


plt.plot(fare,fare_s, 'bo', ls = '-', label = 'Survived')
plt.title('Survival rate(Fare)')
z = np.polyfit(fare, fare_s,1)
p = np.poly1d(z)
plt.plot(fare,p(fare),"r--")
print("y=%.6fx+%.6f"%(z[0],z[1]))


# Podobnie jak w przypadku klasy widać duży wpływ kosztu biletu na przeżywalność

# 2.7. Port początkowy

# In[67]:


train[['Embarked', 'Survived']].groupby(['Embarked'], as_index=False).mean().sort_values(by='Embarked', ascending=True)


# In[68]:


embark_S = train["Survived"][train["Embarked"]=="S"].value_counts(normalize=True).sort_index()*100

embark_C= train["Survived"][train["Embarked"]=="C"].value_counts(normalize=True).sort_index()*100

embark_Q = train["Survived"][train["Embarked"]=="Q"].value_counts(normalize=True).sort_index()*100


# In[71]:


print(embark_C,'\n',embark_Q,'\n', embark_S)


# In[72]:


embark=['S','C','Q']
embark_s = [embark_S[1],embark_C[1],embark_Q[1]]


# In[73]:


plt.plot(embark,embark_s, 'bo', label = 'Survived')
plt.title('Survival rate(Embarkement)')


# <b>Podsumowanie</b>
# 
# 
# W rozważaniach uznaję imiona, numery kabin oraz numery biletów za nieistotne dla przeprowadzanej analizy. Numery biletów i kabin mogą być usunięte.

# In[74]:


train.drop(['Ticket', 'Cabin','rozmiar_rodziny'], axis = 1,inplace = True)
test.drop(['Ticket', 'Cabin'], axis = 1, inplace = True)


# In[75]:


train['Survived'] = train['Survived'].astype(int)


# <h3>3. Przygotowanie danych do modelu</h3>

# In[122]:


data = train.append(test,sort=False)


# Uzupełnienie brakujących danych średnią dla danej "klasy", płci oraz portu.

# In[78]:


data[['Pclass', 'Age', 'Sex']].groupby(['Pclass','Sex'], as_index=True).mean()


# In[79]:


data['Sex'] = data['Sex'].map({'female': 0, 'male': 1}).astype(int)


# In[80]:


for i in range(1,1310):
    if data.loc[i, 'Sex'] == 1 and (data.loc[i, 'Pclass'] == 1):
        if np.isnan(data.loc[i, 'Age']):
            data.loc[i, 'Age'] = 41
    if data.loc[i, 'Sex'] == 0 and (data.loc[i, 'Pclass'] == 1):
        if np.isnan(data.loc[i, 'Age']):
            data.loc[i, 'Age'] = 37
    if data.loc[i, 'Sex'] == 1 and (data.loc[i, 'Pclass'] == 2):
        if np.isnan(data.loc[i, 'Age']):
            data.loc[i, 'Age'] = 31
    if data.loc[i, 'Sex'] == 0 and (data.loc[i, 'Pclass'] == 2):
        if np.isnan(data.loc[i, 'Age']):
            data.loc[i, 'Age'] = 27
    if data.loc[i, 'Sex'] == 1 and (data.loc[i, 'Pclass'] == 3):
        if np.isnan(data.loc[i, 'Age']):
            data.loc[i, 'Age'] = 26
    if data.loc[i, 'Sex'] == 0 and (data.loc[i, 'Pclass'] == 3):
        if np.isnan(data.loc[i, 'Age']):
            data.loc[i, 'Age'] = 22


# Łączenie rozdziału rodzice-dzieci z rodzęństwo tworząc "Rozmar_rodziny" 

# In[81]:


data['Rozmiar_rodziny'] = data['SibSp'] + data['Parch']


# In[82]:


data.drop(['SibSp', 'Parch'],axis=1,inplace=True)


# Kobieta = 0; Mężczyzna = 1

# Port: S=1; C=2; Q=3. Są 2 brakujące dane - przypisane do S

# In[84]:


data['Embarked'].fillna('S',inplace=True)


# In[85]:


data['Embarked'] = data['Embarked'].map({'S':1, 'C':2, 'Q':3}).astype(int)


# In[86]:


data.drop(['Name'],axis=1,inplace=True)


# In[87]:


for i in range(1,1310):
    if np.isnan(data.loc[i, 'Fare']):
        print(data.loc[i, ['Sex', 'Pclass', 'Age']])


# In[88]:


data[['Pclass', 'Fare', 'Sex']].groupby(['Pclass','Sex'], as_index=True).mean()


# In[89]:


data['Fare'].fillna(12.42,inplace=True)


# In[90]:


data.loc[data['Fare'] <= 7.85, 'Fare'] = 0
data.loc[(data['Fare'] > 7.85) & (data['Fare'] <= 10.5), 'Fare'] = 1
data.loc[(data['Fare'] > 10.5) & (data['Fare'] <= 21.68), 'Fare'] = 2
data.loc[(data['Fare'] > 21.68) & (data['Fare'] <= 41.6), 'Fare'] = 3
data.loc[ data['Fare'] > 41.6, 'Fare'] = 4
data['Fare'] = data['Fare'].astype(int)


# In[91]:


data.loc[data['Age'] <= 2, 'Age'] = 0
data.loc[(data['Age'] > 2) & (data['Age'] <= 5), 'Age'] = 1
data.loc[(data['Age'] > 5) & (data['Age'] <= 10), 'Age'] = 2
data.loc[(data['Age'] > 10) & (data['Age'] <= 15), 'Age'] = 3
data.loc[(data['Age'] > 15) & (data['Age'] <= 20), 'Age'] = 4
data.loc[(data['Age'] > 20) & (data['Age'] <= 25), 'Age'] = 5
data.loc[(data['Age'] > 25) & (data['Age'] <= 30), 'Age'] = 6
data.loc[(data['Age'] > 30) & (data['Age'] <= 40), 'Age'] = 7
data.loc[(data['Age'] > 40) & (data['Age'] <= 50), 'Age'] = 8
data.loc[(data['Age'] > 50) & (data['Age'] <= 65), 'Age'] = 9
data.loc[ data['Age'] > 65, 'Age'] = 10
data['Age'] = data['Age'].astype(int)


# In[92]:


data.drop(data.iloc[:, 9:270].head(0).columns, axis=1, inplace = True)


# In[100]:


nauka = data.iloc[:891,:]

wynik = data.iloc[891:,:]

nauka_S = train['Survived']

nauka_S = nauka_S.reset_index()
nauka_S = nauka_S.drop(['PassengerId'], axis = 1)

nauka_D = nauka.drop(['Survived'], axis = 1)

nauka_D = nauka_D.reset_index()
nauka_D = nauka_D.drop(['PassengerId'], axis = 1)

wynik_D = wynik.drop(['Survived'], axis = 1)

wynik_D = wynik_D.reset_index()
wynik_D = wynik_D.drop(['PassengerId'], axis = 1)


# In[103]:


import matplotlib.pyplot as plot
plot.pcolor(data.corr())
plot.show()


# In[104]:


data = data[['Sex', 'Pclass', 'Fare', 'Embarked', 'Age','Rozmiar_rodziny','Survived']]


# <h3> 4. Model </h3>

# <h4>4.1 Regresja logiczna </h4>

# In[105]:


# machine learning
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import Perceptron
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier


# In[106]:


reglog = LogisticRegression()


# In[107]:


reglog.fit(nauka_D, nauka_S)
Y_pred = reglog.predict(wynik_D)
acc_log = round(reglog.score(nauka_D, nauka_S) * 100, 2)
acc_log


# In[109]:


svc = SVC()
svc.fit(nauka_D, nauka_S)
Y_pred = svc.predict(wynik_D)
acc_svc = round(svc.score(nauka_D, nauka_S) * 100, 2)
acc_svc


# In[110]:


knn = KNeighborsClassifier(n_neighbors = 3)
knn.fit(nauka_D, nauka_S)
Y_pred = knn.predict(wynik_D)
acc_knn = round(knn.score(nauka_D, nauka_S) * 100, 2)
acc_knn


# In[111]:


gaussian = GaussianNB()
gaussian.fit(nauka_D, nauka_S)
Y_pred = gaussian.predict(wynik_D)
acc_gaussian = round(gaussian.score(nauka_D, nauka_S) * 100, 2)
acc_gaussian


# In[112]:


perceptron = Perceptron()
perceptron.fit(nauka_D, nauka_S)
Y_pred = perceptron.predict(wynik_D)
acc_perceptron = round(perceptron.score(nauka_D, nauka_S) * 100, 2)
acc_perceptron


# In[113]:


linear_svc = LinearSVC()
linear_svc.fit(nauka_D, nauka_S)
Y_pred = linear_svc.predict(wynik_D)
acc_linear_svc = round(linear_svc.score(nauka_D, nauka_S) * 100, 2)
acc_linear_svc


# In[114]:


sgd = SGDClassifier()
sgd.fit(nauka_D, nauka_S)
Y_pred = sgd.predict(wynik_D)
acc_sgd = round(sgd.score(nauka_D, nauka_S) * 100, 2)
acc_sgd


# In[115]:


decision_tree = DecisionTreeClassifier()
decision_tree.fit(nauka_D, nauka_S)
Y_pred_d = decision_tree.predict(wynik_D)
acc_decision_tree = round(decision_tree.score(nauka_D, nauka_S) * 100, 2)
acc_decision_tree


# In[120]:


random_forest1 = RandomForestClassifier(n_estimators = 550, oob_score = True, n_jobs = -1, random_state =5)
random_forest1.fit(nauka_D, nauka_S)
Y_pred1 = random_forest1.predict(wynik_D)
random_forest1.score(nauka_D, nauka_S)
acc_random_forest1 = round(random_forest1.score(nauka_D, nauka_S) * 100, 2)
acc_random_forest1


# In[118]:


submission = pd.DataFrame({
        "PassengerId": test_w["PassengerId"],
        "Survived": Y_pred2
    })


# In[119]:


submission.to_csv('/home/jan/Dokumenty/Python_kaggle/Tytanic/submission.csv', index=False)

