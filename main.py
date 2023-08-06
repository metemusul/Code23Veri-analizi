import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import seaborn as sns
from imblearn.over_sampling import RandomOverSampler
from sklearn.preprocessing import StandardScaler
from catboost import CatBoostClassifier
import plotly.graph_objects as go

seed_value = 0
np.random.seed(seed_value)

data = pd.read_csv('/kaggle/input/diabetes-data-set/diabetes.csv')

data.info()
data.head()
sns.pairplot(data)

sns.violinplot ( data= data ["BMI"], color="pink", split=False, cut=0, bw=.3, inner="stick", scale="count")
plt.show()

# countplot
plt.figure()
sns.countplot(x = data["Outcome"], data = data, saturation = 1)
plt.title("Count of Outcome Values")

plt.figure(figsize=(30, 6))
sns.boxplot(data=data)
plt.xticks(rotation=45)
plt.show()

plt.figure(figsize=(50, 25))
sns.heatmap(data.corr(), annot=True, cmap='coolwarm')
plt.show()

plt.figure(figsize=(20, 10))
sns.scatterplot(x='Glucose', y='BMI', hue='Outcome', data=data)
plt.show()

plt.figure(figsize=(20, 10))
sns.countplot(x='Outcome', data=data)
plt.show()

plt.figure(figsize=(6, 6))
data['Outcome'].value_counts().plot(kind='pie', autopct='%1.1f%%', colors=['lightblue', 'lightcoral'])
plt.axis('equal')
plt.show()

# Violin Grafiği
plt.figure(figsize=(8, 6))
sns.violinplot(x='Outcome', y='Age', data=data)
plt.show()


# 3B Grafikler (örneğin, plotly ile)
import plotly.express as px
fig = px.scatter_3d(data, x='Glucose', y='BMI', z='Insulin', color='Outcome')
fig.show()

data[['Glucose', 'BloodPressure', 'BMI', 'Insulin', 'SkinThickness']] = data[['Glucose', 'BloodPressure', 'BMI', 'Insulin', 'SkinThickness']].replace(0, np.nan)
#Eksik değerleri işleyelim Seçili sütunlardaki 0 değerlerini NaN ile değiştirelim

data[['Glucose', 'BloodPressure', 'BMI', 'Insulin', 'SkinThickness']] = data[['Glucose', 'BloodPressure', 'BMI', 'Insulin', 'SkinThickness']].fillna(data[['Glucose', 'BloodPressure', 'BMI', 'Insulin', 'SkinThickness']].mean())


#NaN değerlerini ilgili sütunların ortalaması ile dolduralım

X = data.drop(columns=['Outcome'])
y = data['Outcome']


#Veriyi özellikler (X) ve hedef (y) olarak ayıralım

ros = RandomOverSampler(random_state=0)
X_ros, y_ros = ros.fit_resample(X, y)

## Sınıf dengesizliğini ele almak için RandomOverSampler uygulayalım

X_train, X_test, y_train, y_test = train_test_split(X_ros, y_ros, test_size=0.25, random_state=0)

#Eğitim ve test verilerini ayıralım

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

##Eğitim verilerinde özellik ölçeklendirmesi yapalım

model = CatBoostClassifier(iterations=100, learning_rate=0.1, depth=8, random_seed=seed_value)
model.fit(X_train, y_train)

# CatBoostClassifier'ı başlatıp eğitelim

y_pred = model.predict(X_test)


#Test verileri üzerinde tahmin yapalım

accuracy = accuracy_score(y_test, y_pred)
print("accuracy :", accuracy)
print("Karmaşıklık Matrisi:")
print(confusion_matrix(y_test, y_pred))
print("Sınıflandırma Raporu:")
print(classification_report(y_test, y_pred))