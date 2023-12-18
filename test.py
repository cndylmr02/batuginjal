
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_recall_curve
from sklearn.preprocessing import MinMaxScaler

#Title
st.title('KELOMPOK 9')
("""
#### Cindy Laundiya Maretha (210411100037)
#### Laili Nur Afifah (210411100174)
#### Nur Anisa (210411100185)
#### Fiza Ayu Rahma Sari (210411100215)
""")
st.title('Prediksi Batu Ginjal Berdasarkan Analisis Urine ')
st.write("""
#### Dengan 4 Pilihan Metode Klasifikasi
""")
st.write("""
###### Dataset diambil dari halaman https://www.kaggle.com/competitions/playground-series-s3e12/data
###### Untuk Tipe Datanya sendiri berupa tipe data Numerik
###### Data tersebut berisi hasil Analisis dari tes Urine dimana terdapat atribut atau fitur seperti gravity, ph, osmo, cond, urea, calc, dan terdapat target yang berisi hasil klasifikasinya 
""")
# Dataset
dataset = pd.read_csv(r"data.csv")

st.write('')
st.write('## Dataset')
st.dataframe(data=dataset)

# Missing Value
st.write('## Missing Value')
st.write(dataset.isna().sum())


fitur = dataset[['id', 'gravity', 'ph', 'osmo', 'cond', 'urea', 'calc', 'target']]
fitur = dataset.iloc[:, :7]
target = dataset['target']


# Normalisasi Data
st.write("## Dataset Normalisasi")
fitur_tanpa_target = fitur.drop('id', axis=1)
min_max_scaler = MinMaxScaler(feature_range=(0, 1))
fitur = min_max_scaler.fit_transform(fitur_tanpa_target)
st.dataframe(fitur)

#Tambah Klasifikasi
algoritma = st.sidebar.selectbox(
    'Pilih Algoritma', 
    ('KNN', 'Decision Tree', 'Naive Bayes')
)

#Fungsi tambah parameter
def tambah_parameter(nama_algoritma):
    params = dict()
    if nama_algoritma == 'KNN':
        metrics = st.sidebar.selectbox(
            'Pilih Metrics',
            ('euclidean', 'manhattan', 'minkowski')
        )
        params['metrics_select'] = metrics
        K = st.sidebar.slider ('K', 1, 10)
        params['K'] = K

    elif nama_algoritma == 'Decision Tree':
        max_depth = st.sidebar.slider('max_depth', 2, 15)
        params['max_depth'] = max_depth
    return params

params = tambah_parameter(algoritma)

#Fungsi pilih klasifikasi
def pilih_klasifikasi(nama_algoritma, params):
    algo = None
    if nama_algoritma == 'KNN':
        algo = KNeighborsClassifier(n_neighbors=params['K'], metric=params['metrics_select'])
    elif nama_algoritma == 'Decision Tree':
        algo = DecisionTreeClassifier(max_depth=params['max_depth'])
    elif nama_algoritma == 'Naive Bayes':
        algo = GaussianNB()
    return algo

algo = pilih_klasifikasi(algoritma, params)

### PROSES KLASIFIKASI ###

xTrain, xTest, yTrain, yTest = train_test_split(fitur, target, test_size=0.2, random_state=1234)

algo.fit(xTrain, yTrain)
pred = algo.predict(xTest)
acc = accuracy_score(yTest, pred)

col1, col2 = st.columns(2)
with col1:
    st.write(f'#### Algoritma = {algoritma}')
with col2:
    st.write('#### Akurasi = {}%'.format(round(acc*100)))

precision, recall, threshold = precision_recall_curve(yTest, pred)

with col1:
    st.write(f"Precision  =  {precision[0]}")
with col2:
    st.write(f"Recall  =  {recall[0]}")


# Menampilkan Grafik Performa
st.set_option('deprecation.showPyplotGlobalUse', False)
st.write(f"## Grafik Performa")
plt.plot(threshold, precision[:-1], label='Precision')
plt.plot(threshold, recall[:-1], label='Recall')
plt.plot(acc*np.ones_like(threshold), label='Accuracy')
plt.xlabel('Threshold')
plt.ylabel('Score')
plt.title('Performance Metrics')
plt.legend()
st.pyplot(plt.show())

# Confusion Matrix
st.write(f"## Confusion Matrix")
f, ax = plt.subplots(figsize=(8, 5))
sns.heatmap(confusion_matrix(yTest, pred), annot=True, fmt=".0f", ax=ax)
plt.xlabel("Data Predict")
plt.ylabel("Data Train")
st.set_option('deprecation.showPyplotGlobalUse', False)
st.pyplot(plt.show())

# The Prediksi
st.write('# Prediksi Data Baru')

col1, col2 = st.columns(2)

with col1:
    gravity = st.text_input("gravity")
    ph = st.text_input("ph (4 - 8)")
    osmo = st.text_input(" osmo (100 - 1200)")
with col2:
    cond = st.text_input("cond (5 - 40)")
    urea = st.text_input("urea (10 - 550)")
    calc = st.text_input("calc (0 - 15)")

# Button Predict
predicted_class = ''
if st.button("Prediksi"):
    process_data = [[float(gravity), float(ph), float(osmo), float(cond), float(urea), float(calc)]]
    process_data = min_max_scaler.fit_transform(process_data)
    predict_DM = algo.predict(process_data)
    if predict_DM[0] == 1:
        predicted_class = 'negatif'
    elif predict_DM[0] == 0:
        predicted_class = 'positif'
    st.success(predicted_class)
