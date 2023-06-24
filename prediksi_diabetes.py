# Aziza Azka Sajida
# Fidela Azzahra
# Salsabila Ayu Anjelina
# Salma Shafira Fatya Ardyani

import streamlit as st # library untuk import streamlit nya
import pandas as pd # library untuk membaca dataset dan analisis datanya
import seaborn as sns # library untuk visualisasi data dan membuat heatmap
import matplotlib.pyplot as plt # library untuk visualisasi data
import matplotlib.ticker as ticker # library untuk mengatur penanda sumbu plot
from sklearn.neighbors import KNeighborsClassifier # library scikit-learn yang digunakan untuk mengimplementasikan algoritma K-Nearest Neighbors (KNN)
from sklearn.model_selection import train_test_split # fungsi dalam library scikit-learn yang digunakan untuk membagi dataset menjadi subset train dan test secara acak
from sklearn.preprocessing import StandardScaler # kelas dalam library scikit-learn yang digunakan untuk melakukan penskalaan fitur pada dataset

# def main merupakan fungsi main. Penanda titik utama program dijalankan
def main():
    st.title('Prediksi Diabetes dengan KNN')
    st.subheader('')

    # Membaca dataset diabetes dari file CSV dengan menggunakan fungsi read_csv dari pandas. Dataset akan disimpan dalam variabel data.
    data = pd.read_csv('diabetes.csv')

    # pada baris kode ini user diminta untuk memasukkan data
    k = st.number_input('Masukkan nilai dari K', value=5, step=1)
    user_glucose = st.number_input('Masukkan level glukosa', value=60)
    user_insulin = st.number_input('Masukkan level insulin', value=35)
    user_bmi = st.number_input('Masukkan nilai BMI', value=25.0)
    user_age = st.number_input('Masukkan umur', value=30)

    # membuat variabel user_data yang isinya user_glucose, user_insulin, user_bmi, user_age yang akan di input oleh pengguna
    user_data = [[user_glucose, user_insulin, user_bmi, user_age]] 
    X = data[['Glucose', 'Insulin', 'BMI', 'Age']] # X digunakan untuk menyimpan data dari glucose, insulin, dll
    y = data['Outcome'] # y digunakan untuk menyimpan data Outcome

    # preprocessing
    scaler = StandardScaler()  # Membuat objek scaler dari kelas StandardScaler
    X_scaled = scaler.fit_transform(X)  # Menyimpan data X yang telah diubah skala menggunakan scaler
    user_data_scaled = scaler.transform(user_data)  # Mengubah skala data pengguna (user_data) menggunakan scaler yang sama

    # dataset di split dengan menggunakan train test split
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    if st.button('Cek Kesehatan'): # ketika button cek kesehatan ditekan
        st.subheader('')
        st.subheader('')

        # model untuk KNN
        knn = KNeighborsClassifier(n_neighbors=k)  # Membuat objek KNeighborsClassifier dengan jumlah tetangga k yang ditentukan
        knn.fit(X_train, y_train)  # Melatih model KNN menggunakan data latih X_train dan label y_train

        # untuk memprediksi status kesehatan dari user
        user_prediction = knn.predict(user_data_scaled)  # Memprediksi status kesehatan pengguna menggunakan data pengguna yang telah diubah skala

        st.subheader('')
        st.subheader('Hasil Prediksi') 
        if user_prediction[0] == 1: # melakukan pengecekan. Kalau 1 berarti mendertia. Kalau 0 tidak
            st.error('Berdasarkan dataset yang diberikan, pengguna diprediksi MENGIDAP DIABETES')
        else:
            st.success('Berdasarkan dataset yang diberikan, pengguna diprediksi TIDAK MENGIDAP DIABETES')

        st.subheader('')
        st.subheader('')
        st.subheader('Data Terdekat')
        # menampilkan tabel yang berisi data terdekat dengan data pengguna berdasarkan metode KNN.
        # data terdekat diambil dari data menggunakan indeks yang dihasilkan dari knn.kneighbors(user_data_scaled, return_distance=False).flatten().
        
        st.table(data.loc[knn.kneighbors(user_data_scaled, return_distance=False).flatten()][['Glucose', 'Insulin', 'BMI', 'Age', 'Outcome']])
        st.subheader('')

        st.subheader('Heatmap Prediksi Diabetes')
        fig, ax = plt.subplots(figsize=(8, 6)) # Membuat objek gambar (fig) dan objek sumbu (ax) dengan ukuran (8, 6) dalam satuan inci.
        # Membuat heatmap menggunakan sns.heatmap.
        # data yang digunakan adalah data terdekat dengan data pengguna berdasarkan metode KNN.
        # hanya kolom 'Glucose', 'Insulin', 'BMI', dan 'Age' yang digunakan.
        # corr() digunakan untuk menghitung korelasi antara kolom-kolom tersebut.
        # annot = True mengaktifkan anotasi nilai pada heatmap.
        # cmap = 'coolwarm' menentukan skema warna untuk heatmap.
        # linewidths=0.5 menentukan ketebalan garis antar sel pada heatmap.
        # ax=ax mengarahkan heatmap untuk digambar pada sumbu yang telah dibuat sebelumnya.
        heatmap = sns.heatmap(data.loc[knn.kneighbors(user_data_scaled, return_distance=False).flatten()][['Glucose', 'Insulin', 'BMI', 'Age']].corr(), annot=True, cmap='coolwarm', linewidths=0.5, ax=ax)
        heatmap.xaxis.set_major_locator(ticker.IndexLocator(base=1, offset=0.5)) # menentukan penempatan penanda sumbu x pada setiap label kolom.
        heatmap.yaxis.set_major_locator(ticker.IndexLocator(base=1, offset=0.5)) # menentukan penempatan penanda sumbu y pada setiap label baris.
        plt.xticks(rotation=45) # memutar label sumbu x sebesar 45 derajat.
        plt.yticks(rotation=0) # sumbu y tidak diputar / tetap
        st.pyplot(fig) # Menampilkan gambar (heatmap) dalam antarmuka aplikasi Streamlit menggunakan fungsi st.pyplot.


if __name__ == '__main__':
    main()
