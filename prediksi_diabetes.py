import streamlit as st # library untuk import streamlit nya
import pandas as pd # library untuk membaca dataset dan analisis datanya
import seaborn as sns # library untuk visualisasi data dan membuat heatmap
import matplotlib.pyplot as plt # library untuk visualisasi data
import matplotlib.ticker as ticker # library untuk mengatur penanda sumbu plot

from sklearn.neighbors import KNeighborsClassifier # library scikit-learn yang digunakan untuk mengimplementasikan algoritma K-Nearest Neighbors (KNN)
from sklearn.model_selection import train_test_split # fungsi dalam library scikit-learn yang digunakan untuk membagi dataset menjadi subset train dan test secara acak
from sklearn.preprocessing import StandardScaler # kelas dalam library scikit-learn yang digunakan untuk melakukan penskalaan fitur pada dataset
from sklearn.metrics import accuracy_score #  metrics digunakan untuk menghitung nilai akursi


# def main merupakan fungsi main. Penanda titik utama program dijalankan
def main():
    st.title('Prediksi Diabetes Algoritma KNN')
    st.subheader('')

    # Membaca dataset diabetes dari file CSV dengan menggunakan fungsi read_csv dari pandas. Dataset akan disimpan dalam variabel data.
    data = pd.read_csv('diabetes.csv')

    # disini user akan menginput data. value merupakan contoh nilai default nya
    user_pregnancies = st.number_input('Masukkan nilai kehamilan', value=1)
    user_glucose = st.number_input('Masukkan level glukosa', value=85)
    user_blood_pressure = st.number_input('Masukkan tekanan darah', value=66)
    user_skin_thickness = st.number_input('Masukkan ketebalan kulit', value=29)
    user_insulin = st.number_input('Masukkan level insulin', value=0)
    user_bmi = st.number_input('Masukkan nilai BMI', value=26.60)
    user_diabetes_pedigree = st.number_input('Masukkan nilai Diabetes Pedigree Function', value=0.351)
    user_age = st.number_input('Masukkan umur', value=31)

    # membuat variabel user_data yang isinya user_glucose, user_insulin, dll yang akan di input oleh pengguna
    user_data = [[user_pregnancies, user_glucose, user_blood_pressure, user_skin_thickness, user_insulin, user_bmi, user_diabetes_pedigree, user_age]]
    # membuat variabel X yang isinya Pregnancies dan lain lain
    X = data[['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']]
    y = data['Outcome'] # membuat variabel y yang isinya outcome

    # preprocessing
    scaler = StandardScaler() # membuat objek scaler dari kelas StandardScaler
    X_scaled = scaler.fit_transform(X) # Menyimpan data X yang telah diubah skala menggunakan scaler
    user_data_scaled = scaler.transform(user_data) # mengubah skala data pengguna (user_data) menggunakan scaler yang sama

    # dataset di split dengan menggunakan train test split
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    if st.button('Cek Kesehatan'): # jika button cek kesehatan ditekan maka...
        st.subheader('')

        knn = KNeighborsClassifier(n_neighbors=5, metric='manhattan')  # Menggunakan metrik Manhattan
        # mengolah model menggunakan data latih X_train dan label y_train
        # metode fit digunakan untuk melatih model dengan data yang ada.
        knn.fit(X_train, y_train) 

        # KNN yang telah diolah digunakan untuk melakukan prediksi terhadap data pengguna yang telah diubah (user_data_scaled). 
        # metode predict digunakan untuk menghasilkan prediksi berdasarkan data yang user input
        user_prediction = knn.predict(user_data_scaled) 

        st.subheader('')
        st.subheader('')
        st.subheader('Hasil Prediksi')

        if user_prediction == 0: # jika 0 maka tidak mengidap diabetes
            st.success('Berdasarkan dataset yang diberikan, pengguna diprediksi TIDAK MENGIDAP DIABETES dengan tingkat akurasi sebesar {:.2f}%'.format(accuracy_score(y_test, knn.predict(X_test)) * 100))
        else: # selain itu mengidap
            st.error('Berdasarkan dataset yang diberikan, pengguna diprediksi MENGIDAP DIABETES dengan tingkat akurasi sebesar {:.2f}%'.format(accuracy_score(y_test, knn.predict(X_test)) * 100))

        st.subheader('')
        st.subheader('')
        st.subheader('Heatmap Prediksi Diabetes')
        fig, ax = plt.subplots(figsize=(12, 9)) # mengatur ukuran dari heatmap 
        
        # memilih kolom yang relevan berdasarkan hasil prediksi untuk heatmap
        if user_prediction == 0: # data komplit ditampilkan jika pengguna tidak mengidap diabetes
            selected_columns = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']
        else: # data yang ditampilkan kurang jika pengguna mengidap diabetes
            selected_columns = ['Glucose', 'BloodPressure', 'SkinThickness', 'BMI', 'Age']
        
        heatmap = sns.heatmap(data[selected_columns].corr(), annot=True, cmap='coolwarm', linewidths=0.5, ax=ax)
        heatmap.xaxis.set_major_locator(ticker.IndexLocator(base=1, offset=0.5))
        heatmap.yaxis.set_major_locator(ticker.IndexLocator(base=1, offset=0.5))
        plt.xticks(rotation=45)
        plt.yticks(rotation=0)
        st.pyplot(fig) # menampilkan gambar (heatmap) dalam antarmuka aplikasi Streamlit menggunakan fungsi st.pyplot.
        
    
if __name__ == '__main__':
    main()
