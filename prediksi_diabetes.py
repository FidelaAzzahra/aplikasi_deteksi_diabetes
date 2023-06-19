# Aziza Azka Sajida
# Fidela Azzahra
# Salsabila Ayu Anjelina
# Salma Shafira Fatya Ardyani

import streamlit as st # import library streamlit
import pandas as pd # library untuk membaca dataset dan analisis datanya
import seaborn as sns # library untuk visualisasi data dan membuat heatmap
import matplotlib.pyplot as plt # library untuk visualisasi data
import matplotlib.ticker as ticker # library untuk mengatur penanda sumbu plot

# def main merupakan fungsi main. Penanda titik utama program dijalankan
def main():
    st.title('Prediksi Diabetes Dengan KNN') # untuk judul
    st.subheader('')

    data = pd.read_csv('diabetes.csv') # Membaca dataset diabetes dari file CSV dengan menggunakan fungsi read_csv dari pandas. Dataset akan disimpan dalam variabel data.


    # Pengguna diminta untuk memasukkan nilai k, level glucose, insulin, bmi, dan umur
    k = st.number_input('Masukkan nilai dari K', value=5, step=1)
    user_glucose = st.number_input('Masukkan level glukosa', value=60)
    user_insulin = st.number_input('Masukkan level insulin', value=35)
    user_bmi = st.number_input('Masukkan nilai BMI', value=25.0)
    user_age = st.number_input('Masukkan umur', value=30)


    user_data = [user_glucose, user_insulin, user_bmi, user_age] # membuat variabel user_data yang isinya user_glucose, user_insulin, user_bmi, user_age yang akan di input oleh pengguna
    data_arr = [] # untuk menyimpan data dari glucose, insulin, dll
    dist_arr = [] # untuk menyimpan jarak antar data pengguna dengan dataset


    if st.button('Cek Kesehatan'): # kalau prediksi ditekan
        st.subheader('')
        st.subheader('')
        for i in range(len(data)): #  perulangan for akan dilakukan sebanyak jumlah data dalam dataset.
            for j in ['Glucose', 'Insulin', 'BMI', 'Age']: # data glucose, dll akan dimasukkan kedalam data_arr
                elem = data[j][i]
                data_arr.append(elem)
            #  jarak antara data pengguna dengan data pada dataset dihitung menggunakan euclidean distance dan disimpan dalam dist_arr
            dist = ((user_data[0] - data_arr[0]) ** 2 + (user_data[1] - data_arr[1]) ** 2 +
                    (user_data[2] - data_arr[2]) ** 2 + (user_data[3] - data_arr[3]) ** 2) ** 0.5
            dist_arr.append(dist)
            data_arr = [] # data akan dikosongkan untuk perhitungan selanjutnya
            

        result = pd.DataFrame(dist_arr, columns=['Distance']) # untuk pembentukan data frame baru dengan menggunakan dist_arr sebagai kolom distance.
        
        # kemudian menambahkan kolom glucose, insulin, bmi, umur dari dataset kedalam result
        result['Glucose'] = data['Glucose']
        result['Insulin'] = data['Insulin']
        result['BMI'] = data['BMI']
        result['Age'] = data['Age']
        result['Diabetes'] = data['Outcome']


        # mengambil K baris terkecil dari data frame result berdasarkan kolom distance dengan fungsi nsmallest dari library pandas. Untuk hasil disimpan di result_smallest
        result_smallest = result.nsmallest(k, 'Distance')


        # untuk menampilkan pesan hasil prediksi diabetes
        # Jika nilai result_smallest > 0 = mengidap diabetes
        # selain itu = tidak mengidap diabetes
        st.subheader('')
        st.subheader('Hasil Prediksi')
        if result_smallest['Diabetes'].sum() > 0:
            st.error('Berdasarkan dataset yang diberikan, pengguna diprediksi MENGIDAP DIABETES')
        else:
            st.success('Berdasarkan dataset yang diberikan, pengguna diprediksi TIDAK MENGIDAP DIABETES')
        
        
        st.subheader('')
        st.subheader('')
        st.subheader('Data Terdekat') # tampilkan judul data terdekat
        st.table(result_smallest[['Glucose', 'Insulin', 'BMI', 'Age', 'Diabetes']]) # untuk menampilkan keterangan pengguna serta data yang diambil dari nilai K untuk glucose, insulin, dan hasil diabetes (1 atau 0)
        st.subheader('')
        

        st.subheader('Heatmap Prediksi Diabetes') # untuk menampilkan heatmap / peta panas
        fig, ax = plt.subplots(figsize=(8, 6))
        heatmap = sns.heatmap(result_smallest[['Glucose', 'Insulin', 'BMI', 'Age']].corr(), annot=True,
                              cmap='coolwarm', linewidths=0.5, ax=ax)
        heatmap.xaxis.set_major_locator(ticker.IndexLocator(base=1, offset=0.5))
        heatmap.yaxis.set_major_locator(ticker.IndexLocator(base=1, offset=0.5))
        plt.xticks(rotation=45)
        plt.yticks(rotation=0)
        st.pyplot(fig) # menampilkan heatmap menggunakan st.pyplot() 

        
# memanggil fungsi main jika program dijalankan / run
if __name__ == '__main__':
    main()
