import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix
import seaborn as sns
from sklearn.tree import plot_tree

# Memuat model yang telah disimpan
model = joblib.load("WebApps/decision_tree_model.pkl")

# Memuat data yang digunakan untuk pelatihan
df = pd.read_csv('Data/Monkey_Pox_Cases_Worldwide_Predict.csv')
df = df.dropna()  # Menghapus baris dengan nilai NaN

# Menentukan level risiko berdasarkan median
median_cases = df['Confirmed_Cases'].median()
df['Risk_Level'] = df['Confirmed_Cases'].apply(lambda x: 'High' if x > median_cases else 'Low')

# Menentukan fitur dan target
X = df[['Suspected_Cases', 'Hospitalized', 'Travel_History_Yes', 'Travel_History_No']]
y = df['Risk_Level']

# Melakukan prediksi dengan model yang sudah dilatih
y_pred = model.predict(X)

# Sidebar untuk navigasi menggunakan selectbox yang lebih ringkas
st.sidebar.title("Navigation")
st.sidebar.write("Go to")

# Membuat selectbox di sidebar dengan opsi yang lebih ringkas
selection = st.sidebar.selectbox("Pilih Halaman", [
    "Home",
    "Dataset",
    "Hasil Analisa Data",
    "Visualisasi Tree",
    "Model Prediksi Interaktif",
    "Kesimpulan",
    "About Us"
])

# Home
if selection == "Home":
    st.title("Monkeypox Risk Prediction Dashboard")
    
    st.write("""
             Monkeypox, atau cacar monyet, adalah penyakit zoonosis yang disebabkan oleh virus Monkeypox dari genus *Orthopoxvirus* dalam keluarga *Poxviridae*. 
             Pertama kali terdeteksi pada monyet di Copenhagen pada tahun 1958, kasus cacar monyet kembali muncul pada 6 Mei 2022, dimulai dari seorang individu 
             yang melakukan perjalanan ke Nigeria. Pada 21 Mei 2022, kasus dilaporkan di Eropa, Amerika, Asia, Afrika Utara, dan Australia. WHO mendeklarasikan 
             wabah ini sebagai "Evolving Health Threat" pada 27 Juni 2022.

            Di Indonesia, kasus pertama terdeteksi pada individu yang memiliki riwayat perjalanan luar negeri dengan dugaan penularan melalui kontak erat. 
            Karena potensi hewan penular ada di Indonesia, kewaspadaan dan edukasi masyarakat diperlukan untuk mengantisipasi penyakit ini. Studi ini bertujuan 
            untuk memberikan informasi dan edukasi agar masyarakat lebih waspada terhadap cacar monyet.
             """)
# Dashboard: Menampilkan dataset
elif selection == "Dataset":
    st.title("Dataset Overview")
    
    # Menambahkan deskripsi tentang dataset
    st.write("""
        Dataset ini berisi data tentang kasus penyakit cacar monyet (monkeypox) di berbagai negara. Berikut adalah deskripsi kolom-kolom yang ada di dataset ini:
        
        - **Country:** Nama negara yang melaporkan kasus cacar monyet.
        - **Confirmed_Cases:** Jumlah kasus cacar monyet yang telah terkonfirmasi di negara tersebut.
        - **Suspected_Cases:** Jumlah kasus yang dicurigai (suspected) sebagai cacar monyet namun belum dikonfirmasi.
        - **Hospitalized:** Jumlah pasien cacar monyet yang dirawat di rumah sakit.
        - **Travel_History_Yes:** Jumlah kasus yang memiliki riwayat perjalanan (mungkin terkait ke daerah endemik atau terkena penularan dari luar negeri).
        - **Travel_History_No:** Jumlah kasus yang tidak memiliki riwayat perjalanan.
        
        Dataset ini terdiri dari 129 baris, yang menunjukkan data dari berbagai negara di seluruh dunia. Informasi digunakan untuk menganalisis penyebaran cacar monyet berdasarkan faktor-faktor seperti jumlah kasus terkonfirmasi, dugaan kasus, jumlah pasien yang dirawat, dan pengaruh riwayat perjalanan terhadap penyebaran kasus.
    """)
    
    # Menampilkan dataset tanpa scroll samping
    st.dataframe(df, use_container_width=True)
    
    # Menampilkan jumlah total records
    st.write(f"Total records: {len(df)} data")

# Hasil Analisa Data
elif selection == "Hasil Analisa Data":
    st.title("Hasil Analisa Data")
    st.write("Hasil analisa data yang digunakan untuk memprediksi risiko kasus Monkeypox.")
    
    # Menyaring dan menampilkan negara dengan risiko tinggi dan rendah
    high_risk_countries = df[df['Predicted_Risk'] == 'High']
    low_risk_countries = df[df['Predicted_Risk'] == 'Low']

    st.subheader("Negara dengan Risiko Tinggi:")
    st.write(high_risk_countries[['Country', 'Confirmed_Cases', 'Hospitalized', 'Travel_History_Yes', 'Predicted_Risk']])

    # Penjelasan tentang negara berisiko tinggi
    st.write("""
    **Jumlah Total Negara Berisiko Tinggi:** Terdapat 44 negara yang dikategorikan memiliki risiko tinggi terhadap penyebaran penyakit monkeypox.
    
    **Beberapa Negara Berisiko Tinggi dengan Kasus Terkonfirmasi Tinggi:**
    - Spanyol memiliki 7.083 kasus terkonfirmasi
    - Amerika Serikat memiliki 24.403 kasus terkonfirmasi
    - Brazil dengan jumlah kasus terkonfirmasi sebesar 7.300

    **Faktor Risiko:**
    - Riwayat Perjalanan Tinggi: Banyak individu memiliki riwayat perjalanan internasional, misalnya Amerika Serikat (41) dan Brazil (20)
    - Hospitalized Data: Data jumlah pasien yang dirawat di rumah sakit bervariasi
    """)

    st.subheader("Negara dengan Risiko Rendah:")
    st.write(low_risk_countries[['Country', 'Confirmed_Cases', 'Hospitalized', 'Travel_History_Yes', 'Predicted_Risk']])

    # Penjelasan tentang negara berisiko rendah
    st.write("""
    **Jumlah Total Negara Berisiko Rendah:** Terdapat 85 negara yang dikategorikan memiliki risiko rendah terhadap penyebaran penyakit monkeypox.

    **Negara Berisiko Rendah dengan Kasus Terkonfirmasi:**
    - Kanada dengan 1.388 kasus terkonfirmasi.
    - Sweden dengan 186 kasus terkonfirmasi.
    - Netherlands dengan 1.211 kasus terkonfirmasi.

    **Faktor Risiko:**
    - Riwayat Perjalanan Rendah: Beberapa negara seperti Kanada dan Netherlands memiliki riwayat perjalanan yang sedikit atau nol
    - Hospitalized Data Rendah: Sebagian besar negara dengan risiko rendah memiliki nol pasien rawat inap
    """)

    # Penjelasan perbandingan umum
    st.subheader("Perbandingan Umum")
    st.write("""
    - **Jumlah Negara:** Terdapat lebih banyak negara yang dikategorikan sebagai berisiko rendah (85 negara) dibandingkan dengan yang berisiko tinggi (44 negara).
    - **Faktor Kunci:** Meskipun jumlah kasus terkonfirmasi tinggi, riwayat perjalanan dan jumlah rawat inap memiliki pengaruh lebih besar dalam pengkategorian tingkat risiko.
    """)

elif selection == "Visualisasi Tree":
    st.title("Decision Tree Visualization and Model Evaluation")
    
    # Visualisasi Pohon Keputusan
    st.subheader("Visualisasi Decision Tree: ")
    fig, ax = plt.subplots(figsize=(20, 10))
    plot_tree(model, filled=True, feature_names=X.columns, class_names=['Low', 'High'], rounded=True, ax=ax)
    st.pyplot(fig)
    
    # Penjelasan tentang Decision Tree
    st.write("""
    **Gambar Visualisasi Decision Tree**
    **Struktur Decision Tree:** Decision tree yang ditampilkan memiliki node dan cabang yang mengindikasikan faktor-faktor risiko untuk mengklasifikasikan data menjadi dua kategori: risiko tinggi (High) dan risiko rendah (Low).

    **Cabang Keputusan:**
    - **Travel_History_No dan Travel_History_Yes** sering muncul di beberapa cabang awal. Hal ini menunjukkan bahwa riwayat perjalanan merupakan faktor signifikan dalam menentukan prediksi risiko. Jika seseorang memiliki riwayat perjalanan, peluang mereka untuk diklasifikasikan ke dalam risiko tinggi atau rendah dipengaruhi oleh faktor-faktor lain, seperti jumlah kasus terduga (**Suspected_Cases**) dan jumlah rawat inap (**Hospitalized**).
    - **Suspected_Cases:** Faktor ini juga muncul beberapa kali, menunjukkan bahwa jumlah kasus terduga berperan dalam klasifikasi.
    - **Hospitalized:** Angka rawat inap menjadi indikator yang membantu model memutuskan klasifikasi di beberapa jalur.
    - **Nilai Gini:** Setiap node memiliki nilai Gini yang menunjukkan seberapa bersih atau murni node tersebut dalam membagi data menjadi kelas yang berbeda. Semakin kecil nilai Gini, semakin murni pembagiannya.

    **Interpretasi Cabang:**
    - Misalnya, di awal pohon, jika **Travel_History_No <= 17.5** dan **Hospitalized <= 0.5**, maka model memprediksi kelas sebagai risiko tinggi (dengan Gini 0.301).
    - Di sisi lain, jika jalur berlanjut ke **Suspected_Cases <= 2.5** dan variabel lain seperti **Travel_History_Yes** dipertimbangkan, model bisa memutuskan risiko sebagai rendah atau tinggi berdasarkan distribusi data di node tersebut.
    """)

    # Menampilkan metrik evaluasi model
    st.subheader("Model Evaluation")
    st.write(f"Akurasi model pada data uji: {accuracy_score(y, y_pred):.2f}")
    
    st.subheader("Confusion Matrix: ")
    cm = confusion_matrix(y, y_pred, labels=['Low', 'High'])
    fig, ax = plt.subplots(figsize=(5, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Low', 'High'], yticklabels=['Low', 'High'], ax=ax)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    st.pyplot(fig)

    # Penjelasan tentang evaluasi model
    st.write("""
    **Gambar Evaluasi Model (Confusion Matrix)**
    
    **Akurasi Model:** Akurasi model ditampilkan sebesar 0.78, atau 78%. Ini menunjukkan bahwa model mampu mengklasifikasikan data dengan tingkat keakuratan yang baik, meskipun tidak sempurna.

    **Confusion Matrix:**
    - **True Positives (TP):** Sebanyak 38 sampel diklasifikasikan dengan benar sebagai risiko tinggi.
    - **True Negatives (TN):** Sebanyak 62 sampel diklasifikasikan dengan benar sebagai risiko rendah.
    - **False Positives (FP):** 6 sampel diklasifikasikan sebagai risiko tinggi tetapi sebenarnya risiko rendah.
    - **False Negatives (FN):** 23 sampel diklasifikasikan sebagai risiko rendah tetapi sebenarnya risiko tinggi.

    **Interpretasi Confusion Matrix:**
    Model ini lebih cenderung mengklasifikasikan dengan benar sampel yang termasuk kategori risiko rendah dibandingkan kategori risiko tinggi, karena jumlah **False Negatives (FN)** (23) lebih tinggi daripada **False Positives (FP)** (6).
    
    Ini berarti model memiliki tingkat kesalahan yang relatif lebih tinggi dalam memprediksi risiko tinggi.
    """)

# Model Prediksi Interaktif
elif selection == "Model Prediksi Interaktif":
    st.title("Interactive Risk Level Prediction")
    st.header("Input form for interactive prediction")
    
    # Input untuk prediksi interaktif
    suspected_cases = st.number_input("Suspected Cases", min_value=0, value=0)
    hospitalized = st.number_input("Hospitalized", min_value=0, value=0)
    travel_history_yes = st.number_input("Number of People with Travel History (Yes)", min_value=0, value=0)
    travel_history_no = st.number_input("Number of People without Travel History (No)", min_value=0, value=0)

    # Prediksi risiko saat tombol ditekan
    if st.button("Predict Risk Level"):
        # Persiapkan data untuk prediksi
        input_data = np.array([[suspected_cases, hospitalized, travel_history_yes, travel_history_no]])

        # Prediksi menggunakan model
        predicted_risk = model.predict(input_data)[0]  # Ambil nilai prediksi

        # Menampilkan hasil prediksi
        st.header(f"Predicted Risk Level: {predicted_risk}")

elif selection == "Kesimpulan":
    st.title("Kesimpulan")
    st.write("""
        Negara Berisiko Tinggi: Sebanyak 44 negara dikategorikan sebagai negara dengan risiko tinggi penyebaran penyakit monkeypox. Negara-negara ini umumnya memiliki jumlah kasus terkonfirmasi yang tinggi serta faktor risiko 
        lain seperti riwayat perjalanan internasional dan jumlah pasien rawat inap. Contohnya, Amerika Serikat, Spanyol, dan Brazil mencatatkan jumlah kasus terkonfirmasi yang sangat tinggi. Faktor risiko utama pada negara-negara 
        berisiko tinggi termasuk riwayat perjalanan internasional yang tinggi dan variabilitas pada angka pasien yang dirawat inap.

        Negara Berisiko Rendah: Sebanyak 85 negara diklasifikasikan memiliki risiko rendah terhadap penyebaran monkeypox. Meskipun beberapa negara seperti Kanada dan Belanda mencatatkan jumlah kasus terkonfirmasi yang signifikan, 
        mereka tetap masuk kategori risiko rendah karena rendahnya riwayat perjalanan serta jumlah pasien rawat inap yang minimal atau nol.

        Perbandingan Umum: Secara keseluruhan, terdapat lebih banyak negara yang dikategorikan berisiko rendah dibandingkan dengan negara yang berisiko tinggi. Faktor utama yang mempengaruhi tingkat risiko ini meliputi riwayat 
        perjalanan internasional serta data rawat inap, yang cenderung lebih menentukan dibandingkan jumlah kasus terkonfirmasi semata.

        Faktor Risiko Utama: Riwayat perjalanan dan jumlah pasien rawat inap memainkan peran penting dalam menentukan risiko suatu negara. Negara dengan mobilitas penduduk internasional yang tinggi dan rawat inap cenderung masuk 
        kategori risiko tinggi meskipun jumlah kasus terkonfirmasinya mungkin bervariasi.
        
        Penilaian risiko penyebaran monkeypox di berbagai negara lebih dipengaruhi oleh faktor mobilitas global dan kapasitas kesehatan untuk menangani pasien. Negara dengan riwayat perjalanan yang tinggi dan jumlah pasien rawat 
        inap yang signifikan lebih rentan terhadap penyebaran infeksi, terlepas dari total kasus terkonfirmasi.
    """)

# About Us
elif selection == "About Us":
    st.title("About Us")
    st.write("""
        Selamat datang di Dashboard kami. Kami tim TЯ¥20!
    """)
    st.write("""
        Wildan Hasanah Fitrah as Wailden
    """)
    st.write("""
        Muhammad Dwino AlQadri as Dx
    """)
