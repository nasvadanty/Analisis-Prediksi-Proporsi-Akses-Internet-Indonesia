import streamlit as st
import pandas as pd
import joblib
import seaborn as sns
import matplotlib.pyplot as plt

# LOAD MODEL DAN DATA
model = joblib.load("model_loreg.pkl")
scaler = joblib.load("scaler.pkl")
df = pd.read_csv("hasil_cluster_internet.csv")

st.set_page_config(page_title="Analisis Akses Internet SMA/MA", page_icon="📡", layout="wide")

st.title("📡 Analisis Akses Internet SMA/MA di Indonesia")
st.markdown("""
Dashboard interaktif ini menampilkan hasil *Clustering* dan *Prediksi Logistic Regression*
berdasarkan proporsi sekolah SMA/MA dengan akses internet di Indonesia.
""")

# Rule-based cluster untuk df utama
df['Persentase_Tersedia'] = pd.to_numeric(df['Persentase_Tersedia'], errors='coerce')
df['Cluster_Rule'] = df['Persentase_Tersedia'].apply(lambda x: 0 if pd.notna(x) and x >= 90 else 1)
df['Cluster_Label'] = df['Cluster_Rule'].map({0: 'Baik', 1: 'Tertinggal'})

# TAMPILKAN DATA AWAL
st.header("📊 Dataset Awal")
st.dataframe(df.head(10))

# EKSPLORASI DATA (EDA)
st.header("🔍 Eksplorasi Data")
col1, col2 = st.columns(2)

with col1:
    st.subheader("Distribusi Persentase Akses Internet")
    plt.figure(figsize=(8, 5))
    sns.histplot(df['Persentase_Tersedia'], kde=True, color='skyblue')
    st.pyplot(plt)

with col2:
    st.subheader("Boxplot Persentase Akses Internet")
    plt.figure(figsize=(8, 5))
    sns.boxplot(x=df['Persentase_Tersedia'], color='lightgreen')
    st.pyplot(plt)

# VISUALISASI CLUSTER
st.header("🧩 Visualisasi Hasil Clustering")
plt.figure(figsize=(10, 6))
sns.scatterplot(
    data=df,
    x='Persentase_Tersedia',
    y='cluster',
    hue='cluster',
    palette='viridis',
    s=80
)
plt.title("Distribusi Cluster Berdasarkan Persentase Akses Internet")
st.pyplot(plt)

# PREDIKSI LOGISTIC REGRESSION
st.header("🤖 Prediksi Menggunakan Logistic Regression")

st.markdown("""
Masukkan jumlah total sekolah di provinsi dan jumlah sekolah yang memiliki akses internet.
Sistem akan menghitung **proporsinya secara otomatis**, kemudian memprediksi apakah wilayah tersebut
termasuk dalam cluster **Tertinggal (1)** atau **Baik (0)**.
""")

# Input jumlah sekolah
total_schools = st.number_input("🏫 Masukkan Jumlah Total Sekolah di Provinsi", min_value=1, step=1)
internet_schools = st.number_input("🌐 Masukkan Jumlah Sekolah dengan Akses Internet", min_value=0, step=1)

if st.button("🔮 Prediksi Cluster"):
    if internet_schools > total_schools:
        st.error("❌ Jumlah sekolah dengan akses internet tidak boleh lebih besar dari total sekolah.")
    else:
        proporsi = (internet_schools / total_schools) * 100
        st.info(f"📊 Persentase Akses Internet Sekolah: **{proporsi:.2f}%**")

        # RULE-BASED CLUSTERING (90% threshold)
        threshold = 90
        
        # Hitung proporsi
        proporsi = (internet_schools / total_schools) * 100
        st.info(f"📊 Persentase Akses Internet Sekolah: **{proporsi:.2f}%**")
        
        # Hitung probabilitas berbasis jarak ke threshold
        distance = abs(proporsi - threshold)
        
        # Normalisasi jarak (maks 90)
        prob = min(0.50 + (distance / 90) * 0.49, 0.99)
        prob = round(prob, 2)
        
        # Prediksi kategori
        if proporsi >= threshold:
            prediction = 0  # Baik
            st.success(f"✅ Wilayah diprediksi **BAIK** — Probabilitas: {prob:.2f}")
            st.markdown("🌐 *Insight:* Wilayah ini sudah memiliki infrastruktur digital yang cukup baik untuk mendukung pembelajaran daring.")
        else:
            prediction = 1  # Tertinggal
            st.error(f"🚨 Wilayah diprediksi **TERTINGGAL** — Probabilitas: {prob:.2f}")
            st.markdown("💡 *Rekomendasi:* Perlu peningkatan infrastruktur jaringan internet dan pelatihan TIK untuk sekolah di wilayah ini.")

# UPLOAD DATASET UNTUK PREDIKSI MASSAL
st.header("📂 Upload Dataset untuk Prediksi Batch")

st.markdown("""
Upload file **Excel (.xlsx)** atau **CSV (.csv)** yang berisi kolom:
- `Provinsi`
- `Jumlah_Sekolah`
- `Jumlah_Sekolah_Tersedia_Internet`
""")

# Tambahkan ini agar Streamlit tidak rerun ke atas setiap tombol ditekan
uploaded_file = st.file_uploader("Unggah file data baru", type=["xlsx", "csv"], key="upload_file")

if uploaded_file is not None:
    try:
        # Baca file sesuai format
        if uploaded_file.name.endswith('.csv'):
            new_df = pd.read_csv(uploaded_file)
        else:
            new_df = pd.read_excel(uploaded_file)

        st.success("✅ Data Berhasil Diupload!")
        st.dataframe(new_df.head())

        # Validasi kolom
        required_cols = {'Provinsi', 'Jumlah_Sekolah', 'Jumlah_Sekolah_Tersedia_Internet'}
        if not required_cols.issubset(new_df.columns):
            st.error(f"❌ File harus memiliki kolom: {required_cols}")
        else:
            # Merge Koordinat Provinsi
            new_df = new_df.merge(coord_df, on='Provinsi', how='left')

            missing_coords = new_df[new_df['Latitude'].isna()]['Provinsi'].unique()
            if len(missing_coords) > 0:
                st.warning(f"Provinsi berikut tidak memiliki koordinat: {missing_coords}")

            # Proses prediksi
            new_df['Persentase_Tersedia'] = (new_df['Jumlah_Sekolah_Tersedia_Internet'] / new_df['Jumlah_Sekolah']) * 100
            scaled = scaler.transform(new_df[['Persentase_Tersedia']])
            new_df['Prediksi_Cluster'] = model.predict(scaled)
            new_df['Probabilitas'] = model.predict_proba(scaled).max(axis=1)

            st.dataframe(new_df[['Provinsi', 'Persentase_Tersedia', 'Prediksi_Cluster', 'Probabilitas']].head(10))

            # Rule-based prediction untuk new_df
            new_df['Persentase_Tersedia'] = pd.to_numeric(new_df['Persentase_Tersedia'], errors='coerce')
            new_df['Prediksi_Cluster_Rule'] = new_df['Persentase_Tersedia'].apply(lambda x: 0 if pd.notna(x) and x >= 90 else 1)
            new_df['Prediksi_Label'] = new_df['Prediksi_Cluster_Rule'].map({0: 'Baik', 1: 'Tertinggal'})

            # Visualisasi hasil (Diagram)
            st.subheader("📊 Distribusi Prediksi per Provinsi")
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.barplot(data=new_df, x='Provinsi', y='Persentase_Tersedia', hue='Prediksi_Cluster', palette='coolwarm', ax=ax)
            plt.xticks(rotation=90)
            st.pyplot(fig)

            # Unduh hasil
            csv = new_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="⬇️ Unduh Hasil Prediksi (CSV)",
                data=csv,
                file_name="hasil_prediksi_batch.csv",
                mime="text/csv"
            )
    except Exception as e:
        st.error(f"Terjadi kesalahan saat memproses file: {e}")
else:
    st.info("📤 Silakan upload file untuk melakukan prediksi batch.")

# INSIGHT UMUM
st.header("📈 Insight Umum")
col3, col4 = st.columns(2)

with col3:
    avg_internet = df['Persentase_Scaled'].mean()
    st.metric("Rata-rata Akses Internet Nasional (%)", f"{avg_internet:.2f}")

with col4:
    cluster_counts = df['cluster'].value_counts()
    st.metric("Jumlah Cluster", len(cluster_counts))
    st.write(cluster_counts)

st.markdown("---")
st.markdown("🧠 *Dibuat dengan Python, Streamlit, dan Scikit-learn — oleh Tim Penelitian SMA/MA Internet Analysis 2025*")

