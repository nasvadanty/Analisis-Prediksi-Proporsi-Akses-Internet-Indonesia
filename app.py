

import streamlit as st
import pandas as pd
import joblib
import seaborn as sns
import matplotlib.pyplot as plt
from streamlit_folium import st_folium
import folium


# === 1. LOAD MODEL DAN DATA ===
model = joblib.load("model_loreg.pkl")
scaler = joblib.load("scaler.pkl")
df = pd.read_csv("hasil_cluster_internet.csv")

st.set_page_config(page_title="Analisis Akses Internet SMA/MA", page_icon="📡", layout="wide")

st.title("📡 Analisis Akses Internet SMA/MA di Indonesia")
st.markdown("""
Dashboard interaktif ini menampilkan hasil *Clustering* dan *Prediksi Logistic Regression*
berdasarkan proporsi sekolah SMA/MA dengan akses internet di Indonesia.
""")

# === 2. TAMPILKAN DATA AWAL ===
st.header("📊 Dataset Awal")
st.dataframe(df.head(10))

# === 3. EKSPLORASI DATA (EDA) ===
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

# === 4. VISUALISASI CLUSTER ===
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

# === 4b. MAP INTERAKTIF ===
st.header("🗺️ Peta Persebaran Akses Internet (Interaktif)")

# Pastikan ada kolom koordinat
if {'Latitude', 'Longitude'}.issubset(df.columns):
    m = folium.Map(location=[-2.5, 118], zoom_start=5)

    for _, row in df.iterrows():
        color = "green" if row['cluster'] == 0 else "red"
        popup_text = f"""
        <b>Provinsi:</b> {row['Provinsi']}<br>
        <b>Persentase Akses Internet:</b> {row['Persentase_Tersedia']:.2f}%<br>
        <b>Cluster:</b> {'Baik' if row['cluster']==0 else 'Tertinggal'}
        """

        folium.CircleMarker(
            location=[row['Latitude'], row['Longitude']],
            radius=7,
            color=color,
            fill=True,
            fill_color=color,
            popup=popup_text,
            tooltip=row['Provinsi']
        ).add_to(m)

    st_folium(m, width=900, height=500)
else:
    st.warning("Dataset utama belum memiliki kolom Latitude & Longitude untuk memvisualisasikan peta.")


# === 5. PREDIKSI LOGISTIC REGRESSION (versi baru) ===
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

        # ===== RULE-BASED CLUSTERING (90% menjadi batasnya) =====
        if proporsi >= 90:
            prediction = 0  # Baik
            prob = 1.0
            st.success(f"✅ Wilayah diprediksi **BAIK** — Probabilitas: {prob:.2f}")
            st.markdown("🌐 *Insight:* Wilayah ini sudah memiliki infrastruktur digital yang cukup baik untuk mendukung pembelajaran daring.")
        else:
            prediction = 1  # Tertinggal
            prob = 1.0
            st.error(f"🚨 Wilayah diprediksi **TERTINGGAL** — Probabilitas: {prob:.2f}")
            st.markdown("💡 *Rekomendasi:* Perlu peningkatan infrastruktur jaringan internet dan pelatihan TIK untuk sekolah di wilayah ini.")

# === 6. UPLOAD DATASET UNTUK PREDIKSI MASSAL ===
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
            # Proses prediksi
            new_df['Persentase_Tersedia'] = (new_df['Jumlah_Sekolah_Tersedia_Internet'] / new_df['Jumlah_Sekolah']) * 100
            scaled = scaler.transform(new_df[['Persentase_Tersedia']])
            new_df['Prediksi_Cluster'] = model.predict(scaled)
            new_df['Probabilitas'] = model.predict_proba(scaled).max(axis=1)

            st.dataframe(new_df[['Provinsi', 'Persentase_Tersedia', 'Prediksi_Cluster', 'Probabilitas']].head(10))

            # Visualisasi hasil (Diagram)
            st.subheader("📊 Distribusi Prediksi per Provinsi")
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.barplot(data=new_df, x='Provinsi', y='Persentase_Tersedia', hue='Prediksi_Cluster', palette='coolwarm', ax=ax)
            plt.xticks(rotation=90)
            st.pyplot(fig)

            # Visualisasi hasil (Map)
            st.subheader("🗺️ Peta Persebaran Prediksi (Interaktif)")
            
            if {'Latitude', 'Longitude'}.issubset(new_df.columns):
                m2 = folium.Map(location=[-2.5, 118], zoom_start=5)
            
                for _, row in new_df.iterrows():
                    color = "green" if row['Prediksi_Cluster'] == 0 else "red"
                    popup_text = f"""
                    <b>Provinsi:</b> {row['Provinsi']}<br>
                    <b>Persentase Akses Internet:</b> {row['Persentase_Tersedia']:.2f}%<br>
                    <b>Hasil Prediksi:</b> {'Baik' if row['Prediksi_Cluster']==0 else 'Tertinggal'}
                    <br><b>Probabilitas:</b> {row['Probabilitas']:.2f}
                    """
            
                    folium.Marker(
                        location=[row['Latitude'], row['Longitude']],
                        popup=popup_text,
                        tooltip=row['Provinsi'],
                        icon=folium.Icon(color="green" if row['Prediksi_Cluster']==0 else "red")
                    ).add_to(m2)
            
                st_folium(m2, width=900, height=500)
            else:
                st.warning("Dataset upload belum memiliki kolom Latitude & Longitude sehingga peta tidak dapat ditampilkan.")


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

# === 7. INSIGHT UMUM ===
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

