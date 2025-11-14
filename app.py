import streamlit as st
import pandas as pd
import joblib
import seaborn as sns
import matplotlib.pyplot as plt
from streamlit_folium import st_folium
import folium

# Tabel Koordinat Provinsi Indonesia
provinsi_coords = {
    "Aceh": (-3.644, 96.225),
    "Sumatera Utara": (2.115, 99.545),
    "Sumatera Barat": (-0.739, 100.800),
    "Riau": (0.293, 101.707),
    "Jambi": (-1.485, 102.438),
    "Sumatera Selatan": (-3.319, 103.914),
    "Bengkulu": (-3.792, 102.260),
    "Lampung": (-4.558, 105.406),
    "Kepulauan Bangka Belitung": (-2.741, 106.440),
    "Kepulauan Riau": (3.945, 108.142),
    "DKI Jakarta": (-6.2088, 106.8456),
    "Jawa Barat": (-6.889, 107.640),
    "Jawa Tengah": (-7.150, 110.140),
    "DI Yogyakarta": (-7.797, 110.370),
    "Jawa Timur": (-7.536, 112.238),
    "Banten": (-6.405, 106.064),
    "Bali": (-8.340, 115.092),
    "Nusa Tenggara Barat": (-8.653, 117.361),
    "Nusa Tenggara Timur": (-10.177, 123.593),
    "Kalimantan Barat": (-0.278, 109.335),
    "Kalimantan Tengah": (-1.681, 113.382),
    "Kalimantan Selatan": (-3.319, 114.592),
    "Kalimantan Timur": (0.538, 116.419),
    "Kalimantan Utara": (3.020, 116.207),
    "Sulawesi Utara": (1.474, 124.842),
    "Sulawesi Tengah": (-1.430, 121.445),
    "Sulawesi Selatan": (-5.147, 119.432),
    "Sulawesi Tenggara": (-4.013, 122.529),
    "Gorontalo": (0.699, 122.446),
    "Sulawesi Barat": (-2.844, 118.876),
    "Maluku": (-3.238, 130.145),
    "Maluku Utara": (1.570, 127.808),
    "Papua Barat": (-1.336, 133.174),
    "Papua": (-4.269, 138.080)
}


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

# Menambah koordinat ke dataframe
coord_df = pd.DataFrame.from_dict(provinsi_coords, orient='index', columns=['Latitude', 'Longitude'])
coord_df.reset_index(inplace=True)
coord_df.rename(columns={'index': 'Provinsi'}, inplace=True)

# Merge berdasarkan nama provinsi
df = df.merge(coord_df, on='Provinsi', how='left')


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

# MAP INTERAKTIF
st.header("🗺️ Peta Persebaran Akses Internet (Interaktif)")

if {'Latitude', 'Longitude'}.issubset(df.columns):

    # Tampilkan provinsi yang hilang koordinat (debug)
    missing_main = df[df['Latitude'].isna()]['Provinsi'].unique()
    if len(missing_main) > 0:
        st.warning(f"Provinsi berikut tidak memiliki koordinat dan tidak ditampilkan di peta: {missing_main}")

    # Drop provinsi tanpa koordinat agar tidak error Folium
    map_df_main = df.dropna(subset=['Latitude', 'Longitude'])

    # Buat map
    m = folium.Map(location=[-2.5, 118], zoom_start=5)

    for _, row in map_df_main.iterrows():
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

        # RULE-BASED CLUSTERING (90% menjadi batasnya)
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

            # Visualisasi hasil (Diagram)
            st.subheader("📊 Distribusi Prediksi per Provinsi")
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.barplot(data=new_df, x='Provinsi', y='Persentase_Tersedia', hue='Prediksi_Cluster', palette='coolwarm', ax=ax)
            plt.xticks(rotation=90)
            st.pyplot(fig)

            # Visualisasi hasil (Map)
            st.subheader("🗺️ Peta Persebaran Prediksi (Interaktif)")
            
            if {'Latitude', 'Longitude'}.issubset(new_df.columns):
            
                # Filter hanya baris dengan koordinat valid
                map_df = new_df.dropna(subset=['Latitude', 'Longitude'])
            
                # Peringatan jika ada provinsi tidak punya koordinat
                missing_coords = new_df[new_df['Latitude'].isna()]['Provinsi'].unique()
                if len(missing_coords) > 0:
                    st.warning(f"Provinsi berikut tidak memiliki koordinat sehingga tidak muncul di peta: {missing_coords}")
            
                if map_df.empty:
                    st.error("❌ Tidak ada baris dengan koordinat valid untuk ditampilkan di peta.")
                else:
                    m2 = folium.Map(location=[-2.5, 118], zoom_start=5)
            
                    for _, row in map_df.iterrows():
                        color = "green" if row['Prediksi_Cluster'] == 0 else "red"
                        popup_text = f"""
                        <b>Provinsi:</b> {row['Provinsi']}<br>
                        <b>Persentase Akses Internet:</b> {row['Persentase_Tersedia']:.2f}%<br>
                        <b>Hasil Prediksi:</b> {'Baik' if row['Prediksi_Cluster']==0 else 'Tertinggal'}<br>
                        <b>Probabilitas:</b> {row['Probabilitas']:.2f}
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

