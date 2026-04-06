import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

# ================================
# KONFIGURASI HALAMAN
# ================================
st.set_page_config(
    page_title="Prediksi Stok Barang",
    layout="wide"
)

# ================================
# LOAD MODEL
# ================================
@st.cache_resource
def load_model():
    model_path = "model_pipeline_fix.joblib"

    if not os.path.exists(model_path):
        st.error(f"❌ Model tidak ditemukan: {model_path}")
        st.stop()

    return joblib.load(model_path)

model = load_model()

# ================================
# THRESHOLD
# ================================
UNDERSTOCK_THRESHOLD = 5
OVERSTOCK_THRESHOLD = 50

# ================================
# FUNGSI KLASIFIKASI
# ================================
def classify_stock(stock):
    if stock < UNDERSTOCK_THRESHOLD:
        return "Understock"
    elif stock > OVERSTOCK_THRESHOLD:
        return "Overstock"
    else:
        return "Normal"

# ================================
# HEADER
# ================================
st.title("📊 Sistem Prediksi Stok Barang")

st.markdown("""
Aplikasi ini digunakan untuk memprediksi stok barang menggunakan Machine Learning  
untuk menghindari kondisi:

- ⚠️ **Understock** (< 5)  
- 📦 **Overstock** (> 50)
""")

# ================================
# SIDEBAR INPUT MANUAL
# ================================
st.sidebar.header("🔧 Input Data Manual")

jumlah_penjualan = st.sidebar.number_input("Jumlah Penjualan", min_value=0, value=10)
stok_awal = st.sidebar.number_input("Stok Awal", min_value=0, value=20)
stok_masuk = st.sidebar.number_input("Stok Masuk", min_value=0, value=15)

prediksi_btn = st.sidebar.button("🔍 Prediksi")

# ================================
# PREDIKSI MANUAL
# ================================
if prediksi_btn:

    try:
        input_data = np.array([[jumlah_penjualan, stok_awal, stok_masuk]])

        prediction = model.predict(input_data)[0]

        status = classify_stock(prediction)

        st.subheader("📈 Hasil Prediksi")

        col1, col2 = st.columns(2)

        with col1:
            st.metric("Prediksi Stok Akhir", int(prediction))

        with col2:
            st.metric("Status Stok", status)

        if status == "Understock":
            st.error("⚠️ Stok terlalu sedikit! Segera restock.")

        elif status == "Overstock":
            st.warning("📦 Stok terlalu banyak! Kurangi pembelian.")

        else:
            st.success("✅ Stok dalam kondisi aman.")

    except Exception as e:
        st.error(f"Terjadi error: {e}")

# ================================
# UPLOAD DATA
# ================================
st.markdown("---")
st.subheader("📂 Prediksi Banyak Data (Upload File)")

st.info("Kolom yang diperlukan: jumlah_penjualan, stok_awal, stok_masuk")

uploaded_file = st.file_uploader(
    "Upload CSV atau Excel",
    type=["csv", "xlsx"]
)

# ================================
# PROSES FILE
# ================================
if uploaded_file is not None:

    try:

        # DETEKSI FORMAT FILE
        if uploaded_file.name.endswith(".csv"):
            df = pd.read_csv(uploaded_file)

        else:
            df = pd.read_excel(uploaded_file)

        # ============================
        # MEMBERSIHKAN DATASET
        # ============================

        # ubah nama kolom jadi lowercase
        df.columns = df.columns.str.lower()

        # hapus kolom unnamed
        df = df.loc[:, ~df.columns.str.contains('^unnamed')]

        # isi nilai kosong stok masuk
        if "stok_masuk" in df.columns:
            df["stok_masuk"] = df["stok_masuk"].fillna(0)

        st.subheader("📄 Preview Data")
        st.dataframe(df.head())

        # ============================
        # CEK KOLOM
        # ============================

        required_columns = [
            "jumlah_penjualan",
            "stok_awal",
            "stok_masuk"
        ]

        if not all(col in df.columns for col in required_columns):

            st.error(
                "Kolom tidak sesuai! Harus ada: jumlah_penjualan, stok_awal, stok_masuk"
            )

        else:

            if st.button("🚀 Proses Prediksi"):

                X = df[required_columns]

                predictions = model.predict(X)

                df["predicted_stock"] = predictions

                df["status"] = df["predicted_stock"].apply(classify_stock)

                st.success("✅ Prediksi berhasil!")

                # ========================
                # TAMPILKAN HASIL
                # ========================

                st.subheader("📊 Hasil Prediksi")

                st.dataframe(df)

                # ========================
                # VISUALISASI
                # ========================

                col1, col2 = st.columns(2)

                with col1:
                    st.write("Distribusi Status")
                    st.bar_chart(df["status"].value_counts())

                with col2:
                    st.write("Prediksi Stok")
                    st.line_chart(df["predicted_stock"])

                # ========================
                # DOWNLOAD HASIL
                # ========================

                csv = df.to_csv(index=False).encode("utf-8")

                st.download_button(
                    label="📥 Download Hasil Prediksi",
                    data=csv,
                    file_name="hasil_prediksi_stok.csv",
                    mime="text/csv"
                )

    except Exception as e:
        st.error(f"Terjadi error saat membaca file: {e}")

# ================================
# FOOTER
# ================================
st.markdown("---")
st.caption("© 2025 - Sistem Prediksi Stok Barang | Machine Learning Project")
