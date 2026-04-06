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
        st.error(f"Model tidak ditemukan: {model_path}")
        st.stop()

    return joblib.load(model_path)

model = load_model()

# ================================
# THRESHOLD
# ================================
UNDERSTOCK_THRESHOLD = 5
OVERSTOCK_THRESHOLD = 50

# ================================
# KLASIFIKASI STOK
# ================================
def classify_stock(stock):

    if stock < UNDERSTOCK_THRESHOLD:
        return "Understock"

    elif stock > OVERSTOCK_THRESHOLD:
        return "Overstock"

    else:
        return "Normal"


# ================================
# FEATURE ENGINEERING
# ================================
def create_features(df):

    df["tanggal"] = pd.to_datetime(df["tanggal"])

    df = df.sort_values("tanggal")

    # ==========================
    # FITUR WAKTU
    # ==========================

    df["tahun"] = df["tanggal"].dt.year
    df["bulan"] = df["tanggal"].dt.month
    df["hari"] = df["tanggal"].dt.day
    df["hari_dalam_minggu"] = df["tanggal"].dt.weekday
    df["minggu_ke"] = df["tanggal"].dt.isocalendar().week.astype(int)

    df["is_weekend"] = df["hari_dalam_minggu"].isin([5,6]).astype(int)

    # ==========================
    # LAG FEATURES
    # ==========================

    df["lag_penjualan_1"] = df["jumlah_penjualan"].shift(1)
    df["lag_penjualan_2"] = df["jumlah_penjualan"].shift(2)
    df["lag_penjualan_3"] = df["jumlah_penjualan"].shift(3)

    df["lag_stok_awal_1"] = df["stok_awal"].shift(1)
    df["lag_stok_masuk_1"] = df["stok_masuk"].shift(1)

    # ==========================
    # ROLLING FEATURES
    # ==========================

    df["rolling_mean_penjualan_7"] = df["jumlah_penjualan"].rolling(7).mean()
    df["rolling_std_penjualan_7"] = df["jumlah_penjualan"].rolling(7).std()
    df["rolling_min_penjualan_7"] = df["jumlah_penjualan"].rolling(7).min()
    df["rolling_max_penjualan_7"] = df["jumlah_penjualan"].rolling(7).max()

    # ==========================
    # TREND
    # ==========================

    df["tren_penjualan"] = df["jumlah_penjualan"].diff()

    df = df.fillna(0)

    return df


# ================================
# HEADER
# ================================
st.title("📊 Sistem Prediksi Stok Barang")

st.markdown("""
Aplikasi ini digunakan untuk memprediksi stok barang menggunakan Machine Learning
untuk menghindari kondisi:

⚠️ Understock (<5)  
📦 Overstock (>50)
""")


# ================================
# UPLOAD FILE
# ================================
st.subheader("Upload Dataset")

uploaded_file = st.file_uploader(
    "Upload CSV atau Excel",
    type=["csv","xlsx"]
)

# ================================
# PROSES FILE
# ================================
if uploaded_file is not None:

    try:

        # ==========================
        # BACA FILE
        # ==========================

        if uploaded_file.name.endswith(".csv"):
            df = pd.read_csv(uploaded_file)

        else:
            df = pd.read_excel(uploaded_file)

        # ==========================
        # BERSIHKAN DATA
        # ==========================

        df.columns = df.columns.str.lower()

        df = df.loc[:, ~df.columns.str.contains("^unnamed")]

        st.subheader("Preview Data")
        st.dataframe(df.head())

        required_columns = [
            "tanggal",
            "jenis_produk",
            "jumlah_penjualan",
            "stok_awal",
            "stok_masuk"
        ]

        if not all(col in df.columns for col in required_columns):

            st.error("Kolom dataset tidak lengkap")

        else:

            # ==========================
            # BUAT FITUR ML
            # ==========================

            df_features = create_features(df)

            # ==========================
            # PREDIKSI
            # ==========================

            if st.button("Proses Prediksi"):

                X = df_features.drop(
                    columns=["stok_akhir"],
                    errors="ignore"
                )

                predictions = model.predict(X)

                df_features["predicted_stock"] = predictions

                df_features["status"] = df_features[
                    "predicted_stock"
                ].apply(classify_stock)

                st.success("Prediksi berhasil")

                st.subheader("Hasil Prediksi")

                st.dataframe(df_features)

                # ==========================
                # VISUALISASI
                # ==========================

                col1,col2 = st.columns(2)

                with col1:
                    st.write("Distribusi Status")
                    st.bar_chart(
                        df_features["status"].value_counts()
                    )

                with col2:
                    st.write("Prediksi Stok")
                    st.line_chart(
                        df_features["predicted_stock"]
                    )

                # ==========================
                # DOWNLOAD
                # ==========================

                csv = df_features.to_csv(
                    index=False
                ).encode("utf-8")

                st.download_button(
                    label="Download Hasil Prediksi",
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
st.caption("Machine Learning Stock Prediction System")
