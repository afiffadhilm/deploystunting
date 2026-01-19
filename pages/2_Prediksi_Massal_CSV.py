import streamlit as st
import pandas as pd
import numpy as np
import pickle

import re

def usia_text_ke_hari(text):
    if pd.isna(text):
        return np.nan

    text = text.lower()
    tahun = bulan = hari = 0

    t = re.search(r"(\d+)\s*tahun", text)
    b = re.search(r"(\d+)\s*bulan", text)
    h = re.search(r"(\d+)\s*hari", text)

    if t: tahun = int(t.group(1))
    if b: bulan = int(b.group(1))
    if h: hari = int(h.group(1))

    return tahun * 365 + bulan * 30 + hari

# =========================
# LOAD MODEL, SCALER, FEATURE NAMES
# =========================
model = pickle.load(open("model_knn.pkl", "rb"))
scaler = pickle.load(open("scaler_knn.pkl", "rb"))
feature_names = pickle.load(open("feature_names.pkl", "rb"))
# ["berat","tinggi","usia","jenis_kelamin","kenaikan_berat"]

# =========================
# UI
# =========================
st.title("Prediksi Massal (Upload Excel atau CSV)")
st.write("Upload file data balita untuk prediksi massal status gizi.")
st.write('Pastikan kolom berisi [berat, tinggi, usia , jenis_kelamin, kenaikan_berat].')
st.divider()

uploaded_file = st.file_uploader("Upload file CSV dan Excel", type=["csv", "xlsx"])

if uploaded_file is not None:

    # =========================
    # LOAD FILE (CSV / EXCEL)
    # =========================
    if uploaded_file.name.endswith(".csv"):
        df_csv = pd.read_csv(
            uploaded_file,
            sep=";",
            engine="python",
            encoding="latin1",
            on_bad_lines="skip"
        )
    else:
        df_csv = pd.read_excel(uploaded_file)


    # DROP KOLOM TIDAK DIPAKAI
    df_csv = df_csv.drop(columns=["usia_hari", "status_gizi", "z_score"], errors="ignore")

    # buat usia_hari dari usia
    if "usia" in df_csv.columns:
      df_csv["usia_hari"] = df_csv["usia"].apply(usia_text_ke_hari)

    # =========================
    # NORMALISASI NAMA KOLOM
    # =========================
    df_csv.columns = df_csv.columns.str.strip().str.lower()

    # =========================
    # ENCODING KATEGORIK
    # =========================
    # =========================
    # ENCODING AMAN (ANTI NaN)
    # =========================
    if "jenis_kelamin" in df_csv.columns:
        df_csv["jenis_kelamin"] = (
          df_csv["jenis_kelamin"]
          .astype(str)
          .str.strip()
          .str.upper()
          .replace({"L": 1, "P": 0})
    )

    if "kenaikan_berat" in df_csv.columns:
        df_csv["kenaikan_berat"] = (
          df_csv["kenaikan_berat"]
          .astype(str)
          .str.strip()
          .str.upper()
          .replace({"T": 1, "O": 0, "N": 0})
    )


    # =========================
    # VALIDASI KOLOM MODEL
    # =========================
    missing_cols = set(feature_names) - set(df_csv.columns)
    if missing_cols:
        st.error(f"Kolom berikut tidak ada / tidak valid: {missing_cols}")
        st.stop()
    else:
        st.success("Struktur file valid")

    # =========================
    # AMBIL FITUR & PAKSA NUMERIK
    # =========================
    X = df_csv[feature_names]
    X = X.apply(pd.to_numeric, errors="coerce")

    if X.isnull().any().any():
        st.error("Data mengandung nilai non-numerik atau kosong.")
        st.dataframe(df_csv[feature_names])
        st.stop()

    # =========================
    # SCALING & PREDIKSI
    # =========================
    X_scaled = scaler.transform(X)
    preds = model.predict(X_scaled)

    label_map = {
    0: "Normal",
    1: "Berisiko",
    2: "Sangat Berisiko"
}

    df_csv["Prediksi_Status_Gizi"] = pd.Series(preds).map(label_map)


    # =========================
    # OUTPUT
    # =========================
    st.subheader("Hasil Prediksi")
    st.dataframe(df_csv)

    csv_out = df_csv.to_csv(index=False).encode("utf-8")
    st.download_button(
        "Download hasil prediksi",
        csv_out,
        "hasil_prediksi_gizi.csv",
        "text/csv"
    )