import streamlit as st
import pandas as pd
import numpy as np
import pickle

# =========================
# LOAD MODEL & SCALER
# =========================
model = pickle.load(open("model_knn.pkl", "rb"))
scaler = pickle.load(open("scaler_knn.pkl", "rb"))

# =========================
# STREAMLIT UI
# =========================
st.title("Prediksi Status Gizi Balita")
st.caption("Menggunakan Algoritma KNN")

Berat = st.number_input("Berat Badan (kg):", min_value=1.0, max_value=50.0)
Tinggi = st.number_input("Tinggi Badan (cm):", min_value=30.0, max_value=150.0)
usia_bulan = st.number_input("Usia (bulan):", min_value=0, max_value=60)

jenis_kelamin_text = st.selectbox("Jenis Kelamin:", ["Laki-laki", "Perempuan"])
kenaikan_berat_text = st.selectbox(
    "Kenaikan Berat Badan:",
    ["T (Tetap)", "N (Naik)"]
)

# =========================
# MAPPING KE MODEL
# =========================
jenis_kelamin = 1 if jenis_kelamin_text == "Laki-laki" else 2
kenaikan_berat = 1 if "T" in kenaikan_berat_text else 0
usia_hari = usia_bulan * 30.4375

# =========================
# PREDIKSI
# =========================
if st.button("Prediksi"):
    input_data = pd.DataFrame([{
        "berat": Berat,
        "tinggi": Tinggi,
        "usia_hari": usia_hari,
        "jenis_kelamin": jenis_kelamin,
        "kenaikan_berat": kenaikan_berat
    }])

    input_scaled = scaler.transform(input_data)
    pred = model.predict(input_scaled)[0]

    label_map = {
        0: "Normal",
        1: "Berisiko",
        2: "Sangat Berisiko"
    }

    st.subheader("Hasil Prediksi")
    st.write(f"**Status Gizi: {label_map.get(pred, 'Tidak Diketahui')}**")

st.caption(
    "Hasil prediksi bersifat indikatif dan tidak menggantikan pemeriksaan medis."
)