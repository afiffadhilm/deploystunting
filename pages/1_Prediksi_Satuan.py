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
# LOAD WHO TABLE
# =========================
boys = pd.read_excel("boys_zscores.xlsx")
girls = pd.read_excel("girls_zscores.xlsx")

# pastikan kolom numeric
for df in [boys, girls]:
    for col in ["L", "M", "S"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

# =========================
# FUNCTION HITUNG Z-SCORE
# =========================
def hitung_zscore_tb_u(usia_bulan, jenis_kelamin, tinggi):
    # pilih tabel sesuai gender WHO
    if jenis_kelamin == 1:  # 1 = laki-laki
        df = boys
    else:                   # 2 = perempuan
        df = girls

    row = df[df["Month"] == usia_bulan]

    if row.empty:
        return None

    L = row["L"].values[0]
    M = row["M"].values[0]
    S = row["S"].values[0]

    z = ((tinggi / M)**L - 1) / (L * S)
    return z


# =========================
# STREAMLIT UI
# =========================
st.title("Prediksi Status Gizi Balita")
st.caption("Menggunakan KNN + WHO LMS Z-Score")

Berat = st.number_input("Berat Badan (kg):", min_value=1.0, max_value=50.0)
Tinggi = st.number_input("Tinggi Badan (cm):", min_value=30.0, max_value=150.0)
usia_bulan = st.number_input("Usia (bulan):", min_value=0, max_value=60)

jenis_kelamin_text = st.selectbox("Jenis Kelamin:", ["Laki-laki", "Perempuan"])
kenaikan_berat_text = st.selectbox("Kenaikan Berat Badan: T(Tetap) dan N(Naik)", ["T", "N"])

# === mapping ke model ===
jenis_kelamin_model = 1 if jenis_kelamin_text == "Laki-laki" else 2
kenaikan_berat = 1 if kenaikan_berat_text == "T" else 0

# === mapping ke WHO ===
jenis_kelamin_who = 1 if jenis_kelamin_text == "Laki-laki" else 0


# =========================
# Hitung Z-Score TB/U
# =========================
zscore_tb_u = hitung_zscore_tb_u(usia_bulan, jenis_kelamin_who, Tinggi)

if zscore_tb_u is None:
    st.error("Usia tidak tersedia di tabel WHO (0-60 bulan)")
else:
    st.success(f"Z-Score TB/U = **{zscore_tb_u:.2f}**")


st.caption("Hasil prediksi ini bersifat indikatif dan tidak menggantikan pemeriksaan medis.\nUntuk penegakan diagnosis dan penanganan lebih lanjut, silakan konsultasikan dengan tenaga kesehatan atau dokter.")
# =========================
# PREDIKSI
# =========================
if st.button("Prediksi"):
    if zscore_tb_u is None:
        st.error("Z-score tidak bisa dihitung, prediksi dibatalkan.")
    else:
        usia_hari = usia_bulan * 30.4375

        input_data = pd.DataFrame([{
            "berat": Berat,
            "tinggi": Tinggi,
            "usia_hari": usia_hari,
            "jenis_kelamin": jenis_kelamin_model,
            "kenaikan_berat": kenaikan_berat
        }])
        
        
        # scaling
        input_scaled = scaler.transform(input_data)

        # prediction
        pred = model.predict(input_scaled)[0]

        label_map = {
        0: "Normal",
        1: "Beresiko",
        2: "Sangat Berisiko"
        }

        label = label_map.get(pred, "Tidak Diketahui")
        st.subheader("Hasil Prediksi")
        st.write(f"**Status Gizi: {label}**")

        


