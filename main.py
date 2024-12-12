import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import AdaBoostClassifier

# Judul aplikasi
st.title("Klasifikasi Tumbuh Kembang Anak")
st.subheader("Metode: Naïve Bayes dengan Ensemble AdaBoost")

# Sidebar untuk mengunggah dataset
st.sidebar.header("Unggah Dataset")
uploaded_file = st.sidebar.file_uploader("Unggah file CSV dataset", type=["csv"])

if uploaded_file:
    # Membaca dataset
    data = pd.read_csv(uploaded_file)
    st.write("### Dataset yang Diunggah")
    st.dataframe(data)

    # Memilih fitur dan target
    st.sidebar.subheader("Konfigurasi Data")
    features = st.sidebar.multiselect("Pilih Fitur (X)", options=data.columns)
    target = st.sidebar.selectbox("Pilih Target (Y)", options=data.columns)

    if features and target:
        X = data[features]
        y = data[target]

        # Membagi data secara otomatis (di belakang layar)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42
        )

        # Model Naive Bayes dengan AdaBoost
        base_model = GaussianNB()
        model = AdaBoostClassifier(estimator=base_model, n_estimators=50)
        model.fit(X_train, y_train)

        # Prediksi
        y_pred = model.predict(X_test)

        # Antarmuka prediksi manual
        st.write("### Prediksi Manual")
        user_input = {}
        for feature in features:
            user_input[feature] = st.number_input(f"Masukkan {feature}", value=0.0)

        if st.button("Prediksi"):
            user_df = pd.DataFrame([user_input])
            prediction = model.predict(user_df)
            st.write(f"**Hasil Prediksi:** {prediction[0]}")
else:
    st.write("Silakan unggah dataset terlebih dahulu melalui sidebar.")
