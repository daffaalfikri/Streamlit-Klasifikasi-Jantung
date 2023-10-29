import pickle
import streamlit as st

model = pickle.load(open('Klasifikasi-Penyakit-Jantung.sav', 'rb'))

st.title('Estimasi Penyakit Jantung')

age = st.number_input('Masukan umur', step=0, max_value=100, min_value=1)
sex = st.number_input(
    'Masukan Gender,laki laki (Input1),perempuan (Input2)', step=0, max_value=2, min_value=1)
cp = st.number_input(
    'Masukan atypical angina: nyeri dada tidak berhubungan dengan jantung', step=0, max_value=150, min_value=1)
trestbps = st.number_input('Masukan chest pain type',
                           step=0, max_value=250, min_value=1)
chol = st.number_input('Masukan tekanan darah %',
                       step=0, max_value=260, min_value=1)
fbs = st.number_input('Masukan nomer serum cholestoral dalam mg/dl',
                      step=0, max_value=50, min_value=1)
restecg = st.number_input('Masukan fasting gula darah',
                          step=0, max_value=300, min_value=1)
thalach = st.number_input(
    'Masukan sinyal detak jantung yang tidak normal', step=0, max_value=200, min_value=1)
exang = st.number_input(
    'Masukan denyut jantung maksimum tercapai', step=0, max_value=250, min_value=1)
oldpeak = st.number_input('Masukan diinduksi angina', min_value=20.0, step=0.1)
slope = st.number_input(
    'Masukan Depresi yang pernah anda capai', step=0, max_value=250, min_value=1)
ca = st.number_input('Masukan jumlah pembuluh darah utama',
                     step=0, max_value=250, min_value=1)
thal = st.number_input('Masukan hasil stres thalium',
                       step=0, max_value=250, min_value=1)
target = st.number_input('Masukan hasil target',
                         step=0, max_value=500, min_value=1)

predict = ''

if st.button(' Estimasi Klasifikasi Jantung'):
    predict = model.predict(
        [[age, sex, cp, trestbps, chol, fbs, restecg,
            thalach, exang, oldpeak, slope, ca, thal, target]]
    )
    st.write('Estimasi Klasifikasi PenyakitmJantung Dalam TES: ', predict)
    st.write('Estimasi Klasifikasi Penyakit Jantung Keseluruhan: ', predict*2000)
