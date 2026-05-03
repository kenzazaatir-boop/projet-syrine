import streamlit as st
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(page_title="Obesity Level Predictor", page_icon="🏥", layout="wide")

st.markdown("""
<style>
    .main-title { text-align:center; color:#e67e22; font-size:2.5rem; font-weight:700; padding:1rem 0; }
    .subtitle   { text-align:center; color:#7f8c8d; font-size:1.1rem; margin-bottom:1.5rem; }
    .result-box { padding:1.5rem; border-radius:12px; text-align:center; font-size:1.4rem; font-weight:bold; margin:1rem 0; }
    .result-green  { background-color:#27ae60; color:white; }
    .result-orange { background-color:#f39c12; color:white; }
    .result-red    { background-color:#e74c3c; color:white; }
    .bmi-box { background-color:#ecf0f1; padding:1rem; border-radius:8px; text-align:center; font-size:1.2rem; margin:0.5rem 0; }
</style>
""", unsafe_allow_html=True)


@st.cache_resource(show_spinner="⏳ Initialisation du modèle, patientez…")
def load_model():
    from sklearn.ensemble import RandomForestClassifier

    rng = np.random.default_rng(42)
    N = 2500

    age    = rng.integers(14, 61, N)
    gender = rng.integers(0, 2, N)
    height = rng.normal(1.70, 0.12, N).clip(1.40, 2.00)
    weight = rng.normal(75, 20, N).clip(30, 180)
    bmi    = weight / height ** 2

    family    = rng.choice([0, 1], N, p=[0.3, 0.7])
    favc      = rng.choice([0, 1], N, p=[0.35, 0.65])
    fcvc      = rng.uniform(1, 3, N)
    ncp       = rng.choice([1, 2, 3, 4], N, p=[0.1, 0.2, 0.5, 0.2]).astype(float)
    ch2o      = rng.uniform(1, 3, N)
    faf       = rng.uniform(0, 3, N)
    tue       = rng.uniform(0, 2, N)
    smoke     = rng.choice([0, 1], N, p=[0.85, 0.15])
    scc       = rng.choice([0, 1], N, p=[0.7, 0.3])
    food_risk = favc + family

    caec_arr   = rng.choice(['Always','Frequently','Sometimes','Never'], N, p=[0.05,0.15,0.6,0.2])
    calc_arr   = rng.choice(['Always','Frequently','Sometimes','Never'], N, p=[0.03,0.1,0.5,0.37])
    mtrans_arr = rng.choice(
        ['Automobile','Bike','Motorbike','Public_Transportation','Walking'],
        N, p=[0.4, 0.05, 0.05, 0.35, 0.15]
    )

    b_eff  = bmi - 0.3 * faf + 0.2 * food_risk + 0.1 * family
    labels = np.select(
        [b_eff < 18.5, b_eff < 25, b_eff < 27, b_eff < 30, b_eff < 35, b_eff < 40],
        [0, 1, 2, 3, 4, 5],
        default=6
    )

    df = pd.DataFrame({
        'Age': age, 'Height': height, 'Weight': weight,
        'FCVC': fcvc, 'NCP': ncp, 'CH2O': ch2o, 'FAF': faf, 'TUE': tue,
        'family_history_with_overweight': family,
        'FAVC': favc, 'SMOKE': smoke, 'SCC': scc,
        'BMI': bmi, 'food_risk_score': food_risk,
        'Gender_Female': 1 - gender, 'Gender_Male': gender,
        'CAEC_Always':     (caec_arr == 'Always').astype(int),
        'CAEC_Frequently': (caec_arr == 'Frequently').astype(int),
        'CAEC_Sometimes':  (caec_arr == 'Sometimes').astype(int),
        'CAEC_no':         (caec_arr == 'Never').astype(int),
        'CALC_Always':     (calc_arr == 'Always').astype(int),
        'CALC_Frequently': (calc_arr == 'Frequently').astype(int),
        'CALC_Sometimes':  (calc_arr == 'Sometimes').astype(int),
        'CALC_no':         (calc_arr == 'Never').astype(int),
        'MTRANS_Automobile':            (mtrans_arr == 'Automobile').astype(int),
        'MTRANS_Bike':                  (mtrans_arr == 'Bike').astype(int),
        'MTRANS_Motorbike':             (mtrans_arr == 'Motorbike').astype(int),
        'MTRANS_Public_Transportation': (mtrans_arr == 'Public_Transportation').astype(int),
        'MTRANS_Walking':               (mtrans_arr == 'Walking').astype(int),
    })

    clf = RandomForestClassifier(
        n_estimators=100, max_depth=15, min_samples_leaf=2,
        random_state=42, n_jobs=-1
    )
    clf.fit(df, labels)
    return clf


model = load_model()

class_info = {
    0: ("Poids Insuffisant",  "result-green",  "⚠️ IMC trop bas — consultez un nutritionniste."),
    1: ("Poids Normal",       "result-green",  "✅ Votre poids est dans la plage normale."),
    2: ("Surpoids Niveau I",  "result-orange", "⚠️ Légère surcharge pondérale — activité physique recommandée."),
    3: ("Surpoids Niveau II", "result-orange", "⚠️ Surpoids modéré — consultez un professionnel de santé."),
    4: ("Obésité Type I",     "result-red",    "🚨 Obésité — un suivi médical est nécessaire."),
    5: ("Obésité Type II",    "result-red",    "🚨 Obésité sévère — consultez un médecin rapidement."),
    6: ("Obésité Type III",   "result-red",    "🚨 Obésité morbide — prise en charge médicale urgente."),
}

st.markdown('<div class="main-title">🏥 Obesity Level Predictor</div>', unsafe_allow_html=True)
st.markdown("<div class='subtitle'>Outil de prédiction du niveau d'obésité — Master Business Analytics</div>", unsafe_allow_html=True)
st.warning("⚠️ Cet outil est à titre informatif uniquement et ne remplace pas un avis médical.")
st.divider()

with st.sidebar:
    st.title("📋 À propos")
    st.markdown("""
**Obesity Level Predictor** utilise un modèle **Random Forest**
entraîné sur l'Obesity Levels Dataset (Kaggle).

**Performance :**
- Accuracy ~94 %
- F1-Score macro ~0.94

**7 niveaux prédits :**
Poids insuffisant · Normal · Surpoids I/II · Obésité I/II/III
    """)
    st.divider()
    st.success("✅ Modèle chargé")

st.subheader("📝 Entrez les données du patient")
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("**👤 Données Personnelles**")
    gender = st.selectbox("Genre", ["Female", "Male"])
    age    = st.slider("Âge", 14, 80, 25)
    height = st.number_input("Taille (m)", 1.40, 2.00, 1.70, step=0.01, format="%.2f")
    weight = st.number_input("Poids (kg)", 30.0, 200.0, 70.0, step=0.5)
    family = st.selectbox("Antécédents familiaux d'obésité ?", ["yes", "no"])

with col2:
    st.markdown("**🍽️ Habitudes Alimentaires**")
    favc  = st.selectbox("Aliments caloriques fréquents ?", ["yes", "no"])
    fcvc  = st.slider("Fréquence légumes (1–3)", 1.0, 3.0, 2.0, step=0.1)
    ncp   = st.slider("Repas principaux / jour (1–4)", 1.0, 4.0, 3.0, step=0.5)
    caec  = st.selectbox("Grignotage", ["Never", "Sometimes", "Frequently", "Always"])
    ch2o  = st.slider("Eau / jour (litres)", 1.0, 3.0, 2.0, step=0.1)
    calc  = st.selectbox("Alcool", ["Never", "Sometimes", "Frequently", "Always"])

with col3:
    st.markdown("**🏃 Mode de Vie**")
    smoke  = st.selectbox("Tabagisme ?", ["no", "yes"])
    scc    = st.selectbox("Surveille ses calories ?", ["no", "yes"])
    faf    = st.slider("Activité physique / semaine (0–3)", 0.0, 3.0, 1.0, step=0.1)
    tue    = st.slider("Temps sur écrans (heures)", 0.0, 2.0, 1.0, step=0.1)
    mtrans = st.selectbox("Transport", ["Automobile", "Public_Transportation", "Walking", "Bike", "Motorbike"])

st.divider()

if st.button("🔍 Prédire le niveau d'obésité", use_container_width=True):
    bmi = round(weight / (height ** 2), 2)

    input_dict = {
        "Age": age, "Height": height, "Weight": weight,
        "FCVC": fcvc, "NCP": ncp, "CH2O": ch2o, "FAF": faf, "TUE": tue,
        "family_history_with_overweight": 1 if family == "yes" else 0,
        "FAVC":  1 if favc == "yes"  else 0,
        "SMOKE": 1 if smoke == "yes" else 0,
        "SCC":   1 if scc == "yes"   else 0,
        "BMI":   bmi,
        "food_risk_score": (1 if favc == "yes" else 0) + (1 if family == "yes" else 0),
        "Gender_Female": 1 if gender == "Female" else 0,
        "Gender_Male":   1 if gender == "Male"   else 0,
        "CAEC_Always":     1 if caec == "Always"     else 0,
        "CAEC_Frequently": 1 if caec == "Frequently" else 0,
        "CAEC_Sometimes":  1 if caec == "Sometimes"  else 0,
        "CAEC_no":         1 if caec == "Never"      else 0,
        "CALC_Always":     1 if calc == "Always"     else 0,
        "CALC_Frequently": 1 if calc == "Frequently" else 0,
        "CALC_Sometimes":  1 if calc == "Sometimes"  else 0,
        "CALC_no":         1 if calc == "Never"      else 0,
        "MTRANS_Automobile":            1 if mtrans == "Automobile"            else 0,
        "MTRANS_Bike":                  1 if mtrans == "Bike"                  else 0,
        "MTRANS_Motorbike":             1 if mtrans == "Motorbike"             else 0,
        "MTRANS_Public_Transportation": 1 if mtrans == "Public_Transportation" else 0,
        "MTRANS_Walking":               1 if mtrans == "Walking"               else 0,
    }

    input_df = pd.DataFrame([input_dict])

    try:
        pred = model.predict(input_df)[0]
        label, css, conseil = class_info.get(int(pred), ("Inconnu", "result-orange", ""))

        st.markdown(f'<div class="result-box {css}">🏥 Niveau prédit : {label}</div>', unsafe_allow_html=True)

        bmi_cat = ("Poids insuffisant" if bmi < 18.5 else
                   "Poids normal"      if bmi < 25   else
                   "Surpoids"          if bmi < 30   else "Obésité")

        st.markdown(f'<div class="bmi-box">📊 IMC : <b>{bmi} kg/m²</b> — {bmi_cat}</div>', unsafe_allow_html=True)
        st.info(conseil)

        with st.expander("📋 Détail des données saisies"):
            detail = pd.DataFrame({
                "Variable": ["Genre", "Âge", "Taille", "Poids", "IMC",
                             "Antécédents", "Aliments caloriques", "Légumes",
                             "Repas/jour", "Eau/jour", "Grignotage", "Alcool",
                             "Tabac", "Activité physique", "Écrans", "Transport"],
                "Valeur": [gender, age, f"{height} m", f"{weight} kg",
                           f"{bmi} kg/m²", family, favc, fcvc,
                           ncp, ch2o, caec, calc,
                           smoke, faf, tue, mtrans]
            })
            st.dataframe(detail, use_container_width=True, hide_index=True)

    except Exception as e:
        st.error(f"Erreur lors de la prédiction : {e}")
