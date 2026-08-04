import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
import streamlit as st
from src.prediction import RENTPrediction

st.set_page_config(
    page_title="Prédiction de Loyer",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Style CSS
st.markdown("""
    <style>
    /* Arrière-plan global sombre */
    .stApp {
        background-color: #0F172A;
        color: #F8FAFC;
        font-family: 'Inter', system-ui, -apple-system, sans-serif;
    }
    
    /* Carte de prédiction */
    .metric-card {
        background-color: #1E293B;
        border: 2px solid #6366F1;
        border-radius: 12px;
        padding: 24px;
        box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.5);
        text-align: center;
        max-width: 450px;
        margin: 20px auto 30px auto;
    }
    .metric-label {
        font-size: 0.9rem;
        color: #94A3B8;
        font-weight: 600;
        margin-bottom: 8px;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    .metric-value {
        font-size: 2.5rem;
        color: #818CF8;
        font-weight: 800;
    }

    .section-title {
        font-size: 1.25rem;
        font-weight: 600;
        color: #F8FAFC;
        margin-bottom: 20px;
        text-align: center;
    }
    
    label {
        color: #CBD5E1 !important;
    }
    
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    </style>
""", unsafe_allow_html=True)

# Initialisation des états
if 'form_data' not in st.session_state:
    st.session_state['form_data'] = {
        'Quartier': "Kinindo",
        'Superficie_m2': 180.0,
        'Chambres': 4,
        'Meuble': "Oui",
        'Jardin': "Non"
    }

if 'prediction_val' not in st.session_state:
    st.session_state['prediction_val'] = 0.0

liste_quartiers = [
    "Rohero", "Kiriri", "Gihosha", "Kinindo", 
    "Ngagara", "Nyakabiga", "Buyenzi", "Kamenge"
]

@st.cache_resource
def load_prediction_model():
    return RENTPrediction(file_path="models/final_model.pkl")

rent_model = load_prediction_model()

metric_placeholder = st.empty()

st.markdown('<div class="section-title">Caractéristiques du logement</div>', unsafe_allow_html=True)

with st.form("property_form"):
    col1, col2 = st.columns(2)
    
    with col1:
        quartier = st.selectbox(
            "Quartier",
            options=liste_quartiers,
            index=liste_quartiers.index(st.session_state['form_data']['Quartier']) if st.session_state['form_data']['Quartier'] in liste_quartiers else 0
        )
        
        superficie = st.number_input(
            "Superficie (m²)",
            value=float(st.session_state['form_data']['Superficie_m2']),
            min_value=10.0,
            step=5.0
        )
        
        chambres = st.number_input(
            "Nombre de chambres",
            value=int(st.session_state['form_data']['Chambres']),
            min_value=1,
            step=1
        )
    
    with col2:
        meuble = st.selectbox(
            "Meublé ?",
            options=["Non", "Oui"],
            index=1 if st.session_state['form_data']['Meuble'] == "Oui" else 0
        )
        
        jardin = st.selectbox(
            "Jardin ?",
            options=["Non", "Oui"],
            index=1 if st.session_state['form_data']['Jardin'] == "Oui" else 0
        )

    st.markdown("<br>", unsafe_allow_html=True)
    submitted = st.form_submit_button("Calculer la prédiction", use_container_width=True, type="primary")
    
    if submitted:
        st.session_state['form_data'] = {
            'Quartier': quartier,
            'Superficie_m2': superficie,
            'Chambres': chambres,
            'Meuble': meuble,
            'Jardin': jardin
        }
        
        input_df = pd.DataFrame([st.session_state['form_data']])
        
        pred = rent_model.predict(input_df)
        print()
        print(f"Prediction (expm1 scale): {pred}")
        print()
        st.session_state['prediction_val'] = pred[0]

metric_placeholder.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">Loyer Estimé</div>
        <div class="metric-value">{st.session_state['prediction_val']:,.0f} BIF</div>
    </div>
""", unsafe_allow_html=True)