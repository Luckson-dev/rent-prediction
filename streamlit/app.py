import streamlit as st
import pandas as pd
import numpy as np

# 1. Configuration de la page
st.set_page_config(
    page_title="Tableau de bord - Prédiction de Loyer",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# 2. Injection de CSS personnalisé pour un look épuré, lumineux et moderne (Flat / Light Mode)
st.markdown("""
    <style>
    /* Arrière-plan global */
    .stApp {
        background-color: #F8FAFC;
        font-family: 'Inter', system-ui, -apple-system, sans-serif;
    }
    
    /* Cartes d'indicateurs (KPIs) */
    .metric-card {
        background-color: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 8px;
        padding: 20px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }
    .metric-label {
        font-size: 0.875rem;
        color: #64748B;
        font-weight: 500;
        margin-bottom: 6px;
    }
    .metric-value {
        font-size: 1.75rem;
        color: #0F172A;
        font-weight: 700;
    }
    .metric-subtitle {
        font-size: 0.75rem;
        color: #94A3B8;
        margin-top: 4px;
    }

    /* Blocs de section */
    .content-box {
        background-color: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 8px;
        padding: 24px;
        margin-top: 20px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }
    
    /* Titres */
    .section-title {
        font-size: 1.1rem;
        font-weight: 600;
        color: #1E293B;
        margin-bottom: 16px;
    }
    
    /* Masquer le menu Streamlit et le footer par défaut */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    </style>
""", unsafe_allow_html=True)

# 3. Initialisation de l'état de la session (Formulaire)
if 'form_data' not in st.session_state:
    st.session_state['form_data'] = {
        'Quartier_Target': 0.0021,
        'Superficie_m2': 180.0,
        'Chambres': 4,
        'Meuble_Bin': "Oui",
        'Jardin_Bin': "Non"
    }

# 4. Formulaire dans une fenêtre modale (@st.dialog)
@st.dialog("Saisir les caractéristiques du logement")
def open_form():
    with st.form("property_form"):
        quartier = st.selectbox(
            "Quartier",
            options=[],
            value=float(st.session_state['form_data']['Quartier_Target']),
            format="%.6f"
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
        meuble = st.selectbox(
            "Meublé ?",
            options=["Non", "Oui"],
            index=1 if st.session_state['form_data']['Meuble_Bin'] == "Oui" else 0
        )
        jardin = st.selectbox(
            "Jardin ?",
            options=["Non", "Oui"],
            index=1 if st.session_state['form_data']['Jardin_Bin'] == "Oui" else 0
        )
        
        submitted = st.form_submit_button("Calculer la prédiction", use_container_width=True)
        if submitted:
            st.session_state['form_data'] = {
                'Quartier_Target': quartier,
                'Superficie_m2': superficie,
                'Chambres': chambres,
                'Meuble_Bin': meuble,
                'Jardin_Bin': jardin
            }
            st.rerun()

# 5. En-tête de la page
col_header, col_btn = st.columns([4, 1], vertical_alignment="center")

with col_header:
    st.title("Tableau de Bord — Modèle de Prédiction de Loyer")
    st.caption("Évaluation des performances et simulation d'estimation immobilière.")

with col_btn:
    if st.button("➕ Saisir un bien", use_container_width=True, type="primary"):
        open_form()

st.divider()

# 6. Données statiques (Exemple de démonstration)
prediction_val = 1_250_000  # Loyer estimé en BIF
erreur_cumulee_val = 1_245_260  # Prédiction - Erreur (Ex: erreur = 4 740)
score_prediction = 99.61  # R² ou score de confiance (%)

# --- SECTION 1 : Les 3 Cards d'indicateurs ---
c1, c2, c3 = st.columns(3)

with c1:
    st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Loyer Estimé (Prédiction)</div>
            <div class="metric-value">{prediction_val:,.0f} BIF</div>
            <div class="metric-subtitle">Valeur prédite par Gradient Boosting</div>
        </div>
    """, unsafe_allow_html=True)

with c2:
    st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Prédiction Ajustée (Prédiction - Erreur)</div>
            <div class="metric-value">{erreur_cumulee_val:,.0f} BIF</div>
            <div class="metric-subtitle">Valeur corrigée de l'erreur moyenne</div>
        </div>
    """, unsafe_allow_html=True)

with c3:
    st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Score de Prédiction (R²)</div>
            <div class="metric-value">{score_prediction:.2f}%</div>
            <div class="metric-subtitle">Précision en validation croisée</div>
        </div>
    """, unsafe_allow_html=True)

# Métriques de Performance & Statistiques ---
st.markdown('<div class="content-box">', unsafe_allow_html=True)
st.markdown('<div class="section-title">Performances du Modèle & Statistiques Clés</div>', unsafe_allow_html=True)

m1, m2, m3, m4 = st.columns(4)

with m1:
    st.metric(label="MSE (Mean Squared Error)", value="0.000065")
with m2:
    st.metric(label="MAE (Mean Absolute Error)", value="0.003791")
with m3:
    st.metric(label="MEDAE (Median Abs Error)", value="0.003430")
with m4:
    st.metric(label="RMSE (Root Mean Sq Error)", value="0.011695")

st.markdown("<br>", unsafe_allow_html=True)

# Graphiques simples de démonstration
chart_col1, chart_col2 = st.columns(2)

# Graphique 1: Erreurs Résiduelles
with chart_col1:
    st.caption("Distribution des résidus (Erreurs de prédiction)")
    np.random.seed(42)
    chart_data = pd.DataFrame({
        'Échantillons': np.arange(1, 51),
        'Erreur Résiduelle': np.random.normal(loc=0, scale=0.01, size=50)
    })
    st.line_chart(chart_data, x='Échantillons', y='Erreur Résiduelle', height=200)

# Graphique 2: Importance des variables
with chart_col2:
    st.caption("Importance relative des variables dans le modèle")
    importance_data = pd.DataFrame({
        'Variable': ['Superficie_m2', 'Quartier_Target', 'Chambres', 'Jardin_Bin', 'Meuble_Bin'],
        'Importance': [0.45, 0.30, 0.15, 0.06, 0.04]
    }).set_index('Variable')
    st.bar_chart(importance_data, height=200)

st.markdown('</div>', unsafe_allow_html=True)

# --- SECTION 3 : Tableau des Variables du Formulaire ---
st.markdown('<div class="content-box">', unsafe_allow_html=True)
st.markdown('<div class="section-title">Variables d\'Entrée du Logement</div>', unsafe_allow_html=True)

data_inputs = {
    "Variable": [
        "Quartier_Target", 
        "Superficie_m2", 
        "Chambres", 
        "Meuble_Bin", 
        "Jardin_Bin"
    ],
    "Description": [
        "Target encoding de la localisation",
        "Superficie totale en mètres carrés",
        "Nombre complet de chambres",
        "Présence de meubles (1 = Oui, 0 = Non)",
        "Présence d'un jardin (1 = Oui, 0 = Non)"
    ],
    "Valeur Saisie": [
        f"{st.session_state['form_data']['Quartier_Target']:.6f}",
        f"{st.session_state['form_data']['Superficie_m2']} m²",
        st.session_state['form_data']['Chambres'],
        1 if st.session_state['form_data']['Meuble_Bin'] == "Oui" else 0,
        1 if st.session_state['form_data']['Jardin_Bin'] == "Oui" else 0
    ]
}

df_display = pd.DataFrame(data_inputs)
st.dataframe(df_display, use_container_width=True, hide_index=True)

st.markdown('</div>', unsafe_allow_html=True)