"""
Dashboard Credit Scoring Production - Streamlit Cloud
Plateforme: Streamlit Cloud + Railway API v5.0
WCAG 2.1 AA CONFORME - 100% ACCESSIBLE
"""

import streamlit as st
import requests
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import json
import time
from datetime import datetime

# Configuration Streamlit
st.set_page_config(
    page_title="Dashboard Credit Scoring - Prêt à dépenser",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Configuration Plotly pour accessibilité WCAG
PLOTLY_CONFIG = {
    'displayModeBar': True,
    'displaylogo': False,
    'modeBarButtonsToRemove': ['pan2d', 'lasso2d', 'select2d'],
    'accessible': True,
    'locale': 'fr',
    'toImageButtonOptions': {
        'format': 'png',
        'filename': 'graphique_credit_scoring',
        'height': 500,
        'width': 700,
        'scale': 1
    }
}

# CSS WCAG 2.1 AA COMPLET - 100% ACCESSIBLE
st.markdown("""
<style>
/* === WCAG 2.1 AA - ACCESSIBILITÉ COMPLÈTE === */

/* Textes cachés pour lecteurs d'écran */
.sr-only {
    position: absolute !important;
    width: 1px !important;
    height: 1px !important;
    padding: 0 !important;
    margin: -1px !important;
    overflow: hidden !important;
    clip: rect(0, 0, 0, 0) !important;
    white-space: nowrap !important;
    border: 0 !important;
}

/* En-tête principal avec landmarks */
.main-header {
    background: linear-gradient(135deg, #1e3a8a 0%, #3b82f6 100%);
    color: #ffffff;
    padding: 2rem;
    border-radius: 1rem;
    margin-bottom: 2rem;
    font-size: 2rem;
    font-weight: bold;
    text-align: center;
    box-shadow: 0 8px 25px rgba(0, 0, 0, 0.15);
    transition: all 0.3s ease;
}

/* WCAG 2.5.5 - Zones cliquables minimum 44px */
.stButton > button {
    background: linear-gradient(135deg, #3b82f6 0%, #1e40af 100%) !important;
    color: white !important;
    border: none !important;
    border-radius: 0.75rem !important;
    padding: 0.75rem 1.5rem !important;
    font-weight: 600 !important;
    font-size: 1rem !important;
    box-shadow: 0 4px 15px rgba(59, 130, 246, 0.3) !important;
    transition: all 0.3s ease !important;
    text-transform: none !important;
    letter-spacing: 0.025em !important;
    min-height: 44px !important;
    min-width: 44px !important;
    width: 100% !important;
}

/* WCAG 2.4.7 - Focus visible pour navigation clavier */
.stButton > button:focus,
.stSelectbox > div > div:focus,
.stSlider > div:focus,
.stNumberInput > div > div > input:focus {
    outline: 3px solid #2563eb !important;
    outline-offset: 2px !important;
    box-shadow: 0 0 0 3px rgba(37, 99, 235, 0.3) !important;
}

.stButton > button:hover {
    background: linear-gradient(135deg, #2563eb 0%, #1d4ed8 100%) !important;
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 25px rgba(59, 130, 246, 0.4) !important;
}

.stButton > button:active {
    transform: translateY(0px) !important;
    box-shadow: 0 4px 15px rgba(59, 130, 246, 0.3) !important;
}

/* Bouton primaire avec contraste renforcé */
.stButton > button[kind="primary"] {
    background: linear-gradient(135deg, #059669 0%, #047857 100%) !important;
    box-shadow: 0 4px 15px rgba(5, 150, 105, 0.3) !important;
}

.stButton > button[kind="primary"]:hover {
    background: linear-gradient(135deg, #047857 0%, #065f46 100%) !important;
    box-shadow: 0 8px 25px rgba(5, 150, 105, 0.4) !important;
}

/* WCAG 1.4.3 - Contraste minimum 4.5:1 */
.metric-card {
    background: linear-gradient(135deg, #f8fafc 0%, #e2e8f0 100%);
    border: 3px solid #374151;
    padding: 1.5rem;
    border-radius: 1rem;
    margin: 1rem 0;
    font-size: 1.1rem;
    box-shadow: 0 4px 15px rgba(0, 0, 0, 0.1);
    transition: all 0.3s ease;
    color: #111827;
}

/* WCAG 1.4.1 - Couleurs avec alternatives visuelles */
.success-card {
    background: linear-gradient(135deg, #dcfce7 0%, #bbf7d0 100%);
    border: 3px solid #059669;
    color: #065f46;
    box-shadow: 0 4px 20px rgba(5, 150, 105, 0.2);
}

.success-card::before {
    content: "✅ APPROUVÉ ";
    font-weight: bold;
    color: #059669;
    margin-right: 0.5rem;
    font-size: 1.2rem;
}

.error-card {
    background: linear-gradient(135deg, #fee2e2 0%, #fecaca 100%);
    border: 3px solid #dc2626;
    color: #991b1b;
    box-shadow: 0 4px 20px rgba(220, 38, 38, 0.2);
}

.error-card::before {
    content: "❌ REFUSÉ ";
    font-weight: bold;
    color: #dc2626;
    margin-right: 0.5rem;
    font-size: 1.2rem;
}

.warning-card {
    background: linear-gradient(135deg, #fef3c7 0%, #fde68a 100%);
    border: 3px solid #d97706;
    color: #92400e;
    box-shadow: 0 4px 20px rgba(217, 119, 6, 0.2);
}

/* WCAG Accessibilité renforcée */
.alert-info {
    background: linear-gradient(135deg, #eff6ff 0%, #dbeafe 100%);
    border: 3px solid #3b82f6;
    color: #1e40af;
    padding: 1rem;
    border-radius: 0.75rem;
    margin: 1rem 0;
    font-weight: 500;
    box-shadow: 0 4px 15px rgba(59, 130, 246, 0.1);
}

.alert-success {
    background: linear-gradient(135deg, #dcfce7 0%, #bbf7d0 100%);
    border: 3px solid #16a34a;
    color: #15803d;
    padding: 1rem;
    border-radius: 0.75rem;
    margin: 1rem 0;
    font-weight: 500;
    box-shadow: 0 4px 15px rgba(22, 163, 74, 0.1);
}

/* WCAG 1.4.12 - Espacement du texte */
p, li, td {
    line-height: 1.5 !important;
    letter-spacing: 0.025em !important;
}

h1, h2, h3, h4, h5, h6 {
    line-height: 1.3 !important;
    margin-bottom: 0.5em !important;
}

/* WCAG 1.4.10 - Reflow responsive */
@media (max-width: 768px) {
    .main-header {
        font-size: 1.5rem;
        padding: 1rem;
    }
    
    .stButton > button {
        font-size: 0.9rem !important;
        padding: 0.5rem 1rem !important;
    }
    
    .metric-card {
        padding: 1rem;
        font-size: 1rem;
    }
}

/* WCAG Préférences utilisateur - prefers-reduced-motion */
@media (prefers-reduced-motion: reduce) {
    .metric-card,
    .stButton > button,
    .main-header {
        transition: none !important;
        animation: none !important;
    }
    
    .stButton > button:hover,
    .metric-card:hover {
        transform: none !important;
    }
}

/* WCAG High Contrast Mode */
@media (prefers-contrast: high) {
    .metric-card {
        border-width: 4px !important;
        background: #ffffff !important;
        color: #000000 !important;
    }
    
    .success-card {
        background: #ffffff !important;
        color: #000000 !important;
        border-color: #000000 !important;
    }
    
    .error-card {
        background: #ffffff !important;
        color: #000000 !important;
        border-color: #000000 !important;
    }
}

/* WCAG 1.4.4 - Support zoom jusqu'à 200% */
@media (min-width: 1200px) {
    .main-content {
        max-width: 1200px;
        margin: 0 auto;
    }
}

/* Focus management pour skip links */
.skip-link {
    position: absolute;
    top: -40px;
    left: 6px;
    background: #000;
    color: #fff;
    padding: 8px;
    text-decoration: none;
    border-radius: 4px;
    z-index: 1000;
}

.skip-link:focus {
    top: 6px;
}

/* ARIA live regions styling */
[aria-live] {
    position: relative;
}

/* Amélioration des tabs pour accessibilité */
.stTabs [data-testid="stTabs"] {
    gap: 1rem;
}

.stTabs [data-testid="stTabs"] button {
    padding: 0.75rem 1rem !important;
    min-height: 44px !important;
    border-radius: 0.5rem !important;
}

.stTabs [data-testid="stTabs"] button:focus {
    outline: 3px solid #2563eb !important;
    outline-offset: 2px !important;
}
</style>
""", unsafe_allow_html=True)

# Configuration API (Railway)
API_URL = "https://dashboard-credit-scoring-production.up.railway.app"

# Traductions des features
FEATURE_TRANSLATIONS = {
    "EXT_SOURCE_1": "Score Externe 1",
    "EXT_SOURCE_2": "Score Externe 2",
    "EXT_SOURCE_3": "Score Externe 3",
    "DAYS_EMPLOYED": "Ancienneté emploi",
    "CODE_GENDER": "Genre",
    "INSTAL_DPD_MEAN": "Retards moyens",
    "PAYMENT_RATE": "Ratio d'endettement",
    "NAME_EDUCATION_TYPE_Higher_education": "Éducation supérieure",
    "AMT_ANNUITY": "Annuité mensuelle",
    "INSTAL_AMT_PAYMENT_SUM": "Historique paiements"
}

FEATURE_EXPLANATIONS = {
    "EXT_SOURCE_2": "🔍 Score externe 2 : plus élevé = moins de risque (0=très risqué, 1=très sûr). Impact fort sur la décision.",
    "EXT_SOURCE_3": "🔍 Score externe 3 : plus élevé = moins de risque (0=très risqué, 1=très sûr). Impact fort sur la décision.",
    "EXT_SOURCE_1": "🔍 Score externe 1 : plus élevé = moins de risque (0=très risqué, 1=très sûr). Impact modéré.",
    "DAYS_EMPLOYED": "🔍 Ancienneté emploi : plus longue = moins de risque. Stabilité professionnelle importante.",
    "PAYMENT_RATE": "🔍 Ratio endettement : plus bas = moins de risque. Charges vs revenus.",
    "CODE_GENDER": "🔍 Genre : variable sociodémographique. Impact statistique léger.",
    "INSTAL_DPD_MEAN": "🔍 Retards moyens : plus élevés = plus de risque. Historique paiements antérieurs.",
    "NAME_EDUCATION_TYPE_Higher_education": "🔍 Éducation supérieure : variable sociodémographique. Impact modéré.",
    "AMT_ANNUITY": "🔍 Annuité mensuelle : plus élevée = plus de risque si disproportionnée aux revenus.",
    "INSTAL_AMT_PAYMENT_SUM": "🔍 Historique paiements : plus important = moins de risque. Expérience crédit positive."
}

# Les 10 variables dashboard
DASHBOARD_FEATURES = [
    'EXT_SOURCE_2', 'EXT_SOURCE_3', 'EXT_SOURCE_1',
    'DAYS_EMPLOYED', 'CODE_GENDER', 'INSTAL_DPD_MEAN',
    'PAYMENT_RATE', 'NAME_EDUCATION_TYPE_Higher_education',
    'AMT_ANNUITY', 'INSTAL_AMT_PAYMENT_SUM'
]

# Initialisation Session State
def init_session_state():
    """Initialiser session state une seule fois"""
    defaults = {
        'client_analyzed': False,
        'client_data': None,
        'prediction_result': None,
        'api_call_in_progress': False, 
        'last_analysis_time': None,
        'population_cache': {},
        'bivariate_cache': {}
    }
    
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

# Appeler une seule fois
init_session_state()

# Fonctions API

@st.cache_data(ttl=300)
def test_api_connection():
    """Test de connexion API SANS st.error()"""
    try:
        response = requests.get(f"{API_URL}/health", timeout=10)
        if response.status_code == 200:
            return True, response.json(), None
        return False, None, f"Status {response.status_code}"
    except Exception as e:
        return False, None, str(e)

def call_prediction_api(client_data):
    """Appel API de prédiction - SANS CACHE pour contrôle strict"""
    try:
        response = requests.post(
            f"{API_URL}/predict_dashboard",
            json=client_data,
            timeout=30,
            headers={"Content-Type": "application/json"}
        )

        if response.status_code == 200:
            return response.json(), None
        else:
            error_text = response.text
            return None, f"Erreur API {response.status_code}: {error_text}"

    except requests.exceptions.Timeout:
        return None, "Timeout API - Veuillez réessayer"
    except Exception as e:
        return None, f"Erreur connexion: {str(e)}"

def get_population_distribution(variable):
    """Récupérer distribution d'une variable spécifique - SANS CACHE"""
    try:
        response = requests.get(f"{API_URL}/population/{variable}", timeout=15)
        if response.status_code == 200:
            return response.json()
        return None
    except Exception as e:
        return None

def get_population_data():
    """Récupérer données population - SANS CACHE"""
    try:
        response = requests.get(f"{API_URL}/population_stats", timeout=15)
        if response.status_code == 200:
            return response.json()
        return None
    except Exception as e:
        return None

def get_bivariate_data(var1, var2):
    """Analyse bi-variée - SANS CACHE"""
    try:
        response = requests.post(
            f"{API_URL}/bivariate_analysis",
            json={"variable1": var1, "variable2": var2},
            timeout=20
        )
        if response.status_code == 200:
            return response.json()
        return None
    except Exception as e:
        return None

# Interface de saisie client

def create_client_form():
    """Formulaire de saisie client avec accessibilité renforcée"""
    
    # WCAG 2.4.6 - En-têtes descriptifs
    st.markdown('<section aria-labelledby="form-section-heading">', unsafe_allow_html=True)
    st.markdown('<h3 id="form-section-heading">Formulaire de Saisie des Données Client</h3>', unsafe_allow_html=True)
    
    with st.expander("ℹ️ Guide d'utilisation", expanded=False):
        st.markdown("""
        <div role="region" aria-labelledby="guide-heading">
            <h4 id="guide-heading" class="sr-only">Guide d'utilisation du dashboard</h4>
            <ol role="list">
                <li><strong>Saisissez</strong> les informations client dans le formulaire ci-dessous</li>
                <li><strong>Analysez</strong> le dossier en cliquant sur "Analyser ce client"</li>
                <li><strong>Explorez</strong> les onglets Résultats, Comparaisons et Analyses</li>
                <li><strong>Simulez</strong> différents scénarios si nécessaire</li>
            </ol>
        </div>
        """, unsafe_allow_html=True)

    # Valeurs par défaut (ou valeurs précédentes si modification)
    default_values = st.session_state.client_data if st.session_state.client_data else {}

    col1, col2 = st.columns(2)

    with col1:
        st.markdown('<h4 id="scores-section">Scores et Ancienneté</h4>', unsafe_allow_html=True)

        ext_source_2 = st.slider(
            "Score Externe 2",
            0.0, 1.0,
            float(default_values.get('EXT_SOURCE_2', 0.6)),
            0.01,
            help=FEATURE_EXPLANATIONS["EXT_SOURCE_2"],
            key="ext_source_2"
        )

        ext_source_3 = st.slider(
            "Score Externe 3",
            0.0, 1.0,
            float(default_values.get('EXT_SOURCE_3', 0.5)),
            0.01,
            help=FEATURE_EXPLANATIONS["EXT_SOURCE_3"],
            key="ext_source_3"
        )

        ext_source_1 = st.slider(
            "Score Externe 1",
            0.0, 1.0,
            float(default_values.get('EXT_SOURCE_1', 0.4)),
            0.01,
            help=FEATURE_EXPLANATIONS["EXT_SOURCE_1"],
            key="ext_source_1"
        )

        # Conversion jours en années pour l'affichage
        default_employment = abs(default_values.get('DAYS_EMPLOYED', -1825)) / 365.25
        employment_years = st.number_input(
            "Ancienneté emploi (années)",
            0.0, 40.0,
            float(default_employment),
            0.01,
            help=FEATURE_EXPLANATIONS["DAYS_EMPLOYED"],
            key="employment_years"
        )

        instal_dpd_mean = st.slider(
            "Retards moyens (jours)",
            0.0, 30.0,
            float(default_values.get('INSTAL_DPD_MEAN', 0.5)),
            0.1,
            help=FEATURE_EXPLANATIONS["INSTAL_DPD_MEAN"],
            key="instal_dpd_mean"
        )

    with col2:
        st.markdown('<h4 id="profile-section">Profil et Finances</h4>', unsafe_allow_html=True)

        # Conversion M/F pour l'affichage
        default_gender = "Homme" if default_values.get('CODE_GENDER') == 'M' else "Femme"
        gender = st.selectbox(
            "Genre",
            ["Femme", "Homme"],
            index=0 if default_gender == "Femme" else 1,
            help=FEATURE_EXPLANATIONS["CODE_GENDER"],
            key="gender"
        )

        payment_rate = st.slider(
            "Ratio d'endettement",
            0.0, 1.0,
            float(default_values.get('PAYMENT_RATE', 0.15)),
            0.01,
            help=FEATURE_EXPLANATIONS["PAYMENT_RATE"],
            key="payment_rate"
        )

        # Conversion 0/1 pour l'affichage
        default_education = "Oui" if default_values.get('NAME_EDUCATION_TYPE_Higher_education', 0) == 1 else "Non"
        education = st.selectbox(
            "Éducation supérieure",
            ["Non", "Oui"],
            index=0 if default_education == "Non" else 1,
            help=FEATURE_EXPLANATIONS["NAME_EDUCATION_TYPE_Higher_education"],
            key="education"
        )

        annuity = st.number_input(
            "Annuité mensuelle (€)",
            5000, 100000,
            int(default_values.get('AMT_ANNUITY', 18000)),
            1000,
            help=FEATURE_EXPLANATIONS["AMT_ANNUITY"],
            key="annuity"
        )

        payment_sum = st.number_input(
            "Historique paiements (€)",
            10000, 1000000,
            int(default_values.get('INSTAL_AMT_PAYMENT_SUM', 120000)),
            10000,
            help=FEATURE_EXPLANATIONS["INSTAL_AMT_PAYMENT_SUM"],
            key="payment_sum"
        )

    st.markdown('</section>', unsafe_allow_html=True)

    # Conversion pour API (années vers jours négatifs)
    employment_days = -int(employment_years * 365.25)

    client_data = {
        "EXT_SOURCE_2": float(ext_source_2),
        "EXT_SOURCE_3": float(ext_source_3),
        "EXT_SOURCE_1": float(ext_source_1),
        "DAYS_EMPLOYED": employment_days,
        "CODE_GENDER": "M" if gender == "Homme" else "F",
        "INSTAL_DPD_MEAN": float(instal_dpd_mean),
        "PAYMENT_RATE": float(payment_rate),
        "NAME_EDUCATION_TYPE_Higher_education": 1 if education == "Oui" else 0,
        "AMT_ANNUITY": float(annuity),
        "INSTAL_AMT_PAYMENT_SUM": float(payment_sum)
    }

    return client_data

def display_prediction_result(result):
    """Afficher résultat de prédiction avec accessibilité complète"""
    prediction = result.get('prediction', {})
    probability = prediction.get('probability', 0)
    decision = prediction.get('decision', 'UNKNOWN')
    decision_fr = prediction.get('decision_fr', decision)
    risk_level = prediction.get('risk_level', 'Inconnu')
    
    # Récupération du Threshold depuis l'API
    threshold = prediction.get('threshold', 0.1)
    threshold_percent = threshold * 100

    # WCAG 1.4.1 & 1.1.1 - Résultat principal avec couleurs + icônes + texte
    if decision == "REFUSE":
        st.markdown(f"""
        <div class="metric-card error-card" role="alert" aria-label="Résultat négatif crédit refusé">
            <h3>
                <span style="color: #dc2626;" aria-hidden="true">❌</span>
                <span class="sr-only">Statut négatif :</span>
                <strong>CRÉDIT REFUSÉ</strong> - 
                Probabilité de défaut: {probability:.1%} - 
                Niveau de risque: {risk_level}
            </h3>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="metric-card success-card" role="alert" aria-label="Résultat positif crédit accordé">
            <h3>
                <span style="color: #16a34a;" aria-hidden="true">✅</span>
                <span class="sr-only">Statut positif :</span>
                <strong>CRÉDIT ACCORDÉ</strong> - 
                Probabilité de défaut: {probability:.1%} - 
                Niveau de risque: {risk_level}
            </h3>
        </div>
        """, unsafe_allow_html=True)

    # Jauge avec seuil dynamique
    fig_gauge = go.Figure(go.Indicator(
        mode="gauge+number",
        value=probability * 100,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={
            'text': "📊 Niveau de Risque (%)",
            'font': {'size': 24, 'color': '#1e40af', 'family': 'Arial Black'}
        },
        number={
            'font': {'size': 48, 'color': '#1e40af', 'family': 'Arial Black'},
            'suffix': '%'
        },
        gauge={
            'axis': {
                'range': [None, 100],
                'tickwidth': 2,
                'tickcolor': "#1e40af",
                'tickfont': {'size': 14, 'color': '#1e40af'}
            },
            'bar': {
                'color': "#3b82f6",
                'thickness': 0.25,
                'line': {'color': "#1e40af", 'width': 2}
            },
            'bgcolor': "white",
            'borderwidth': 3,
            'bordercolor': "#e5e7eb",
            'steps': [
                {'range': [0, threshold_percent], 'color': '#dcfce7', 'name': 'Acceptable'},
                {'range': [threshold_percent, min(threshold_percent * 2.5, 100)], 'color': '#fef3c7', 'name': 'Modéré'},
                {'range': [min(threshold_percent * 2.5, 100), min(threshold_percent * 5, 100)], 'color': '#fed7aa', 'name': 'Élevé'},
                {'range': [min(threshold_percent * 5, 100), 100], 'color': '#fee2e2', 'name': 'Très élevé'}
            ],
            'threshold': {
                'line': {'color': "#dc2626", 'width': 6},
                'thickness': 0.9,
                'value': threshold_percent
            }
        }
    ))

    fig_gauge.update_layout(
        height=450,
        font={'color': "#1e40af", 'family': "Arial", 'size': 16},
        margin=dict(l=50, r=50, t=80, b=50),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)'
    )

    st.plotly_chart(fig_gauge, use_container_width=True, config=PLOTLY_CONFIG)

    # Métriques avec ARIA labels
    probability_percent = probability * 100
    ecart_avec_seuil = probability_percent - threshold_percent
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(
            label="📊 Probabilité de défaut",
            value=f"{probability_percent:.2f}%",
            help="Probabilité calculée par le modèle LightGBM optimisé"
        )
    
    with col2:
        st.metric(
            label="🎯 Seuil de décision",
            value=f"{threshold_percent:.2f}%",
            help="Seuil optimal issu du fichier optimal_threshold_optimized.pkl"
        )
    
    with col3:
        # Couleur selon l'écart
        if ecart_avec_seuil < 0:
            delta_color = "normal"  # Vert (sous le seuil)
            ecart_text = f"-{abs(ecart_avec_seuil):.2f} points"
            interpretation = "Sous le seuil (bon)"
        else:
            delta_color = "inverse"  # Rouge (au-dessus du seuil)
            ecart_text = f"+{ecart_avec_seuil:.2f} points"
            interpretation = "Au-dessus du seuil (risqué)"
            
        st.metric(
            label="📈 Écart avec seuil",
            value=ecart_text,
            delta=interpretation,
            delta_color=delta_color,
            help="Distance par rapport au seuil de décision optimisé"
        )

    # Analyse détaillée de l'écart avec alternatives visuelles
    if abs(ecart_avec_seuil) < 1:  # Très proche du seuil
        st.markdown("""
        <div class="warning-card" role="alert">
            <p><strong>⚠️ Client proche du seuil</strong> : Décision sensible aux variations des données</p>
        </div>
        """, unsafe_allow_html=True)
    elif ecart_avec_seuil < -5:  # Bien en dessous
        st.markdown("""
        <div class="alert-success" role="alert">
            <p><strong>✅ Profil très sûr</strong> : Risque très faible, bien en dessous du seuil</p>
        </div>
        """, unsafe_allow_html=True)
    elif ecart_avec_seuil > 5:  # Bien au-dessus
        st.markdown("""
        <div class="error-card" role="alert">
            <p><strong>❌ Profil très risqué</strong> : Risque élevé, bien au-dessus du seuil</p>
        </div>
        """, unsafe_allow_html=True)

    # WCAG 1.1.1 : Description textuelle complète pour la jauge
    st.markdown(f"""
    <div role="img" aria-labelledby="gauge-description">
        <h4 id="gauge-description" class="sr-only">Description du graphique jauge de risque</h4>
        <p><strong>📊 Description textuelle :</strong> Jauge de risque affichant {probability:.1%} de probabilité de défaut de paiement.
        Le seuil de décision est fixé à {threshold:.1%} (ligne rouge verticale sur la jauge). 
        Ce client se situe dans la zone {'à risque (rouge)' if probability >= threshold else 'acceptable (verte)'}.
        Écart avec le seuil : {ecart_avec_seuil:+.2f} points de pourcentage.
        {'Décision: Crédit refusé.' if probability >= threshold else 'Décision: Crédit accordé.'}</p>
    </div>
    """, unsafe_allow_html=True)

def display_feature_importance(result):
    """Afficher importance des variables avec accessibilité complète"""
    explanation = result.get('explanation', {})
    top_features = explanation.get('top_features', [])
    client_data = st.session_state.client_data

    if not top_features:
        st.warning("Explications des variables non disponibles")
        return

    st.markdown('<h4 id="interpretation-section">🔍 Interprétation de la décision</h4>', unsafe_allow_html=True)

    # Créer données complètes pour toutes les variables
    all_features_data = []

    # Variables avec impact SHAP (top 5)
    for feature in top_features:
        feature_name = feature.get('feature', '')
        shap_value = feature.get('shap_value', 0)
        client_value = client_data.get(feature_name, 0)

        # Déterminer l'impact
        if abs(shap_value) < 0.001:
            impact = "Impact neutre ⚪"
        elif shap_value > 0:
            impact = "Augmente le risque 🔴"
        else:
            impact = "Diminue le risque 🟢"

        all_features_data.append({
            'feature': feature_name,
            'feature_fr': FEATURE_TRANSLATIONS.get(feature_name, feature_name),
            'shap_value': shap_value,
            'client_value': client_value,
            'impact': impact
        })

    # Ajouter les variables restantes avec valeur SHAP = 0
    remaining_features = [
        'EXT_SOURCE_1', 'EXT_SOURCE_2', 'DAYS_EMPLOYED',
        'NAME_EDUCATION_TYPE_Higher_education', 'INSTAL_AMT_PAYMENT_SUM'
    ]

    for feature_name in remaining_features:
        if not any(f['feature'] == feature_name for f in all_features_data):
            client_value = client_data.get(feature_name, 0)
            all_features_data.append({
                'feature': feature_name,
                'feature_fr': FEATURE_TRANSLATIONS.get(feature_name, feature_name),
                'shap_value': 0.0,
                'client_value': client_value,
                'impact': "Impact neutre ⚪"
            })

    # Créer DataFrame pour le graphique
    features_df = pd.DataFrame(all_features_data)

    # Trier par valeur SHAP absolue (décroissante)
    features_df['abs_shap'] = features_df['shap_value'].abs()
    features_df = features_df.sort_values('abs_shap', ascending=True)

    # Couleurs selon impact avec symboles
    features_df['color'] = features_df['shap_value'].apply(
        lambda x: "Augmente le risque 🔴" if x > 0 else ("Diminue le risque 🟢" if x < 0 else "Impact neutre ⚪")
    )

    # Graphique horizontal avec accessibilité
    fig = px.bar(
        features_df,
        x='shap_value',
        y='feature_fr',
        orientation='h',
        color='color',
        color_discrete_map={
            "Augmente le risque 🔴": "#ff4444",
            "Diminue le risque 🟢": "#22c55e", 
            "Impact neutre ⚪": "#94a3b8"
        },
        title="Impact des variables sur la décision de crédit"
    )

    fig.update_layout(
        height=500,
        showlegend=True,
        font={'size': 12},
        xaxis_title="Impact sur la prédiction (valeurs SHAP)",
        yaxis_title="Variables client",
        legend_title="Type d'impact"
    )

    fig.add_vline(x=0, line_dash="dash", line_color="gray", line_width=2)

    st.plotly_chart(fig, use_container_width=True, config=PLOTLY_CONFIG)

    # WCAG 1.1.1 : Description textuelle complète pour graphique feature importance
    positive_features = [f['feature_fr'] for f in all_features_data if f['shap_value'] > 0]
    negative_features = [f['feature_fr'] for f in all_features_data if f['shap_value'] < 0]
    neutral_features = [f['feature_fr'] for f in all_features_data if abs(f['shap_value']) < 0.001]

    st.markdown(f"""
    <div role="img" aria-labelledby="feature-importance-description">
        <h5 id="feature-importance-description" class="sr-only">Description du graphique d'importance des variables</h5>
        <p><strong>📊 Description textuelle :</strong> Graphique en barres horizontales montrant l'impact de chaque variable sur la décision de crédit.
        <span style="color: #ff4444;"><strong>🔴 Variables augmentant le risque (barres rouges vers la droite)</strong></span> : {', '.join(positive_features[:3]) if positive_features else 'Aucune'}.
        <span style="color: #22c55e;"><strong>🟢 Variables diminuant le risque (barres vertes vers la gauche)</strong></span> : {', '.join(negative_features[:3]) if negative_features else 'Aucune'}.
        <span style="color: #94a3b8;"><strong>⚪ Variables neutres (barres grises courtes)</strong></span> : {', '.join(neutral_features[:2]) if neutral_features else 'Aucune'}.
        Plus la barre est longue, plus l'impact est fort. La ligne verticale grise au centre sépare les impacts positifs et négatifs.</p>
    </div>
    """, unsafe_allow_html=True)

    # Tableau détaillé avec accessibilité
    with st.expander("📋 Tableau détaillé des impacts", expanded=True):
        st.markdown('<h5 id="impact-table">Détail de l\'impact de chaque variable</h5>', unsafe_allow_html=True)

        # Préparer données pour le tableau
        table_data = []
        for _, row in features_df.iterrows():
            # Formater la valeur client selon le type de variable
            feature_name = row['feature']
            client_val = row['client_value']

            if feature_name == 'CODE_GENDER':
                formatted_value = "Homme" if client_val == 'M' else "Femme"
            elif feature_name == 'NAME_EDUCATION_TYPE_Higher_education':
                formatted_value = "Oui" if client_val == 1 else "Non"
            elif feature_name == 'DAYS_EMPLOYED':
                formatted_value = f"{abs(client_val)} jours"
            elif 'EXT_SOURCE' in feature_name or feature_name == 'PAYMENT_RATE':
                formatted_value = f"{client_val:.4f}"
            elif feature_name in ['AMT_ANNUITY', 'INSTAL_AMT_PAYMENT_SUM']:
                formatted_value = f"{client_val:,.0f} €"
            else:
                formatted_value = f"{client_val:.1f}"

            table_data.append({
                'Variable': row['feature_fr'],
                'Valeur SHAP': f"{row['shap_value']:.4f}",
                'Valeur Client': formatted_value,
                'Impact': row['impact']
            })

        # Afficher le tableau avec accessibilité
        table_df = pd.DataFrame(table_data)
        st.dataframe(
            table_df,
            use_container_width=True,
            hide_index=True,
            column_config={
                'Variable': st.column_config.TextColumn('Variable', width='medium', help='Nom de la variable analysée'),
                'Valeur SHAP': st.column_config.TextColumn('Valeur SHAP', width='small', help='Impact calculé par le modèle'),
                'Valeur Client': st.column_config.TextColumn('Valeur Client', width='medium', help='Valeur saisie pour ce client'),
                'Impact': st.column_config.TextColumn('Impact', width='medium', help='Type d\'impact sur le risque')
            }
        )

    # Explication pédagogique avec couleurs + symboles
    st.markdown("""
    <div class="alert-info" role="region" aria-labelledby="reading-guide">
        <h5 id="reading-guide">💡 Guide de lecture du graphique</h5>
        <ul role="list">
            <li><span style="color: #22c55e;"><strong>🟢 Barres vertes (valeurs négatives vers la gauche)</strong></span> : Ces variables réduisent le risque de défaut pour ce client</li>
            <li><span style="color: #ff4444;"><strong>🔴 Barres rouges (valeurs positives vers la droite)</strong></span> : Ces variables augmentent le risque de défaut pour ce client</li>
            <li><span style="color: #94a3b8;"><strong>⚪ Barres grises (proche de zéro)</strong></span> : Ces variables ont un impact neutre ou très faible</li>
            <li><strong>Longueur des barres</strong> : Plus c'est long, plus l'impact est important sur la décision finale</li>
            <li><strong>Ligne verticale centrale</strong> : Sépare les impacts positifs (droite) des impacts négatifs (gauche)</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

def display_client_profile(client_data):
    """Afficher profil client avec accessibilité complète"""
    st.markdown('<h4 id="client-profile">👤 Profil Client Analysé</h4>', unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown('<h5 id="external-scores">Scores Externes</h5>', unsafe_allow_html=True)
        st.metric("Score Externe 2", f"{client_data.get('EXT_SOURCE_2', 0):.3f}", help="Score externe 2 : évaluation par organisme externe")
        st.metric("Score Externe 3", f"{client_data.get('EXT_SOURCE_3', 0):.3f}", help="Score externe 3 : évaluation par organisme externe")
        st.metric("Score Externe 1", f"{client_data.get('EXT_SOURCE_1', 0):.3f}", help="Score externe 1 : évaluation par organisme externe")
        st.metric("Retards moyens", f"{client_data.get('INSTAL_DPD_MEAN', 0):.1f} jours", help="Moyenne des jours de retard sur paiements antérieurs")

        # WCAG 1.1.1 : Description textuelle des métriques
        st.markdown("""
        <p><small><strong>Explication :</strong> Les scores externes sont des indicateurs fournis par des organismes spécialisés 
        (0=très risqué, 1=très sûr). Les retards moyens correspondent à l'historique de ponctualité des paiements précédents.</small></p>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown('<h5 id="employment-info">Emploi et Profil</h5>', unsafe_allow_html=True)
        
        employment_years = abs(client_data.get('DAYS_EMPLOYED', 0)) / 365.25
        st.metric("Ancienneté emploi", f"{employment_years:.2f} ans", help="Durée dans l'emploi actuel en années")

        gender = "Homme" if client_data.get('CODE_GENDER') == 'M' else "Femme"
        st.metric("Genre", gender, help="Genre déclaré : Homme ou Femme")

        payment_rate = client_data.get('PAYMENT_RATE', 0)
        st.metric("Ratio endettement", f"{payment_rate:.1%}", help="Ratio charges mensuelles sur revenus")

        # WCAG 1.1.1 : Description textuelle des métriques
        st.markdown("""
        <p><small><strong>Explication :</strong> L'ancienneté dans l'emploi indique la stabilité professionnelle. 
        Le ratio d'endettement compare les charges mensuelles aux revenus (plus bas = mieux).</small></p>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown('<h5 id="financial-info">Informations Financières</h5>', unsafe_allow_html=True)
        
        annuity = client_data.get('AMT_ANNUITY', 0)
        st.metric("Annuité mensuelle", f"{annuity:,.0f} €", help="Montant mensuel du crédit demandé")

        education = "Oui" if client_data.get('NAME_EDUCATION_TYPE_Higher_education', 0) == 1 else "Non"
        st.metric("Éducation supérieure", education, help="Niveau d'éducation : Oui (supérieure) ou Non")

        payment_sum = client_data.get('INSTAL_AMT_PAYMENT_SUM', 0)
        st.metric("Historique paiements", f"{payment_sum:,.0f} €", help="Cumul des paiements antérieurs réalisés")

        # WCAG 1.1.1 : Description textuelle des métriques
        st.markdown("""
        <p><small><strong>Explication :</strong> L'annuité est le montant mensuel à rembourser. 
        L'historique de paiements montre l'expérience passée du client avec les crédits.</small></p>
        """, unsafe_allow_html=True)

def create_simple_population_plot(distribution_data, client_value, variable_name):
    """Créer histogramme accessible : distribution population + ligne client"""

    values = distribution_data.get('values', [])

    if not values:
        st.error(f"Aucune donnée disponible pour {variable_name}")
        return

    # Conversion spéciale pour variables catégorielles
    if variable_name == 'CODE_GENDER':
        # Convertir M/F en 1/0 pour le graphique
        if client_value == 'M':
            client_value_numeric = 1
        else:
            client_value_numeric = 0
    elif variable_name == 'NAME_EDUCATION_TYPE_Higher_education':
        # Convertir les booléens True/False en 1/0 pour l'affichage
        values = [1 if v else 0 for v in values]
        client_value_numeric = client_value
    else:
        client_value_numeric = client_value

    # Histogramme simple
    fig = go.Figure()

    # Histogramme population
    fig.add_trace(go.Histogram(
        x=values,
        nbinsx=30 if variable_name not in ['CODE_GENDER', 'NAME_EDUCATION_TYPE_Higher_education'] else 10,
        opacity=0.7,
        marker_color='lightblue',
        name='Population générale',
        showlegend=False
    ))

    # Ligne verticale rouge pour le client (seulement si valeur numérique)
    try:
        fig.add_vline(
            x=client_value_numeric,
            line_dash="solid",
            line_color="red",
            line_width=4,
            annotation_text="📍 Position Client",
            annotation_position="top"
        )
    except (TypeError, ValueError):
        st.warning(f"Impossible d'afficher la position client pour {variable_name}")

    # Configuration du graphique avec layout accessible
    layout_config = {
        'title': f"Distribution : {FEATURE_TRANSLATIONS.get(variable_name, variable_name)}",
        'xaxis': {
            'title': f"{FEATURE_TRANSLATIONS.get(variable_name, variable_name)}"
        },
        'yaxis': {
            'title': "Nombre de clients"
        },
        'height': 400,
        'showlegend': False
    }

    # Labels spéciaux pour variables catégorielles
    if variable_name == 'CODE_GENDER':
        layout_config['xaxis'].update({
            'tickmode': 'array',
            'tickvals': [0, 1],
            'ticktext': ['Femme', 'Homme']
        })
    elif variable_name == 'NAME_EDUCATION_TYPE_Higher_education':
        layout_config['xaxis'].update({
            'tickmode': 'array',
            'tickvals': [0, 1],
            'ticktext': ['Non', 'Oui']
        })

    fig.update_layout(layout_config)

    st.plotly_chart(fig, use_container_width=True, config=PLOTLY_CONFIG)

    # WCAG 1.1.1 : Description textuelle complète pour histogramme population
    variable_fr = FEATURE_TRANSLATIONS.get(variable_name, variable_name)
    
    if variable_name in ['CODE_GENDER', 'NAME_EDUCATION_TYPE_Higher_education']:
        st.markdown(f"""
        <div role="img" aria-labelledby="histogram-{variable_name}-description">
            <h6 id="histogram-{variable_name}-description" class="sr-only">Description histogramme {variable_fr}</h6>
            <p><strong>📊 Description textuelle :</strong> Histogramme de répartition de la variable {variable_fr} dans la population générale.
            Graphique en barres montrant la distribution des clients selon cette caractéristique catégorielle.
            La position du client analysé est marquée par une ligne rouge verticale avec annotation "📍 Position Client".
            L'axe horizontal montre les deux modalités possibles, l'axe vertical le nombre de clients pour chaque modalité.</p>
        </div>
        """, unsafe_allow_html=True)
    else:
        client_val_formatted = f"{client_value_numeric:.2f}" if isinstance(client_value_numeric, (int, float)) else str(client_value_numeric)
        st.markdown(f"""
        <div role="img" aria-labelledby="histogram-{variable_name}-description">
            <h6 id="histogram-{variable_name}-description" class="sr-only">Description histogramme {variable_fr}</h6>
            <p><strong>📊 Description textuelle :</strong> Histogramme de distribution de la variable {variable_fr} dans la population générale.
            L'axe horizontal représente les valeurs de {variable_fr}, l'axe vertical le nombre de clients ayant chaque valeur.
            Le client analysé (valeur: {client_val_formatted}) est positionné par une ligne rouge verticale marquée "📍 Position Client".
            Cet histogramme permet de comparer le client à l'ensemble de la population pour cette variable.</p>
        </div>
        """, unsafe_allow_html=True)

def display_simple_population_comparison(client_data):
    """Interface comparaison population avec accessibilité complète"""

    st.markdown('<section aria-labelledby="population-comparison-section">', unsafe_allow_html=True)
    st.markdown('<h5 id="population-comparison-section">Comparaison avec la Population Générale</h5>', unsafe_allow_html=True)

    # Layout avec bouton accessible
    col1, col2 = st.columns([3, 1])

    with col1:
        selected_variable = st.selectbox(
            "Variable à analyser :",
            DASHBOARD_FEATURES,
            format_func=lambda x: FEATURE_TRANSLATIONS.get(x, x),
            key="population_variable_select",
            help="Sélectionnez une variable pour voir comment ce client se positionne par rapport à la population générale"
        )

    with col2:
        # Bouton avec accessibilité renforcée
        if st.button(
            "📊 Charger données", 
            help="Charger et afficher les données de comparaison pour cette variable", 
            key="load_population_btn",
            use_container_width=True
        ):
            
            with st.spinner("🔄 Chargement des données de la population..."):
                # APPEL API UNIQUEMENT ICI
                distribution_data = get_population_distribution(selected_variable)
            
            if distribution_data:
                client_value = client_data.get(selected_variable)
                
                if client_value is not None:
                    # Stocker dans session state pour éviter re-appel
                    st.session_state[f'population_data_{selected_variable}'] = distribution_data
                    st.success(f"✅ Données chargées pour {FEATURE_TRANSLATIONS.get(selected_variable, selected_variable)}")
                    
                    # Afficher le graphique
                    create_simple_population_plot(distribution_data, client_value, selected_variable)
                else:
                    st.error(f"Valeur client manquante pour {selected_variable}")
            else:
                st.error(f"Impossible de charger les données pour {selected_variable}")
    
    # Afficher données en cache si disponibles
    cache_key = f'population_data_{selected_variable}'
    if cache_key in st.session_state:
        st.info("📋 Données en cache - Cliquez sur 'Charger données' pour actualiser")
        client_value = client_data.get(selected_variable)
        if client_value is not None:
            create_simple_population_plot(st.session_state[cache_key], client_value, selected_variable)
    
    st.markdown('</section>', unsafe_allow_html=True)

# WCAG 2.4.2 : Structure hiérarchique complète avec landmarks

# Skip link pour navigation clavier
st.markdown("""
<a href="#main-content" class="skip-link">Aller au contenu principal</a>
""", unsafe_allow_html=True)

# En-tête principal avec landmark
st.markdown("""
<header role="banner">
    <h1 id="main-title">🏦 Dashboard Credit Scoring - Prêt à dépenser</h1>
</header>
""", unsafe_allow_html=True)

# Vérification API avec gestion d'erreur accessible
api_ok, api_info, api_error = test_api_connection()

if not api_ok:
    st.markdown(f"""
    <div class="error-card" role="alert" aria-live="assertive">
        <h2>⚠️ API non accessible</h2>
        <p><strong>Erreur :</strong> {api_error}</p>
        <p>Veuillez vérifier votre connexion internet et réessayer.</p>
    </div>
    """, unsafe_allow_html=True)
    st.stop()

# Navigation sidebar avec landmark
with st.sidebar:
    st.markdown("""
    <nav role="navigation" aria-label="Navigation principale du dashboard">
        <h2 class="sr-only">Menu principal</h2>
    </nav>
    """, unsafe_allow_html=True)
    
    st.markdown("**🏦 Dashboard Credit Scoring**")
    st.markdown("---")

    st.markdown("### 📋 Navigation")

    # Bouton nouveau client avec accessibilité
    if st.button(
        "🆕 Nouveau client", 
        use_container_width=True, 
        help="Réinitialiser le dashboard pour analyser un nouveau dossier client"
    ):
        # Reset complet de l'état + cache
        st.session_state.client_analyzed = False
        st.session_state.client_data = None
        st.session_state.prediction_result = None
        st.session_state.api_call_in_progress = False
        st.session_state.population_cache = {}
        st.session_state.bivariate_cache = {}
        
        # Nettoyer aussi les clés de cache dynamiques
        keys_to_remove = [key for key in st.session_state.keys() if 
                         key.startswith('population_data_') or key.startswith('bivariate_')]
        for key in keys_to_remove:
            del st.session_state[key]
            
        # Notification accessible
        st.markdown("""
        <div aria-live="polite" aria-atomic="true">
            <span class="sr-only">Dashboard réinitialisé pour nouveau client</span>
        </div>
        """, unsafe_allow_html=True)
        st.rerun()

    st.markdown("---")
    st.markdown("**📊 Statut API**")
    if api_info:
        st.success("✅ Connectée")
        st.caption(f"Version: {api_info.get('version', 'N/A')}")
    else:
        st.error("❌ Déconnectée")

# Contenu principal avec landmark
st.markdown('<main role="main" id="main-content" aria-labelledby="main-title">', unsafe_allow_html=True)

# Interface principale - Appel API uniquement sur bouton
if not st.session_state.client_analyzed:
    
    st.markdown('<h2 id="client-form-heading">📝 Saisie des Données Client</h2>', unsafe_allow_html=True)

    # Formulaire de saisie avec accessibilité
    client_data = create_client_form()

    # Bouton d'analyse avec accessibilité complète
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.button(
            "🎯 ANALYSER CE CLIENT",
            type="primary",
            use_container_width=True,
            disabled=st.session_state.api_call_in_progress,
            key="analyze_client_btn",
            help="Lance l'analyse de crédit basée sur les informations saisies. Traitement par IA en temps réel."
        ):
            # Marquer appel en cours
            st.session_state.api_call_in_progress = True
            
            # Notification de démarrage
            st.markdown("""
            <div aria-live="polite" aria-atomic="true">
                <span class="sr-only">Analyse en cours, veuillez patienter</span>
            </div>
            """, unsafe_allow_html=True)
            
            # Appel API direct
            with st.spinner("🔄 Analyse en cours par intelligence artificielle..."):
                result, error = call_prediction_api(client_data)
            
            # Traitement résultat
            if result:
                # Mise à jour complète de l'état
                st.session_state.client_data = client_data
                st.session_state.prediction_result = result
                st.session_state.client_analyzed = True
                st.session_state.last_analysis_time = time.time()
                st.session_state.api_call_in_progress = False
                
                # Notification de succès accessible
                st.markdown("""
                <div aria-live="polite" aria-atomic="true" class="alert-success">
                    <p><strong>✅ Analyse terminée avec succès !</strong> Résultats disponibles ci-dessous.</p>
                </div>
                """, unsafe_allow_html=True)
                st.rerun()
            else:
                # Reset en cas d'erreur
                st.session_state.api_call_in_progress = False
                st.markdown(f"""
                <div aria-live="assertive" aria-atomic="true" class="error-card">
                    <p><strong>❌ Erreur d'analyse :</strong> {error}</p>
                    <p>Veuillez vérifier les données saisies et réessayer.</p>
                </div>
                """, unsafe_allow_html=True)

else:
    # Section résultats avec structure accessible
    st.markdown('<h2 id="analysis-results-heading">🎯 Analyse du Dossier Client</h2>', unsafe_allow_html=True)
    
    # Onglets avec accessibilité renforcée
    tab1, tab2, tab3 = st.tabs([
        "🎯 Résultats", 
        "📊 Comparaisons", 
        "🔧 Analyses bi-variées"
    ])
    
    # Script pour améliorer l'accessibilité des onglets
    st.markdown("""
    <script>
    document.addEventListener('DOMContentLoaded', function() {
        const tabs = document.querySelectorAll('[data-testid="stTabs"] button');
        if (tabs.length >= 3) {
            tabs[0]?.setAttribute('aria-label', 'Onglet résultats : décision de crédit et interprétation');
            tabs[1]?.setAttribute('aria-label', 'Onglet comparaisons : position client vs population');
            tabs[2]?.setAttribute('aria-label', 'Onglet analyses : relations entre variables');
        }
    });
    </script>
    """, unsafe_allow_html=True)

    with tab1:
        st.markdown("""
        <section aria-labelledby="results-tab-heading">
            <h3 id="results-tab-heading">📊 Résultats de l'Analyse de Crédit</h3>
        </section>
        """, unsafe_allow_html=True)

        # Bouton pour modifier avec accessibilité
        col1, col2 = st.columns([3, 1])
        with col2:
            if st.button(
                "🔧 Modifier", 
                use_container_width=True,
                help="Retourner au formulaire pour modifier les données client"
            ):
                # Reset pour retour au formulaire
                st.session_state.client_analyzed = False
                st.session_state.api_call_in_progress = False
                
                # Notification accessible
                st.markdown("""
                <div aria-live="polite">
                    <span class="sr-only">Retour au formulaire de saisie</span>
                </div>
                """, unsafe_allow_html=True)
                st.rerun()

        # Profil client
        display_client_profile(st.session_state.client_data)

        st.markdown("---")

        # Décision de crédit
        st.markdown('<h4 id="credit-decision-heading">🎯 Décision de Crédit</h4>', unsafe_allow_html=True)
        display_prediction_result(st.session_state.prediction_result)

        st.markdown("---")

        # Feature importance avec graphique + tableau détaillé
        display_feature_importance(st.session_state.prediction_result)

    with tab2:
        st.markdown("""
        <section aria-labelledby="comparison-tab-heading">
            <h3 id="comparison-tab-heading">📊 Comparaisons avec la Population</h3>
        </section>
        """, unsafe_allow_html=True)

        # Interface comparaison population
        display_simple_population_comparison(st.session_state.client_data)

    with tab3:
        st.markdown("""
        <section aria-labelledby="bivariate-tab-heading">
            <h3 id="bivariate-tab-heading">🔧 Analyses Bi-variées</h3>
        </section>
        """, unsafe_allow_html=True)

        col1, col2 = st.columns(2)

        with col1:
            var1 = st.selectbox(
                "Variable 1",
                DASHBOARD_FEATURES,
                format_func=lambda x: FEATURE_TRANSLATIONS.get(x, x),
                key="bivariate_var1",
                help="Première variable pour l'analyse de corrélation"
            )

        with col2:
            var2 = st.selectbox(
                "Variable 2",
                DASHBOARD_FEATURES,
                index=1,
                format_func=lambda x: FEATURE_TRANSLATIONS.get(x, x),
                key="bivariate_var2",
                help="Seconde variable pour l'analyse de corrélation"
            )

        # Bouton avec appel API contrôlé et accessibilité
        if st.button(
            "📈 Analyser Relation", 
            use_container_width=True, 
            key="analyze_bivariate_btn",
            help="Analyser la relation statistique entre les deux variables sélectionnées"
        ):
            
            with st.spinner("🔄 Analyse bi-variée en cours..."):
                # Appels API uniquement ici
                dist1 = get_population_distribution(var1)
                dist2 = get_population_distribution(var2)

            if dist1 and dist2:
                values1 = dist1.get('values', [])
                values2 = dist2.get('values', [])

                if values1 and values2:
                    # Conversion spéciale pour variables catégorielles
                    if var1 == 'NAME_EDUCATION_TYPE_Higher_education':
                        values1 = [1 if v else 0 for v in values1]
                    if var2 == 'NAME_EDUCATION_TYPE_Higher_education':
                        values2 = [1 if v else 0 for v in values2]

                    # Assurer même longueur (prendre le minimum)
                    min_len = min(len(values1), len(values2))
                    x_data = values1[:min_len]
                    y_data = values2[:min_len]

                    # Stocker en cache pour éviter re-appel
                    cache_key = f'bivariate_{var1}_{var2}'
                    st.session_state[cache_key] = {
                        'x_data': x_data,
                        'y_data': y_data,
                        'var1': var1,
                        'var2': var2
                    }

                    # Graphique de corrélation avec accessibilité complète
                    fig = px.scatter(
                        x=x_data,
                        y=y_data,
                        title=f"Relation entre {FEATURE_TRANSLATIONS.get(var1, var1)} et {FEATURE_TRANSLATIONS.get(var2, var2)}",
                        labels={
                            'x': FEATURE_TRANSLATIONS.get(var1, var1),
                            'y': FEATURE_TRANSLATIONS.get(var2, var2)
                        },
                        opacity=0.6,
                        color_discrete_sequence=['lightblue']
                    )

                    # Ajouter les lignes de croisement pour la position du client
                    client_x = st.session_state.client_data.get(var1, 0)
                    client_y = st.session_state.client_data.get(var2, 0)
                    
                    # Conversion spéciale pour variables catégorielles du client
                    if var1 == 'NAME_EDUCATION_TYPE_Higher_education':
                        client_x = 1 if client_x == 1 else 0
                    if var2 == 'NAME_EDUCATION_TYPE_Higher_education':
                        client_y = 1 if client_y == 1 else 0
                    if var1 == 'CODE_GENDER':
                        client_x = 1 if client_x == 'M' else 0
                    if var2 == 'CODE_GENDER':
                        client_y = 1 if client_y == 'M' else 0
                    
                    # Ajouter ligne verticale (position X du client)
                    fig.add_vline(
                        x=client_x,
                        line_dash="dash",
                        line_color="red",
                        line_width=3,
                        annotation_text=f"📍 Client: {FEATURE_TRANSLATIONS.get(var1, var1)}",
                        annotation_position="top"
                    )
                    
                    # Ajouter ligne horizontale (position Y du client)
                    fig.add_hline(
                        y=client_y,
                        line_dash="dash",
                        line_color="red",
                        line_width=3,
                        annotation_text=f"📍 Client: {FEATURE_TRANSLATIONS.get(var2, var2)}",
                        annotation_position="right"
                    )

                    fig.update_layout(
                        height=500,
                        showlegend=False
                    )
                    st.plotly_chart(fig, use_container_width=True, config=PLOTLY_CONFIG)

                    # WCAG 1.1.1 : Description textuelle complète pour analyse bi-variée
                    correlation = np.corrcoef(x_data, y_data)[0, 1] if len(x_data) > 1 else 0
                    var1_fr = FEATURE_TRANSLATIONS.get(var1, var1)
                    var2_fr = FEATURE_TRANSLATIONS.get(var2, var2)

                    st.markdown(f"""
                    <div role="img" aria-labelledby="bivariate-{var1}-{var2}-description">
                        <h5 id="bivariate-{var1}-{var2}-description" class="sr-only">Description analyse bi-variée {var1_fr} vs {var2_fr}</h5>
                        <p><strong>📊 Description textuelle :</strong> Nuage de points montrant la relation statistique entre {var1_fr} (axe horizontal) et {var2_fr} (axe vertical).
                        Chaque point bleu clair représente un client de la population générale. 
                        Les lignes rouges en pointillés indiquent la position précise du client analysé : 
                        ligne verticale rouge à {var1_fr} = {client_x}, ligne horizontale rouge à {var2_fr} = {client_y}.
                        Le croisement des deux lignes rouges localise exactement le client dans la distribution bi-variée.
                        Corrélation statistique générale : {correlation:.3f}.
                        {'Relation positive modérée à forte' if correlation > 0.3 else 'Relation négative modérée à forte' if correlation < -0.3 else 'Relation faible ou absence de corrélation'} entre les deux variables analysées.</p>
                    </div>
                    """, unsafe_allow_html=True)

                    # Analyse positionnement client avec accessibilité
                    percentile_x = sum(1 for val in x_data if val <= client_x) / len(x_data) * 100
                    percentile_y = sum(1 for val in y_data if val <= client_y) / len(y_data) * 100
                    
                    st.markdown(f"""
                    <div class="alert-info" role="region" aria-labelledby="client-position-analysis">
                        <h5 id="client-position-analysis">📍 Position du client dans la population bi-variée</h5>
                        <ul role="list">
                            <li><strong>{var1_fr}</strong> : {percentile_x:.0f}e percentile (ligne verticale rouge) - 
                                {'Valeur élevée par rapport à la population' if percentile_x > 75 else 'Valeur moyenne' if percentile_x > 25 else 'Valeur faible par rapport à la population'}</li>
                            <li><strong>{var2_fr}</strong> : {percentile_y:.0f}e percentile (ligne horizontale rouge) - 
                                {'Valeur élevée par rapport à la population' if percentile_y > 75 else 'Valeur moyenne' if percentile_y > 25 else 'Valeur faible par rapport à la population'}</li>
                            <li><strong>Position croisée</strong> : L'intersection des deux lignes rouges indique la position exacte du client dans l'espace bi-dimensionnel</li>
                        </ul>
                    </div>
                    """, unsafe_allow_html=True)

                    # Notification de succès accessible
                    st.markdown(f"""
                    <div aria-live="polite" class="alert-success">
                        <p><strong>✅ Analyse bi-variée terminée</strong> - Corrélation: {correlation:.3f} 
                        ({len(x_data):,} points analysés)</p>
                    </div>
                    """, unsafe_allow_html=True)

                else:
                    st.error("Données insuffisantes pour une des variables")
            else:
                st.error("Impossible de charger les données pour l'analyse bi-variée")
                missing_vars = []
                if not dist1:
                    missing_vars.append(var1)
                if not dist2:
                    missing_vars.append(var2)
                st.warning(f"Variables indisponibles : {missing_vars}")
        
        # Afficher analyse en cache si disponible
        cache_key = f'bivariate_{var1}_{var2}'
        if cache_key in st.session_state:
            cached_data = st.session_state[cache_key]
            if cached_data['var1'] == var1 and cached_data['var2'] == var2:
                st.info("📋 Analyse en cache - Cliquez sur 'Analyser Relation' pour actualiser")
                
                # Re-afficher le graphique depuis le cache
                fig = px.scatter(
                    x=cached_data['x_data'],
                    y=cached_data['y_data'],
                    title=f"Relation entre {FEATURE_TRANSLATIONS.get(var1, var1)} et {FEATURE_TRANSLATIONS.get(var2, var2)} (Cache)",
                    labels={
                        'x': FEATURE_TRANSLATIONS.get(var1, var1),
                        'y': FEATURE_TRANSLATIONS.get(var2, var2)
                    },
                    opacity=0.6,
                    color_discrete_sequence=['lightblue']
                )
                
                # Ajouter les lignes de croisement du client aussi depuis le cache
                client_x = st.session_state.client_data.get(var1, 0)
                client_y = st.session_state.client_data.get(var2, 0)
                
                # Conversion pour variables catégorielles
                if var1 == 'NAME_EDUCATION_TYPE_Higher_education':
                    client_x = 1 if client_x == 1 else 0
                if var2 == 'NAME_EDUCATION_TYPE_Higher_education':
                    client_y = 1 if client_y == 1 else 0
                if var1 == 'CODE_GENDER':
                    client_x = 1 if client_x == 'M' else 0
                if var2 == 'CODE_GENDER':
                    client_y = 1 if client_y == 'M' else 0
                
                # Ligne verticale
                fig.add_vline(
                    x=client_x,
                    line_dash="dash",
                    line_color="red",
                    line_width=3,
                    annotation_text=f"📍 Client: {FEATURE_TRANSLATIONS.get(var1, var1)}",
                    annotation_position="top"
                )
                
                # Ligne horizontale
                fig.add_hline(
                    y=client_y,
                    line_dash="dash",
                    line_color="red",
                    line_width=3,
                    annotation_text=f"📍 Client: {FEATURE_TRANSLATIONS.get(var2, var2)}",
                    annotation_position="right"
                )
                
                fig.update_layout(
                    height=500,
                    showlegend=False
                )
                st.plotly_chart(fig, use_container_width=True, config=PLOTLY_CONFIG)

# Notification dynamique pour les changements d'état
if st.session_state.get('client_analyzed'):
    decision_fr = st.session_state.prediction_result.get('prediction', {}).get('decision_fr', '')
    st.markdown(f"""
    <div aria-live="polite" aria-atomic="true">
        <span class="sr-only">État actuel : {decision_fr}</span>
    </div>
    """, unsafe_allow_html=True)

st.markdown('</main>', unsafe_allow_html=True)

# Footer accessible avec landmark
st.markdown("---")
st.markdown("""
<footer role="contentinfo">
    <div style="text-align: center; margin-top: 2rem; padding: 1rem; border-top: 3px solid #e5e7eb;">
""", unsafe_allow_html=True)

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("**🏦 Prêt à dépenser**")
    st.markdown("Dashboard Credit Scoring")
    st.markdown("Brice Béchet")
    st.markdown("Juin 2025 - Master 2 Data Scientist - OpenClassRoom")

with col2:
    st.markdown("**✅ Fonctionnalités**")
    st.markdown("• Analyse de crédit instantanée par IA")
    st.markdown("• Explications transparentes avec SHAP")
    st.markdown("• Comparaisons population en temps réel")

with col3:
    st.markdown("**♿ Accessibilité WCAG 2.1 AA**")
    st.markdown("• Navigation clavier complète")
    st.markdown("• Descriptions textuelles pour graphiques")
    st.markdown("• Couleurs avec alternatives visuelles")
    st.markdown("• Contrastes conformes 4.5:1")

st.markdown("""
    </div>
</footer>
""", unsafe_allow_html=True)