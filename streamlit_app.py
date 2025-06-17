"""
Dashboard Credit Scoring Production - Streamlit Cloud
Version: Production v2.1 - CORRIGÉ ANTI-BOUCLE
Plateforme: Streamlit Cloud + Railway API v5.0
Fonctionnalités: Interface chargés relation client + graphique simple population
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
    'toImageButtonOptions': {
        'format': 'png',
        'filename': 'graphique_credit_scoring',
        'height': 500,
        'width': 700,
        'scale': 1
    }
}

# CSS WCAG pour production
st.markdown("""
<style>
/* Styles WCAG conformes */
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

.main-header:hover {
    transform: translateY(-2px);
    box-shadow: 0 12px 35px rgba(0, 0, 0, 0.2);
}

/* BOUTONS UNIFORMISÉS */
.stButton > button {
    background: linear-gradient(135deg, #3b82f6 0%, #1e40af 100%) !important;
    color: white !important;
    border: none !important;
    border-radius: 12px !important;
    padding: 0.75rem 1.5rem !important;
    font-weight: 600 !important;
    font-size: 1rem !important;
    box-shadow: 0 4px 15px rgba(59, 130, 246, 0.3) !important;
    transition: all 0.3s ease !important;
    text-transform: none !important;
    letter-spacing: 0.025em !important;
    min-height: 3rem !important;
    width: 100% !important;
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

/* BOUTON PRIMAIRE SPÉCIAL */
.stButton > button[kind="primary"] {
    background: linear-gradient(135deg, #059669 0%, #047857 100%) !important;
    box-shadow: 0 4px 15px rgba(5, 150, 105, 0.3) !important;
}

.stButton > button[kind="primary"]:hover {
    background: linear-gradient(135deg, #047857 0%, #065f46 100%) !important;
    box-shadow: 0 8px 25px rgba(5, 150, 105, 0.4) !important;
}

.metric-card {
    background: linear-gradient(135deg, #f8fafc 0%, #e2e8f0 100%);
    border: 2px solid #e2e8f0;
    padding: 1.5rem;
    border-radius: 1rem;
    margin: 1rem 0;
    font-size: 1.1rem;
    box-shadow: 0 4px 15px rgba(0, 0, 0, 0.1);
    transition: all 0.3s ease;
}

.metric-card:hover {
    transform: translateY(-2px);
    box-shadow: 0 8px 25px rgba(0, 0, 0, 0.15);
}

.success-card {
    background: linear-gradient(135deg, #dcfce7 0%, #bbf7d0 100%);
    border: 3px solid #16a34a;
    color: #15803d;
    box-shadow: 0 4px 20px rgba(22, 163, 74, 0.2);
}

.warning-card {
    background: linear-gradient(135deg, #fef3c7 0%, #fde68a 100%);
    border: 3px solid #d97706;
    color: #92400e;
    box-shadow: 0 4px 20px rgba(217, 119, 6, 0.2);
}

.error-card {
    background: linear-gradient(135deg, #fee2e2 0%, #fecaca 100%);
    border: 3px solid #dc2626;
    color: #991b1b;
    box-shadow: 0 4px 20px rgba(220, 38, 38, 0.2);
}

/* WCAG Accessibilité */
.approved::before { content: "✅ "; font-weight: bold; }
.refused::before { content: "❌ "; font-weight: bold; }

.alert-info {
    background: linear-gradient(135deg, #eff6ff 0%, #dbeafe 100%);
    border: 2px solid #3b82f6;
    color: #1d4ed8;
    padding: 1rem;
    border-radius: 0.75rem;
    margin: 1rem 0;
    font-weight: 500;
    box-shadow: 0 4px 15px rgba(59, 130, 246, 0.1);
}

.alert-success {
    background: linear-gradient(135deg, #dcfce7 0%, #bbf7d0 100%);
    border: 2px solid #16a34a;
    color: #15803d;
    padding: 1rem;
    border-radius: 0.75rem;
    margin: 1rem 0;
    font-weight: 500;
    box-shadow: 0 4px 15px rgba(22, 163, 74, 0.1);
}

/* Responsive design */
@media (max-width: 768px) {
    .main-header {
        font-size: 1.5rem;
        padding: 1rem;
    }
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
    "EXT_SOURCE_2": "Un score externe 2 élevé diminue le risque de défaut",
    "EXT_SOURCE_3": "Un score externe 3 élevé diminue le risque de défaut", 
    "EXT_SOURCE_1": "Un score externe 1 élevé diminue le risque de défaut",
    "DAYS_EMPLOYED": "Une ancienneté dans l'emploi actuel élevée diminue le risque de défaut",
    "PAYMENT_RATE": "Un ratio d'endettement bas diminue le risque de défaut",
    "CODE_GENDER": "Un client homme augmente légèrement le risque de défaut par rapport à une femme",
    "INSTAL_DPD_MEAN": "Des retards moyens élevés sur paiements antérieurs augmentent le risque de défaut",
    "NAME_EDUCATION_TYPE_Higher_education": "Une éducation supérieure augmente légèrement le risque de défaut",
    "AMT_ANNUITY": "Une annuité mensuelle élevée augmente le risque de défaut",
    "INSTAL_AMT_PAYMENT_SUM": "Un historique de paiements important diminue le risque de défaut"
}

# Les 10 variables dashboard
DASHBOARD_FEATURES = [
    'EXT_SOURCE_2', 'EXT_SOURCE_3', 'EXT_SOURCE_1',
    'DAYS_EMPLOYED', 'CODE_GENDER', 'INSTAL_DPD_MEAN',
    'PAYMENT_RATE', 'NAME_EDUCATION_TYPE_Higher_education',
    'AMT_ANNUITY', 'INSTAL_AMT_PAYMENT_SUM'
]

# INITIALISATION SESSION STATE SÉCURISÉE
def init_session_state():
    """Initialiser session state une seule fois"""
    defaults = {
        'client_analyzed': False,
        'client_data': None,
        'prediction_result': None,
        'refresh_comparison': False,
        'form_submitted': False,
        'last_analysis_time': None
    }
    
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

# Appeler une seule fois
init_session_state()

# Fonctions API SANS EFFETS DE BORD

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
    """Appel API de prédiction"""
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

@st.cache_data(ttl=600)
def get_population_distribution(variable):
    """Récupérer distribution d'une variable spécifique"""
    try:
        response = requests.get(f"{API_URL}/population/{variable}", timeout=15)
        if response.status_code == 200:
            return response.json()
        return None
    except Exception as e:
        return None

@st.cache_data(ttl=600)
def get_population_data():
    """Récupérer données population avec cache (pour bi-variée)"""
    try:
        response = requests.get(f"{API_URL}/population_stats", timeout=15)
        if response.status_code == 200:
            return response.json()
        return None
    except Exception as e:
        return None

def get_bivariate_data(var1, var2):
    """Analyse bi-variée"""
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
    """Formulaire de saisie client SANS rerun automatique"""
    
    with st.expander("ℹ️ Guide d'utilisation", expanded=False):
        st.markdown("""
        ### 🚀 **Prêt à commencer ?**
        1. **Saisissez** les informations client dans le formulaire ci-dessous
        2. **Analysez** le dossier en cliquant sur "Analyser ce client"  
        3. **Explorez** les onglets Résultats, Comparaisons et Analyses
        4. **Simulez** différents scénarios si nécessaire        
        """)
    
    # Valeurs par défaut (ou valeurs précédentes si modification)
    default_values = st.session_state.client_data if st.session_state.client_data else {}
    
    col1, col2 = st.columns(2)
    
    with col1:
        
        ext_source_2 = st.slider(
            "Score Externe 2", 
            0.0, 1.0, 
            float(default_values.get('EXT_SOURCE_2', 0.6)), 
            0.01,
            help=FEATURE_EXPLANATIONS["EXT_SOURCE_2"]
        )
        
        ext_source_3 = st.slider(
            "Score Externe 3", 
            0.0, 1.0, 
            float(default_values.get('EXT_SOURCE_3', 0.5)), 
            0.01,
            help=FEATURE_EXPLANATIONS["EXT_SOURCE_3"]
        )
        
        ext_source_1 = st.slider(
            "Score Externe 1", 
            0.0, 1.0, 
            float(default_values.get('EXT_SOURCE_1', 0.4)), 
            0.01,
            help=FEATURE_EXPLANATIONS["EXT_SOURCE_1"]
        )
        
        # Conversion jours en années pour l'affichage
        default_employment = abs(default_values.get('DAYS_EMPLOYED', -1825)) / 365.25
        employment_years = st.number_input(
            "Ancienneté emploi (années)", 
            0.0, 40.0, 
            float(default_employment), 
            0.01,
            help=FEATURE_EXPLANATIONS["DAYS_EMPLOYED"]
        )
        
        instal_dpd_mean = st.slider(
            "Retards moyens (jours)", 
            0.0, 30.0, 
            float(default_values.get('INSTAL_DPD_MEAN', 0.5)), 
            0.1,
            help=FEATURE_EXPLANATIONS["INSTAL_DPD_MEAN"]
        )
        
    with col2:
        st.markdown("**💼 Informations Complémentaires**")
        
        # Conversion M/F pour l'affichage
        default_gender = "Homme" if default_values.get('CODE_GENDER') == 'M' else "Femme"
        gender = st.selectbox(
            "Genre", 
            ["Femme", "Homme"],
            index=0 if default_gender == "Femme" else 1
        )
        
        payment_rate = st.slider(
            "Ratio d'endettement", 
            0.0, 1.0, 
            float(default_values.get('PAYMENT_RATE', 0.15)), 
            0.01,
            help=FEATURE_EXPLANATIONS["PAYMENT_RATE"]
        )
        
        # Conversion 0/1 pour l'affichage
        default_education = "Oui" if default_values.get('NAME_EDUCATION_TYPE_Higher_education', 0) == 1 else "Non"
        education = st.selectbox(
            "Éducation supérieure", 
            ["Non", "Oui"],
            index=0 if default_education == "Non" else 1
        )
        
        annuity = st.number_input(
            "Annuité mensuelle (€)", 
            5000, 100000, 
            int(default_values.get('AMT_ANNUITY', 18000)), 
            1000,
            help=FEATURE_EXPLANATIONS["AMT_ANNUITY"]
        )
        
        payment_sum = st.number_input(
            "Historique paiements (€)", 
            10000, 1000000, 
            int(default_values.get('INSTAL_AMT_PAYMENT_SUM', 120000)), 
            10000,
            help="Somme des paiements antérieurs"
        )
    
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

# Affichage des résultats

def display_prediction_result(result):
    """Afficher résultat de prédiction avec jauge modernisée"""
    prediction = result.get('prediction', {})
    probability = prediction.get('probability', 0)
    decision = prediction.get('decision', 'UNKNOWN')
    decision_fr = prediction.get('decision_fr', decision)
    risk_level = prediction.get('risk_level', 'Inconnu')
    
    # Résultat principal
    if decision == "REFUSE":
        st.markdown(f"""
        <div class="metric-card error-card refused">
            <h2>CRÉDIT REFUSÉ : Probabilité de défaut: {probability:.1%} - Niveau de risque: {risk_level}</h2>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="metric-card success-card approved">
            <h2>CRÉDIT ACCORDÉ : Probabilité de défaut: {probability:.1%} - Niveau de risque: {risk_level}</h2>
        </div>
        """, unsafe_allow_html=True)
    
    # JAUGE MODERNISÉE
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
                {'range': [0, 10], 'color': '#dcfce7', 'name': 'Faible'},
                {'range': [10, 25], 'color': '#fef3c7', 'name': 'Modéré'},
                {'range': [25, 50], 'color': '#fed7aa', 'name': 'Élevé'},
                {'range': [50, 100], 'color': '#fee2e2', 'name': 'Très élevé'}
            ],
            'threshold': {
                'line': {'color': "#dc2626", 'width': 6},
                'thickness': 0.9,
                'value': 10
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
    
    # WCAG 1.1.1 : Texte alternatif pour la jauge
    st.markdown(f"""
    **Description graphique :** Jauge de risque affichant {probability:.1%} de probabilité de défaut de paiement. 
    Le seuil de décision est fixé à 10%. Ce client se situe dans la zone {'rouge (risque élevé)' if probability >= 0.1 else 'verte (risque faible)'}.
    """)

def display_feature_importance(result):
    """Afficher importance des variables avec graphique et tableau détaillé"""
    explanation = result.get('explanation', {})
    top_features = explanation.get('top_features', [])
    client_data = st.session_state.client_data
    
    if not top_features:
        st.warning("Explications des variables non disponibles")
        return
    
    st.markdown("#### 🔍 Interprétation de la décision")
    
    # Créer données complètes pour toutes les variables
    all_features_data = []
    
    # Variables avec impact SHAP (top 5)
    for feature in top_features:
        feature_name = feature.get('feature', '')
        shap_value = feature.get('shap_value', 0)
        client_value = client_data.get(feature_name, 0)
        
        # Déterminer l'impact
        if abs(shap_value) < 0.001:
            impact = "Impact neutre"
        elif shap_value > 0:
            impact = "Augmente le risque"
        else:
            impact = "Diminue le risque"
        
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
                'impact': "Impact neutre"
            })
    
    # Créer DataFrame pour le graphique
    features_df = pd.DataFrame(all_features_data)
    
    # Trier par valeur SHAP absolue (décroissante)
    features_df['abs_shap'] = features_df['shap_value'].abs()
    features_df = features_df.sort_values('abs_shap', ascending=True)
    
    # Couleurs selon impact
    features_df['color'] = features_df['shap_value'].apply(
        lambda x: "Augmente le risque" if x > 0 else ("Diminue le risque" if x < 0 else "Impact neutre")
    )
    
    # Graphique horizontal
    fig = px.bar(
        features_df,
        x='shap_value',
        y='feature_fr',
        orientation='h',
        color='color',
        color_discrete_map={
            "Augmente le risque": "#ff4444",
            "Diminue le risque": "#22c55e",
            "Impact neutre": "#94a3b8"
        },
        title="Impact des variables sur la décision"
    )
    
    fig.update_layout(
        height=500,
        showlegend=True,
        font={'size': 12},
        xaxis_title="Impact sur la prédiction",
        yaxis_title="Variables"
    )
    
    fig.add_vline(x=0, line_dash="dash", line_color="gray", line_width=2)
    
    st.plotly_chart(fig, use_container_width=True, config=PLOTLY_CONFIG)
    
    # WCAG 1.1.1 : Texte alternatif pour graphique feature importance
    positive_features = [f['feature_fr'] for f in all_features_data if f['shap_value'] > 0]
    negative_features = [f['feature_fr'] for f in all_features_data if f['shap_value'] < 0]
    
    st.markdown(f"""
    **Description graphique :** Graphique en barres horizontales montrant l'impact de chaque variable sur la décision. 
    Variables augmentant le risque (barres rouges) : {', '.join(positive_features[:3]) if positive_features else 'Aucune'}. 
    Variables diminuant le risque (barres vertes) : {', '.join(negative_features[:3]) if negative_features else 'Aucune'}.
    """)
    
    # Tableau détaillé
    with st.expander("📋 Tableau détaillé", expanded=True):
        
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
        
        # Afficher le tableau
        table_df = pd.DataFrame(table_data)
        st.dataframe(
            table_df,
            use_container_width=True,
            hide_index=True,
            column_config={
                'Variable': st.column_config.TextColumn('Variable', width='medium'),
                'Valeur SHAP': st.column_config.TextColumn('Valeur SHAP', width='small'),
                'Valeur Client': st.column_config.TextColumn('Valeur Client', width='medium'), 
                'Impact': st.column_config.TextColumn('Impact', width='medium')
            }
        )
    
    # Explication pédagogique
    st.markdown("""
    <div class="alert-info">
        <strong>💡 Lecture du graphique des variables :</strong><br>
        • <span style="color: #22c55e;"><strong>Barres vertes (valeurs négatives)</strong></span> : Ces variables réduisent le risque de défaut<br>
        • <span style="color: #ff4444;"><strong>Barres rouges (valeurs positives)</strong></span> : Ces variables augmentent le risque de défaut<br>
        • <span style="color: #94a3b8;"><strong>Barres grises (proche de zéro)</strong></span> : Ces variables ont un impact neutre ou très faible<br>
        • <strong>Longueur des barres</strong> : Plus c'est long, plus l'impact est important<br>
        • <strong>Toutes ces variables peuvent être ajustées dans l'onglet "Simulations"</strong>
    </div>
    """, unsafe_allow_html=True)

def display_client_profile(client_data):
    """Afficher profil client complet"""
    st.markdown("#### 👤 Profil Client")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Score Externe 2", f"{client_data.get('EXT_SOURCE_2', 0):.3f}")
        st.metric("Score Externe 3", f"{client_data.get('EXT_SOURCE_3', 0):.3f}")
        st.metric("Score Externe 1", f"{client_data.get('EXT_SOURCE_1', 0):.3f}")
        st.metric("Retards moyens", f"{client_data.get('INSTAL_DPD_MEAN', 0):.1f} jours")
        
        # WCAG 1.1.1 : Description textuelle des métriques
        st.caption("Scores externes : indicateurs de solvabilité (0=risqué, 1=sûr). Retards : moyenne des jours de retard sur paiements antérieurs.")
    
    with col2:
        employment_years = abs(client_data.get('DAYS_EMPLOYED', 0)) / 365.25
        st.metric("Ancienneté emploi", f"{employment_years:.2f} ans")
        
        gender = "Homme" if client_data.get('CODE_GENDER') == 'M' else "Femme"
        st.metric("Genre", gender)
        
        payment_rate = client_data.get('PAYMENT_RATE', 0)
        st.metric("Ratio endettement", f"{payment_rate:.1%}")
        
        # WCAG 1.1.1 : Description textuelle des métriques
        st.caption("Ancienneté emploi : durée dans le poste actuel. Ratio endettement : charges mensuelles / revenus.")
    
    with col3:
        annuity = client_data.get('AMT_ANNUITY', 0)
        st.metric("Annuité mensuelle", f"{annuity:,.0f} €")
        
        education = "Oui" if client_data.get('NAME_EDUCATION_TYPE_Higher_education', 0) == 1 else "Non"
        st.metric("Éducation supérieure", education)
        
        payment_sum = client_data.get('INSTAL_AMT_PAYMENT_SUM', 0)
        st.metric("Hist. paiements", f"{payment_sum:,.0f} €")
        
        # WCAG 1.1.1 : Description textuelle des métriques
        st.caption("Annuité : montant mensuel du crédit. Historique : cumul des paiements antérieurs.")

def create_simple_population_plot(distribution_data, client_value, variable_name):
    """Créer histogramme simple : distribution population + ligne client"""
    
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
        client_value_numeric = client_value  # Déjà 0 ou 1
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
        name='Population',
        showlegend=False
    ))
    
    # Ligne verticale rouge pour le client (seulement si valeur numérique)
    try:
        fig.add_vline(
            x=client_value_numeric,
            line_dash="solid",
            line_color="red",
            line_width=4,
            annotation_text="⭐ Client",
            annotation_position="top"
        )
    except (TypeError, ValueError):
        st.warning(f"Impossible d'afficher la position client pour {variable_name}")
    
    # Configuration du graphique avec layout
    layout_config = {
        'title': f"{FEATURE_TRANSLATIONS.get(variable_name, variable_name)}",
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
    
    # WCAG 1.1.1 : Texte alternatif pour histogramme population
    variable_fr = FEATURE_TRANSLATIONS.get(variable_name, variable_name)
    
    if variable_name in ['CODE_GENDER', 'NAME_EDUCATION_TYPE_Higher_education']:
        st.markdown(f"""
        **Description graphique :** Histogramme de répartition de la variable {variable_fr} dans la population. 
        Graphique en barres montrant la distribution des clients selon cette caractéristique. 
        La position du client analysé est marquée par une ligne rouge verticale.
        """)
    else:
        client_val_formatted = f"{client_value_numeric:.2f}" if isinstance(client_value_numeric, (int, float)) else str(client_value_numeric)
        st.markdown(f"""
        **Description graphique :** Histogramme de distribution de la variable {variable_fr} dans la population. 
        L'axe horizontal représente les valeurs de {variable_fr}, l'axe vertical le nombre de clients. 
        Le client analysé (valeur: {client_val_formatted}) est positionné par une ligne rouge verticale.
        """)

def display_simple_population_comparison(client_data):
    """Interface SANS rerun forcé"""
       
    # Layout avec bouton MAIS sans rerun
    col1, col2 = st.columns([3, 1])
    
    with col1:
        selected_variable = st.selectbox(
            "Variable à analyser :",
            DASHBOARD_FEATURES,
            format_func=lambda x: FEATURE_TRANSLATIONS.get(x, x)
        )
    
    with col2:
        # Bouton qui change juste un flag SANS st.rerun()
        if st.button("📊 Actualiser", key="refresh_comparison", help="Actualiser le graphique"):
            st.session_state.refresh_comparison = not st.session_state.refresh_comparison
    
    # Récupérer les données de distribution
    distribution_data = get_population_distribution(selected_variable)
    
    if distribution_data:
        client_value = client_data.get(selected_variable)
        
        if client_value is not None:
            # Afficher le graphique simple
            create_simple_population_plot(distribution_data, client_value, selected_variable)
        else:
            st.error(f"Valeur client manquante pour {selected_variable}")
    else:
        st.error(f"Impossible de charger les données pour {selected_variable}")
        
# Interface principale

st.markdown('<div class="main-header">🏦 Dashboard Credit Scoring<br>Prêt à dépenser</div>', unsafe_allow_html=True)

# Vérification API SANS effet de bord
api_ok, api_info, api_error = test_api_connection()

if not api_ok:
    st.error(f"⚠️ **API non accessible**: {api_error}")
    st.stop()

# Sidebar AVEC PROTECTION
with st.sidebar:
    st.markdown("**🏦 Dashboard Credit Scoring<br>Prêt à dépenser**")
    st.markdown("---")

    st.markdown("### 📋 Navigation")
    
    # NOUVEAU CLIENT avec protection
    if st.button("🆕 Nouveau client", use_container_width=True):
        # RESET COMPLET SANS rerun forcé
        for key in ['client_analyzed', 'client_data', 'prediction_result', 'form_submitted']:
            if key in st.session_state:
                st.session_state[key] = False if 'analyzed' in key or 'submitted' in key else None
    
    st.markdown("---")
    st.markdown("**📊 Statut API**")
    if api_info:
        st.success("✅ Connectée")
        st.caption(f"Version: {api_info.get('version', 'N/A')}")
    else:
        st.error("❌ Déconnectée")

# Interface principale AVEC PROTECTION ANTI-BOUCLE
if not st.session_state.client_analyzed:
    # Étape 1 : Saisie client
    st.markdown("### 📝 Nouveau client")
    
    client_data = create_client_form()
    
    # PROTECTION ANTI-DOUBLE-CLIC
    if st.button("🎯 ANALYSER CE CLIENT", type="primary", use_container_width=True, disabled=st.session_state.form_submitted):
        
        # Marquer comme soumis IMMÉDIATEMENT
        st.session_state.form_submitted = True
        
        with st.spinner("🔄 Analyse en cours..."):
            result, error = call_prediction_api(client_data)
        
        if result:
            # Mettre à jour TOUT L'ÉTAT en une fois
            st.session_state.client_data = client_data
            st.session_state.prediction_result = result
            st.session_state.client_analyzed = True
            st.session_state.last_analysis_time = time.time()
            st.session_state.form_submitted = False  # Reset pour prochaine fois
            
            st.success("✅ Client analysé avec succès !")
            
            # RERUN UNE SEULE FOIS
            st.rerun()
        else:
            st.session_state.form_submitted = False  # Reset en cas d'erreur
            st.error(f"❌ Erreur d'analyse : {error}")

else:
    # Étape 2 : Résultats et analyses
    tab1, tab2, tab3 = st.tabs(["🎯 Résultats", "📊 Comparaisons", "🔧 Analyses bi-variées"])
    
    with tab1:
        st.markdown("### 🎯 Résultat de l'analyse")
        
        # Bouton pour modifier SANS rerun automatique
        col1, col2 = st.columns([3, 1])
        with col2:
            if st.button("🔧 Modifier", use_container_width=True):
                # RESET COMPLET sans rerun immédiat
                st.session_state.client_analyzed = False
                st.session_state.form_submitted = False
                # Le rerun se fera naturellement au prochain cycle
        
        # Profil client
        display_client_profile(st.session_state.client_data)
        
        st.markdown("---")
        
        # Résultat scoring
        display_prediction_result(st.session_state.prediction_result)
        
        st.markdown("---")
        
        # Feature importance avec graphique + tableau détaillé
        display_feature_importance(st.session_state.prediction_result)
    
    with tab2:
        st.markdown("### 📊 Comparaisons avec la base clients")
        
        # Interface avec bouton actualiser SANS rerun
        display_simple_population_comparison(st.session_state.client_data)
    
    with tab3:
        st.markdown("### 🔧 Analyse bi-variée")
        
        col1, col2 = st.columns(2)
        
        with col1:
            var1 = st.selectbox(
                "Variable 1",
                DASHBOARD_FEATURES,
                format_func=lambda x: FEATURE_TRANSLATIONS.get(x, x)
            )
        
        with col2:
            var2 = st.selectbox(
                "Variable 2", 
                DASHBOARD_FEATURES,
                index=1,
                format_func=lambda x: FEATURE_TRANSLATIONS.get(x, x)
            )
        
        if st.button("📈 Analyser Relation", use_container_width=True):
            with st.spinner("Analyse en cours..."):
                # Récupérer les vraies distributions pour les 2 variables
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
                    
                    # Graphique de corrélation avec TOUT l'échantillon
                    fig = px.scatter(
                        x=x_data,
                        y=y_data,
                        title=f"Relation entre {FEATURE_TRANSLATIONS.get(var1, var1)} et {FEATURE_TRANSLATIONS.get(var2, var2)}",
                        labels={
                            'x': FEATURE_TRANSLATIONS.get(var1, var1),
                            'y': FEATURE_TRANSLATIONS.get(var2, var2)
                        },
                        opacity=0.6
                    )
                    
                    fig.update_layout(height=500)
                    st.plotly_chart(fig, use_container_width=True, config=PLOTLY_CONFIG)
                    
                    # WCAG 1.1.1 : Texte alternatif pour analyse bi-variée
                    correlation = np.corrcoef(x_data, y_data)[0, 1] if len(x_data) > 1 else 0
                    var1_fr = FEATURE_TRANSLATIONS.get(var1, var1)
                    var2_fr = FEATURE_TRANSLATIONS.get(var2, var2)
                    
                    st.markdown(f"""
                    **Description graphique :** Nuage de points montrant la relation entre {var1_fr} (axe horizontal) et {var2_fr} (axe vertical). 
                    Chaque point représente un client. Corrélation : {correlation:.3f}. 
                    {'Relation positive' if correlation > 0.3 else 'Relation négative' if correlation < -0.3 else 'Relation faible'} entre les deux variables.
                    """)
                    
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

# Footer
st.markdown("---")
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("**🏦 Prêt à dépenser**")
    st.markdown("Dashboard Credit Scoring")
    st.markdown("Brice Béchet")
    st.markdown("Juin 2025 - Master 2 Data Scientist - OpenClassRoom")

with col2:
    st.markdown("**✅ Fonctionnalités**")
    st.markdown("• Analyse de crédit instantanée")
    st.markdown("• Explications transparentes")
    st.markdown("• Comparaisons population")
    st.markdown("• Interface chargé relation client")

with col3:
    st.markdown("**♿ Accessibilité WCAG 2.1**")
    st.markdown("• Navigation clavier optimisée")
    st.markdown("• Contrastes élevés")
    st.markdown("• Textes alternatifs")
    st.markdown("• Interface responsive")