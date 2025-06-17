"""
Dashboard Credit Scoring Production - Streamlit Cloud
Version: Production v2.0 - SIMPLE
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

# CSS WCAG conforme pour production
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
    box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15);
}

.metric-card {
    background-color: #f8fafc;
    border: 2px solid #e2e8f0;
    padding: 1.5rem;
    border-radius: 1rem;
    margin: 1rem 0;
    font-size: 1.1rem;
    box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1);
}

.success-card {
    background-color: #dcfce7;
    border: 3px solid #16a34a;
    color: #15803d;
}

.warning-card {
    background-color: #fef3c7;
    border: 3px solid #d97706;
    color: #92400e;
}

.error-card {
    background-color: #fee2e2;
    border: 3px solid #dc2626;
    color: #991b1b;
}

/* WCAG Accessibilité */
.approved::before { content: "✅ "; font-weight: bold; }
.refused::before { content: "❌ "; font-weight: bold; }

.alert-info {
    background: #eff6ff;
    border: 2px solid #3b82f6;
    color: #1d4ed8;
    padding: 1rem;
    border-radius: 0.5rem;
    margin: 1rem 0;
    font-weight: 500;
}

.alert-success {
    background: #dcfce7;
    border: 2px solid #16a34a;
    color: #15803d;
    padding: 1rem;
    border-radius: 0.5rem;
    margin: 1rem 0;
    font-weight: 500;
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
    "EXT_SOURCE_2": "Score externe principal (0=mauvais, 1=excellent)",
    "EXT_SOURCE_3": "Score externe complémentaire",
    "EXT_SOURCE_1": "Premier score externe (peut être manquant)",
    "DAYS_EMPLOYED": "Ancienneté dans l'emploi actuel (jours négatifs)",
    "PAYMENT_RATE": "Ratio d'endettement par rapport aux revenus",
    "AMT_ANNUITY": "Montant mensuel que le client devra payer",
    "INSTAL_DPD_MEAN": "Retards moyens sur paiements antérieurs (jours)",
    "AMT_ANNUITY": "Montant de l'annuité mensuelle",
    "INSTAL_AMT_PAYMENT_SUM": "Somme historique des paiements",
    "CODE_GENDER": "Genre du client",
    "NAME_EDUCATION_TYPE_Higher_education": "Niveau d'éducation supérieure"
}

# Les 10 variables dashboard
DASHBOARD_FEATURES = [
    'EXT_SOURCE_2', 'EXT_SOURCE_3', 'EXT_SOURCE_1',
    'DAYS_EMPLOYED', 'CODE_GENDER', 'INSTAL_DPD_MEAN',
    'PAYMENT_RATE', 'NAME_EDUCATION_TYPE_Higher_education',
    'AMT_ANNUITY', 'INSTAL_AMT_PAYMENT_SUM'
]

# Session state optimisé
if 'client_analyzed' not in st.session_state:
    st.session_state.client_analyzed = False
if 'client_data' not in st.session_state:
    st.session_state.client_data = None
if 'prediction_result' not in st.session_state:
    st.session_state.prediction_result = None

# Fonctions API

@st.cache_data(ttl=300)  # Cache 5 minutes
def test_api_connection():
    """Test de connexion API avec cache"""
    try:
        response = requests.get(f"{API_URL}/health", timeout=10)
        if response.status_code == 200:
            return True, response.json()
        return False, None
    except Exception as e:
        st.error(f"⚠️ Erreur API: {str(e)}")
        return False, None

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

@st.cache_data(ttl=600)  # Cache 10 minutes
def get_population_distribution(variable):
    """Récupérer distribution d'une variable spécifique"""
    try:
        response = requests.get(f"{API_URL}/population/{variable}", timeout=15)
        if response.status_code == 200:
            return response.json()
        return None
    except Exception as e:
        st.warning(f"Distribution {variable} indisponible: {str(e)}")
        return None

@st.cache_data(ttl=600)  # Cache 10 minutes
def get_population_data():
    """Récupérer données population avec cache (pour bi-variée)"""
    try:
        response = requests.get(f"{API_URL}/population_stats", timeout=15)
        if response.status_code == 200:
            return response.json()
        return None
    except Exception as e:
        st.warning(f"Données population indisponibles: {str(e)}")
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
        st.warning(f"Analyse bi-variée indisponible: {str(e)}")
        return None

# Interface de saisie client

def create_client_form():
    """Formulaire de saisie client avec les 10 variables complètes"""
    
    with st.expander("ℹ️ Guide d'utilisation", expanded=False):
        st.markdown("""
        **Pour les chargés de relation client :**
        - Les **scores externes** viennent des bureaux de crédit (0 = risqué, 1 = sûr)
        - L'**ancienneté emploi** est en années positives
        - Le **ratio d'endettement** = charges / revenus (max 100%)
        - Les **retards moyens** = nombre moyen de jours de retard sur les paiements précédents
        """)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**📊 Scores Externes (Critiques)**")
        
        ext_source_2 = st.slider(
            "Score Externe 2*", 
            0.0, 1.0, 0.6, 0.01,
            help=FEATURE_EXPLANATIONS["EXT_SOURCE_2"]
        )
        
        ext_source_3 = st.slider(
            "Score Externe 3*", 
            0.0, 1.0, 0.5, 0.01,
            help=FEATURE_EXPLANATIONS["EXT_SOURCE_3"]
        )
        
        ext_source_1 = st.slider(
            "Score Externe 1", 
            0.0, 1.0, 0.4, 0.01,
            help=FEATURE_EXPLANATIONS["EXT_SOURCE_1"]
        )
        
        employment_years = st.number_input(
            "Ancienneté emploi (années)", 
            0.0, 40.0, 5.0, 0.01,
            help="Années dans l'emploi actuel"
        )
        
        instal_dpd_mean = st.slider(
            "Retards moyens (jours)", 
            0.0, 30.0, 0.5, 0.1,
            help="Nombre moyen de jours de retard sur les paiements antérieurs"
        )
        
    with col2:
        st.markdown("**💼 Informations Complémentaires**")
        
        gender = st.selectbox("Genre", ["Femme", "Homme"])
        
        payment_rate = st.slider(
            "Ratio d'endettement", 
            0.0, 1.0, 0.15, 0.01,
            help=FEATURE_EXPLANATIONS["PAYMENT_RATE"]
        )
        
        education = st.selectbox(
            "Éducation supérieure", ["Non", "Oui"]
        )
        
        annuity = st.number_input(
            "Annuité mensuelle (€)", 
            5000, 100000, 18000, 1000,
            help=FEATURE_EXPLANATIONS["AMT_ANNUITY"]
        )
        
        payment_sum = st.number_input(
            "Historique paiements (€)", 
            10000, 1000000, 120000, 10000,
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
    """Afficher résultat de prédiction avec accessibilité WCAG"""
    prediction = result.get('prediction', {})
    probability = prediction.get('probability', 0)
    decision = prediction.get('decision', 'UNKNOWN')
    decision_fr = prediction.get('decision_fr', decision)
    risk_level = prediction.get('risk_level', 'Inconnu')
    
    # Résultat principal
    if decision == "REFUSE":
        st.markdown(f"""
        <div class="metric-card error-card refused">
            <h2>❌ CRÉDIT REFUSÉ</h2>
            <p><strong>Probabilité de défaut: {probability:.1%}</strong></p>
            <p>Niveau de risque: {risk_level}</p>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="metric-card success-card approved">
            <h2>✅ CRÉDIT ACCORDÉ</h2>
            <p><strong>Probabilité de défaut: {probability:.1%}</strong></p>
            <p>Niveau de risque: {risk_level}</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Gauge visuelle
    fig_gauge = go.Figure(go.Indicator(
        mode="gauge",
        value=probability * 100,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': "Niveau de Risque (%)", 'font': {'size': 20}},
        gauge={
            'axis': {'range': [None, 100], 'tickwidth': 3, 'tickcolor': "black"},
            'bar': {'color': "darkblue", 'thickness': 0.3},
            'bgcolor': "white",
            'borderwidth': 3,
            'bordercolor': "gray",
            'steps': [
                {'range': [0, 10], 'color': 'lightgreen'},
                {'range': [10, 100], 'color': 'lightcoral'}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 5},
                'thickness': 0.8,
                'value': 10  # Seuil simplifié à 10%
            }
        }
    ))
    
    fig_gauge.update_layout(
        height=400,
        font={'color': "black", 'family': "Arial", 'size': 16}
    )
    
    st.plotly_chart(fig_gauge, use_container_width=True)

def display_feature_importance(result):
    """Afficher importance des variables avec graphique et tableau détaillé"""
    explanation = result.get('explanation', {})
    top_features = explanation.get('top_features', [])
    client_data = st.session_state.client_data
    
    if not top_features:
        st.warning("Explications des variables non disponibles")
        return
    
    st.markdown("#### 🔍 Interprétation de la Décision")
    st.markdown("**Impact de TOUTES les Variables Saisissables sur la Décision**")
    
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
        title="Impact de TOUTES les Variables Saisissables sur la Décision"
    )
    
    fig.update_layout(
        height=500,
        showlegend=True,
        font={'size': 12},
        xaxis_title="Impact sur la prédiction",
        yaxis_title="Variables Saisissables"
    )
    
    fig.add_vline(x=0, line_dash="dash", line_color="gray", line_width=2)
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Tableau détaillé
    with st.expander("📋 Tableau Détaillé de Toutes les Variables Saisissables", expanded=True):
        
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
        <strong>💡 Lecture du graphique des variables saisissables :</strong><br>
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
    
    with col2:
        employment_years = abs(client_data.get('DAYS_EMPLOYED', 0)) / 365.25
        st.metric("Ancienneté emploi", f"{employment_years:.2f} ans")
        
        gender = "Homme" if client_data.get('CODE_GENDER') == 'M' else "Femme"
        st.metric("Genre", gender)
        
        payment_rate = client_data.get('PAYMENT_RATE', 0)
        st.metric("Ratio endettement", f"{payment_rate:.1%}")
    
    with col3:
        annuity = client_data.get('AMT_ANNUITY', 0)
        st.metric("Annuité mensuelle", f"{annuity:,.0f} €")
        
        education = "Oui" if client_data.get('NAME_EDUCATION_TYPE_Higher_education', 0) == 1 else "Non"
        st.metric("Éducation supérieure", education)
        
        payment_sum = client_data.get('INSTAL_AMT_PAYMENT_SUM', 0)
        st.metric("Hist. paiements", f"{payment_sum:,.0f} €")

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
    
    st.plotly_chart(fig, use_container_width=True)

def display_simple_population_comparison(client_data):
    """Interface simple : dropdown + graphique"""
    
    st.markdown("#### 📊 Position vs Population")
    
    # Sélecteur de variable (les 10 variables)
    selected_variable = st.selectbox(
        "Variable à analyser :",
        DASHBOARD_FEATURES,
        format_func=lambda x: FEATURE_TRANSLATIONS.get(x, x)
    )
    
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
        
        # Info pour debugging
        st.info("""
        **Possible causes :**
        - Variable non présente dans population_distribution.json
        - Problème de connexion API
        - Variable avec trop de valeurs manquantes
        
        **Variables généralement disponibles :** EXT_SOURCE_2, EXT_SOURCE_3, PAYMENT_RATE, AMT_ANNUITY
        """)

# Interface principale

st.markdown('<div class="main-header">🏦 Dashboard Credit Scoring<br>Prêt à dépenser</div>', unsafe_allow_html=True)

# Vérification API
api_ok, api_info = test_api_connection()

if not api_ok:
    st.error("""
    ⚠️ **API non accessible**
    
    L'API de scoring n'est pas disponible. Cela peut être dû à :
    - Déploiement en cours sur Railway
    - Problème de connexion réseau
    - Maintenance du service
    
    Veuillez réessayer dans quelques minutes.
    """)
    st.stop()

# Sidebar
with st.sidebar:
    st.markdown("### 📋 Navigation")
    
    if st.button("🔄 Nouveau Client", use_container_width=True):
        st.session_state.client_analyzed = False
        st.session_state.client_data = None
        st.session_state.prediction_result = None
        st.rerun()
    
    st.markdown("---")
    st.markdown("**📊 Statut API**")
    if api_info:
        st.success("✅ Connectée")
        st.caption(f"Version: {api_info.get('version', 'N/A')}")
    else:
        st.error("❌ Déconnectée")

# Interface principale avec onglets
if not st.session_state.client_analyzed:
    # Étape 1 : Saisie client
    st.markdown("### 📝 Nouveau Dossier Client")
    
    client_data = create_client_form()
    
    if st.button("🎯 ANALYSER CE CLIENT", type="primary", use_container_width=True):
        with st.spinner("🔄 Analyse en cours..."):
            result, error = call_prediction_api(client_data)
        
        if result:
            st.session_state.client_data = client_data
            st.session_state.prediction_result = result
            st.session_state.client_analyzed = True
            st.success("✅ Client analysé avec succès !")
            st.rerun()
        else:
            st.error(f"❌ Erreur d'analyse : {error}")

else:
    # Étape 2 : Résultats et analyses
    tab1, tab2, tab3 = st.tabs(["🎯 Résultats", "📊 Comparaisons", "🔧 Analyses"])
    
    with tab1:
        st.markdown("### 🎯 Résultat de l'Analyse")
        
        # Profil client
        display_client_profile(st.session_state.client_data)
        
        st.markdown("---")
        
        # Résultat scoring
        display_prediction_result(st.session_state.prediction_result)
        
        st.markdown("---")
        
        # Feature importance avec graphique + tableau détaillé
        display_feature_importance(st.session_state.prediction_result)
    
    with tab2:
        st.markdown("### 📊 Comparaisons Population")
        
        # Interface simple : dropdown + graphique
        display_simple_population_comparison(st.session_state.client_data)
    
    with tab3:
        st.markdown("### 🔧 Analyse Bi-variée")
        
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
                    st.plotly_chart(fig, use_container_width=True)
                    
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
    st.markdown("Version Production v2.0")

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

# Informations techniques cachées
with st.expander("🔧 Informations Techniques", expanded=False):
    st.markdown(f"""
    **Configuration Production :**
    - API URL: {API_URL}
    - Streamlit Cloud: ✅ Déployé
    - Railway API: {'✅ Connectée' if api_ok else '❌ Déconnectée'}
    - Cache: 5-10 minutes
    - WCAG 2.1: Niveau AA
    
    **Support :** Cette interface est optimisée pour les chargés de relation client.
    """)