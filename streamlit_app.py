"""
Dashboard Credit Scoring Production - Streamlit Cloud
Version: Production v1.0
Plateforme: Streamlit Cloud + Railway API
Fonctionnalités: Interface chargés relation client optimisée
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
API_URL = st.secrets.get("API_URL", "https://dashboard-credit-scoring-production.up.railway.app")

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
}

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
def get_population_data():
    """Récupérer données population avec cache"""
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
    """Formulaire de saisie client simplifié"""
    
    with st.expander("ℹ️ Guide d'utilisation", expanded=False):
        st.markdown("""
        **Pour les chargés de relation client :**
        - Les **scores externes** viennent des bureaux de crédit (0 = risqué, 1 = sûr)
        - L'**ancienneté emploi** est en jours négatifs (-1825 = 5 ans)
        - Le **ratio d'endettement** = charges / revenus
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
        
        employment_days = st.number_input(
            "Ancienneté emploi (jours)", 
            -15000, 0, -1825, 100,
            help="Jours dans l'emploi actuel (négatif)"
        )
        
    with col2:
        st.markdown("**💼 Informations Complémentaires**")
        
        gender = st.selectbox("Genre", ["Femme", "Homme"])
        
        payment_rate = st.slider(
            "Ratio d'endettement", 
            0.0, 0.8, 0.15, 0.01,
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
    
    # Conversion pour API
    client_data = {
        "EXT_SOURCE_2": float(ext_source_2),
        "EXT_SOURCE_3": float(ext_source_3),
        "EXT_SOURCE_1": float(ext_source_1),
        "DAYS_EMPLOYED": int(employment_days),
        "CODE_GENDER": "M" if gender == "Homme" else "F",
        "INSTAL_DPD_MEAN": 0.0,  # Simplifié pour production
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
    """Afficher importance des variables"""
    explanation = result.get('explanation', {})
    top_features = explanation.get('top_features', [])
    
    if not top_features:
        st.warning("Explications des variables non disponibles")
        return
    
    st.markdown("#### 🔍 Variables Influentes sur la Décision")
    
    # Préparer données pour graphique
    features_df = pd.DataFrame(top_features)
    if features_df.empty:
        return
    
    # Traduction des noms
    features_df['feature_fr'] = features_df['feature'].map(
        lambda x: FEATURE_TRANSLATIONS.get(x, x.replace('_', ' ').title())
    )
    
    # Couleurs selon impact
    features_df['color'] = features_df['shap_value'].apply(
        lambda x: "Augmente le risque" if x > 0 else "Diminue le risque"
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
            "Diminue le risque": "#44aa44"
        },
        title="Impact des Variables sur la Décision",
        labels={'shap_value': 'Impact', 'feature_fr': 'Variables'}
    )
    
    fig.update_layout(
        height=400,
        showlegend=True,
        font={'size': 12}
    )
    
    fig.add_vline(x=0, line_dash="dash", line_color="gray")
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Explication pédagogique
    st.markdown("""
    <div class="alert-info">
        <strong>💡 Lecture du graphique :</strong><br>
        • <span style="color: #44aa44;"><strong>Vert</strong></span> : Variables qui diminuent le risque<br>
        • <span style="color: #ff4444;"><strong>Rouge</strong></span> : Variables qui augmentent le risque<br>
        • <strong>Longueur</strong> : Plus c'est long, plus l'impact est fort
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
    
    with col2:
        employment_years = abs(client_data.get('DAYS_EMPLOYED', 0)) / 365.25
        st.metric("Ancienneté emploi", f"{employment_years:.1f} ans")
        
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

def display_population_comparison(client_data, population_data):
    """Comparaisons avec la population"""
    if not population_data:
        st.warning("Données de population non disponibles")
        return
    
    st.markdown("#### 📊 Position vs Population")
    
    graph_data = population_data.get('graph_data', {})
    key_vars = ['EXT_SOURCE_2', 'EXT_SOURCE_3', 'PAYMENT_RATE']
    
    for i in range(0, len(key_vars), 2):
        col1, col2 = st.columns(2)
        
        for j, col in enumerate([col1, col2]):
            if i + j < len(key_vars):
                var = key_vars[i + j]
                
                with col:
                    if var in graph_data and var in client_data:
                        data = graph_data[var]
                        values = data['values']
                        stats = data['stats']
                        client_value = client_data[var]
                        
                        # Histogramme
                        fig = go.Figure()
                        
                        fig.add_trace(go.Histogram(
                            x=values,
                            nbinsx=20,
                            name='Population',
                            opacity=0.7,
                            marker_color='lightblue'
                        ))
                        
                        # Ligne client
                        fig.add_vline(
                            x=client_value,
                            line_dash="solid",
                            line_color="red",
                            line_width=3,
                            annotation_text="Client"
                        )
                        
                        # Ligne moyenne
                        fig.add_vline(
                            x=stats['mean'],
                            line_dash="dash",
                            line_color="blue",
                            annotation_text="Moyenne"
                        )
                        
                        fig.update_layout(
                            title=FEATURE_TRANSLATIONS.get(var, var),
                            height=300,
                            showlegend=False
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Indicateur position
                        diff_pct = ((client_value - stats['mean']) / stats['mean']) * 100
                        if abs(diff_pct) < 10:
                            st.success(f"🟢 Proche moyenne ({diff_pct:+.1f}%)")
                        elif diff_pct > 0:
                            st.info(f"🔵 Au-dessus moyenne ({diff_pct:+.1f}%)")
                        else:
                            st.warning(f"🟡 En-dessous moyenne ({diff_pct:+.1f}%)")

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
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
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
    
    with col2:
        st.markdown("**ℹ️ Aide**")
        st.caption("Remplissez au minimum les scores externes 2 et 3")

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
        
        # Feature importance
        display_feature_importance(st.session_state.prediction_result)
    
    with tab2:
        st.markdown("### 📊 Comparaisons Population")
        
        population_data = get_population_data()
        if population_data:
            display_population_comparison(st.session_state.client_data, population_data)
        else:
            st.warning("Données de population temporairement indisponibles")
    
    with tab3:
        st.markdown("### 🔧 Analyse Bi-variée")
        
        population_data = get_population_data()
        if population_data:
            available_vars = population_data.get('variables_available', [])
            
            if len(available_vars) >= 2:
                col1, col2 = st.columns(2)
                
                with col1:
                    var1 = st.selectbox(
                        "Variable 1",
                        available_vars,
                        format_func=lambda x: FEATURE_TRANSLATIONS.get(x, x)
                    )
                
                with col2:
                    var2 = st.selectbox(
                        "Variable 2", 
                        available_vars,
                        index=1,
                        format_func=lambda x: FEATURE_TRANSLATIONS.get(x, x)
                    )
                
                if st.button("📈 Analyser Relation", use_container_width=True):
                    with st.spinner("Analyse en cours..."):
                        bivariate_data = get_bivariate_data(var1, var2)
                    
                    if bivariate_data:
                        x_data = bivariate_data['data_points']['x']
                        y_data = bivariate_data['data_points']['y']
                        correlation = bivariate_data['correlation']
                        
                        # Graphique de corrélation
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
                        
                        # Métrique de corrélation
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            st.metric("Corrélation", f"{correlation:.3f}")
                        
                        with col2:
                            if abs(correlation) > 0.7:
                                strength = "Forte"
                            elif abs(correlation) > 0.3:
                                strength = "Modérée"
                            else:
                                strength = "Faible"
                            st.metric("Force", strength)
                        
                        with col3:
                            direction = "Positive" if correlation > 0 else "Négative"
                            st.metric("Direction", direction)
                    else:
                        st.error("Erreur lors de l'analyse bi-variée")
            else:
                st.warning("Pas assez de variables disponibles pour l'analyse")
        else:
            st.warning("Données indisponibles pour l'analyse bi-variée")

# Footer
st.markdown("---")
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("**🏦 Prêt à dépenser**")
    st.markdown("Dashboard Credit Scoring")
    st.markdown("Version Production v1.0")

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