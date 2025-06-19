"""
Dashboard Credit Scoring - Version Refactorisée Complète
Architecture modulaire avec toutes les fonctionnalités originales
"""

import streamlit as st
import requests
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import time
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass, asdict
from enum import Enum


# =============================================================================
# CONFIGURATION ET CONSTANTES
# =============================================================================

class Config:
    """Configuration centralisée de l'application"""
    API_URL = "https://dashboard-credit-scoring-production.up.railway.app"
    
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
    
    DASHBOARD_FEATURES = [
        'EXT_SOURCE_2', 'EXT_SOURCE_3', 'EXT_SOURCE_1',
        'DAYS_EMPLOYED', 'CODE_GENDER', 'INSTAL_DPD_MEAN',
        'PAYMENT_RATE', 'NAME_EDUCATION_TYPE_Higher_education',
        'AMT_ANNUITY', 'INSTAL_AMT_PAYMENT_SUM'
    ]
    
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


# =============================================================================
# MODÈLES DE DONNÉES
# =============================================================================

@dataclass
class ClientData:
    """Modèle pour les données client"""
    EXT_SOURCE_2: float
    EXT_SOURCE_3: float
    EXT_SOURCE_1: float
    DAYS_EMPLOYED: int
    CODE_GENDER: str
    INSTAL_DPD_MEAN: float
    PAYMENT_RATE: float
    NAME_EDUCATION_TYPE_Higher_education: int
    AMT_ANNUITY: float
    INSTAL_AMT_PAYMENT_SUM: float
    
    def to_dict(self) -> Dict[str, Any]:
        """Convertit en dictionnaire pour l'API"""
        return asdict(self)


@dataclass
class FeatureImportance:
    """Modèle pour l'importance des features"""
    feature: str
    shap_value: float
    feature_value: float
    impact: str


@dataclass
class PredictionResult:
    """Modèle pour les résultats de prédiction"""
    probability: float
    decision: str
    decision_fr: str
    risk_level: str
    threshold: float
    feature_importance: List[FeatureImportance]
    population_comparison: Dict[str, Any]
    processing_time: float
    
    @classmethod
    def from_api_response(cls, data: Dict) -> 'PredictionResult':
        """Crée une instance depuis la réponse API"""
        pred = data.get('prediction', {})
        expl = data.get('explanation', {})
        comp = data.get('population_comparison', {})
        meta = data.get('metadata', {})
        
        # Parse feature importance
        features = []
        for f in expl.get('top_features', []):
            features.append(FeatureImportance(
                feature=f.get('feature', ''),
                shap_value=f.get('shap_value', 0),
                feature_value=f.get('feature_value', 0),
                impact=f.get('impact', 'neutral')
            ))
        
        return cls(
            probability=pred.get('probability', 0),
            decision=pred.get('decision', 'UNKNOWN'),
            decision_fr=pred.get('decision_fr', ''),
            risk_level=pred.get('risk_level', ''),
            threshold=pred.get('threshold', 0),
            feature_importance=features,
            population_comparison=comp,
            processing_time=meta.get('processing_time_seconds', 0)
        )


class DecisionType(Enum):
    """Types de décision possibles"""
    APPROVE = "APPROVE"
    REFUSE = "REFUSE"


# =============================================================================
# SERVICES
# =============================================================================

class APIService:
    """Service pour les appels API"""
    
    def __init__(self, base_url: str):
        self.base_url = base_url
    
    @st.cache_data(ttl=300)
    def test_connection(_self) -> Tuple[bool, Optional[Dict], Optional[str]]:
        """Test de connexion API avec cache"""
        try:
            response = requests.get(f"{_self.base_url}/health", timeout=10)
            if response.status_code == 200:
                return True, response.json(), None
            return False, None, f"Status {response.status_code}"
        except Exception as e:
            return False, None, str(e)
    
    def predict(self, client_data: ClientData) -> Tuple[Optional[PredictionResult], Optional[str]]:
        """Appel API de prédiction"""
        try:
            response = requests.post(
                f"{self.base_url}/predict_dashboard",
                json=client_data.to_dict(),
                timeout=30,
                headers={"Content-Type": "application/json"}
            )
            
            if response.status_code == 200:
                data = response.json()
                return PredictionResult.from_api_response(data), None
            else:
                error_text = response.text
                return None, f"Erreur API {response.status_code}: {error_text}"
                
        except requests.exceptions.Timeout:
            return None, "Timeout API - Veuillez réessayer"
        except Exception as e:
            return None, f"Erreur connexion: {str(e)}"
    
    def get_population_distribution(self, variable: str) -> Optional[Dict]:
        """Récupère la distribution d'une variable"""
        try:
            response = requests.get(f"{self.base_url}/population/{variable}", timeout=15)
            return response.json() if response.status_code == 200 else None
        except Exception:
            return None
    
    def get_population_stats(self) -> Optional[Dict]:
        """Récupère les statistiques de population"""
        try:
            response = requests.get(f"{self.base_url}/population_stats", timeout=15)
            return response.json() if response.status_code == 200 else None
        except Exception:
            return None
    
    def get_bivariate_analysis(self, var1: str, var2: str) -> Optional[Dict]:
        """Analyse bi-variée"""
        try:
            response = requests.post(
                f"{self.base_url}/bivariate_analysis",
                json={"variable1": var1, "variable2": var2},
                timeout=20
            )
            return response.json() if response.status_code == 200 else None
        except Exception:
            return None


class SessionManager:
    """Gestionnaire d'état de session"""
    
    @staticmethod
    def init_session():
        """Initialise l'état de session"""
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
    
    @staticmethod
    def reset_session():
        """Remet à zéro la session"""
        st.session_state.client_analyzed = False
        st.session_state.client_data = None
        st.session_state.prediction_result = None
        st.session_state.api_call_in_progress = False
        st.session_state.population_cache = {}
        st.session_state.bivariate_cache = {}
        
        # Nettoyer cache dynamique
        keys_to_remove = [
            key for key in st.session_state.keys() 
            if key.startswith(('population_data_', 'bivariate_'))
        ]
        for key in keys_to_remove:
            del st.session_state[key]


class StyleManager:
    """Gestionnaire des styles CSS"""
    
    @staticmethod
    def load_styles():
        """Charge tous les styles CSS"""
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


# =============================================================================
# COMPOSANTS UI
# =============================================================================

class UIComponents:
    """Composants d'interface utilisateur réutilisables"""
    
    @staticmethod
    def render_header():
        """Affiche l'en-tête principal"""
        st.markdown(
            '<div class="main-header">🏦 Dashboard Credit Scoring<br>Prêt à dépenser</div>', 
            unsafe_allow_html=True
        )
    
    @staticmethod
    def render_guide():
        """Affiche le guide d'utilisation"""
        with st.expander("ℹ️ Guide d'utilisation", expanded=False):
            st.markdown("""
            ### 🚀 **Prêt à commencer ?**
            1. **Saisissez** les informations client dans le formulaire ci-dessous
            2. **Analysez** le dossier en cliquant sur "Analyser ce client"
            3. **Explorez** les onglets Résultats, Comparaisons et Analyses
            4. **Simulez** différents scénarios si nécessaire
            """)


class ClientForm:
    """Formulaire de saisie client"""
    
    def __init__(self):
        self.default_values = st.session_state.client_data.to_dict() if st.session_state.client_data else {}
    
    def render(self) -> ClientData:
        """Affiche le formulaire et retourne les données client"""
        UIComponents.render_guide()
        
        col1, col2 = st.columns(2)
        
        with col1:
            ext_source_2 = st.slider(
                "Score Externe 2", 0.0, 1.0,
                float(self.default_values.get('EXT_SOURCE_2', 0.6)), 0.01,
                help=Config.FEATURE_EXPLANATIONS["EXT_SOURCE_2"]
            )
            
            ext_source_3 = st.slider(
                "Score Externe 3", 0.0, 1.0,
                float(self.default_values.get('EXT_SOURCE_3', 0.5)), 0.01,
                help=Config.FEATURE_EXPLANATIONS["EXT_SOURCE_3"]
            )
            
            ext_source_1 = st.slider(
                "Score Externe 1", 0.0, 1.0,
                float(self.default_values.get('EXT_SOURCE_1', 0.4)), 0.01,
                help=Config.FEATURE_EXPLANATIONS["EXT_SOURCE_1"]
            )
            
            default_employment = abs(self.default_values.get('DAYS_EMPLOYED', -1825)) / 365.25
            employment_years = st.number_input(
                "Ancienneté emploi (années)", 0.0, 40.0,
                float(default_employment), 0.01,
                help=Config.FEATURE_EXPLANATIONS["DAYS_EMPLOYED"]
            )
            
            instal_dpd_mean = st.slider(
                "Retards moyens (jours)", 0.0, 30.0,
                float(self.default_values.get('INSTAL_DPD_MEAN', 0.5)), 0.1,
                help=Config.FEATURE_EXPLANATIONS["INSTAL_DPD_MEAN"]
            )
        
        with col2:
            st.markdown("**💼 Informations Complémentaires**")
            
            default_gender = "Homme" if self.default_values.get('CODE_GENDER') == 'M' else "Femme"
            gender = st.selectbox("Genre", ["Femme", "Homme"], 
                                index=0 if default_gender == "Femme" else 1)
            
            payment_rate = st.slider(
                "Ratio d'endettement", 0.0, 1.0,
                float(self.default_values.get('PAYMENT_RATE', 0.15)), 0.01,
                help=Config.FEATURE_EXPLANATIONS["PAYMENT_RATE"]
            )
            
            default_education = "Oui" if self.default_values.get('NAME_EDUCATION_TYPE_Higher_education', 0) == 1 else "Non"
            education = st.selectbox("Éducation supérieure", ["Non", "Oui"],
                                   index=0 if default_education == "Non" else 1)
            
            annuity = st.number_input(
                "Annuité mensuelle (€)", 5000, 100000,
                int(self.default_values.get('AMT_ANNUITY', 18000)), 1000,
                help=Config.FEATURE_EXPLANATIONS["AMT_ANNUITY"]
            )
            
            payment_sum = st.number_input(
                "Historique paiements (€)", 10000, 1000000,
                int(self.default_values.get('INSTAL_AMT_PAYMENT_SUM', 120000)), 10000,
                help="Somme des paiements antérieurs"
            )
        
        return ClientData(
            EXT_SOURCE_2=float(ext_source_2),
            EXT_SOURCE_3=float(ext_source_3),
            EXT_SOURCE_1=float(ext_source_1),
            DAYS_EMPLOYED=-int(employment_years * 365.25),
            CODE_GENDER="M" if gender == "Homme" else "F",
            INSTAL_DPD_MEAN=float(instal_dpd_mean),
            PAYMENT_RATE=float(payment_rate),
            NAME_EDUCATION_TYPE_Higher_education=1 if education == "Oui" else 0,
            AMT_ANNUITY=float(annuity),
            INSTAL_AMT_PAYMENT_SUM=float(payment_sum)
        )


class ResultsDisplay:
    """Affichage des résultats de prédiction"""
    
    @staticmethod
    def render_client_profile(client_data: ClientData):
        """Affiche le profil client complet"""
        st.markdown("#### 👤 Profil Client")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Score Externe 2", f"{client_data.EXT_SOURCE_2:.3f}")
            st.metric("Score Externe 3", f"{client_data.EXT_SOURCE_3:.3f}")
            st.metric("Score Externe 1", f"{client_data.EXT_SOURCE_1:.3f}")
            st.metric("Retards moyens", f"{client_data.INSTAL_DPD_MEAN:.1f} jours")
            st.caption("Scores externes : indicateurs externes (0=risqué, 1=sûr). Retards : moyenne des jours de retard sur paiements antérieurs.")
        
        with col2:
            employment_years = abs(client_data.DAYS_EMPLOYED) / 365.25
            st.metric("Ancienneté emploi", f"{employment_years:.2f} ans")
            
            gender = "Homme" if client_data.CODE_GENDER == 'M' else "Femme"
            st.metric("Genre", gender)
            
            st.metric("Ratio endettement", f"{client_data.PAYMENT_RATE:.1%}")
            st.caption("Ancienneté emploi : durée dans le poste actuel. Genre : Homme ou Femme. Ratio endettement : charges mensuelles / revenus.")
        
        with col3:
            st.metric("Annuité mensuelle", f"{client_data.AMT_ANNUITY:,.0f} €")
            
            education = "Oui" if client_data.NAME_EDUCATION_TYPE_Higher_education == 1 else "Non"
            st.metric("Éducation supérieure", education)
            
            st.metric("Hist. paiements", f"{client_data.INSTAL_AMT_PAYMENT_SUM:,.0f} €")
            st.caption("Annuité : montant mensuel du crédit. Education supérieure : Oui ou Non. Historique : cumul des paiements antérieurs.")
    
    @staticmethod
    def render_prediction_result(result: PredictionResult):
        """Affiche le résultat de prédiction avec jauge modernisée"""
        probability = result.probability
        decision = result.decision
        decision_fr = result.decision_fr
        risk_level = result.risk_level
        threshold = result.threshold
        threshold_percent = threshold * 100
        
        # Résultat principal
        if decision == "REFUSE":
            st.markdown(f"""
            <div class="metric-card error-card refused">
                <h2>❌ CRÉDIT REFUSÉ - <strong>Probabilité de défaut: {probability:.1%}</strong> Niveau de risque: {risk_level}</h2>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="metric-card success-card approved">
                <h2>✅ CRÉDIT ACCORDÉ - <strong>Probabilité de défaut: {probability:.1%}</strong> Niveau de risque: {risk_level}</h2>
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
        
        st.plotly_chart(fig_gauge, use_container_width=True, config=Config.PLOTLY_CONFIG)
        
        # Métriques détaillées
        probability_percent = probability * 100
        ecart_avec_seuil = probability_percent - threshold_percent
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric(
                label="📊 Probabilité de défaut",
                value=f"{probability_percent:.2f}%",
                help="Probabilité calculée par le modèle"
            )
        
        with col2:
            st.metric(
                label="🎯 Seuil de décision",
                value=f"{threshold_percent:.2f}%",
                help="Seuil optimal issu du fichier optimal_threshold_optimized.pkl"
            )
        
        with col3:
            if ecart_avec_seuil < 0:
                delta_color = "normal"
                ecart_text = f"-{abs(ecart_avec_seuil):.2f} points"
                interpretation = "Sous le seuil"
            else:
                delta_color = "inverse"
                ecart_text = f"+{ecart_avec_seuil:.2f} points"
                interpretation = "Au-dessus du seuil"
                
            st.metric(
                label="📈 Écart avec seuil",
                value=ecart_text,
                delta=interpretation,
                delta_color=delta_color,
                help="Distance par rapport au seuil de décision"
            )
        
        # Analyse de l'écart
        if abs(ecart_avec_seuil) < 1:
            st.warning(f"""
            ⚠️ **Client proche du seuil** : Écart de seulement {abs(ecart_avec_seuil):.2f} points
            → Décision sensible aux variations des données
            """)
        elif ecart_avec_seuil < -5:
            st.success(f"""
            ✅ **Profil très sûr** : {abs(ecart_avec_seuil):.2f} points sous le seuil 
            → Risque très faible
            """)
        elif ecart_avec_seuil > 5:
            st.error(f"""
            ❌ **Profil très risqué** : {ecart_avec_seuil:.2f} points au-dessus du seuil 
            → Risque élevé
            """)
        
        # WCAG 1.1.1 : Texte alternatif
        st.markdown(f"""
        **Description graphique :** Jauge de risque affichant {probability:.1%} de probabilité de défaut de paiement.
        Le seuil de décision est fixé à {threshold:.1%} (ligne rouge). Ce client se situe dans la zone {'à risque ' if probability >= threshold else 'verte (risque faible)'}.
        Écart avec le seuil : {ecart_avec_seuil:+.2f} points.
        """)
    
    @staticmethod
    def render_feature_importance(result: PredictionResult, client_data: ClientData):
        """Affiche l'importance des variables avec graphique et tableau détaillé"""
        if not result.feature_importance:
            st.warning("Explications des variables non disponibles")
            return
        
        st.markdown("#### 🔍 Interprétation de la décision")
        
        # Créer données complètes pour toutes les variables
        all_features_data = []
        client_dict = client_data.to_dict()
        
        # Variables avec impact SHAP
        for feature in result.feature_importance:
            all_features_data.append({
                'feature': feature.feature,
                'feature_fr': Config.FEATURE_TRANSLATIONS.get(feature.feature, feature.feature),
                'shap_value': feature.shap_value,
                'client_value': client_dict.get(feature.feature, 0),
                'impact': feature.impact
            })
        
        # Ajouter les variables restantes
        remaining_features = [
            'EXT_SOURCE_1', 'EXT_SOURCE_2', 'DAYS_EMPLOYED',
            'NAME_EDUCATION_TYPE_Higher_education', 'INSTAL_AMT_PAYMENT_SUM'
        ]
        
        for feature_name in remaining_features:
            if not any(f['feature'] == feature_name for f in all_features_data):
                client_value = client_dict.get(feature_name, 0)
                all_features_data.append({
                    'feature': feature_name,
                    'feature_fr': Config.FEATURE_TRANSLATIONS.get(feature_name, feature_name),
                    'shap_value': 0.0,
                    'client_value': client_value,
                    'impact': "Impact neutre"
                })
        
        # Créer DataFrame
        features_df = pd.DataFrame(all_features_data)
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
        
        st.plotly_chart(fig, use_container_width=True, config=Config.PLOTLY_CONFIG)
        
        # WCAG 1.1.1 : Texte alternatif
        positive_features = [f['feature_fr'] for f in all_features_data if f['shap_value'] > 0]
        negative_features = [f['feature_fr'] for f in all_features_data if f['shap_value'] < 0]
        
        st.markdown(f"""
        **Description graphique :** Graphique en barres horizontales montrant l'impact de chaque variable sur la décision.
        Variables augmentant le risque (barres rouges) : {', '.join(positive_features[:3]) if positive_features else 'Aucune'}.
        Variables diminuant le risque (barres vertes) : {', '.join(negative_features[:3]) if negative_features else 'Aucune'}.
        """)
        
        # Tableau détaillé
        with st.expander("📋 Tableau détaillé", expanded=True):
            table_data = []
            for _, row in features_df.iterrows():
                feature_name = row['feature']
                client_val = row['client_value']
                
                # Formater la valeur client
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


class PopulationComparison:
    """Composant pour les comparaisons avec la population"""
    
    def __init__(self, api_service: APIService):
        self.api_service = api_service
    
    def render(self, client_data: ClientData):
        """Interface comparaison population avec contrôle API"""
        col1, col2 = st.columns([3, 1])
        
        with col1:
            selected_variable = st.selectbox(
                "Variable à analyser :",
                Config.DASHBOARD_FEATURES,
                format_func=lambda x: Config.FEATURE_TRANSLATIONS.get(x, x),
                key="population_variable_select"
            )
        
        with col2:
            if st.button("📊 Charger données", help="Charger les données de cette variable", key="load_population_btn"):
                with st.spinner("🔄 Chargement des données population..."):
                    distribution_data = self.api_service.get_population_distribution(selected_variable)
                
                if distribution_data:
                    client_value = getattr(client_data, selected_variable)
                    
                    if client_value is not None:
                        st.session_state[f'population_data_{selected_variable}'] = distribution_data
                        st.success(f"✅ Données chargées pour {Config.FEATURE_TRANSLATIONS.get(selected_variable, selected_variable)}")
                        self._create_population_plot(distribution_data, client_value, selected_variable)
                    else:
                        st.error(f"Valeur client manquante pour {selected_variable}")
                else:
                    st.error(f"Impossible de charger les données pour {selected_variable}")
        
        # Afficher données en cache si disponibles
        cache_key = f'population_data_{selected_variable}'
        if cache_key in st.session_state:
            st.info("📋 Données en cache - Cliquez sur 'Charger données' pour actualiser")
            client_value = getattr(client_data, selected_variable)
            if client_value is not None:
                self._create_population_plot(st.session_state[cache_key], client_value, selected_variable)
    
    def _create_population_plot(self, distribution_data: Dict, client_value: Any, variable_name: str):
        """Créer histogramme simple : distribution population + ligne client"""
        values = distribution_data.get('values', [])
        
        if not values:
            st.error(f"Aucune donnée disponible pour {variable_name}")
            return
        
        # Conversion spéciale pour variables catégorielles
        if variable_name == 'CODE_GENDER':
            client_value_numeric = 1 if client_value == 'M' else 0
        elif variable_name == 'NAME_EDUCATION_TYPE_Higher_education':
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
            name='Population',
            showlegend=False
        ))
        
        # Ligne verticale rouge pour le client
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
        
        # Configuration du graphique
        layout_config = {
            'title': f"{Config.FEATURE_TRANSLATIONS.get(variable_name, variable_name)}",
            'xaxis': {'title': f"{Config.FEATURE_TRANSLATIONS.get(variable_name, variable_name)}"},
            'yaxis': {'title': "Nombre de clients"},
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
        st.plotly_chart(fig, use_container_width=True, config=Config.PLOTLY_CONFIG)
        
        # WCAG 1.1.1 : Texte alternatif
        variable_fr = Config.FEATURE_TRANSLATIONS.get(variable_name, variable_name)
        
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


class BivariateAnalysis:
    """Composant pour l'analyse bi-variée"""
    
    def __init__(self, api_service: APIService):
        self.api_service = api_service
    
    def render(self, client_data: ClientData):
        """Interface d'analyse bi-variée"""
        col1, col2 = st.columns(2)
        
        with col1:
            var1 = st.selectbox(
                "Variable 1",
                Config.DASHBOARD_FEATURES,
                format_func=lambda x: Config.FEATURE_TRANSLATIONS.get(x, x),
                key="bivariate_var1"
            )
        
        with col2:
            var2 = st.selectbox(
                "Variable 2",
                Config.DASHBOARD_FEATURES,
                index=1,
                format_func=lambda x: Config.FEATURE_TRANSLATIONS.get(x, x),
                key="bivariate_var2"
            )
        
        if st.button("📈 Analyser Relation", use_container_width=True, key="analyze_bivariate_btn"):
            with st.spinner("🔄 Analyse bi-variée en cours..."):
                dist1 = self.api_service.get_population_distribution(var1)
                dist2 = self.api_service.get_population_distribution(var2)
            
            if dist1 and dist2:
                self._create_bivariate_plot(dist1, dist2, var1, var2, client_data)
            else:
                st.error("Impossible de charger les données pour l'analyse bi-variée")
        
        # Afficher analyse en cache
        cache_key = f'bivariate_{var1}_{var2}'
        if cache_key in st.session_state:
            cached_data = st.session_state[cache_key]
            if cached_data['var1'] == var1 and cached_data['var2'] == var2:
                st.info("📋 Analyse en cache - Cliquez sur 'Analyser Relation' pour actualiser")
                self._display_cached_bivariate(cached_data, client_data)
    
    def _create_bivariate_plot(self, dist1: Dict, dist2: Dict, var1: str, var2: str, client_data: ClientData):
        """Créer le graphique bi-varié"""
        values1 = dist1.get('values', [])
        values2 = dist2.get('values', [])
        
        if not values1 or not values2:
            st.error("Données insuffisantes pour l'analyse")
            return
        
        # Conversion pour variables catégorielles
        if var1 == 'NAME_EDUCATION_TYPE_Higher_education':
            values1 = [1 if v else 0 for v in values1]
        if var2 == 'NAME_EDUCATION_TYPE_Higher_education':
            values2 = [1 if v else 0 for v in values2]
        
        # Assurer même longueur
        min_len = min(len(values1), len(values2))
        x_data = values1[:min_len]
        y_data = values2[:min_len]
        
        # Stocker en cache
        cache_key = f'bivariate_{var1}_{var2}'
        st.session_state[cache_key] = {
            'x_data': x_data,
            'y_data': y_data,
            'var1': var1,
            'var2': var2
        }
        
        self._display_bivariate_plot(x_data, y_data, var1, var2, client_data)
    
    def _display_bivariate_plot(self, x_data: List, y_data: List, var1: str, var2: str, client_data: ClientData):
        """Afficher le graphique bi-varié"""
        # Graphique de corrélation
        fig = px.scatter(
            x=x_data,
            y=y_data,
            title=f"Relation entre {Config.FEATURE_TRANSLATIONS.get(var1, var1)} et {Config.FEATURE_TRANSLATIONS.get(var2, var2)}",
            labels={
                'x': Config.FEATURE_TRANSLATIONS.get(var1, var1),
                'y': Config.FEATURE_TRANSLATIONS.get(var2, var2)
            },
            opacity=0.6,
            color_discrete_sequence=['lightblue']
        )
        
        # Position du client
        client_x = getattr(client_data, var1)
        client_y = getattr(client_data, var2)
        
        # Conversion pour variables catégorielles
        if var1 == 'NAME_EDUCATION_TYPE_Higher_education':
            client_x = 1 if client_x == 1 else 0
        if var2 == 'NAME_EDUCATION_TYPE_Higher_education':
            client_y = 1 if client_y == 1 else 0
        if var1 == 'CODE_GENDER':
            client_x = 1 if client_x == 'M' else 0
        if var2 == 'CODE_GENDER':
            client_y = 1 if client_y == 'M' else 0
        
        # Lignes de position client
        fig.add_vline(
            x=client_x,
            line_dash="dash",
            line_color="red",
            line_width=3,
            annotation_text=f"📍 Client: {Config.FEATURE_TRANSLATIONS.get(var1, var1)}",
            annotation_position="top"
        )
        
        fig.add_hline(
            y=client_y,
            line_dash="dash",
            line_color="red",
            line_width=3,
            annotation_text=f"📍 Client: {Config.FEATURE_TRANSLATIONS.get(var2, var2)}",
            annotation_position="right"
        )
        
        fig.update_layout(height=500, showlegend=False)
        st.plotly_chart(fig, use_container_width=True, config=Config.PLOTLY_CONFIG)
        
        # Analyse statistique
        correlation = np.corrcoef(x_data, y_data)[0, 1] if len(x_data) > 1 else 0
        var1_fr = Config.FEATURE_TRANSLATIONS.get(var1, var1)
        var2_fr = Config.FEATURE_TRANSLATIONS.get(var2, var2)
        
        st.markdown(f"""
        **Description graphique :** Nuage de points montrant la relation entre {var1_fr} (axe horizontal) et {var2_fr} (axe vertical).
        Chaque point bleu représente un client de la population. Les lignes rouges en pointillés indiquent la position du client analysé : 
        ligne verticale à {var1_fr} = {client_x}, ligne horizontale à {var2_fr} = {client_y}.
        Le croisement des deux lignes localise précisément le client dans la distribution.
        Corrélation générale : {correlation:.3f}.
        {'Relation positive' if correlation > 0.3 else 'Relation négative' if correlation < -0.3 else 'Relation faible'} entre les deux variables.
        """)
        
        # Position du client
        percentile_x = sum(1 for val in x_data if val <= client_x) / len(x_data) * 100
        percentile_y = sum(1 for val in y_data if val <= client_y) / len(y_data) * 100
        
        st.info(f"""
        📍 **Position du client dans la population :**
        • {var1_fr} : {percentile_x:.0f}e percentile (ligne verticale rouge)
        • {var2_fr} : {percentile_y:.0f}e percentile (ligne horizontale rouge)
        • **Croisement** : intersection des deux lignes = position exacte du client
        """)
        
        st.success(f"✅ Analyse terminée - Corrélation: {correlation:.3f}")
    
    def _display_cached_bivariate(self, cached_data: Dict, client_data: ClientData):
        """Afficher l'analyse bi-variée depuis le cache"""
        self._display_bivariate_plot(
            cached_data['x_data'],
            cached_data['y_data'],
            cached_data['var1'],
            cached_data['var2'],
            client_data
        )


class Footer:
    """Composant footer"""
    
    @staticmethod
    def render():
        """Affiche le footer"""
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


# =============================================================================
# APPLICATION PRINCIPALE
# =============================================================================

class CreditScoringDashboard:
    """Application principale du dashboard"""
    
    def __init__(self):
        self.api_service = APIService(Config.API_URL)
        self.session_manager = SessionManager()
        
    def run(self):
        """Lance l'application"""
        # Configuration Streamlit
        st.set_page_config(
            page_title="Dashboard Credit Scoring - Prêt à dépenser",
            page_icon="🏦",
            layout="wide",
            initial_sidebar_state="expanded"
        )
        
        # Initialisation
        StyleManager.load_styles()
        self.session_manager.init_session()
        
        # Vérification API
        api_ok, api_info, api_error = self.api_service.test_connection()
        if not api_ok:
            st.error(f"⚠️ **API non accessible**: {api_error}")
            st.stop()
        
        # Interface principale
        UIComponents.render_header()
        self._render_sidebar(api_info)
        self._render_main_content()
        Footer.render()
    
    def _render_sidebar(self, api_info: Optional[Dict]):
        """Affiche la barre latérale"""
        with st.sidebar:
            st.markdown("**🏦 Dashboard Credit Scoring**")
            st.markdown("---")
            st.markdown("### 📋 Navigation")
            
            if st.button("🆕 Nouveau client", use_container_width=True):
                self.session_manager.reset_session()
                st.rerun()
            
            st.markdown("---")
            st.markdown("**📊 Statut API**")
            if api_info:
                st.success("✅ Connectée")
                st.caption(f"Version: {api_info.get('version', 'N/A')}")
            else:
                st.error("❌ Déconnectée")
    
    def _render_main_content(self):
        """Affiche le contenu principal"""
        if not st.session_state.client_analyzed:
            self._render_client_input()
        else:
            self._render_results()
    
    def _render_client_input(self):
        """Affiche le formulaire de saisie client"""
        st.markdown("### 📝 Nouveau client")
        
        form = ClientForm()
        client_data = form.render()
        
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            if st.button(
                "🎯 ANALYSER CE CLIENT",
                type="primary",
                use_container_width=True,
                disabled=st.session_state.api_call_in_progress,
                key="analyze_client_btn"
            ):
                self._analyze_client(client_data)
    
    def _analyze_client(self, client_data: ClientData):
        """Analyse un client via l'API"""
        st.session_state.api_call_in_progress = True
        
        with st.spinner("🔄 Analyse en cours..."):
            result, error = self.api_service.predict(client_data)
        
        if result:
            st.session_state.client_data = client_data
            st.session_state.prediction_result = result
            st.session_state.client_analyzed = True
            st.session_state.last_analysis_time = time.time()
            st.session_state.api_call_in_progress = False
            st.success("✅ Client analysé avec succès !")
            st.rerun()
        else:
            st.session_state.api_call_in_progress = False
            st.error(f"❌ Erreur d'analyse : {error}")
    
    def _render_results(self):
        """Affiche les résultats d'analyse"""
        result = st.session_state.prediction_result
        client_data = st.session_state.client_data
        
        tab1, tab2, tab3 = st.tabs(["🎯 Résultats", "📊 Comparaisons", "🔧 Analyses bi-variées"])
        
        with tab1:
            st.markdown("### 🎯 Résultat de l'analyse")
            
            col1, col2 = st.columns([3, 1])
            with col2:
                if st.button("🔧 Modifier", use_container_width=True):
                    st.session_state.client_analyzed = False
                    st.session_state.api_call_in_progress = False
                    st.rerun()
            
            # Profil client
            ResultsDisplay.render_client_profile(client_data)
            st.markdown("---")
            
            # Résultat scoring
            ResultsDisplay.render_prediction_result(result)
            st.markdown("---")
            
            # Feature importance
            ResultsDisplay.render_feature_importance(result, client_data)
        
        with tab2:
            st.markdown("### 📊 Comparaisons avec la base clients")
            population_comp = PopulationComparison(self.api_service)
            population_comp.render(client_data)
        
        with tab3:
            st.markdown("### 🔧 Analyse bi-variée")
            bivariate_analysis = BivariateAnalysis(self.api_service)
            bivariate_analysis.render(client_data)


# =============================================================================
# POINT D'ENTRÉE
# =============================================================================

if __name__ == "__main__":
    app = CreditScoringDashboard()
    app.run()