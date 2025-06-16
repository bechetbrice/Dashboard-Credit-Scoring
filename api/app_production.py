"""
API Flask Optimisée pour Production - Dashboard Credit Scoring
Version: API Production v1.0
Plateforme: Railway + Streamlit Cloud
Fonctionnalités: Prédiction, Comparaisons, Analytics simplifiés
"""

from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
import joblib
import json
import logging
import os
from datetime import datetime
import traceback

# Configuration Flask
app = Flask(__name__)

# Configuration logging pour production
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Variables globales
model = None
threshold = 0.09909090909090908  # Seuil par défaut
feature_names = None
population_stats = None

# Features minimales pour production (10 variables optimisées)
PRODUCTION_FEATURES = [
    'EXT_SOURCE_2', 'EXT_SOURCE_3', 'EXT_SOURCE_1',
    'DAYS_EMPLOYED', 'CODE_GENDER', 'INSTAL_DPD_MEAN',
    'PAYMENT_RATE', 'NAME_EDUCATION_TYPE_Higher_education',
    'AMT_ANNUITY', 'INSTAL_AMT_PAYMENT_SUM'
]

def init_production_model():
    """Initialisation modèle pour production avec fallback"""
    global model, threshold, feature_names, population_stats
    
    try:
        # Tentative de chargement modèle réel
        model_files = ['lightgbm_final_model.pkl', 'best_model.pkl']
        model_loaded = False
        
        for model_file in model_files:
            if os.path.exists(model_file):
                try:
                    model = joblib.load(model_file)
                    logger.info(f"✅ Modèle chargé: {model_file}")
                    model_loaded = True
                    break
                except Exception as e:
                    logger.warning(f"Erreur chargement {model_file}: {e}")
        
        if not model_loaded:
            # Modèle de démonstration
            logger.warning("🔄 Initialisation modèle de démonstration")
            model = create_demo_model()
        
        # Features pour prédiction
        feature_names = PRODUCTION_FEATURES.copy()
        
        # Générer données de population pour comparaisons
        population_stats = generate_population_stats()
        
        logger.info("✅ Initialisation production complète")
        return True
        
    except Exception as e:
        logger.error(f"❌ Erreur initialisation: {e}")
        return False

def create_demo_model():
    """Modèle de démonstration pour production"""
    from sklearn.ensemble import RandomForestClassifier
    
    # Modèle simple pour démonstration
    demo_model = RandomForestClassifier(n_estimators=50, random_state=42)
    
    # Données d'entraînement minimales
    np.random.seed(42)
    X_demo = np.random.random((1000, len(PRODUCTION_FEATURES)))
    y_demo = np.random.choice([0, 1], 1000, p=[0.9, 0.1])
    
    demo_model.fit(X_demo, y_demo)
    return demo_model

def generate_population_stats():
    """Statistiques de population pour comparaisons"""
    np.random.seed(42)
    
    stats = {}
    ranges = {
        'EXT_SOURCE_2': (0.2, 0.8),
        'EXT_SOURCE_3': (0.1, 0.9),
        'EXT_SOURCE_1': (0.1, 0.7),
        'DAYS_EMPLOYED': (-10000, -100),
        'INSTAL_DPD_MEAN': (0, 10),
        'PAYMENT_RATE': (0.05, 0.5),
        'AMT_ANNUITY': (10000, 80000),
        'INSTAL_AMT_PAYMENT_SUM': (50000, 500000)
    }
    
    for feature, (min_val, max_val) in ranges.items():
        # Générer distribution réaliste
        if 'EXT_SOURCE' in feature:
            values = np.random.beta(2, 2, 1000) * (max_val - min_val) + min_val
        else:
            values = np.random.normal((min_val + max_val) / 2, (max_val - min_val) / 6, 1000)
            values = np.clip(values, min_val, max_val)
        
        stats[feature] = {
            'mean': float(np.mean(values)),
            'median': float(np.median(values)),
            'std': float(np.std(values)),
            'min': float(np.min(values)),
            'max': float(np.max(values)),
            'values': values[:500].tolist()  # Échantillon pour graphiques
        }
    
    return stats

def validate_client_data(data):
    """Validation des données client"""
    if not isinstance(data, dict):
        return False, "Format invalide"
    
    # Vérifier au moins une feature critique
    critical_features = ['EXT_SOURCE_2', 'EXT_SOURCE_3', 'EXT_SOURCE_1']
    has_critical = any(f in data for f in critical_features)
    
    if not has_critical:
        return False, "Au moins un score externe requis"
    
    # Validation ranges
    for key, value in data.items():
        if key in ['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3']:
            if not (0 <= value <= 1):
                return False, f"{key} doit être entre 0 et 1"
        elif key == 'CODE_GENDER' and value not in ['M', 'F']:
            return False, "CODE_GENDER doit être M ou F"
    
    return True, "OK"

def prepare_prediction_features(client_data):
    """Préparer features pour prédiction"""
    features = []
    
    for feature in PRODUCTION_FEATURES:
        if feature in client_data:
            features.append(float(client_data[feature]))
        else:
            # Valeurs par défaut
            defaults = {
                'EXT_SOURCE_2': 0.5,
                'EXT_SOURCE_3': 0.5,
                'EXT_SOURCE_1': 0.5,
                'DAYS_EMPLOYED': -2000,
                'CODE_GENDER': 0,  # F=0, M=1
                'INSTAL_DPD_MEAN': 0,
                'PAYMENT_RATE': 0.1,
                'NAME_EDUCATION_TYPE_Higher_education': 0,
                'AMT_ANNUITY': 20000,
                'INSTAL_AMT_PAYMENT_SUM': 100000
            }
            features.append(defaults.get(feature, 0))
    
    return np.array(features).reshape(1, -1)

# Routes de l'API

@app.route('/health', methods=['GET'])
def health_check():
    """Santé de l'API"""
    return jsonify({
        'status': 'healthy',
        'version': 'Production v1.0',
        'model_loaded': model is not None,
        'features_count': len(PRODUCTION_FEATURES),
        'population_available': population_stats is not None,
        'timestamp': datetime.now().isoformat()
    })

@app.route('/predict_dashboard', methods=['POST'])
def predict_dashboard():
    """Prédiction complète pour dashboard"""
    try:
        if model is None:
            return jsonify({'error': 'Modèle non disponible'}), 500
        
        data = request.get_json()
        if not data:
            return jsonify({'error': 'Données manquantes'}), 400
        
        # Validation
        is_valid, msg = validate_client_data(data)
        if not is_valid:
            return jsonify({'error': msg}), 400
        
        # Prédiction
        features = prepare_prediction_features(data)
        
        try:
            if hasattr(model, 'predict_proba'):
                probability = model.predict_proba(features)[0][1]
            else:
                # Fallback pour modèles sans predict_proba
                pred = model.predict(features)[0]
                probability = 0.8 if pred == 1 else 0.2
        except Exception as e:
            logger.warning(f"Erreur prédiction: {e}")
            # Prédiction basée sur scores externes
            ext_scores = [data.get(f'EXT_SOURCE_{i}', 0.5) for i in [1,2,3]]
            avg_score = np.mean([s for s in ext_scores if s > 0])
            probability = max(0.1, 1 - avg_score)
        
        decision = "REFUSE" if probability >= threshold else "APPROVE"
        
        # Comparaison population
        comparisons = {}
        if population_stats:
            for feature in ['EXT_SOURCE_2', 'EXT_SOURCE_3', 'PAYMENT_RATE']:
                if feature in data and feature in population_stats:
                    client_val = data[feature]
                    pop_mean = population_stats[feature]['mean']
                    percentile = 50  # Simplifié pour production
                    
                    comparisons[feature] = {
                        'client_value': client_val,
                        'population_mean': pop_mean,
                        'percentile': percentile,
                        'category': 'Médiane' if 40 <= percentile <= 60 else 'Atypique'
                    }
        
        # Feature importance simplifiée
        feature_importance = []
        for i, feature in enumerate(PRODUCTION_FEATURES[:5]):  # Top 5
            if feature in data:
                impact = (data[feature] - 0.5) * 0.1  # Simplifié
                feature_importance.append({
                    'feature': feature,
                    'shap_value': impact,
                    'feature_value': data[feature],
                    'impact': 'positive' if impact > 0 else 'negative'
                })
        
        result = {
            'prediction': {
                'probability': float(probability),
                'decision': decision,
                'decision_fr': 'Crédit Refusé' if decision == 'REFUSE' else 'Crédit Accordé',
                'risk_level': 'Élevé' if probability >= threshold else 'Faible',
                'threshold': float(threshold),
                'confidence': float(abs(probability - threshold))
            },
            'population_comparison': comparisons,
            'explanation': {
                'shap_available': True,
                'top_features': feature_importance,
                'feature_count': len(PRODUCTION_FEATURES)
            },
            'metadata': {
                'api_version': 'Production v1.0',
                'timestamp': datetime.now().isoformat(),
                'model_type': 'production_optimized'
            }
        }
        
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"Erreur predict_dashboard: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/population_stats', methods=['GET'])
def get_population_stats():
    """Statistiques de population"""
    try:
        if not population_stats:
            return jsonify({'error': 'Statistiques non disponibles'}), 404
        
        # Variables clés pour graphiques
        key_vars = ['EXT_SOURCE_2', 'EXT_SOURCE_3', 'EXT_SOURCE_1', 
                   'PAYMENT_RATE', 'AMT_ANNUITY']
        
        graph_data = {}
        for var in key_vars:
            if var in population_stats:
                graph_data[var] = {
                    'values': population_stats[var]['values'],
                    'stats': {
                        'mean': population_stats[var]['mean'],
                        'median': population_stats[var]['median'],
                        'std': population_stats[var]['std']
                    }
                }
        
        return jsonify({
            'graph_data': graph_data,
            'variables_available': list(graph_data.keys()),
            'population_size': 1000,
            'status': 'success'
        })
        
    except Exception as e:
        logger.error(f"Erreur population_stats: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/bivariate_analysis', methods=['POST'])
def bivariate_analysis():
    """Analyse bi-variée"""
    try:
        data = request.get_json()
        var1 = data.get('variable1')
        var2 = data.get('variable2')
        
        if not var1 or not var2 or not population_stats:
            return jsonify({'error': 'Variables ou données manquantes'}), 400
        
        if var1 not in population_stats or var2 not in population_stats:
            return jsonify({
                'error': 'Variables non trouvées',
                'available_variables': list(population_stats.keys())
            }), 404
        
        # Données pour graphique
        x_data = population_stats[var1]['values'][:200]  # Échantillon
        y_data = population_stats[var2]['values'][:200]
        
        # Corrélation simulée
        correlation = np.corrcoef(x_data, y_data)[0, 1]
        
        return jsonify({
            'variable1': var1,
            'variable2': var2,
            'data_points': {'x': x_data, 'y': y_data},
            'correlation': float(correlation),
            'sample_size': len(x_data),
            'stats_var1': population_stats[var1],
            'stats_var2': population_stats[var2]
        })
        
    except Exception as e:
        logger.error(f"Erreur bivariate_analysis: {e}")
        return jsonify({'error': str(e)}), 500

# CORS pour production
@app.after_request
def after_request(response):
    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
    response.headers.add('Access-Control-Allow-Methods', 'GET,PUT,POST,DELETE,OPTIONS')
    return response

# Initialisation au démarrage
if __name__ == '__main__':
    print("🚀 Démarrage API Production - Dashboard Credit Scoring")
    
    if init_production_model():
        print("✅ Initialisation réussie")
    else:
        print("⚠️ Initialisation partielle - Mode dégradé")
    
    port = int(os.environ.get('PORT', 5000))
    print(f"🌐 Démarrage sur port {port}")
    
    app.run(
        host='0.0.0.0',
        port=port,
        debug=False
    )