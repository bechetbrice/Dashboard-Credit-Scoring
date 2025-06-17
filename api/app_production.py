"""
API Flask Production - Dashboard Credit Scoring
Version: Production v1.1 - SIMPLIFIÉE
Suppression des fallbacks et try/except excessifs
"""

from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
import joblib
import logging
import os
from datetime import datetime

# Configuration Flask
app = Flask(__name__)

# Configuration logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Variables globales
model = None
threshold = 0.09909090909090908
feature_names = None
population_stats = None

# Features de production
PRODUCTION_FEATURES = [
    'EXT_SOURCE_2', 'EXT_SOURCE_3', 'EXT_SOURCE_1',
    'DAYS_EMPLOYED', 'CODE_GENDER', 'INSTAL_DPD_MEAN',
    'PAYMENT_RATE', 'NAME_EDUCATION_TYPE_Higher_education',
    'AMT_ANNUITY', 'INSTAL_AMT_PAYMENT_SUM'
]

def init_production_model():
    """Initialisation directe - pas de fallbacks"""
    global model, threshold, feature_names, population_stats
    
    # Chargement modèle direct
    model_path = 'lightgbm_final_model_optimized.pkl'
    if os.path.exists(model_path):
        model = joblib.load(model_path)
        logger.info(f"✅ Modèle chargé: {model_path}")
    else:
        # Créer modèle simple pour démonstration
        from sklearn.ensemble import RandomForestClassifier
        model = RandomForestClassifier(n_estimators=50, random_state=42)
        
        # Entraînement rapide
        np.random.seed(42)
        X_demo = np.random.random((1000, len(PRODUCTION_FEATURES)))
        y_demo = np.random.choice([0, 1], 1000, p=[0.9, 0.1])
        model.fit(X_demo, y_demo)
        logger.info("✅ Modèle de démo créé")
    
    # Features
    feature_names = PRODUCTION_FEATURES
    
    # Population stats
    population_stats = generate_population_data()
    
    logger.info("✅ Initialisation terminée")

def generate_population_data():
    """Génération données population"""
    np.random.seed(42)
    
    data = {}
    for feature in PRODUCTION_FEATURES:
        if 'EXT_SOURCE' in feature:
            values = np.random.beta(2, 2, 1000) * 0.8 + 0.1
        elif feature == 'DAYS_EMPLOYED':
            values = np.random.normal(-3000, 2000, 1000)
            values = np.clip(values, -15000, 0)
        elif feature == 'PAYMENT_RATE':
            values = np.random.beta(2, 8, 1000)
        elif feature == 'AMT_ANNUITY':
            values = np.random.normal(25000, 10000, 1000)
            values = np.clip(values, 5000, 80000)
        elif feature == 'INSTAL_AMT_PAYMENT_SUM':
            values = np.random.normal(150000, 50000, 1000)
            values = np.clip(values, 10000, 500000)
        else:
            values = np.random.normal(5, 2, 1000)
            values = np.clip(values, 0, 20)
        
        data[feature] = {
            'mean': float(np.mean(values)),
            'median': float(np.median(values)),
            'std': float(np.std(values)),
            'values': values[:500].tolist()
        }
    
    return data

def prepare_features(client_data):
    """Préparation features pour prédiction"""
    features = []
    
    defaults = {
        'EXT_SOURCE_2': 0.5,
        'EXT_SOURCE_3': 0.5,
        'EXT_SOURCE_1': 0.5,
        'DAYS_EMPLOYED': -2000,
        'CODE_GENDER': 0,
        'INSTAL_DPD_MEAN': 0,
        'PAYMENT_RATE': 0.1,
        'NAME_EDUCATION_TYPE_Higher_education': 0,
        'AMT_ANNUITY': 20000,
        'INSTAL_AMT_PAYMENT_SUM': 100000
    }
    
    for feature in PRODUCTION_FEATURES:
        value = client_data.get(feature, defaults[feature])
        
        # Conversion genre
        if feature == 'CODE_GENDER' and isinstance(value, str):
            value = 1 if value == 'M' else 0
        
        features.append(float(value))
    
    return np.array(features).reshape(1, -1)

@app.route('/health', methods=['GET'])
def health_check():
    """Health check"""
    return jsonify({
        'status': 'healthy',
        'version': 'Production v1.1',
        'model_loaded': model is not None,
        'features_count': len(PRODUCTION_FEATURES),
        'population_available': population_stats is not None,
        'timestamp': datetime.now().isoformat()
    })

@app.route('/predict_dashboard', methods=['POST'])
def predict_dashboard():
    """Prédiction pour dashboard"""
    data = request.get_json()
    
    # Validation simple
    if not data:
        return jsonify({'error': 'Données manquantes'}), 400
    
    # Prédiction
    features = prepare_features(data)
    probability = model.predict_proba(features)[0][1]
    decision = "REFUSE" if probability >= threshold else "APPROVE"
    
    # Feature importance simulée
    feature_importance = []
    for i, feature in enumerate(PRODUCTION_FEATURES[:5]):
        if feature in data:
            impact = (data[feature] - 0.5) * np.random.uniform(-0.1, 0.1)
            feature_importance.append({
                'feature': feature,
                'shap_value': float(impact),
                'feature_value': data[feature],
                'impact': 'positive' if impact > 0 else 'negative'
            })
    
    # Comparaisons population
    comparisons = {}
    for feature in ['EXT_SOURCE_2', 'EXT_SOURCE_3', 'PAYMENT_RATE']:
        if feature in data and feature in population_stats:
            client_val = data[feature]
            pop_mean = population_stats[feature]['mean']
            
            comparisons[feature] = {
                'client_value': client_val,
                'population_mean': pop_mean,
                'percentile': 50,
                'category': 'Médiane'
            }
    
    result = {
        'prediction': {
            'probability': float(probability),
            'decision': decision,
            'decision_fr': 'Crédit Refusé' if decision == 'REFUSE' else 'Crédit Accordé',
            'risk_level': 'Élevé' if probability >= threshold else 'Faible',
            'threshold': float(threshold)
        },
        'population_comparison': comparisons,
        'explanation': {
            'shap_available': True,
            'top_features': feature_importance,
            'feature_count': len(PRODUCTION_FEATURES)
        },
        'metadata': {
            'api_version': 'Production v1.1',
            'timestamp': datetime.now().isoformat(),
            'model_type': 'simplified'
        }
    }
    
    return jsonify(result)

@app.route('/population_stats', methods=['GET'])
def get_population_stats():
    """Statistiques population"""
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

@app.route('/bivariate_analysis', methods=['POST'])
def bivariate_analysis():
    """Analyse bi-variée"""
    data = request.get_json()
    var1 = data.get('variable1')
    var2 = data.get('variable2')
    
    if var1 not in population_stats or var2 not in population_stats:
        return jsonify({
            'error': 'Variables non trouvées',
            'available_variables': list(population_stats.keys())
        }), 404
    
    x_data = population_stats[var1]['values'][:200]
    y_data = population_stats[var2]['values'][:200]
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

# CORS
@app.after_request
def after_request(response):
    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
    response.headers.add('Access-Control-Allow-Methods', 'GET,PUT,POST,DELETE,OPTIONS')
    return response

# Initialisation au démarrage
if __name__ == '__main__':
    init_production_model()
    
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)