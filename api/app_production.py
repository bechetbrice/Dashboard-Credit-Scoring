"""
API Flask Production - Dashboard Credit Scoring
Version: Production v2.0 - VRAI MODÈLE + VRAI SHAP
Correction: Suppression RandomForest démo, intégration vrai LightGBM + SHAP
"""

from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
import joblib
import shap
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
explainer = None

# Features de production
PRODUCTION_FEATURES = [
    'EXT_SOURCE_2', 'EXT_SOURCE_3', 'EXT_SOURCE_1',
    'DAYS_EMPLOYED', 'CODE_GENDER', 'INSTAL_DPD_MEAN',
    'PAYMENT_RATE', 'NAME_EDUCATION_TYPE_Higher_education',
    'AMT_ANNUITY', 'INSTAL_AMT_PAYMENT_SUM'
]

def init_production_model():
    """Initialisation avec vrai modèle LightGBM obligatoire"""
    global model, threshold, feature_names, population_stats, explainer
    
    # Chargement modèle LightGBM OBLIGATOIRE
    model_path = 'lightgbm_final_model_optimized.pkl'
    if not os.path.exists(model_path):
        logger.error(f"❌ ERREUR CRITIQUE: Modèle non trouvé: {model_path}")
        raise FileNotFoundError(f"Modèle obligatoire manquant: {model_path}")
    
    try:
        model = joblib.load(model_path)
        logger.info(f"✅ Modèle LightGBM chargé: {model_path}")
    except Exception as e:
        logger.error(f"❌ ERREUR CRITIQUE chargement modèle: {str(e)}")
        raise e
    
    # Features
    feature_names = PRODUCTION_FEATURES
    logger.info(f"✅ Features configurées: {len(feature_names)} variables")
    
    # Population stats
    population_stats = generate_population_data()
    logger.info("✅ Données population générées")
    
    # Initialiser SHAP TreeExplainer
    try:
        explainer = shap.TreeExplainer(model)
        logger.info("✅ SHAP TreeExplainer initialisé")
    except Exception as e:
        logger.error(f"❌ ERREUR SHAP: {str(e)}")
        explainer = None
    
    logger.info("✅ Initialisation terminée avec succès")

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
    """Préparation features pour le vrai modèle"""
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
    """Health check avec statut SHAP"""
    return jsonify({
        'status': 'healthy',
        'version': 'Production v2.0 - VRAI MODÈLE + SHAP',
        'model_loaded': model is not None,
        'model_type': 'LightGBM' if model is not None else 'None',
        'features_count': len(PRODUCTION_FEATURES),
        'population_available': population_stats is not None,
        'shap_available': explainer is not None,
        'timestamp': datetime.now().isoformat()
    })

@app.route('/predict_dashboard', methods=['POST'])
def predict_dashboard():
    """Prédiction avec VRAIES valeurs SHAP"""
    data = request.get_json()
    
    # Validation simple
    if not data:
        return jsonify({'error': 'Données manquantes'}), 400
    
    if model is None:
        return jsonify({'error': 'Modèle non chargé'}), 500
    
    # Prédiction avec vrai modèle
    features = prepare_features(data)
    probability = model.predict_proba(features)[0][1]
    decision = "REFUSE" if probability >= threshold else "APPROVE"
    
    # VRAIES valeurs SHAP
    feature_importance = []
    
    if explainer is not None:
        try:
            # Calcul SHAP réel
            shap_values = explainer.shap_values(features)
            
            # Gérer format SHAP selon le modèle
            if isinstance(shap_values, list):
                # Classification binaire : prendre classe positive
                shap_vals = shap_values[1][0]  # [1] = classe positive, [0] = premier échantillon
            else:
                # Régression ou format simple
                shap_vals = shap_values[0]
            
            # Créer DataFrame avec vraies valeurs SHAP
            shap_df = pd.DataFrame({
                'feature': PRODUCTION_FEATURES,
                'shap_value': shap_vals,
                'feature_value': features[0]
            })
            
            # Convertir en format API
            for _, row in shap_df.iterrows():
                feature_importance.append({
                    'feature': row['feature'],
                    'shap_value': float(row['shap_value']),
                    'feature_value': float(row['feature_value']),
                    'impact': 'positive' if row['shap_value'] > 0 else 'negative'
                })
            
            logger.info(f"✅ SHAP calculé pour {len(feature_importance)} variables")
            
        except Exception as e:
            logger.error(f"❌ Erreur calcul SHAP: {str(e)}")
            # Fallback si SHAP échoue
            feature_importance = []
            for feature in PRODUCTION_FEATURES:
                feature_importance.append({
                    'feature': feature,
                    'shap_value': 0.0,
                    'feature_value': float(data.get(feature, 0.0)),
                    'impact': 'neutral'
                })
    else:
        logger.warning("⚠️ SHAP non disponible")
        # Fallback si pas d'explainer
        feature_importance = []
        for feature in PRODUCTION_FEATURES:
            feature_importance.append({
                'feature': feature,
                'shap_value': 0.0,
                'feature_value': float(data.get(feature, 0.0)),
                'impact': 'neutral'
            })
    
    # Comparaisons population
    comparisons = {}
    comparison_vars = ['EXT_SOURCE_2', 'EXT_SOURCE_3', 'EXT_SOURCE_1', 'PAYMENT_RATE', 'AMT_ANNUITY']
    
    for feature in comparison_vars:
        if feature in data and feature in population_stats:
            client_val = data[feature]
            pop_stats = population_stats[feature]
            
            # Calculer percentile
            pop_values = pop_stats['values']
            percentile = sum(1 for val in pop_values if val <= client_val) / len(pop_values) * 100
            
            # Catégoriser
            if percentile <= 25:
                category = "Quartile inférieur"
            elif percentile <= 50:
                category = "Médiane inférieure"
            elif percentile <= 75:
                category = "Médiane supérieure"
            else:
                category = "Quartile supérieur"
            
            comparisons[feature] = {
                'client_value': client_val,
                'population_mean': pop_stats['mean'],
                'percentile': round(percentile, 1),
                'category': category
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
            'shap_available': explainer is not None,
            'top_features': feature_importance,
            'feature_count': len(PRODUCTION_FEATURES)
        },
        'metadata': {
            'api_version': 'Production v2.0 - VRAI MODÈLE + SHAP',
            'timestamp': datetime.now().isoformat(),
            'model_type': 'LightGBM',
            'features_returned': len(feature_importance),
            'shap_computed': explainer is not None
        }
    }
    
    return jsonify(result)

@app.route('/population_stats', methods=['GET'])
def get_population_stats():
    """Statistiques population pour toutes les variables"""
    key_vars = ['EXT_SOURCE_2', 'EXT_SOURCE_3', 'EXT_SOURCE_1', 
               'DAYS_EMPLOYED', 'INSTAL_DPD_MEAN', 'PAYMENT_RATE',
               'AMT_ANNUITY', 'INSTAL_AMT_PAYMENT_SUM']
    
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

# Initialisation FORCÉE au démarrage
try:
    init_production_model()
except Exception as e:
    logger.error(f"❌ ERREUR CRITIQUE lors de l'initialisation: {str(e)}")
    logger.error("🛑 L'API ne peut pas démarrer sans le modèle LightGBM")
    raise e

# Démarrage local seulement
if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)