"""
API Flask Production - Dashboard Credit Scoring
Version: Railway v3.0 - MODÈLE 234 FEATURES + SHAP
Correction: Utilisation complète du modèle avec toutes les features
"""

from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
import joblib
import shap
import json
import logging
import os
import time
from datetime import datetime
from pathlib import Path
from functools import lru_cache

# Configuration Flask
app = Flask(__name__)

# Configuration logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Chemins Railway
MODELS_DIR = Path(os.path.join(os.path.dirname(__file__), '..', 'models'))
DATA_DIR = Path(os.path.join(os.path.dirname(__file__), '..', 'data', 'processed'))

# Variables globales
model = None
threshold = None
feature_names = None
population_stats = None
explainer = None

# Features dashboard (10 variables saisissables)
DASHBOARD_FEATURES = [
    'EXT_SOURCE_2', 'EXT_SOURCE_3', 'EXT_SOURCE_1',
    'DAYS_EMPLOYED', 'CODE_GENDER', 'INSTAL_DPD_MEAN',
    'PAYMENT_RATE', 'NAME_EDUCATION_TYPE_Higher_education',
    'AMT_ANNUITY', 'INSTAL_AMT_PAYMENT_SUM'
]

@lru_cache(maxsize=1)
def load_model_cached():
    """Charger modèle avec cache"""
    return joblib.load(MODELS_DIR / "lightgbm_final_model_optimized.pkl")

@lru_cache(maxsize=1)
def load_threshold_cached():
    """Charger seuil avec cache"""
    return joblib.load(MODELS_DIR / "optimal_threshold_optimized.pkl")

@lru_cache(maxsize=1)
def load_features_cached():
    """Charger liste complète des 234 features"""
    with open(DATA_DIR / "final_features_list.json", 'r') as f:
        feature_data = json.load(f)
        if 'selected_features' in feature_data:
            return feature_data['selected_features']
        elif isinstance(feature_data, list):
            return feature_data
        else:
            return list(feature_data.keys())

def init_railway_model():
    """Initialisation modèle Railway avec 234 features"""
    global model, threshold, feature_names, population_stats, explainer
    
    logger.info("🚀 INITIALISATION RAILWAY avec modèle complet...")
    
    try:
        # Chargement modèle obligatoire
        model = load_model_cached()
        logger.info(f"✅ Modèle LightGBM chargé: {model.__class__.__name__}")
        
        # Chargement seuil
        threshold = load_threshold_cached()
        logger.info(f"✅ Seuil optimal: {threshold:.6f}")
        
        # Chargement features complètes
        feature_names = load_features_cached()
        logger.info(f"✅ Features chargées: {len(feature_names)} variables")
        
        # Vérification cohérence modèle
        if hasattr(model, 'n_features_'):
            model_features = model.n_features_
            logger.info(f"✅ Modèle attend: {model_features} features")
            
            if len(feature_names) != model_features:
                logger.warning(f"⚠️ Mismatch: {len(feature_names)} features chargées vs {model_features} attendues")
        
        # Population stats simplifiées
        population_stats = generate_population_data()
        logger.info("✅ Données population générées")
        
        # Initialiser SHAP TreeExplainer
        try:
            explainer = shap.TreeExplainer(model)
            logger.info("✅ SHAP TreeExplainer initialisé avec succès")
        except Exception as e:
            logger.error(f"❌ ERREUR SHAP: {str(e)}")
            explainer = None
        
        logger.info("✅ RAILWAY initialisé avec succès")
        return True
        
    except Exception as e:
        logger.error(f"❌ ERREUR CRITIQUE Railway: {str(e)}")
        raise e

def generate_population_data():
    """Génération données population pour comparaisons"""
    np.random.seed(42)
    
    data = {}
    # Seulement pour les variables d'interface
    for feature in DASHBOARD_FEATURES:
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

def prepare_features_complete(client_data):
    """Préparer vecteur complet de 234 features"""
    
    # Créer vecteur complet avec valeurs par défaut
    features_vector = {}
    
    # Valeurs par défaut pour toutes les features
    for feature_name in feature_names:
        if 'EXT_SOURCE' in feature_name:
            features_vector[feature_name] = 0.5
        elif 'DAYS_' in feature_name:
            if 'BIRTH' in feature_name:
                features_vector[feature_name] = -15000
            elif 'EMPLOYED' in feature_name:
                features_vector[feature_name] = -2000
            else:
                features_vector[feature_name] = -1000
        elif 'AMT_' in feature_name:
            if 'INCOME' in feature_name:
                features_vector[feature_name] = 150000
            elif 'CREDIT' in feature_name:
                features_vector[feature_name] = 500000
            elif 'ANNUITY' in feature_name:
                features_vector[feature_name] = 25000
            elif 'GOODS' in feature_name:
                features_vector[feature_name] = 450000
            else:
                features_vector[feature_name] = 50000
        elif 'CNT_' in feature_name:
            features_vector[feature_name] = 1
        elif feature_name.startswith('CODE_GENDER'):
            features_vector[feature_name] = 0
        elif feature_name.startswith('NAME_') or feature_name.startswith('FLAG_'):
            features_vector[feature_name] = 0
        else:
            features_vector[feature_name] = 0.0
    
    # Mapper les données client saisies
    dashboard_mapping = {
        'EXT_SOURCE_2': client_data.get('EXT_SOURCE_2', 0.5),
        'EXT_SOURCE_3': client_data.get('EXT_SOURCE_3', 0.5),
        'EXT_SOURCE_1': client_data.get('EXT_SOURCE_1', 0.5),
        'DAYS_EMPLOYED': client_data.get('DAYS_EMPLOYED', -2000),
        'INSTAL_DPD_MEAN': client_data.get('INSTAL_DPD_MEAN', 0),
        'PAYMENT_RATE': client_data.get('PAYMENT_RATE', 0.1),
        'NAME_EDUCATION_TYPE_Higher_education': client_data.get('NAME_EDUCATION_TYPE_Higher_education', 0),
        'AMT_ANNUITY': client_data.get('AMT_ANNUITY', 25000),
        'INSTAL_AMT_PAYMENT_SUM': client_data.get('INSTAL_AMT_PAYMENT_SUM', 100000)
    }
    
    # Gestion spéciale du genre
    gender = client_data.get('CODE_GENDER', 'F')
    if isinstance(gender, str):
        if gender in ['M', 'Homme', 'Male']:
            if 'CODE_GENDER_M' in features_vector:
                dashboard_mapping['CODE_GENDER_M'] = 1.0
            if 'CODE_GENDER_F' in features_vector:
                dashboard_mapping['CODE_GENDER_F'] = 0.0
        else:  # Femme par défaut
            if 'CODE_GENDER_M' in features_vector:
                dashboard_mapping['CODE_GENDER_M'] = 0.0
            if 'CODE_GENDER_F' in features_vector:
                dashboard_mapping['CODE_GENDER_F'] = 1.0
    
    # Mettre à jour le vecteur avec les données client
    for feature, value in dashboard_mapping.items():
        if feature in features_vector:
            features_vector[feature] = float(value)
    
    # Convertir en DataFrame avec l'ordre correct
    ordered_values = [features_vector[fname] for fname in feature_names]
    return pd.DataFrame([ordered_values], columns=feature_names)

@app.route('/health', methods=['GET'])
def health_check():
    """Health check avec informations complètes"""
    return jsonify({
        'status': 'healthy',
        'version': 'Railway v3.0 - MODÈLE 234 FEATURES + SHAP',
        'platform': 'Railway Cloud',
        'model_loaded': model is not None,
        'model_type': 'LightGBM Complet',
        'model_features': len(feature_names) if feature_names else 0,
        'dashboard_features': len(DASHBOARD_FEATURES),
        'population_available': population_stats is not None,
        'shap_available': explainer is not None,
        'threshold': float(threshold) if threshold else None,
        'timestamp': datetime.now().isoformat()
    })

@app.route('/predict_dashboard', methods=['POST'])
def predict_dashboard():
    """Prédiction avec modèle complet + SHAP pour dashboard"""
    start_time = time.time()
    
    try:
        data = request.get_json()
        if not data:
            return jsonify({'error': 'Données manquantes'}), 400
        
        if model is None:
            return jsonify({'error': 'Modèle non chargé'}), 500
        
        logger.info(f"Prédiction pour client avec {len(data)} variables saisies")
        
        # Préparer vecteur complet 234 features
        features_df = prepare_features_complete(data)
        logger.info(f"Vecteur de {features_df.shape[1]} features préparé")
        
        # Prédiction avec modèle complet
        probability = model.predict_proba(features_df)[0][1]
        decision = "REFUSE" if probability >= threshold else "APPROVE"
        
        logger.info(f"Prédiction: {decision} (prob: {probability:.4f})")
        
        # SHAP avec modèle complet
        feature_importance = []
        
        if explainer is not None:
            try:
                # Calcul SHAP sur toutes les 234 features
                shap_values = explainer.shap_values(features_df)
                
                if isinstance(shap_values, list):
                    shap_vals = shap_values[1][0]  # Classe positive
                else:
                    shap_vals = shap_values[0]
                
                # Créer DataFrame SHAP complet
                shap_df = pd.DataFrame({
                    'feature': feature_names,
                    'shap_value': shap_vals,
                    'feature_value': features_df.iloc[0].values
                })
                
                # Filtrer pour les 10 variables dashboard
                dashboard_shap = shap_df[shap_df['feature'].isin(DASHBOARD_FEATURES)].copy()
                
                # Ajouter variables dashboard manquantes
                missing_dashboard = set(DASHBOARD_FEATURES) - set(dashboard_shap['feature'])
                for var in missing_dashboard:
                    if var in data:
                        new_row = pd.DataFrame([{
                            'feature': var,
                            'shap_value': 0.0,
                            'feature_value': float(data[var])
                        }])
                        dashboard_shap = pd.concat([dashboard_shap, new_row], ignore_index=True)
                
                # Trier par importance absolue
                dashboard_shap['abs_shap'] = dashboard_shap['shap_value'].abs()
                dashboard_shap = dashboard_shap.sort_values('abs_shap', ascending=False)
                
                # Convertir pour API
                for _, row in dashboard_shap.iterrows():
                    feature_importance.append({
                        'feature': row['feature'],
                        'shap_value': float(row['shap_value']),
                        'feature_value': float(row['feature_value']),
                        'impact': 'positive' if row['shap_value'] > 0 else 'negative'
                    })
                
                logger.info(f"✅ SHAP calculé: {len(feature_importance)} variables dashboard")
                
            except Exception as e:
                logger.error(f"❌ Erreur SHAP: {str(e)}")
                # Fallback sans SHAP
                for feature in DASHBOARD_FEATURES:
                    feature_importance.append({
                        'feature': feature,
                        'shap_value': 0.0,
                        'feature_value': float(data.get(feature, 0.0)),
                        'impact': 'neutral'
                    })
        else:
            logger.warning("SHAP non disponible")
            for feature in DASHBOARD_FEATURES:
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
                
                pop_values = pop_stats['values']
                percentile = sum(1 for val in pop_values if val <= client_val) / len(pop_values) * 100
                
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
        
        processing_time = time.time() - start_time
        
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
                'feature_count': len(DASHBOARD_FEATURES)
            },
            'metadata': {
                'api_version': 'Railway v3.0 - MODÈLE 234 FEATURES + SHAP',
                'timestamp': datetime.now().isoformat(),
                'model_type': 'LightGBM Complet',
                'model_features_used': len(feature_names),
                'dashboard_features_returned': len(feature_importance),
                'processing_time_seconds': round(processing_time, 3),
                'shap_computed': explainer is not None
            }
        }
        
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"Erreur prédiction: {str(e)}")
        return jsonify({
            'error': f'Erreur serveur: {str(e)}',
            'status': 'server_error',
            'timestamp': datetime.now().isoformat()
        }), 500

@app.route('/population_stats', methods=['GET'])
def get_population_stats():
    """Statistiques population pour dashboard"""
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

# Initialisation Railway
try:
    init_railway_model()
except Exception as e:
    logger.error(f"❌ ERREUR CRITIQUE: {str(e)}")
    raise e

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)