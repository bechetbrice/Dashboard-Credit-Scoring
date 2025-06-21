"""
Test des Endpoints API Production - Dashboard Credit Scoring
"""

import requests
import json
import time
from datetime import datetime
import sys

# Configuration
API_BASE_URL = "https://dashboard-credit-scoring-production.up.railway.app"
TIMEOUT = 30

# Couleurs pour l'affichage
class Colors:
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    BOLD = '\033[1m'
    END = '\033[0m'

def print_header(title):
    """Afficher un en-tête de section"""
    print(f"\n{Colors.BLUE}{'='*60}{Colors.END}")
    print(f"{Colors.BOLD}{Colors.BLUE}🔍 {title}{Colors.END}")
    print(f"{Colors.BLUE}{'='*60}{Colors.END}")

def print_success(message):
    """Afficher un message de succès"""
    print(f"{Colors.GREEN}✅ {message}{Colors.END}")

def print_error(message):
    """Afficher un message d'erreur"""
    print(f"{Colors.RED}❌ {message}{Colors.END}")

def print_warning(message):
    """Afficher un avertissement"""
    print(f"{Colors.YELLOW}⚠️ {message}{Colors.END}")

def print_info(message):
    """Afficher une information"""
    print(f"{Colors.BLUE}ℹ️ {message}{Colors.END}")

def test_health_endpoint():
    """Test 1: Health Check"""
    print_header("TEST 1: HEALTH CHECK")
    
    try:
        print_info(f"URL: {API_BASE_URL}/health")
        
        start_time = time.time()
        response = requests.get(f"{API_BASE_URL}/health", timeout=TIMEOUT)
        response_time = time.time() - start_time
        
        print_info(f"Status Code: {response.status_code}")
        print_info(f"Response Time: {response_time:.2f}s")
        
        if response.status_code == 200:
            data = response.json()
            print_success("Health Check RÉUSSI")
            
            # Vérifier les champs critiques
            critical_fields = ['status', 'version', 'model_loaded', 'shap_available']
            for field in critical_fields:
                if field in data:
                    value = data[field]
                    print_info(f"{field}: {value}")
                    if field == 'model_loaded' and value:
                        print_success("Modèle chargé avec succès")
                    elif field == 'shap_available' and value:
                        print_success("SHAP disponible")
                else:
                    print_warning(f"Champ manquant: {field}")
            
            return True, data
        else:
            print_error(f"Health Check ÉCHOUÉ: {response.status_code}")
            print_error(f"Réponse: {response.text}")
            return False, None
            
    except requests.exceptions.Timeout:
        print_error("TIMEOUT - API non accessible")
        return False, None
    except requests.exceptions.ConnectionError:
        print_error("ERREUR CONNEXION - API indisponible")
        return False, None
    except Exception as e:
        print_error(f"ERREUR: {str(e)}")
        return False, None

def test_prediction_endpoint():
    """Test 2: Prédiction Dashboard"""
    print_header("TEST 2: PRÉDICTION DASHBOARD")
    
    # Données client de test
    client_data = {
        "EXT_SOURCE_2": 0.65,
        "EXT_SOURCE_3": 0.55,
        "EXT_SOURCE_1": 0.45,
        "DAYS_EMPLOYED": -2000,
        "CODE_GENDER": "F",
        "INSTAL_DPD_MEAN": 0.5,
        "PAYMENT_RATE": 0.15,
        "NAME_EDUCATION_TYPE_Higher_education": 1,
        "AMT_ANNUITY": 18000,
        "INSTAL_AMT_PAYMENT_SUM": 120000
    }
    
    try:
        print_info(f"URL: {API_BASE_URL}/predict_dashboard")
        print_info("Données client:")
        for key, value in client_data.items():
            print(f"  {key}: {value}")
        
        start_time = time.time()
        response = requests.post(
            f"{API_BASE_URL}/predict_dashboard",
            json=client_data,
            timeout=TIMEOUT,
            headers={"Content-Type": "application/json"}
        )
        response_time = time.time() - start_time
        
        print_info(f"Status Code: {response.status_code}")
        print_info(f"Response Time: {response_time:.2f}s")
        
        if response.status_code == 200:
            data = response.json()
            print_success("Prédiction RÉUSSIE")
            
            # Vérifier les sections principales
            sections = ['prediction', 'explanation', 'metadata']
            for section in sections:
                if section in data:
                    print_success(f"Section '{section}' présente")
                else:
                    print_error(f"Section '{section}' manquante")
            
            # Détails de la prédiction
            if 'prediction' in data:
                pred = data['prediction']
                print_info(f"Décision: {pred.get('decision_fr', 'N/A')}")
                print_info(f"Probabilité: {pred.get('probability', 0):.3f}")
                print_info(f"Niveau de risque: {pred.get('risk_level', 'N/A')}")
            
            # Vérifier SHAP
            if 'explanation' in data:
                expl = data['explanation']
                shap_available = expl.get('shap_available', False)
                top_features = expl.get('top_features', [])
                
                if shap_available:
                    print_success("SHAP disponible")
                    print_info(f"Nombre de features: {len(top_features)}")
                    
                    if len(top_features) == 10:
                        print_success("10 variables retournées (correct)")
                    else:
                        print_warning(f"Seulement {len(top_features)} variables (attendu: 10)")
                    
                    # Afficher top 3 features
                    print_info("Top 3 features importantes:")
                    for i, feature in enumerate(top_features[:3]):
                        name = feature.get('feature', 'N/A')
                        shap_val = feature.get('shap_value', 0)
                        impact = "+" if shap_val > 0 else "-"
                        print(f"  {i+1}. {name}: {impact}{abs(shap_val):.4f}")
                else:
                    print_error("SHAP non disponible")
            
            return True, data
        else:
            print_error(f"Prédiction ÉCHOUÉE: {response.status_code}")
            try:
                error_data = response.json()
                print_error(f"Erreur: {error_data.get('error', 'Erreur inconnue')}")
            except:
                print_error(f"Réponse: {response.text}")
            return False, None
            
    except Exception as e:
        print_error(f"ERREUR: {str(e)}")
        return False, None

def test_population_stats_endpoint():
    """Test 3: Statistiques Population"""
    print_header("TEST 3: STATISTIQUES POPULATION")
    
    try:
        print_info(f"URL: {API_BASE_URL}/population_stats")
        
        start_time = time.time()
        response = requests.get(f"{API_BASE_URL}/population_stats", timeout=TIMEOUT)
        response_time = time.time() - start_time
        
        print_info(f"Status Code: {response.status_code}")
        print_info(f"Response Time: {response_time:.2f}s")
        
        if response.status_code == 200:
            data = response.json()
            print_success("Population Stats RÉUSSIES")
            
            # Vérifier les données
            graph_data = data.get('graph_data', {})
            variables_available = data.get('variables_available', [])
            population_size = data.get('population_size', 0)
            
            print_info(f"Variables disponibles: {len(variables_available)}")
            print_info(f"Taille population: {population_size}")
            
            # Variables attendues (8 variables clés)
            expected_vars = [
                'EXT_SOURCE_2', 'EXT_SOURCE_3', 'EXT_SOURCE_1',
                'DAYS_EMPLOYED', 'INSTAL_DPD_MEAN', 'PAYMENT_RATE',
                'AMT_ANNUITY', 'INSTAL_AMT_PAYMENT_SUM'
            ]
            
            print_info("Variables trouvées:")
            for var in variables_available:
                if var in expected_vars:
                    print_success(f"  ✓ {var}")
                else:
                    print_info(f"  • {var}")
            
            # Variables manquantes
            missing = set(expected_vars) - set(variables_available)
            if missing:
                print_warning(f"Variables manquantes: {missing}")
            else:
                print_success("Toutes les variables attendues présentes")
            
            return True, data
        else:
            print_error(f"Population Stats ÉCHOUÉES: {response.status_code}")
            return False, None
            
    except Exception as e:
        print_error(f"ERREUR: {str(e)}")
        return False, None

def test_bivariate_analysis_endpoint():
    """Test 4: Analyse Bi-variée"""
    print_header("TEST 4: ANALYSE BI-VARIÉE")
    
    # Test avec 2 variables importantes
    analysis_data = {
        "variable1": "EXT_SOURCE_2",
        "variable2": "PAYMENT_RATE"
    }
    
    try:
        print_info(f"URL: {API_BASE_URL}/bivariate_analysis")
        print_info(f"Variables: {analysis_data['variable1']} vs {analysis_data['variable2']}")
        
        start_time = time.time()
        response = requests.post(
            f"{API_BASE_URL}/bivariate_analysis",
            json=analysis_data,
            timeout=TIMEOUT,
            headers={"Content-Type": "application/json"}
        )
        response_time = time.time() - start_time
        
        print_info(f"Status Code: {response.status_code}")
        print_info(f"Response Time: {response_time:.2f}s")
        
        if response.status_code == 200:
            data = response.json()
            print_success("Analyse Bi-variée RÉUSSIE")
            
            # Vérifier les données
            correlation = data.get('correlation', 0)
            sample_size = data.get('sample_size', 0)
            data_points = data.get('data_points', {})
            
            print_info(f"Corrélation: {correlation:.3f}")
            print_info(f"Taille échantillon: {sample_size}")
            
            x_data = data_points.get('x', [])
            y_data = data_points.get('y', [])
            print_info(f"Points X: {len(x_data)}")
            print_info(f"Points Y: {len(y_data)}")
            
            if len(x_data) == len(y_data) and len(x_data) > 0:
                print_success("Données cohérentes")
            else:
                print_warning("Problème avec les données de points")
            
            return True, data
        else:
            print_error(f"Analyse Bi-variée ÉCHOUÉE: {response.status_code}")
            try:
                error_data = response.json()
                print_error(f"Erreur: {error_data.get('error', 'Erreur inconnue')}")
            except:
                print_error(f"Réponse: {response.text}")
            return False, None
            
    except Exception as e:
        print_error(f"ERREUR: {str(e)}")
        return False, None

def run_all_tests():
    """Exécuter tous les tests"""
    print(f"{Colors.BOLD}🚀 TESTS API DASHBOARD CREDIT SCORING{Colors.END}")
    print(f"URL Base: {API_BASE_URL}")
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    results = {}
    
    # Test 1: Health Check
    results['health'] = test_health_endpoint()[0]
    
    # Test 2: Prédiction
    results['prediction'] = test_prediction_endpoint()[0]
    
    # Test 3: Population Stats
    results['population'] = test_population_stats_endpoint()[0]
    
    # Test 4: Analyse Bi-variée
    results['bivariate'] = test_bivariate_analysis_endpoint()[0]
    
    # Résumé final
    print_header("RÉSUMÉ DES TESTS")
    
    total_tests = len(results)
    passed_tests = sum(results.values())
    
    test_names = {
        'health': 'Health Check',
        'prediction': 'Prédiction Dashboard',
        'population': 'Statistiques Population',
        'bivariate': 'Analyse Bi-variée'
    }
    
    for test_key, success in results.items():
        test_name = test_names.get(test_key, test_key)
        if success:
            print_success(f"{test_name}")
        else:
            print_error(f"{test_name}")
    
    print(f"\n{Colors.BOLD}📊 RÉSULTAT GLOBAL: {passed_tests}/{total_tests} tests réussis{Colors.END}")
    
    if passed_tests == total_tests:
        print_success("🎉 TOUS LES TESTS SONT PASSÉS!")
        print_success("✅ API prête pour le dashboard")
        return True
    else:
        print_error("❌ CERTAINS TESTS ONT ÉCHOUÉ")
        print_warning("🔧 Vérifiez les logs d'erreur ci-dessus")
        return False

if __name__ == "__main__":
    try:
        success = run_all_tests()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print(f"\n{Colors.YELLOW}Tests interrompus par l'utilisateur{Colors.END}")
        sys.exit(1)