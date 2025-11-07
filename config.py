# ==========================
# config.py - Configurações do Projeto
# ==========================
import os
from pathlib import Path

# Diretório base do projeto
BASE_DIR = Path(__file__).resolve().parent

class Config:
    """Configurações gerais da aplicação"""
    
    # Flask
    SECRET_KEY = os.environ.get('SECRET_KEY') or 'dev-secret-key-change-in-production'
    DEBUG = True
    
    # Upload de arquivos
    UPLOAD_FOLDER = BASE_DIR / 'data' / 'uploads'
    MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16MB max
    ALLOWED_EXTENSIONS = {'csv', 'xlsx', 'xls'}
    
    # Modelos salvos
    MODELS_FOLDER = BASE_DIR / 'data' / 'models'
    
    # Gráficos gerados
    CHARTS_FOLDER = BASE_DIR / 'data' / 'charts'
    CHARTS_URL = '/static/charts'
    
    # Configurações de ML
    TEST_SIZE = 0.2
    RANDOM_STATE = 42
    CV_FOLDS = 5
    
    # Modelos disponíveis
    AVAILABLE_MODELS = {
        'random_forest': 'Random Forest',
        'gradient_boosting': 'Gradient Boosting',
        'svm': 'Support Vector Machine'
    }
    
    # Parâmetros padrão dos modelos
    MODEL_DEFAULTS = {
        'random_forest': {
            'n_estimators': 500,
            'max_depth': None,
            'min_samples_split': 2,
            'min_samples_leaf': 1,
            'random_state': RANDOM_STATE
        },
        'gradient_boosting': {
            'n_estimators': 200,
            'learning_rate': 0.1,
            'max_depth': 3,
            'random_state': RANDOM_STATE
        },
        'svm': {
            'C': 1.0,
            'kernel': 'rbf',
            'gamma': 'scale',
            'random_state': RANDOM_STATE
        }
    }
    
    # Colunas obrigatórias do dataset
    REQUIRED_COLUMNS = [
        'Common Name',
        'Observed Length (m)',
        'Observed Weight (kg)',
        'Conservation Status',
        'Country/Region',
        'Habitat Type',
        'Genus',
        'Family',
        'Age Class',
        'Sex'
    ]
    
    # Feature engineering
    NUMERICAL_FEATURES = ['Observed Length (m)', 'Observed Weight (kg)', 'Weight_per_Length']
    
    CATEGORICAL_FEATURES = [
        'Region_Group',
        'Habitat_Type_Group',
        'Conservation_Status_Group',
        'Genus',
        'Family',
        'Age Class',
        'Sex',
        'Size_Class'
    ]

    @staticmethod
    def init_app():
        """Inicializa diretórios necessários"""
        for folder in [Config.UPLOAD_FOLDER, Config.MODELS_FOLDER, Config.CHARTS_FOLDER]:
            folder.mkdir(parents=True, exist_ok=True)