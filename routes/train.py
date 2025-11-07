# ==========================
# routes/train.py - Treinamento de Modelos
# ==========================
from flask import Blueprint, render_template, request, redirect, url_for, flash, session
from config import Config
from src.data.loader import load_dataset
from src.data.preprocessor import preprocess_data
from src.models.random_forest import RandomForestModel
from src.models.gradient_boosting import GradientBoostingModel
from src.models.svm_model import SVMModel
from src.visualization.metrics import generate_metrics_plots
import joblib
from datetime import datetime
import numpy as np

train_bp = Blueprint('train', __name__)

@train_bp.route('/', methods=['GET'])
def train_page():
    """Página de configuração e treinamento"""
    if 'dataset_path' not in session:
        flash('Carregue um dataset primeiro!', 'warning')
        return redirect(url_for('upload.upload_page'))
    
    return render_template('train.html')

@train_bp.route('/start', methods=['POST'])
def start_training():
    """Inicia o treinamento do modelo"""
    if 'dataset_path' not in session:
        flash('Nenhum dataset carregado!', 'error')
        return redirect(url_for('upload.upload_page'))
    
    try:
        model_type = request.form.get('model_type', 'random_forest')
        
        # Carregar dados
        df = load_dataset(session['dataset_path'])
        X_train, X_test, y_train, y_test, label_encoder, feature_names = preprocess_data(df)
        
        # RANDOM FOREST
        if model_type == 'random_forest':
            n_est = request.form.get('rf_n_estimators', '500')
            max_d = request.form.get('rf_max_depth', '')
            min_split = request.form.get('rf_min_samples_split', '2')
            min_leaf = request.form.get('rf_min_samples_leaf', '1')
            
            params = {
                'n_estimators': int(n_est) if n_est else 500,
                'min_samples_split': int(min_split) if min_split else 2,
                'min_samples_leaf': int(min_leaf) if min_leaf else 1,
                'max_depth': int(max_d) if max_d and max_d.strip() else None
            }
            model = RandomForestModel(**params)
        
        # GRADIENT BOOSTING
        elif model_type == 'gradient_boosting':
            n_est = request.form.get('gb_n_estimators', '200')
            lr = request.form.get('gb_learning_rate', '0.1')
            max_d = request.form.get('gb_max_depth', '3')
            
            params = {
                'n_estimators': int(n_est) if n_est else 200,
                'learning_rate': float(lr) if lr else 0.1,
                'max_depth': int(max_d) if max_d else 3,
            }
            model = GradientBoostingModel(**params)
        
        # SVM
        elif model_type == 'svm':
            c_val = request.form.get('svm_c_param', '1.0')
            kern = request.form.get('svm_kernel', 'rbf')
            gam = request.form.get('svm_gamma', 'scale')
            
            params = {
                'C': float(c_val) if c_val else 1.0,
                'kernel': kern if kern else 'rbf',
                'gamma': gam if gam else 'scale',
            }
            model = SVMModel(**params)
        
        else:
            flash('Modelo inválido!', 'error')
            return redirect(url_for('train.train_page'))
        
        # Treinar
        model.train(X_train, y_train)
        
        # Avaliar
        metrics = model.evaluate(X_test, y_test, label_encoder.classes_)
        
        # Converter para tipos Python nativos
        def to_python_type(value):
            """Converte numpy/outros tipos para tipos Python nativos"""
            if isinstance(value, (np.integer, np.int64, np.int32)):
                return int(value)
            elif isinstance(value, (np.floating, np.float64, np.float32)):
                return float(value)
            elif isinstance(value, np.ndarray):
                return value.tolist()
            elif isinstance(value, dict):
                return {k: to_python_type(v) for k, v in value.items()}
            elif isinstance(value, list):
                return [to_python_type(v) for v in value]
            else:
                return value
        
        # Métricas simples - apenas valores numéricos básicos
        simple_metrics = {
            'accuracy': round(float(metrics['accuracy']), 4),
            'precision_weighted': round(float(metrics.get('precision_weighted', 0)), 4),
            'recall_weighted': round(float(metrics.get('recall_weighted', 0)), 4),
            'f1_weighted': round(float(metrics.get('f1_weighted', 0)), 4),
            'precision_macro': round(float(metrics.get('precision_macro', 0)), 4),
            'recall_macro': round(float(metrics.get('recall_macro', 0)), 4),
            'f1_macro': round(float(metrics.get('f1_macro', 0)), 4)
        }
        
        # Gráficos
        chart_files = generate_metrics_plots(
            model.model, X_test, y_test, label_encoder.classes_, feature_names
        )
        
        # Salvar modelo
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        model_path = Config.MODELS_FOLDER / f'{model_type}_{timestamp}.pkl'
        
        joblib.dump({
            'model': model.model,
            'label_encoder': label_encoder,
            'feature_names': feature_names,
            'model_type': model_type,
            'params': {k: str(v) for k, v in params.items()},
            'metrics': simple_metrics
        }, model_path)
        
        # Sessão - APENAS caminho do modelo (sem métricas para evitar erro JSON)
        session['model_path'] = str(model_path)
        session['model_type'] = str(model_type)
        
        # Flash com sucesso
        flash(f'Modelo treinado com sucesso! Accuracy: {simple_metrics["accuracy"]:.2%}', 'success')
        return redirect(url_for('train.results'))
        
    except Exception as e:
        import traceback
        print("="*80)
        print("ERRO NO TREINAMENTO:")
        print(traceback.format_exc())
        print("="*80)
        flash(f'Erro no treinamento: {str(e)}', 'error')
        return redirect(url_for('train.train_page'))

@train_bp.route('/results')
def results():
    """Exibe resultados do treinamento"""
    if 'model_path' not in session:
        flash('Nenhum modelo treinado ainda!', 'warning')
        return redirect(url_for('train.train_page'))
    
    try:
        # Carregar modelo salvo
        model_data = joblib.load(session['model_path'])
        
        # Listar gráficos gerados
        chart_files = [f.name for f in Config.CHARTS_FOLDER.glob('*.png')]
        
        return render_template(
            'results.html',
            metrics=model_data['metrics'],
            model_type=model_data['model_type'],
            params=model_data['params'],
            chart_files=chart_files
        )
        
    except Exception as e:
        import traceback
        print("="*80)
        print("ERRO AO CARREGAR RESULTADOS:")
        print(traceback.format_exc())
        print("="*80)
        flash(f'Erro ao carregar resultados: {str(e)}', 'error')
        return redirect(url_for('train.train_page'))