# ==========================
# routes/main.py - Rota Principal
# ==========================
from flask import Blueprint, render_template, session, redirect, url_for
from config import Config
import os

main_bp = Blueprint('main', __name__)

@main_bp.route('/')
def index():
    """Página inicial - Dashboard principal"""
    
    # Verificar se há dados carregados na sessão
    has_data = 'dataset_path' in session
    
    # Verificar se há modelo treinado
    has_model = 'model_path' in session
    
    # Listar datasets disponíveis
    upload_folder = Config.UPLOAD_FOLDER
    datasets = []
    if upload_folder.exists():
        datasets = [f.name for f in upload_folder.iterdir() if f.suffix in ['.csv', '.xlsx', '.xls']]
    
    # Listar modelos salvos
    models_folder = Config.MODELS_FOLDER
    saved_models = []
    if models_folder.exists():
        saved_models = [f.name for f in models_folder.iterdir() if f.suffix == '.pkl']
    
    return render_template(
        'index.html',
        has_data=has_data,
        has_model=has_model,
        datasets=datasets,
        saved_models=saved_models,
        available_models=Config.AVAILABLE_MODELS
    )

@main_bp.route('/clear-session')
def clear_session():
    """Limpa a sessão e recomeça"""
    session.clear()
    return redirect(url_for('main.index'))