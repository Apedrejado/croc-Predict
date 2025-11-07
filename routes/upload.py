# ==========================
# routes/upload.py - Upload e Visualização de Dados
# ==========================
from flask import Blueprint, render_template, request, redirect, url_for, flash, session
from werkzeug.utils import secure_filename
from config import Config
import os
from src.data.loader import load_dataset, validate_dataset
from src.data.preprocessor import get_data_summary
from src.visualization.exploratory import generate_exploratory_plots

upload_bp = Blueprint('upload', __name__)

def allowed_file(filename):
    """Verifica se o arquivo tem extensão permitida"""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in Config.ALLOWED_EXTENSIONS

@upload_bp.route('/', methods=['GET'])
def upload_page():
    """Página de upload de dados"""
    return render_template('upload.html')

@upload_bp.route('/file', methods=['POST'])
def upload_file():
    """Processa upload do arquivo CSV/Excel"""
    
    # Verificar se arquivo foi enviado
    if 'file' not in request.files:
        flash('Nenhum arquivo selecionado', 'error')
        return redirect(url_for('upload.upload_page'))
    
    file = request.files['file']
    
    # Verificar se arquivo tem nome
    if file.filename == '':
        flash('Nenhum arquivo selecionado', 'error')
        return redirect(url_for('upload.upload_page'))
    
    # Verificar extensão
    if not allowed_file(file.filename):
        flash(f'Formato não suportado. Use: {", ".join(Config.ALLOWED_EXTENSIONS)}', 'error')
        return redirect(url_for('upload.upload_page'))
    
    try:
        # Salvar arquivo
        filename = secure_filename(file.filename)
        filepath = Config.UPLOAD_FOLDER / filename
        file.save(filepath)
        
        # Carregar e validar dados
        df = load_dataset(filepath)
        is_valid, message = validate_dataset(df)
        
        if not is_valid:
            os.remove(filepath)  # Remove arquivo inválido
            flash(f'Dataset inválido: {message}', 'error')
            return redirect(url_for('upload.upload_page'))
        
        # Salvar caminho na sessão
        session['dataset_path'] = str(filepath)
        session['dataset_filename'] = filename
        session['dataset_rows'] = len(df)
        session['dataset_cols'] = len(df.columns)
        
        flash(f'Dataset "{filename}" carregado com sucesso!', 'success')
        return redirect(url_for('upload.preview'))
        
    except Exception as e:
        flash(f'Erro ao processar arquivo: {str(e)}', 'error')
        return redirect(url_for('upload.upload_page'))

@upload_bp.route('/preview')
def preview():
    """Exibe prévia dos dados carregados"""
    
    # Verificar se há dados carregados
    if 'dataset_path' not in session:
        flash('Nenhum dataset carregado. Faça upload primeiro.', 'warning')
        return redirect(url_for('upload.upload_page'))
    
    try:
        # Carregar dataset
        df = load_dataset(session['dataset_path'])
        
        # Gerar resumo estatístico
        summary = get_data_summary(df)
        
        # Gerar gráficos exploratórios
        chart_files = generate_exploratory_plots(df)
        
        # Converter primeiras linhas para HTML
        preview_html = df.head(20).to_html(
            classes='table table-striped table-hover',
            index=False,
            border=0
        )
        
        return render_template(
            'preview.html',
            preview_html=preview_html,
            summary=summary,
            chart_files=chart_files,
            dataset_info=session
        )
        
    except Exception as e:
        flash(f'Erro ao visualizar dados: {str(e)}', 'error')
        return redirect(url_for('upload.upload_page'))

@upload_bp.route('/use-example')
def use_example():
    """Usa dataset de exemplo se disponível"""
    
    # Buscar dataset de exemplo nos uploads enviados pelo usuário
    example_files = list(Config.UPLOAD_FOLDER.glob('*.csv'))
    
    if not example_files:
        flash('Nenhum dataset de exemplo disponível. Faça upload de um arquivo.', 'warning')
        return redirect(url_for('upload.upload_page'))
    
    # Usar o primeiro dataset encontrado
    example_path = example_files[0]
    
    try:
        df = load_dataset(example_path)
        
        session['dataset_path'] = str(example_path)
        session['dataset_filename'] = example_path.name
        session['dataset_rows'] = len(df)
        session['dataset_cols'] = len(df.columns)
        
        flash(f'Dataset de exemplo "{example_path.name}" carregado!', 'success')
        return redirect(url_for('upload.preview'))
        
    except Exception as e:
        flash(f'Erro ao carregar exemplo: {str(e)}', 'error')
        return redirect(url_for('upload.upload_page'))