# ==========================
# app.py - Aplicação Flask Principal
# ==========================
from flask import Flask, render_template, session
from config import Config
import os

# Criar aplicação Flask
app = Flask(__name__)
app.config.from_object(Config)

# Inicializar diretórios
Config.init_app()

# Importar rotas
from routes.main import main_bp
from routes.upload import upload_bp
from routes.train import train_bp
from routes.predict import predict_bp

# Registrar blueprints
app.register_blueprint(main_bp)
app.register_blueprint(upload_bp, url_prefix='/upload')
app.register_blueprint(train_bp, url_prefix='/train')
app.register_blueprint(predict_bp, url_prefix='/predict')

# Criar rota para servir gráficos
@app.route('/static/charts/<path:filename>')
def serve_chart(filename):
    """Serve gráficos gerados dinamicamente"""
    from flask import send_from_directory
    return send_from_directory(app.config['CHARTS_FOLDER'], filename)

# Filtro Jinja2 para formatar números
@app.template_filter('format_number')
def format_number(value, decimals=2):
    """Formata número com casas decimais"""
    try:
        return f"{float(value):.{decimals}f}"
    except (ValueError, TypeError):
        return value

# Filtro Jinja2 para formatar porcentagem
@app.template_filter('format_percent')
def format_percent(value, decimals=2):
    """Formata porcentagem"""
    try:
        return f"{float(value) * 100:.{decimals}f}%"
    except (ValueError, TypeError):
        return value

# Contexto global para templates
@app.context_processor
def inject_config():
    """Injeta configurações nos templates"""
    return {
        'available_models': Config.AVAILABLE_MODELS,
        'app_name': 'CrocPredict'
    }

# Página de erro 404
@app.errorhandler(404)
def page_not_found(e):
    return render_template('base.html', error="Página não encontrada"), 404

# Página de erro 500
@app.errorhandler(500)
def internal_error(e):
    return render_template('base.html', error="Erro interno do servidor"), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)