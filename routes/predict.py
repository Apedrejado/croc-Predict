
from flask import Blueprint, render_template, request, redirect, url_for, flash, session
from config import Config
import joblib
import pandas as pd
from src.data.feature_engineering import create_features

predict_bp = Blueprint('predict', __name__)

@predict_bp.route('/', methods=['GET'])
def predict_page():
    """Página de predição"""
    
    # Verificar se há modelo treinado
    if 'model_path' not in session:
        flash('Treine um modelo primeiro!', 'warning')
        return redirect(url_for('train.train_page'))
    
    return render_template('predict.html')

@predict_bp.route('/species', methods=['POST'])
def predict_species():
    """Realiza predição de espécie"""
    
    # Verificar se há modelo
    if 'model_path' not in session:
        flash('Nenhum modelo disponível!', 'error')
        return redirect(url_for('predict.predict_page'))
    
    try:
        # Carregar modelo
        model_data = joblib.load(session['model_path'])
        model = model_data['model']
        label_encoder = model_data['label_encoder']
        feature_names = model_data['feature_names']
        
        # Obter dados do formulário
        input_data = {
            'Observed Length (m)': float(request.form.get('length')),
            'Observed Weight (kg)': float(request.form.get('weight')),
            'Conservation Status': request.form.get('status'),
            'Country/Region': request.form.get('country'),
            'Habitat Type': request.form.get('habitat'),
            'Genus': request.form.get('genus'),
            'Family': request.form.get('family'),
            'Age Class': request.form.get('age_class'),
            'Sex': request.form.get('sex')
        }
        
        # Criar DataFrame
        df_input = pd.DataFrame([input_data])
        
        # Aplicar feature engineering
        df_processed = create_features(df_input)
        
        # Codificar features categóricas
        X_encoded = pd.get_dummies(df_processed)
        
        # Garantir mesmas colunas do treinamento
        for col in feature_names:
            if col not in X_encoded.columns:
                X_encoded[col] = 0
        X_encoded = X_encoded[feature_names]
        
        # Fazer predição
        pred_encoded = model.predict(X_encoded)
        pred_species = label_encoder.inverse_transform(pred_encoded)[0]
        
        # Top 3 espécies mais prováveis
        pred_probs = model.predict_proba(X_encoded)
        top3_idx = pred_probs[0].argsort()[::-1][:3]
        top3_species = label_encoder.inverse_transform(top3_idx)
        top3_probs = pred_probs[0][top3_idx]
        
        predictions = [
            {'species': sp, 'probability': prob}
            for sp, prob in zip(top3_species, top3_probs)
        ]
        
        return render_template(
            'predict.html',
            prediction=pred_species,
            predictions=predictions,
            input_data=input_data
        )
        
    except Exception as e:
        flash(f'Erro na predição: {str(e)}', 'error')
        return redirect(url_for('predict.predict_page'))