# ==========================
# src/models/svm_model.py - Support Vector Machine Classifier
# ==========================
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from src.models.base_model import BaseModel
from config import Config
import numpy as np

class SVMModel(BaseModel):
    """
    Modelo SVM para classificação de espécies
    """
    
    def __init__(self, C=1.0, kernel='rbf', gamma='scale', degree=3, **kwargs):
        """
        Inicializa SVM
        
        Args:
            C: Parâmetro de regularização
            kernel: Tipo de kernel ('linear', 'poly', 'rbf', 'sigmoid')
            gamma: Coeficiente do kernel ('scale', 'auto' ou float)
            degree: Grau do kernel polinomial (se kernel='poly')
        """
        params = {
            'C': C,
            'kernel': kernel,
            'gamma': gamma,
            'degree': degree,
            'random_state': Config.RANDOM_STATE,
            'probability': True  # Habilitar predict_proba
        }
        params.update(kwargs)
        super().__init__(**params)
        
        # SVM é sensível à escala, então vamos usar StandardScaler
        self.scaler = StandardScaler()
        self.use_scaling = True
    
    def _create_model(self):
        """
        Cria instância do SVM
        
        Returns:
            SVC configurado
        """
        return SVC(**self.params)
    
    def train(self, X_train, y_train):
        """
        Treina o SVM (com escalonamento de features)
        
        Args:
            X_train: Features de treino
            y_train: Labels de treino
        """
        # Criar modelo se não existir
        if self.model is None:
            self.model = self._create_model()
        
        # Escalonar features
        if self.use_scaling:
            X_train_scaled = self.scaler.fit_transform(X_train)
        else:
            X_train_scaled = X_train
        
        # Treinar
        self.model.fit(X_train_scaled, y_train)
        self.is_trained = True
        
        return self
    
    def predict(self, X):
        """
        Faz predições (com escalonamento)
        
        Args:
            X: Features para predição
        
        Returns:
            Array com predições
        """
        if not self.is_trained:
            raise ValueError("Modelo não foi treinado ainda!")
        
        # Escalonar features
        if self.use_scaling:
            X_scaled = self.scaler.transform(X)
        else:
            X_scaled = X
        
        return self.model.predict(X_scaled)
    
    def predict_proba(self, X):
        """
        Retorna probabilidades (com escalonamento)
        
        Args:
            X: Features para predição
        
        Returns:
            Array com probabilidades
        """
        if not self.is_trained:
            raise ValueError("Modelo não foi treinado ainda!")
        
        # Escalonar features
        if self.use_scaling:
            X_scaled = self.scaler.transform(X)
        else:
            X_scaled = X
        
        return self.model.predict_proba(X_scaled)
    
    def get_support_vectors(self):
        """
        Retorna vetores de suporte
        
        Returns:
            array: support vectors ou None
        """
        if not self.is_trained:
            return None
        
        if hasattr(self.model, 'support_vectors_'):
            return self.model.support_vectors_
        return None
    
    def get_n_support(self):
        """
        Retorna número de vetores de suporte por classe
        
        Returns:
            array: número de support vectors por classe
        """
        if not self.is_trained:
            return None
        
        if hasattr(self.model, 'n_support_'):
            return self.model.n_support_
        return None
    
    def get_feature_importance(self):
        """
        Para SVM com kernel linear, retorna coeficientes como importância
        Para outros kernels, retorna None
        
        Returns:
            Array com importância ou None
        """
        if not self.is_trained:
            raise ValueError("Modelo não foi treinado ainda!")
        
        if self.params['kernel'] == 'linear' and hasattr(self.model, 'coef_'):
            # Para classificação multiclasse, média dos coeficientes absolutos
            return np.abs(self.model.coef_).mean(axis=0)
        else:
            # Kernels não-lineares não têm importância de features direta
            return None