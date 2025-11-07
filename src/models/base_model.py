# ==========================
# src/models/base_model.py - Classe Base para Modelos
# ==========================
from abc import ABC, abstractmethod
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from config import Config
import numpy as np

class BaseModel(ABC):
    """
    Classe base abstrata para todos os modelos de ML
    """
    
    def __init__(self, **params):
        """
        Inicializa o modelo com parâmetros
        
        Args:
            **params: Parâmetros específicos do modelo
        """
        self.params = params
        self.model = None
        self.is_trained = False
    
    @abstractmethod
    def _create_model(self):
        """
        Cria a instância do modelo específico
        Deve ser implementado pelas subclasses
        
        Returns:
            Modelo do scikit-learn
        """
        pass
    
    def train(self, X_train, y_train):
        """
        Treina o modelo
        
        Args:
            X_train: Features de treino
            y_train: Labels de treino
        """
        # Criar modelo se não existir
        if self.model is None:
            self.model = self._create_model()
        
        # Treinar
        self.model.fit(X_train, y_train)
        self.is_trained = True
        
        return self
    
    def predict(self, X):
        """
        Faz predições
        
        Args:
            X: Features para predição
        
        Returns:
            Array com predições
        """
        if not self.is_trained:
            raise ValueError("Modelo não foi treinado ainda!")
        
        return self.model.predict(X)
    
    def predict_proba(self, X):
        """
        Retorna probabilidades das predições
        
        Args:
            X: Features para predição
        
        Returns:
            Array com probabilidades
        """
        if not self.is_trained:
            raise ValueError("Modelo não foi treinado ainda!")
        
        if hasattr(self.model, 'predict_proba'):
            return self.model.predict_proba(X)
        else:
            raise NotImplementedError("Este modelo não suporta predict_proba")
    
    def evaluate(self, X_test, y_test, class_names=None):
        """
        Avalia o modelo no conjunto de teste
        
        Args:
            X_test: Features de teste
            y_test: Labels de teste
            class_names: Nomes das classes (opcional)
        
        Returns:
            dict com métricas de avaliação
        """
        if not self.is_trained:
            raise ValueError("Modelo não foi treinado ainda!")
        
        # Fazer predições
        y_pred = self.predict(X_test)
        
        # Calcular métricas
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision_macro': precision_score(y_test, y_pred, average='macro', zero_division=0),
            'recall_macro': recall_score(y_test, y_pred, average='macro', zero_division=0),
            'f1_macro': f1_score(y_test, y_pred, average='macro', zero_division=0),
            'precision_weighted': precision_score(y_test, y_pred, average='weighted', zero_division=0),
            'recall_weighted': recall_score(y_test, y_pred, average='weighted', zero_division=0),
            'f1_weighted': f1_score(y_test, y_pred, average='weighted', zero_division=0)
        }
        
        # Relatório de classificação
        if class_names is not None:
            metrics['classification_report'] = classification_report(
                y_test, y_pred, 
                target_names=class_names,
                output_dict=True,
                zero_division=0
            )
        
        return metrics
    
    def cross_validate(self, X, y, cv=None):
        """
        Realiza validação cruzada
        
        Args:
            X: Features completas
            y: Labels completas
            cv: Número de folds ou objeto de CV (default: Config.CV_FOLDS)
        
        Returns:
            dict com scores de CV
        """
        if self.model is None:
            self.model = self._create_model()
        
        if cv is None:
            cv = StratifiedKFold(
                n_splits=Config.CV_FOLDS, 
                shuffle=True, 
                random_state=Config.RANDOM_STATE
            )
        
        # Realizar CV
        scores = cross_val_score(
            self.model, X, y, 
            cv=cv, 
            scoring='accuracy',
            n_jobs=-1
        )
        
        return {
            'scores': scores.tolist(),
            'mean': scores.mean(),
            'std': scores.std(),
            'min': scores.min(),
            'max': scores.max()
        }
    
    def get_feature_importance(self):
        """
        Retorna importância das features (se disponível)
        
        Returns:
            Array com importância das features ou None
        """
        if not self.is_trained:
            raise ValueError("Modelo não foi treinado ainda!")
        
        if hasattr(self.model, 'feature_importances_'):
            return self.model.feature_importances_
        elif hasattr(self.model, 'coef_'):
            # Para modelos lineares, usar coeficientes absolutos
            return np.abs(self.model.coef_).mean(axis=0)
        else:
            return None
    
    def get_params(self):
        """
        Retorna parâmetros do modelo
        
        Returns:
            dict com parâmetros
        """
        return self.params