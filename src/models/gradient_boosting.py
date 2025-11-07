# ==========================
# src/models/gradient_boosting.py - Gradient Boosting Classifier
# ==========================
from sklearn.ensemble import GradientBoostingClassifier
from src.models.base_model import BaseModel
from config import Config

class GradientBoostingModel(BaseModel):
    """
    Modelo Gradient Boosting para classificação de espécies
    """
    
    def __init__(self, n_estimators=200, learning_rate=0.1, max_depth=3, 
                 min_samples_split=2, min_samples_leaf=1, subsample=1.0, **kwargs):
        """
        Inicializa Gradient Boosting
        
        Args:
            n_estimators: Número de boosting stages
            learning_rate: Taxa de aprendizado (shrinkage)
            max_depth: Profundidade máxima das árvores
            min_samples_split: Mínimo de amostras para dividir nó
            min_samples_leaf: Mínimo de amostras em folha
            subsample: Fração de amostras para treinar cada árvore
        """
        params = {
            'n_estimators': n_estimators,
            'learning_rate': learning_rate,
            'max_depth': max_depth,
            'min_samples_split': min_samples_split,
            'min_samples_leaf': min_samples_leaf,
            'subsample': subsample,
            'random_state': Config.RANDOM_STATE
        }
        params.update(kwargs)
        super().__init__(**params)
    
    def _create_model(self):
        """
        Cria instância do Gradient Boosting
        
        Returns:
            GradientBoostingClassifier configurado
        """
        return GradientBoostingClassifier(**self.params)
    
    def get_training_scores(self):
        """
        Retorna scores de treinamento por iteração
        
        Returns:
            array: train scores ou None
        """
        if not self.is_trained:
            return None
        
        if hasattr(self.model, 'train_score_'):
            return self.model.train_score_
        return None
    
    def get_staged_predictions(self, X):
        """
        Retorna predições para cada estágio do boosting
        
        Args:
            X: Features para predição
        
        Returns:
            generator: predições por estágio
        """
        if not self.is_trained:
            raise ValueError("Modelo não foi treinado ainda!")
        
        return self.model.staged_predict(X)