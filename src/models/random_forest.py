# ==========================
# src/models/random_forest.py - Random Forest Classifier
# ==========================
from sklearn.ensemble import RandomForestClassifier
from src.models.base_model import BaseModel
from config import Config

class RandomForestModel(BaseModel):
    """
    Modelo Random Forest para classificação de espécies
    """
    
    def __init__(self, n_estimators=500, max_depth=None, min_samples_split=2, 
                 min_samples_leaf=1, max_features='sqrt', **kwargs):
        """
        Inicializa Random Forest
        
        Args:
            n_estimators: Número de árvores na floresta
            max_depth: Profundidade máxima das árvores (None = sem limite)
            min_samples_split: Mínimo de amostras para dividir nó
            min_samples_leaf: Mínimo de amostras em folha
            max_features: Número de features a considerar para split
        """
        params = {
            'n_estimators': n_estimators,
            'max_depth': max_depth,
            'min_samples_split': min_samples_split,
            'min_samples_leaf': min_samples_leaf,
            'max_features': max_features,
            'random_state': Config.RANDOM_STATE,
            'n_jobs': -1  # Usar todos os cores disponíveis
        }
        params.update(kwargs)
        super().__init__(**params)
    
    def _create_model(self):
        """
        Cria instância do Random Forest
        
        Returns:
            RandomForestClassifier configurado
        """
        return RandomForestClassifier(**self.params)
    
    def get_tree_count(self):
        """
        Retorna número de árvores no modelo
        
        Returns:
            int: número de árvores
        """
        if not self.is_trained:
            return self.params['n_estimators']
        return len(self.model.estimators_)
    
    def get_oob_score(self):
        """
        Retorna Out-of-Bag score se disponível
        
        Returns:
            float: OOB score ou None
        """
        if hasattr(self.model, 'oob_score_'):
            return self.model.oob_score_
        return None