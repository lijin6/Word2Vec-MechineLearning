import pandas as pd
import numpy as np
import logging
from gensim.models import Word2Vec
from xgboost import XGBClassifier
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
import warnings

# 配置日志和忽略警告
warnings.filterwarnings('ignore', category=UserWarning)
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

class XGBoostSentimentAnalyzer:
    def __init__(self, w2v_model_path):
        """Initialize with Word2Vec model"""
        try:
            self.w2v_model = Word2Vec.load(w2v_model_path)
            logging.info(f"Loaded Word2Vec model: {w2v_model_path}")
        except Exception as e:
            logging.error(f"Failed to load model: {str(e)}")
            raise

    def load_data(self, data_path, text_col='words', label_col='type'):
        """Load and preprocess data"""
        try:
            df = pd.read_csv(data_path, sep='\t')
            logging.info(f"Loaded data: {data_path}")
            
            # Clean data
            df = df.dropna(subset=[text_col, label_col])
            df = shuffle(df).reset_index(drop=True)
            
            # Convert labels to 0/1
            df[label_col] = df[label_col].apply(lambda x: 0 if x == -1 else 1)
            
            return df[text_col], df[label_col]
        except Exception as e:
            logging.error(f"Data loading failed: {str(e)}")
            raise

    def text_to_vector(self, texts):
        """Convert texts to Word2Vec average vectors"""
        vectors = []
        for text in texts:
            words = str(text).split()
            word_vecs = [self.w2v_model.wv[word] for word in words if word in self.w2v_model.wv]
            vectors.append(np.mean(word_vecs, axis=0) if word_vecs else np.zeros(self.w2v_model.vector_size))
        return np.array(vectors)

    def train_evaluate(self, X, y, test_size=0.2, cv=5):
        """Train and evaluate XGBoost model"""
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42
        )
        
        # Initialize XGBoost (removed unused parameter)
        model = XGBClassifier(
            n_estimators=200,
            max_depth=5,
            learning_rate=0.1,
            subsample=0.8,
            eval_metric='logloss'
        )
        
        logging.info("Training XGBoost model...")
        model.fit(X_train, y_train)
        
        # Cross-validation
        cv_scores = cross_val_score(model, X, y, cv=cv, scoring='f1')
        logging.info(f"{cv}-fold CV F1: {np.mean(cv_scores):.4f} (±{np.std(cv_scores):.4f})")
        
        # Evaluation
        y_pred = model.predict(X_test)
        self._evaluate_metrics(y_test, y_pred)
        
        # Plotting (English labels only)
        self._plot_feature_importance(model)
        
        return model

    def _evaluate_metrics(self, y_true, y_pred):
        """Calculate evaluation metrics"""
        metrics = {
            'Accuracy': accuracy_score(y_true, y_pred),
            'Precision': precision_score(y_true, y_pred),
            'Recall': recall_score(y_true, y_pred),
            'F1': f1_score(y_true, y_pred)
        }
        
        logging.info("Evaluation Results:")
        for name, score in metrics.items():
            logging.info(f"{name}: {score:.4f}")
        
        return metrics

    def _plot_feature_importance(self, model, top_n=20):
        """Plot feature importance (English only)"""
        plt.figure(figsize=(10, 6))
        feat_imp = pd.Series(model.feature_importances_).sort_values(ascending=False)
        feat_imp[:top_n].plot(kind='barh', color='steelblue')
        
        # English labels only
        plt.title(f'Top {top_n} Important Features')
        plt.xlabel('Importance Score')
        plt.ylabel('Feature Index')
        
        plt.tight_layout()
        plt.savefig('xgboost_feature_importance.png', dpi=300)
        plt.close()
        logging.info("Saved feature importance plot as 'xgboost_feature_importance.png'")

if __name__ == '__main__':
    # Configure paths
    DATA_PATH = './data/2/words_type.csv'
    W2V_MODEL_PATH = './data/2/w2v.bin'
    
    try:
        # Initialize analyzer
        analyzer = XGBoostSentimentAnalyzer(W2V_MODEL_PATH)
        
        # Load data
        texts, labels = analyzer.load_data(DATA_PATH)
        
        # Feature engineering
        X = analyzer.text_to_vector(texts)
        logging.info(f"Feature matrix shape: {X.shape} (samples×features)")
        
        # Train and evaluate
        model = analyzer.train_evaluate(X, labels)
        
        # Example predictions
        test_samples = ["good service", "bad experience"]
        test_vectors = analyzer.text_to_vector(test_samples)
        predictions = model.predict(test_vectors)
        
        logging.info("Prediction Examples:")
        for text, pred in zip(test_samples, predictions):
            sentiment = "Positive" if pred == 1 else "Negative"
            logging.info(f"Text: '{text}' → Prediction: {sentiment}")
            
    except Exception as e:
        logging.error(f"Error: {str(e)}")