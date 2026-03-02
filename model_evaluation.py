"""
Model Evaluation Module
======================
This module handles model evaluation and metrics calculation
for fraud detection models.

Author: Fraud Detection Team
Version: 1.0.0
"""

import pandas as pd
import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report,
    precision_recall_curve, average_precision_score
)
from typing import Dict, Tuple, Optional
import logging
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ModelEvaluator:
    """
    Comprehensive model evaluator for fraud detection.
    
    This class provides detailed evaluation metrics and visualizations
    for assessing model performance.
    """
    
    def __init__(self):
        """Initialize the ModelEvaluator."""
        self.metrics = {}
        
    def calculate_metrics(self, y_true: np.ndarray, 
                         y_pred: np.ndarray,
                         y_pred_proba: Optional[np.ndarray] = None) -> Dict:
        """
        Calculate comprehensive evaluation metrics.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            y_pred_proba: Predicted probabilities (optional)
            
        Returns:
            Dictionary of metrics
        """
        metrics = {
            # Basic metrics
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, zero_division=0),
            'recall': recall_score(y_true, y_pred, zero_division=0),
            'f1_score': f1_score(y_true, y_pred, zero_division=0),
            
            # Confusion matrix
            'tn': int((y_true == 0) & (y_pred == 0).sum()),
            'fp': int((y_true == 0) & (y_pred == 1).sum()),
            'fn': int((y_true == 1) & (y_pred == 0).sum()),
            'tp': int((y_true == 1) & (y_pred == 1).sum()),
        }
        
        # Calculate derived metrics
        metrics['specificity'] = (
            metrics['tn'] / (metrics['tn'] + metrics['fp']) 
            if (metrics['tn'] + metrics['fp']) > 0 else 0
        )
        
        # False positive rate
        metrics['fpr'] = (
            metrics['fp'] / (metrics['fp'] + metrics['tn'])
            if (metrics['fp'] + metrics['tn']) > 0 else 0
        )
        
        # False negative rate
        metrics['fnr'] = (
            metrics['fn'] / (metrics['fn'] + metrics['tp'])
            if (metrics['fn'] + metrics['tp']) > 0 else 0
        )
        
        # ROC-AUC if probabilities provided
        if y_pred_proba is not None:
            try:
                metrics['roc_auc'] = roc_auc_score(y_true, y_pred_proba)
                metrics['average_precision'] = average_precision_score(y_true, y_pred_proba)
            except Exception as e:
                logger.warning(f"Could not calculate AUC: {e}")
                metrics['roc_auc'] = 0.0
                metrics['average_precision'] = 0.0
        
        self.metrics = metrics
        return metrics
    
    def get_confusion_matrix(self, y_true: np.ndarray, 
                           y_pred: np.ndarray) -> pd.DataFrame:
        """
        Get confusion matrix as DataFrame.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            
        Returns:
            Confusion matrix as DataFrame
        """
        cm = confusion_matrix(y_true, y_pred)
        
        cm_df = pd.DataFrame(
            cm,
            index=['Actual Normal', 'Actual Fraud'],
            columns=['Predicted Normal', 'Predicted Fraud']
        )
        
        return cm_df
    
    def get_classification_report(self, y_true: np.ndarray, 
                                 y_pred: np.ndarray) -> str:
        """
        Get detailed classification report.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            
        Returns:
            Classification report as string
        """
        report = classification_report(
            y_true, y_pred,
            target_names=['Normal', 'Fraud'],
            digits=4
        )
        return report
    
    def plot_confusion_matrix_streamlit(self, cm: pd.DataFrame) -> go.Figure:
        """
        Create interactive confusion matrix plot for Streamlit.
        
        Args:
            cm: Confusion matrix DataFrame
            
        Returns:
            Plotly figure
        """
        fig = px.imshow(
            cm.values,
            labels=dict(x="Predicted", y="Actual", color="Count"),
            x=cm.columns.tolist(),
            y=cm.index.tolist(),
            text_auto=True,
            color_continuous_scale='Blues'
        )
        
        fig.update_layout(
            title='Confusion Matrix',
            xaxis_title='Predicted Label',
            yaxis_title='True Label'
        )
        
        return fig
    
    def plot_roc_curve(self, y_true: np.ndarray, 
                      y_pred_proba: np.ndarray) -> go.Figure:
        """
        Create ROC curve plot.
        
        Args:
            y_true: True labels
            y_pred_proba: Predicted probabilities
            
        Returns:
            Plotly figure
        """
        from sklearn.metrics import roc_curve
        
        fpr, tpr, thresholds = roc_curve(y_true, y_pred_proba)
        roc_auc = roc_auc_score(y_true, y_pred_proba)
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=fpr, y=tpr,
            mode='lines',
            name=f'ROC Curve (AUC = {roc_auc:.4f})',
            line=dict(color='blue', width=2)
        ))
        
        fig.add_trace(go.Scatter(
            x=[0, 1], y=[0, 1],
            mode='lines',
            name='Random Classifier',
            line=dict(color='gray', width=1, dash='dash')
        ))
        
        fig.update_layout(
            title='ROC Curve',
            xaxis_title='False Positive Rate',
            yaxis_title='True Positive Rate',
            legend=dict(x=0.5, y=0.05, xanchor='center'),
            showlegend=True
        )
        
        return fig
    
    def plot_precision_recall_curve(self, y_true: np.ndarray, 
                                   y_pred_proba: np.ndarray) -> go.Figure:
        """
        Create Precision-Recall curve plot.
        
        Args:
            y_true: True labels
            y_pred_proba: Predicted probabilities
            
        Returns:
            Plotly figure
        """
        from sklearn.metrics import precision_recall_curve, auc
        
        precision, recall, thresholds = precision_recall_curve(y_true, y_pred_proba)
        pr_auc = auc(recall, precision)
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=recall, y=precision,
            mode='lines',
            name=f'PR Curve (AUC = {pr_auc:.4f})',
            line=dict(color='green', width=2)
        ))
        
        fig.update_layout(
            title='Precision-Recall Curve',
            xaxis_title='Recall',
            yaxis_title='Precision',
            legend=dict(x=0.5, y=0.05, xanchor='center'),
            showlegend=True
        )
        
        return fig
    
    def plot_feature_importance(self, feature_importance: pd.DataFrame,
                               top_n: int = 15) -> go.Figure:
        """
        Create feature importance bar chart.
        
        Args:
            feature_importance: DataFrame with feature and importance columns
            top_n: Number of top features to display
            
        Returns:
            Plotly figure
        """
        # Get top N features
        top_features = feature_importance.head(top_n)
        
        fig = px.bar(
            top_features,
            x='importance',
            y='feature',
            orientation='h',
            title=f'Top {top_n} Feature Importance',
            color='importance',
            color_continuous_scale='viridis'
        )
        
        fig.update_layout(
            xaxis_title='Importance Score',
            yaxis_title='Feature',
            yaxis=dict(autorange='reversed')
        )
        
        return fig
    
    def plot_class_distribution(self, y: np.ndarray) -> go.Figure:
        """
        Create class distribution plot.
        
        Args:
            y: Target variable
            
        Returns:
            Plotly figure
        """
        from collections import Counter
        
        counter = Counter(y)
        labels = ['Normal', 'Fraud']
        values = [counter[0], counter[1]]
        
        fig = go.Figure(data=[
            go.Bar(x=labels, y=values, marker_color=['green', 'red'])
        ])
        
        fig.update_layout(
            title='Class Distribution',
            xaxis_title='Class',
            yaxis_title='Count',
            showlegend=False
        )
        
        return fig
    
    def plot_prediction_distribution(self, y_pred_proba: np.ndarray,
                                    threshold: float = 0.5) -> go.Figure:
        """
        Create prediction probability distribution plot.
        
        Args:
            y_pred_proba: Predicted probabilities
            threshold: Classification threshold
            
        Returns:
            Plotly figure
        """
        fig = go.Figure()
        
        # Histogram of probabilities
        fig.add_trace(go.Histogram(
            x=y_pred_proba,
            nbinsx=50,
            name='Probability Distribution',
            marker_color='blue',
            opacity=0.7
        ))
        
        # Add threshold line
        fig.add_vline(
            x=threshold, 
            line_dash="dash", 
            line_color="red",
            annotation_text=f"Threshold: {threshold}"
        )
        
        fig.update_layout(
            title='Fraud Probability Distribution',
            xaxis_title='Fraud Probability',
            yaxis_title='Count',
            showlegend=False
        )
        
        return fig
    
    def evaluate(self, y_true: np.ndarray,
               y_pred: np.ndarray,
               y_pred_proba: Optional[np.ndarray] = None,
               feature_importance: Optional[pd.DataFrame] = None) -> Dict:
        """
        Complete model evaluation.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            y_pred_proba: Predicted probabilities
            feature_importance: Feature importance DataFrame
            
        Returns:
            Complete evaluation results
        """
        logger.info("Starting model evaluation")
        
        # Calculate metrics
        metrics = self.calculate_metrics(y_true, y_pred, y_pred_proba)
        
        # Generate reports
        confusion_mat = self.get_confusion_matrix(y_true, y_pred)
        report = self.get_classification_report(y_true, y_pred)
        
        results = {
            'metrics': metrics,
            'confusion_matrix': confusion_mat,
            'classification_report': report,
            'feature_importance': feature_importance
        }
        
        logger.info(f"Evaluation complete. ROC-AUC: {metrics.get('roc_auc', 0):.4f}")
        
        return results
    
    def print_evaluation_summary(self, results: Dict) -> None:
        """
        Print evaluation summary to console.
        
        Args:
            results: Evaluation results
        """
        metrics = results['metrics']
        
        print("\n" + "="*50)
        print("MODEL EVALUATION SUMMARY")
        print("="*50)
        
        print(f"\nAccuracy:  {metrics['accuracy']:.4f}")
        print(f"Precision: {metrics['precision']:.4f}")
        print(f"Recall:    {metrics['recall']:.4f}")
        print(f"F1-Score: {metrics['f1_score']:.4f}")
        
        if 'roc_auc' in metrics:
            print(f"ROC-AUC:   {metrics['roc_auc']:.4f}")
        
        if 'average_precision' in metrics:
            print(f"Avg Precision: {metrics['average_precision']:.4f}")
        
        print("\nConfusion Matrix:")
        print(results['confusion_matrix'])
        
        print("\nClassification Report:")
        print(results['classification_report'])
        
        print("="*50 + "\n")


# Convenience function for quick evaluation
def quick_evaluate(y_true: np.ndarray, y_pred: np.ndarray,
                  y_pred_proba: np.ndarray = None) -> Dict:
    """
    Quick model evaluation.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        y_pred_proba: Predicted probabilities
        
    Returns:
        Evaluation metrics
    """
    evaluator = ModelEvaluator()
    return evaluator.calculate_metrics(y_true, y_pred, y_pred_proba)


# Testing the evaluator
if __name__ == "__main__":
    from data_preprocessing import create_sample_dataset
    from model_training import ModelTrainer
    from feature_engineering import FeatureEngineer
    from sklearn.model_selection import train_test_split
    
    print("Testing ModelEvaluator...")
    
    # Create sample data
    df = create_sample_dataset(n_samples=5000, fraud_ratio=0.02)
    
    # Prepare features
    X = df.drop('Class', axis=1)
    y = df['Class']
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Simple model for testing
    trainer = ModelTrainer()
    scale_pos_weight = trainer.calculate_scale_pos_weight(y_train.values)
    params = trainer.get_default_params(scale_pos_weight)
    params['n_estimators'] = 50  # Quick training
    
    from xgboost import XGBClassifier
    model = XGBClassifier(**params)
    model.fit(X_train, y_train)
    
    # Predictions
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    # Evaluate
    evaluator = ModelEvaluator()
    results = evaluator.evaluate(
        y_test.values, 
        y_pred, 
        y_pred_proba,
        feature_importance=pd.DataFrame({
            'feature': X.columns,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)
    )
    
    evaluator.print_evaluation_summary(results)
    
    print("ModelEvaluator test completed successfully!")
