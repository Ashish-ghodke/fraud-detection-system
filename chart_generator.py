"""
Chart Generator Module
=====================
This module handles creating all dashboard charts and visualizations.

Author: Fraud Detection Team
Version: 1.0.0
"""

import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from typing import Dict, List, Optional
import logging

from .constants import (
    COLOR_FRAUD,
    COLOR_NORMAL,
    COLOR_WARNING,
    COLOR_CRITICAL,
    CHART_HEIGHT,
    TOP_SUSPICIOUS_TRANSACTIONS
)

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ChartGenerator:
    """
    Chart generator for fraud detection dashboard.
    
    This class creates all the visualizations needed for the dashboard:
    - Bar charts
    - Pie charts
    - Histograms
    - Line charts
    - Data tables
    """
    
    def __init__(self):
        """Initialize the ChartGenerator."""
        self.chart_height = CHART_HEIGHT
        
    def create_bar_chart(self, df: pd.DataFrame, 
                        title: str = "Fraud vs Non-Fraud") -> go.Figure:
        """
        Create bar chart for fraud distribution.
        
        Args:
            df: DataFrame with predictions
            title: Chart title
            
        Returns:
            Plotly figure
        """
        # Count fraud and non-fraud
        if 'Prediction' in df.columns:
            fraud_count = (df['Prediction'] == 1).sum()
            normal_count = (df['Prediction'] == 0).sum()
        else:
            # Use Class column if available
            fraud_count = (df['Class'] == 1).sum() if 'Class' in df.columns else 0
            normal_count = (df['Class'] == 0).sum() if 'Class' in df.columns else len(df)
        
        fig = go.Figure(data=[
            go.Bar(
                x=['Normal', 'Fraud'],
                y=[normal_count, fraud_count],
                marker_color=[COLOR_NORMAL, COLOR_FRAUD],
                text=[normal_count, fraud_count],
                textposition='auto'
            )
        ])
        
        fig.update_layout(
            title=title,
            xaxis_title='Transaction Type',
            yaxis_title='Count',
            height=self.chart_height,
            showlegend=False
        )
        
        return fig
    
    def create_pie_chart(self, df: pd.DataFrame,
                       title: str = "Fraud Distribution") -> go.Figure:
        """
        Create pie chart for fraud percentage.
        
        Args:
            df: DataFrame with predictions
            title: Chart title
            
        Returns:
            Plotly figure
        """
        # Count fraud and non-fraud
        if 'Prediction' in df.columns:
            fraud_count = (df['Prediction'] == 1).sum()
            normal_count = (df['Prediction'] == 0).sum()
        else:
            fraud_count = (df['Class'] == 1).sum() if 'Class' in df.columns else 0
            normal_count = (df['Class'] == 0).sum() if 'Class' in df.columns else len(df)
        
        labels = ['Normal', 'Fraud']
        values = [normal_count, fraud_count]
        colors = [COLOR_NORMAL, COLOR_FRAUD]
        
        fig = go.Figure(data=[go.Pie(
            labels=labels,
            values=values,
            marker=dict(colors=colors),
            textinfo='label+percent',
            hole=0.4
        )])
        
        fig.update_layout(
            title=title,
            height=self.chart_height,
            showlegend=True
        )
        
        return fig
    
    def create_histogram(self, df: pd.DataFrame,
                        column: str = 'Amount',
                        title: str = "Transaction Amount Distribution") -> go.Figure:
        """
        Create histogram for transaction amounts.
        
        Args:
            df: DataFrame with transactions
            column: Column to plot
            title: Chart title
            
        Returns:
            Plotly figure
        """
        if column not in df.columns:
            logger.warning(f"Column {column} not found in DataFrame")
            return go.Figure()
        
        # Separate fraud and non-fraud
        if 'Prediction' in df.columns:
            fraud_data = df[df['Prediction'] == 1][column]
            normal_data = df[df['Prediction'] == 0][column]
        elif 'Class' in df.columns:
            fraud_data = df[df['Class'] == 1][column]
            normal_data = df[df['Class'] == 0][column]
        else:
            fraud_data = []
            normal_data = df[column]
        
        fig = go.Figure()
        
        # Add normal transactions
        fig.add_trace(go.Histogram(
            x=normal_data,
            name='Normal',
            marker_color=COLOR_NORMAL,
            opacity=0.7,
            nbinsx=50
        ))
        
        # Add fraud transactions
        if len(fraud_data) > 0:
            fig.add_trace(go.Histogram(
                x=fraud_data,
                name='Fraud',
                marker_color=COLOR_FRAUD,
                opacity=0.7,
                nbinsx=50
            ))
        
        fig.update_layout(
            title=title,
            xaxis_title=column,
            yaxis_title='Count',
            height=self.chart_height,
            barmode='overlay',
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        
        return fig
    
    def create_line_chart(self, df: pd.DataFrame,
                         x_col: str = 'Time',
                         y_col: str = 'Fraud_Probability',
                         title: str = "Suspicious Trend Over Time") -> go.Figure:
        """
        Create line chart for trends.
        
        Args:
            df: DataFrame with data
            x_col: X-axis column
            y_col: Y-axis column
            title: Chart title
            
        Returns:
            Plotly figure
        """
        if x_col not in df.columns or y_col not in df.columns:
            logger.warning(f"Required columns not found in DataFrame")
            return go.Figure()
        
        # Sort by time
        df_sorted = df.sort_values(x_col)
        
        fig = px.line(
            df_sorted,
            x=x_col,
            y=y_col,
            title=title,
            height=self.chart_height
        )
        
        fig.update_layout(
            xaxis_title=x_col,
            yaxis_title=y_col
        )
        
        return fig
    
    def create_scatter_plot(self, df: pd.DataFrame,
                          x_col: str = 'Amount',
                          y_col: str = 'Fraud_Probability',
                          color_col: str = 'Prediction',
                          title: str = "Amount vs Fraud Probability") -> go.Figure:
        """
        Create scatter plot.
        
        Args:
            df: DataFrame with data
            x_col: X-axis column
            y_col: Y-axis column
            color_col: Column to color by
            title: Chart title
            
        Returns:
            Plotly figure
        """
        if x_col not in df.columns or y_col not in df.columns:
            logger.warning(f"Required columns not found")
            return go.Figure()
        
        fig = px.scatter(
            df,
            x=x_col,
            y=y_col,
            color=color_col if color_col in df.columns else None,
            title=title,
            height=self.chart_height,
            color_discrete_map={0: COLOR_NORMAL, 1: COLOR_FRAUD}
        )
        
        fig.update_layout(
            xaxis_title=x_col,
            yaxis_title=y_col
        )
        
        return fig
    
    def create_top_suspicious_table(self, df: pd.DataFrame,
                                   n: int = TOP_SUSPICIOUS_TRANSACTIONS) -> pd.DataFrame:
        """
        Create table of top suspicious transactions.
        
        Args:
            df: DataFrame with predictions
            n: Number of top transactions
            
        Returns:
            DataFrame with top suspicious transactions
        """
        if 'Fraud_Probability' not in df.columns:
            logger.warning("No fraud probability column found")
            return pd.DataFrame()
        
        # Get top N suspicious
        top_suspicious = df.nlargest(n, 'Fraud_Probability')
        
        # Select relevant columns
        display_cols = []
        
        # Add available columns
        priority_cols = ['Time', 'Amount', 'Fraud_Probability', 'Prediction', 'Risk_Level']
        for col in priority_cols:
            if col in top_suspicious.columns:
                display_cols.append(col)
        
        # Add V columns if available
        v_cols = [col for col in top_suspicious.columns if str(col).startswith('V')]
        display_cols.extend(v_cols[:5])  # Add first 5 V columns
        
        # Get unique columns
        display_cols = list(dict.fromkeys(display_cols))
        
        # Filter to existing columns
        display_cols = [col for col in display_cols if col in top_suspicious.columns]
        
        if display_cols:
            return top_suspicious[display_cols]
        else:
            return top_suspicious
    
    def create_metrics_cards(self, summary: Dict) -> Dict:
        """
        Create metric cards data.
        
        Args:
            summary: Summary dictionary
            
        Returns:
            Dictionary with metric values
        """
        metrics = {
            'total_transactions': summary.get('total_transactions', 0),
            'fraud_transactions': summary.get('fraud_transactions', 0),
            'fraud_percentage': summary.get('fraud_percentage', 0),
            'normal_transactions': summary.get('normal_transactions', 0),
            'avg_fraud_probability': summary.get('avg_fraud_probability', 0)
        }
        
        return metrics
    
    def create_risk_distribution_chart(self, df: pd.DataFrame) -> go.Figure:
        """
        Create risk level distribution chart.
        
        Args:
            df: DataFrame with risk levels
            
        Returns:
            Plotly figure
        """
        if 'Risk_Level' not in df.columns:
            # Create risk levels based on probability
            df = df.copy()
            df['Risk_Level'] = pd.cut(
                df['Fraud_Probability'],
                bins=[0, 0.2, 0.5, 0.8, 1.0],
                labels=['Low', 'Medium', 'High', 'Critical']
            )
        
        risk_counts = df['Risk_Level'].value_counts()
        
        color_map = {
            'Low': COLOR_NORMAL,
            'Medium': COLOR_WARNING,
            'High': COLOR_FRAUD,
            'Critical': COLOR_CRITICAL
        }
        
        colors = [color_map.get(x, COLOR_NORMAL) for x in risk_counts.index]
        
        fig = go.Figure(data=[
            go.Bar(
                x=risk_counts.index,
                y=risk_counts.values,
                marker_color=colors,
                text=risk_counts.values,
                textposition='auto'
            )
        ])
        
        fig.update_layout(
            title='Risk Level Distribution',
            xaxis_title='Risk Level',
            yaxis_title='Count',
            height=self.chart_height,
            showlegend=False
        )
        
        return fig
    
    def create_feature_importance_chart(self, feature_importance: pd.DataFrame) -> go.Figure:
        """
        Create feature importance chart.
        
        Args:
            feature_importance: DataFrame with feature importance
            
        Returns:
            Plotly figure
        """
        if feature_importance is None or feature_importance.empty:
            return go.Figure()
        
        # Get top 15 features
        top_features = feature_importance.head(15)
        
        fig = px.bar(
            top_features,
            x='importance',
            y='feature',
            orientation='h',
            title='Top 15 Feature Importance',
            color='importance',
            color_continuous_scale='viridis'
        )
        
        fig.update_layout(
            xaxis_title='Importance Score',
            yaxis_title='Feature',
            yaxis=dict(autorange='reversed'),
            height=self.chart_height
        )
        
        return fig


# Testing the chart generator
if __name__ == "__main__":
    from file_handler import load_sample_data
    
    print("Testing ChartGenerator...")
    
    # Load sample data
    df = load_sample_data()
    
    # Add prediction columns
    df['Prediction'] = df['Class']
    df['Fraud_Probability'] = np.random.uniform(0, 1, len(df))
    df['Fraud_Probability'] = np.where(df['Prediction'] == 1, 
                                        np.random.uniform(0.5, 1, len(df)),
                                        np.random.uniform(0, 0.3, len(df)))
    
    # Test chart generation
    generator = ChartGenerator()
    
    # Test bar chart
    bar_fig = generator.create_bar_chart(df)
    print("Bar chart created")
    
    # Test pie chart
    pie_fig = generator.create_pie_chart(df)
    print("Pie chart created")
    
    # Test histogram
    hist_fig = generator.create_histogram(df)
    print("Histogram created")
    
    # Test top suspicious table
    top_suspicious = generator.create_top_suspicious_table(df)
    print(f"Top suspicious: {len(top_suspicious)} rows")
    
    print("\nChartGenerator test completed successfully!")
