"""
Streamlit Dashboard for Boston Housing Price Prediction
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import joblib
import os
import sys

# Add src to path
sys.path.append('src')
from data_loader import BostonHousingDataLoader
from model import BostonHousingModel
from predict import BostonHousingPredictor


def load_data():
    """Load the Boston Housing dataset"""
    data_loader = BostonHousingDataLoader()
    X_train, X_test, y_train, y_test = data_loader.load_data()
    feature_names = data_loader.get_feature_names()
    
    # Create full dataset for visualization
    from sklearn.datasets import load_boston
    boston = load_boston()
    df = pd.DataFrame(boston.data, columns=boston.feature_names)
    df['MEDV'] = boston.target
    
    return df, feature_names, X_train, X_test, y_train, y_test


def load_model():
    """Load the trained model"""
    try:
        predictor = BostonHousingPredictor()
        predictor.load_model_and_scaler()
        return predictor
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None


def main():
    """Main Streamlit app"""
    st.set_page_config(
        page_title="Boston Housing Price Prediction",
        page_icon="🏠",
        layout="wide"
    )
    
    st.title("🏠 Boston Housing Price Prediction Dashboard")
    st.markdown("---")
    
    # Load data
    with st.spinner("Loading data..."):
        df, feature_names, X_train, X_test, y_train, y_test = load_data()
    
    # Sidebar
    st.sidebar.title("Navigation")
    page = st.sidebar.selectbox(
        "Choose a page",
        ["🏠 Home", "📊 Data Exploration", "🤖 Model Prediction", "📈 Model Performance"]
    )
    
    if page == "🏠 Home":
        st.header("Welcome to Boston Housing Price Prediction")
        st.markdown("""
        This dashboard provides insights into the Boston Housing dataset and allows you to predict house prices using a Ridge Regression model.
        
        ### Features:
        - **Data Exploration**: Visualize the dataset and understand feature distributions
        - **Model Prediction**: Input house features to get price predictions
        - **Model Performance**: View model metrics and feature importance
        """)
        
        # Display dataset info
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Houses", len(df))
        with col2:
            st.metric("Features", len(feature_names))
        with col3:
            st.metric("Avg Price", f"${df['MEDV'].mean():.2f}")
    
    elif page == "📊 Data Exploration":
        st.header("Data Exploration")
        
        # Dataset overview
        st.subheader("Dataset Overview")
        st.dataframe(df.head(10))
        
        # Feature distributions
        st.subheader("Feature Distributions")
        selected_features = st.multiselect(
            "Select features to visualize",
            feature_names,
            default=feature_names[:6]
        )
        
        if selected_features:
            fig = make_subplots(
                rows=(len(selected_features) + 2) // 3,
                cols=3,
                subplot_titles=selected_features
            )
            
            for i, feature in enumerate(selected_features):
                row = i // 3 + 1
                col = i % 3 + 1
                fig.add_trace(
                    go.Histogram(x=df[feature], name=feature),
                    row=row, col=col
                )
            
            fig.update_layout(height=300 * ((len(selected_features) + 2) // 3))
            st.plotly_chart(fig, use_container_width=True)
        
        # Correlation heatmap
        st.subheader("Feature Correlation")
        corr_matrix = df.corr()
        fig = px.imshow(
            corr_matrix,
            text_auto=True,
            aspect="auto",
            title="Correlation Matrix"
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Price distribution
        st.subheader("Price Distribution")
        fig = px.histogram(
            df, x='MEDV',
            title="Distribution of House Prices",
            labels={'MEDV': 'Price ($1000s)', 'count': 'Frequency'}
        )
        st.plotly_chart(fig, use_container_width=True)
    
    elif page == "🤖 Model Prediction":
        st.header("House Price Prediction")
        
        # Load model
        predictor = load_model()
        
        if predictor is not None:
            st.success("Model loaded successfully!")
            
            # Input form
            st.subheader("Enter House Features")
            
            # Create input form
            features = {}
            col1, col2 = st.columns(2)
            
            with col1:
                features['CRIM'] = st.number_input("Crime Rate (CRIM)", min_value=0.0, value=0.02731)
                features['ZN'] = st.number_input("Residential Land (ZN)", min_value=0.0, value=0.0)
                features['INDUS'] = st.number_input("Industrial Land (INDUS)", min_value=0.0, value=7.07)
                features['CHAS'] = st.number_input("Charles River (CHAS)", min_value=0.0, max_value=1.0, value=0.0)
                features['NOX'] = st.number_input("Nitric Oxide (NOX)", min_value=0.0, value=0.469)
                features['RM'] = st.number_input("Rooms per Dwelling (RM)", min_value=0.0, value=6.421)
                features['AGE'] = st.number_input("Age (AGE)", min_value=0.0, value=78.9)
            
            with col2:
                features['DIS'] = st.number_input("Distance to Employment (DIS)", min_value=0.0, value=4.9671)
                features['RAD'] = st.number_input("Highway Accessibility (RAD)", min_value=0.0, value=2.0)
                features['TAX'] = st.number_input("Property Tax Rate (TAX)", min_value=0.0, value=242.0)
                features['PTRATIO'] = st.number_input("Pupil-Teacher Ratio (PTRATIO)", min_value=0.0, value=17.8)
                features['B'] = st.number_input("Black Population (B)", min_value=0.0, value=396.90)
                features['LSTAT'] = st.number_input("Lower Status % (LSTAT)", min_value=0.0, value=9.14)
            
            # Predict button
            if st.button("Predict Price", type="primary"):
                try:
                    # Convert to list in correct order
                    feature_values = [features[name] for name in feature_names]
                    prediction = predictor.predict_single(feature_values)
                    
                    st.success(f"Predicted House Price: **${prediction:.2f}** (in $1000s)")
                    st.info(f"Actual Price Range: ${df['MEDV'].min():.2f} - ${df['MEDV'].max():.2f}")
                    
                    # Show prediction confidence
                    price_percentile = (prediction - df['MEDV'].min()) / (df['MEDV'].max() - df['MEDV'].min()) * 100
                    st.metric("Price Percentile", f"{price_percentile:.1f}%")
                    
                except Exception as e:
                    st.error(f"Prediction error: {e}")
        else:
            st.error("Model not found. Please train the model first.")
    
    elif page == "📈 Model Performance":
        st.header("Model Performance")
        
        # Load model
        predictor = load_model()
        
        if predictor is not None:
            # Load feature importance
            try:
                importance_df = pd.read_csv('models/feature_importance.csv')
                
                st.subheader("Feature Importance")
                fig = px.bar(
                    importance_df,
                    x='coefficient',
                    y='feature',
                    orientation='h',
                    title="Feature Coefficients (Ridge Regression)",
                    labels={'coefficient': 'Coefficient Value', 'feature': 'Feature'}
                )
                fig.update_layout(yaxis={'categoryorder': 'total ascending'})
                st.plotly_chart(fig, use_container_width=True)
                
            except FileNotFoundError:
                st.warning("Feature importance file not found. Please train the model first.")
        
        # Model metrics (if available)
        st.subheader("Model Evaluation")
        st.info("Train the model to see performance metrics here.")
        
        # Sample predictions vs actual
        st.subheader("Sample Predictions")
        if st.button("Generate Sample Predictions"):
            try:
                # Get sample predictions
                sample_indices = np.random.choice(len(X_test), 10, replace=False)
                sample_X = X_test[sample_indices]
                sample_y = y_test.iloc[sample_indices]
                
                predictions = predictor.predict_batch(sample_X)
                
                # Create comparison dataframe
                comparison_df = pd.DataFrame({
                    'Actual': sample_y.values,
                    'Predicted': predictions,
                    'Error': np.abs(sample_y.values - predictions)
                })
                
                st.dataframe(comparison_df)
                
                # Visualization
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=sample_y.values,
                    y=predictions,
                    mode='markers',
                    name='Predictions',
                    marker=dict(size=10, color='blue')
                ))
                fig.add_trace(go.Scatter(
                    x=[sample_y.min(), sample_y.max()],
                    y=[sample_y.min(), sample_y.max()],
                    mode='lines',
                    name='Perfect Prediction',
                    line=dict(dash='dash', color='red')
                ))
                fig.update_layout(
                    title="Actual vs Predicted Prices",
                    xaxis_title="Actual Price ($1000s)",
                    yaxis_title="Predicted Price ($1000s)"
                )
                st.plotly_chart(fig, use_container_width=True)
                
            except Exception as e:
                st.error(f"Error generating predictions: {e}")


if __name__ == "__main__":
    main()
