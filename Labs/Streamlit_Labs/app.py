import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.datasets import load_iris, load_wine, load_breast_cancer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib
import os

# Page config
st.set_page_config(
    page_title="ML Model Explorer",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 2px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        padding-left: 20px;
        padding-right: 20px;
    }
</style>
""", unsafe_allow_html=True)

# Load datasets
@st.cache_data
def load_datasets():
    """Load and cache datasets"""
    datasets = {
        'Iris': load_iris(),
        'Wine': load_wine(),
        'Breast Cancer': load_breast_cancer()
    }
    return datasets

# Model training functions
@st.cache_data
def train_models(X, y, dataset_name):
    """Train multiple models and return results"""
    models = {
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
        'SVM': SVC(probability=True, random_state=42)
    }
    
    results = {}
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    for name, model in models.items():
        # Train model
        model.fit(X_train, y_train)
        
        # Predictions
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test) if hasattr(model, 'predict_proba') else None
        
        # Cross-validation score
        cv_scores = cross_val_score(model, X, y, cv=5)
        
        results[name] = {
            'model': model,
            'accuracy': accuracy_score(y_test, y_pred),
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std(),
            'y_test': y_test,
            'y_pred': y_pred,
            'y_pred_proba': y_pred_proba,
            'feature_names': X.columns.tolist() if hasattr(X, 'columns') else [f'Feature_{i}' for i in range(X.shape[1])]
        }
    
    return results, X_test, y_test

def main():
    # Header
    st.markdown('<h1 class="main-header">🤖 ML Model Explorer Dashboard</h1>', unsafe_allow_html=True)
    
    # Sidebar
    st.sidebar.title("⚙️ Configuration")
    
    # Dataset selection
    datasets = load_datasets()
    dataset_name = st.sidebar.selectbox(
        "Select Dataset",
        list(datasets.keys()),
        help="Choose a dataset to explore and model"
    )
    
    # Load selected dataset
    dataset = datasets[dataset_name]
    X = pd.DataFrame(dataset.data, columns=dataset.feature_names)
    y = dataset.target
    target_names = dataset.target_names
    
    # Sidebar info
    st.sidebar.markdown("### Dataset Info")
    st.sidebar.metric("Samples", X.shape[0])
    st.sidebar.metric("Features", X.shape[1])
    st.sidebar.metric("Classes", len(np.unique(y)))
    
    # Main content tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 Data Exploration", 
        "🤖 Model Training", 
        "📈 Model Comparison", 
        "🔮 Predictions", 
        "💾 Model Management"
    ])
    
    with tab1:
        st.header("📊 Data Exploration")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Dataset Overview")
            st.dataframe(X.head(10))
            
            st.subheader("Basic Statistics")
            st.dataframe(X.describe())
        
        with col2:
            st.subheader("Target Distribution")
            target_counts = pd.Series(y).value_counts()
            fig_pie = px.pie(
                values=target_counts.values, 
                names=[target_names[i] for i in target_counts.index],
                title="Class Distribution"
            )
            st.plotly_chart(fig_pie, use_container_width=True)
        
        # Feature visualization
        st.subheader("Feature Analysis")
        
        # Feature selection for plotting
        feature1 = st.selectbox("Select Feature 1", X.columns, key="feat1")
        feature2 = st.selectbox("Select Feature 2", X.columns, key="feat2")
        
        # Scatter plot
        fig_scatter = px.scatter(
            X, x=feature1, y=feature2, 
            color=[target_names[i] for i in y],
            title=f"{feature1} vs {feature2}",
            labels={'color': 'Class'}
        )
        st.plotly_chart(fig_scatter, use_container_width=True)
        
        # Correlation heatmap
        st.subheader("Feature Correlation")
        corr_matrix = X.corr()
        fig_heatmap = px.imshow(
            corr_matrix, 
            text_auto=True, 
            aspect="auto",
            title="Feature Correlation Matrix"
        )
        st.plotly_chart(fig_heatmap, use_container_width=True)
    
    with tab2:
        st.header("🤖 Model Training")
        
        if st.button("🚀 Train All Models", type="primary"):
            with st.spinner("Training models..."):
                results, X_test, y_test = train_models(X, y, dataset_name)
                st.session_state['model_results'] = results
                st.session_state['X_test'] = X_test
                st.session_state['y_test'] = y_test
                st.success("Models trained successfully!")
        
        if 'model_results' in st.session_state:
            st.subheader("Training Results")
            
            # Display results in columns
            cols = st.columns(len(st.session_state['model_results']))
            for i, (name, result) in enumerate(st.session_state['model_results'].items()):
                with cols[i]:
                    st.metric(
                        label=name,
                        value=f"{result['accuracy']:.3f}",
                        delta=f"CV: {result['cv_mean']:.3f} ± {result['cv_std']:.3f}"
                    )
    
    with tab3:
        st.header("📈 Model Comparison")
        
        if 'model_results' in st.session_state:
            results = st.session_state['model_results']
            
            # Accuracy comparison
            model_names = list(results.keys())
            accuracies = [results[name]['accuracy'] for name in model_names]
            cv_means = [results[name]['cv_mean'] for name in model_names]
            cv_stds = [results[name]['cv_std'] for name in model_names]
            
            # Bar chart
            fig_comparison = go.Figure()
            fig_comparison.add_trace(go.Bar(
                name='Test Accuracy',
                x=model_names,
                y=accuracies,
                marker_color='lightblue'
            ))
            fig_comparison.add_trace(go.Bar(
                name='CV Mean',
                x=model_names,
                y=cv_means,
                marker_color='lightcoral'
            ))
            
            fig_comparison.update_layout(
                title="Model Performance Comparison",
                xaxis_title="Models",
                yaxis_title="Accuracy",
                barmode='group'
            )
            st.plotly_chart(fig_comparison, use_container_width=True)
            
            # Detailed results table
            st.subheader("Detailed Results")
            comparison_data = []
            for name, result in results.items():
                comparison_data.append({
                    'Model': name,
                    'Test Accuracy': f"{result['accuracy']:.4f}",
                    'CV Mean': f"{result['cv_mean']:.4f}",
                    'CV Std': f"{result['cv_std']:.4f}"
                })
            
            st.dataframe(pd.DataFrame(comparison_data), use_container_width=True)
            
            # Confusion matrices
            st.subheader("Confusion Matrices")
            selected_model = st.selectbox("Select Model for Confusion Matrix", model_names)
            
            if selected_model:
                cm = confusion_matrix(
                    st.session_state['y_test'], 
                    results[selected_model]['y_pred']
                )
                
                fig_cm = px.imshow(
                    cm,
                    text_auto=True,
                    aspect="auto",
                    title=f"Confusion Matrix - {selected_model}",
                    labels=dict(x="Predicted", y="Actual")
                )
                st.plotly_chart(fig_cm, use_container_width=True)
        else:
            st.info("Please train models first in the 'Model Training' tab.")
    
    with tab4:
        st.header("🔮 Make Predictions")
        
        if 'model_results' in st.session_state:
            results = st.session_state['model_results']
            selected_model = st.selectbox("Select Model for Prediction", list(results.keys()))
            
            if selected_model:
                model = results[selected_model]['model']
                feature_names = results[selected_model]['feature_names']
                
                st.subheader("Input Features")
                
                # Create input form
                input_data = {}
                cols = st.columns(2)
                
                for i, feature in enumerate(feature_names):
                    with cols[i % 2]:
                        input_data[feature] = st.number_input(
                            f"{feature}",
                            value=float(X[feature].mean()),
                            step=0.01,
                            key=f"input_{feature}"
                        )
                
                if st.button("🔮 Predict", type="primary"):
                    # Prepare input
                    input_array = np.array([list(input_data.values())]).reshape(1, -1)
                    
                    # Make prediction
                    prediction = model.predict(input_array)[0]
                    prediction_proba = model.predict_proba(input_array)[0] if hasattr(model, 'predict_proba') else None
                    
                    # Display results
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.metric("Predicted Class", target_names[prediction])
                        st.metric("Class Index", prediction)
                    
                    with col2:
                        if prediction_proba is not None:
                            st.metric("Confidence", f"{max(prediction_proba):.3f}")
                            
                            # Probability distribution
                            prob_df = pd.DataFrame({
                                'Class': target_names,
                                'Probability': prediction_proba
                            })
                            
                            fig_prob = px.bar(
                                prob_df, x='Class', y='Probability',
                                title="Prediction Probabilities"
                            )
                            st.plotly_chart(fig_prob, use_container_width=True)
        else:
            st.info("Please train models first in the 'Model Training' tab.")
    
    with tab5:
        st.header("💾 Model Management")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Save Model")
            if 'model_results' in st.session_state:
                model_name = st.selectbox("Select Model to Save", list(st.session_state['model_results'].keys()))
                
                if st.button("💾 Save Model"):
                    os.makedirs('models', exist_ok=True)
                    model = st.session_state['model_results'][model_name]['model']
                    joblib.dump(model, f'models/{model_name.lower().replace(" ", "_")}.pkl')
                    st.success(f"Model saved as models/{model_name.lower().replace(' ', '_')}.pkl")
            else:
                st.info("No models available to save.")
        
        with col2:
            st.subheader("Load Model")
            if os.path.exists('models'):
                model_files = [f for f in os.listdir('models') if f.endswith('.pkl')]
                if model_files:
                    selected_file = st.selectbox("Select Model File", model_files)
                    
                    if st.button("📂 Load Model"):
                        model = joblib.load(f'models/{selected_file}')
                        st.success(f"Model loaded: {selected_file}")
                        st.json({"Model Type": str(type(model).__name__)})
                else:
                    st.info("No saved models found.")
            else:
                st.info("No models directory found.")

if __name__ == "__main__":
    main()
