import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.datasets import load_iris, load_wine, load_breast_cancer
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import plotly.figure_factory as ff

def main():
    st.set_page_config(
        page_title="Advanced Analytics",
        page_icon="📊",
        layout="wide"
    )
    
    st.title("📊 Advanced Analytics")
    st.markdown("Dimensionality reduction, clustering, and advanced visualizations")
    
    # Load datasets
    @st.cache_data
    def load_datasets():
        datasets = {
            'Iris': load_iris(),
            'Wine': load_wine(),
            'Breast Cancer': load_breast_cancer()
        }
        return datasets
    
    datasets = load_datasets()
    dataset_name = st.selectbox("Select Dataset", list(datasets.keys()))
    dataset = datasets[dataset_name]
    
    X = pd.DataFrame(dataset.data, columns=dataset.feature_names)
    y = dataset.target
    target_names = dataset.target_names
    
    # Sidebar controls
    st.sidebar.header("Analysis Options")
    
    # Dimensionality Reduction
    st.header("🔍 Dimensionality Reduction")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("PCA Analysis")
        n_components = st.slider("Number of Components", 2, min(10, X.shape[1]), 2)
        
        if st.button("Run PCA"):
            pca = PCA(n_components=n_components)
            X_pca = pca.fit_transform(X)
            
            # Explained variance
            explained_variance = pca.explained_variance_ratio_
            
            fig_var = px.bar(
                x=[f"PC{i+1}" for i in range(n_components)],
                y=explained_variance,
                title="Explained Variance by Component"
            )
            st.plotly_chart(fig_var, use_container_width=True)
            
            # PCA scatter plot
            if n_components >= 2:
                pca_df = pd.DataFrame(X_pca[:, :2], columns=['PC1', 'PC2'])
                pca_df['Class'] = [target_names[i] for i in y]
                
                fig_pca = px.scatter(
                    pca_df, x='PC1', y='PC2', color='Class',
                    title="PCA Visualization (First 2 Components)"
                )
                st.plotly_chart(fig_pca, use_container_width=True)
    
    with col2:
        st.subheader("t-SNE Analysis")
        perplexity = st.slider("Perplexity", 5, 50, 30)
        learning_rate = st.slider("Learning Rate", 10, 1000, 200)
        
        if st.button("Run t-SNE"):
            with st.spinner("Running t-SNE (this may take a moment)..."):
                tsne = TSNE(n_components=2, perplexity=perplexity, learning_rate=learning_rate, random_state=42)
                X_tsne = tsne.fit_transform(X)
                
                tsne_df = pd.DataFrame(X_tsne, columns=['t-SNE 1', 't-SNE 2'])
                tsne_df['Class'] = [target_names[i] for i in y]
                
                fig_tsne = px.scatter(
                    tsne_df, x='t-SNE 1', y='t-SNE 2', color='Class',
                    title="t-SNE Visualization"
                )
                st.plotly_chart(fig_tsne, use_container_width=True)
    
    # Clustering Analysis
    st.header("🎯 Clustering Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("K-Means Clustering")
        n_clusters = st.slider("Number of Clusters", 2, 10, 3)
        
        if st.button("Run K-Means"):
            kmeans = KMeans(n_clusters=n_clusters, random_state=42)
            cluster_labels = kmeans.fit_predict(X)
            
            # Create 2D visualization using PCA
            pca_2d = PCA(n_components=2)
            X_pca_2d = pca_2d.fit_transform(X)
            
            cluster_df = pd.DataFrame(X_pca_2d, columns=['PC1', 'PC2'])
            cluster_df['Cluster'] = cluster_labels
            cluster_df['True_Class'] = [target_names[i] for i in y]
            
            fig_cluster = px.scatter(
                cluster_df, x='PC1', y='PC2', color='Cluster',
                title=f"K-Means Clustering (k={n_clusters})",
                hover_data=['True_Class']
            )
            st.plotly_chart(fig_cluster, use_container_width=True)
    
    with col2:
        st.subheader("Cluster Analysis")
        if 'cluster_labels' in locals():
            # Cluster statistics
            cluster_stats = []
            for i in range(n_clusters):
                cluster_mask = cluster_labels == i
                cluster_data = X[cluster_mask]
                
                stats = {
                    'Cluster': i,
                    'Size': cluster_mask.sum(),
                    'Avg_Feature_1': cluster_data.iloc[:, 0].mean(),
                    'Avg_Feature_2': cluster_data.iloc[:, 1].mean()
                }
                cluster_stats.append(stats)
            
            st.dataframe(pd.DataFrame(cluster_stats))
    
    # Feature Importance Analysis
    st.header("📈 Feature Importance")
    
    if st.button("Analyze Feature Importance"):
        from sklearn.ensemble import RandomForestClassifier
        
        rf = RandomForestClassifier(n_estimators=100, random_state=42)
        rf.fit(X, y)
        
        feature_importance = pd.DataFrame({
            'Feature': X.columns,
            'Importance': rf.feature_importances_
        }).sort_values('Importance', ascending=True)
        
        fig_importance = px.bar(
            feature_importance, x='Importance', y='Feature',
            title="Feature Importance (Random Forest)",
            orientation='h'
        )
        st.plotly_chart(fig_importance, use_container_width=True)
    
    # Correlation Analysis
    st.header("🔗 Correlation Analysis")
    
    # Correlation heatmap
    corr_matrix = X.corr()
    
    fig_corr = px.imshow(
        corr_matrix,
        text_auto=True,
        aspect="auto",
        title="Feature Correlation Matrix",
        color_continuous_scale='RdBu_r'
    )
    st.plotly_chart(fig_corr, use_container_width=True)
    
    # Distribution Analysis
    st.header("📊 Distribution Analysis")
    
    selected_feature = st.selectbox("Select Feature for Distribution Analysis", X.columns)
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Histogram
        fig_hist = px.histogram(
            X, x=selected_feature,
            title=f"Distribution of {selected_feature}",
            nbins=30
        )
        st.plotly_chart(fig_hist, use_container_width=True)
    
    with col2:
        # Box plot by class
        box_df = X.copy()
        box_df['Class'] = [target_names[i] for i in y]
        
        fig_box = px.box(
            box_df, x='Class', y=selected_feature,
            title=f"{selected_feature} by Class"
        )
        st.plotly_chart(fig_box, use_container_width=True)

if __name__ == "__main__":
    main()
