from flask import Flask, request, jsonify, render_template
import joblib
import numpy as np
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import os

app = Flask(__name__)

# Global model variable
model = None
target_names = None

def load_or_train_model():
    """Load existing model or train a new one"""
    global model, target_names
    
    model_path = 'models/iris_model.pkl'
    
    if os.path.exists(model_path):
        model = joblib.load(model_path)
        print("Model loaded from file")
    else:
        print("Training new model...")
        # Load iris dataset
        iris = load_iris()
        X = iris.data
        y = iris.target
        target_names = iris.target_names
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Train model
        model = LogisticRegression(max_iter=1000, multi_class='auto')
        model.fit(X_train, y_train)
        
        # Evaluate
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        print(f"Model accuracy: {accuracy:.4f}")
        
        # Save model
        os.makedirs('models', exist_ok=True)
        joblib.dump(model, model_path)
        print(f"Model saved to {model_path}")
    
    # Get target names if not already loaded
    if target_names is None:
        iris = load_iris()
        target_names = iris.target_names

@app.route('/')
def home():
    """Home page with prediction form"""
    return render_template('index.html')

@app.route('/api/health')
def health():
    """Health check endpoint"""
    return jsonify({"status": "healthy", "model_loaded": model is not None})

@app.route('/api/predict', methods=['POST'])
def predict():
    """Prediction endpoint"""
    try:
        if model is None:
            return jsonify({"error": "Model not loaded"}), 500
        
        data = request.get_json()
        
        # Check if data is None or empty
        if not data:
            return jsonify({"error": "No data provided"}), 400
        
        # Validate input
        required_fields = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']
        for field in required_fields:
            if field not in data:
                return jsonify({"error": f"Missing field: {field}"}), 400
        
        # Prepare features
        features = np.array([[
            data['sepal_length'],
            data['sepal_width'], 
            data['petal_length'],
            data['petal_width']
        ]])
        
        # Make prediction
        prediction = model.predict(features)[0]
        probabilities = model.predict_proba(features)[0]
        
        # Get class name
        class_name = target_names[prediction] if target_names is not None else str(prediction)
        confidence = float(np.max(probabilities))
        
        return jsonify({
            "prediction": int(prediction),
            "class_name": class_name,
            "confidence": confidence,
            "probabilities": {
                target_names[i]: float(prob) for i, prob in enumerate(probabilities)
            } if target_names is not None else {}
        })
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/retrain', methods=['POST'])
def retrain():
    """Retrain model endpoint"""
    try:
        global model, target_names
        
        # Load fresh data
        iris = load_iris()
        X = iris.data
        y = iris.target
        target_names = iris.target_names
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Train new model
        model = LogisticRegression(max_iter=1000, multi_class='auto')
        model.fit(X_train, y_train)
        
        # Evaluate
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        # Save model
        os.makedirs('models', exist_ok=True)
        joblib.dump(model, 'models/iris_model.pkl')
        
        return jsonify({
            "message": "Model retrained successfully",
            "accuracy": accuracy
        })
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    load_or_train_model()
    app.run(host='0.0.0.0', port=8080, debug=True)
