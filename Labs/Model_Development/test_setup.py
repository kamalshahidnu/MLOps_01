"""
Quick setup verification script
"""
import sys
import os

def test_imports():
    """Test if all required packages can be imported"""
    print("Testing imports...")
    try:
        import pandas
        import numpy
        import sklearn
        import xgboost
        import streamlit
        import plotly
        import joblib
        print("✅ All packages imported successfully")
        return True
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False

def test_data_loader():
    """Test data loader"""
    print("\nTesting data loader...")
    try:
        sys.path.append('src')
        from data_loader import HeartDiseaseDataLoader
        
        loader = HeartDiseaseDataLoader()
        X_train, X_test, y_train, y_test = loader.load_data()
        
        print(f"✅ Data loaded successfully")
        print(f"   Training samples: {len(X_train)}")
        print(f"   Test samples: {len(X_test)}")
        print(f"   Features: {len(loader.get_feature_names())}")
        return True
    except Exception as e:
        print(f"❌ Data loader error: {e}")
        return False

def test_model():
    """Test model initialization"""
    print("\nTesting model...")
    try:
        sys.path.append('src')
        from model import HeartDiseaseModel
        
        model = HeartDiseaseModel()
        print("✅ Model initialized successfully")
        return True
    except Exception as e:
        print(f"❌ Model error: {e}")
        return False

def main():
    """Run all tests"""
    print("=" * 60)
    print("Model Development Lab - Setup Verification")
    print("=" * 60)
    
    results = []
    results.append(("Imports", test_imports()))
    results.append(("Data Loader", test_data_loader()))
    results.append(("Model", test_model()))
    
    print("\n" + "=" * 60)
    print("Verification Results")
    print("=" * 60)
    
    all_passed = True
    for name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{name}: {status}")
        if not result:
            all_passed = False
    
    print("=" * 60)
    if all_passed:
        print("✅ All tests passed! Setup is complete.")
        print("\nNext steps:")
        print("1. Train the model: python src/train.py")
        print("2. Run the dashboard: streamlit run app.py")
    else:
        print("❌ Some tests failed. Please check the errors above.")
        print("Make sure to install dependencies: pip install -r requirements.txt")
    print("=" * 60)

if __name__ == "__main__":
    main()

