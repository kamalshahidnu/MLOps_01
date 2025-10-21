#!/usr/bin/env python3
"""
Test script to verify the Boston Housing prediction setup
"""
import sys
import os

# Add src to path
sys.path.append('src')

def test_imports():
    """Test if all required modules can be imported"""
    try:
        from data_loader import BostonHousingDataLoader
        from model import BostonHousingModel
        from predict import BostonHousingPredictor
        print("✅ All modules imported successfully")
        return True
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False

def test_data_loading():
    """Test data loading functionality"""
    try:
        from data_loader import BostonHousingDataLoader
        data_loader = BostonHousingDataLoader()
        X_train, X_test, y_train, y_test = data_loader.load_data()
        
        print(f"✅ Data loaded successfully")
        print(f"   Training set: {X_train.shape}")
        print(f"   Test set: {X_test.shape}")
        return True
    except Exception as e:
        print(f"❌ Data loading error: {e}")
        return False

def test_model_training():
    """Test model training"""
    try:
        from data_loader import BostonHousingDataLoader
        from model import BostonHousingModel
        
        # Load data
        data_loader = BostonHousingDataLoader()
        X_train, X_test, y_train, y_test = data_loader.load_data()
        
        # Train model
        model = BostonHousingModel(alpha=1.0)
        model.train(X_train, y_train)
        
        # Test prediction
        prediction = model.predict(X_test[:1])
        print(f"✅ Model training successful")
        print(f"   Sample prediction: {prediction[0]:.2f}")
        return True
    except Exception as e:
        print(f"❌ Model training error: {e}")
        return False

def main():
    """Run all tests"""
    print("🧪 Testing Boston Housing Prediction Setup")
    print("=" * 50)
    
    tests = [
        ("Import Test", test_imports),
        ("Data Loading Test", test_data_loading),
        ("Model Training Test", test_model_training)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n🔍 Running {test_name}...")
        if test_func():
            passed += 1
        else:
            print(f"❌ {test_name} failed")
    
    print(f"\n📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Setup is ready.")
        return True
    else:
        print("⚠️  Some tests failed. Please check the errors above.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
