#!/usr/bin/env python3
"""
Test script to verify the breast cancer diagnosis app is working correctly.
Run this before starting the Streamlit app to ensure all components are functional.
"""

import sys
import os

def test_imports():
    """Test all required imports"""
    print("🔍 Testing imports...")
    try:
        import streamlit as st
        import torch
        import torch.nn as nn
        import torchvision.transforms as transforms
        import numpy as np
        import pandas as pd
        import matplotlib.pyplot as plt
        import json
        import hashlib
        from torchvision import models
        from PIL import Image
        from lime import lime_image
        from skimage.segmentation import mark_boundaries
        import plotly.express as px
        import plotly.graph_objects as go
        from datetime import datetime
        from captum.attr import GradientShap, Saliency, LayerGradCam, LayerLRP, IntegratedGradients, NoiseTunnel
        from captum.attr import visualization as viz
        import matplotlib.cm as cm
        import base64
        import io
        from fairlearn.postprocessing import ThresholdOptimizer
        from fairlearn.metrics import MetricFrame, equalized_odds_difference, demographic_parity_difference
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, precision_score, recall_score
        from sklearn.linear_model import LogisticRegression
        import warnings
        print("✅ All imports successful")
        return True
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False

def test_model_loading():
    """Test model loading"""
    print("🔍 Testing model loading...")
    try:
        from torchvision import models
        import torch
        import torch.nn as nn
        
        # Check if model file exists
        if not os.path.exists("Model/best_model.pt"):
            print("❌ Model file 'Model/best_model.pt' not found")
            return False
        
        # Load model
        model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, 2)
        model.load_state_dict(torch.load("Model/best_model.pt", map_location=torch.device('cpu')))
        model.eval()
        
        print("✅ Model loaded successfully")
        print(f"   - Model type: {type(model)}")
        print(f"   - Model in eval mode: {not model.training}")
        print(f"   - Final layer output size: {model.fc.out_features}")
        return True
    except Exception as e:
        print(f"❌ Model loading error: {e}")
        return False

def test_transformations():
    """Test image transformations"""
    print("🔍 Testing transformations...")
    try:
        import torchvision.transforms as transforms
        
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])
        ])
        print("✅ Image transformations defined successfully")
        return True
    except Exception as e:
        print(f"❌ Transformation error: {e}")
        return False

def main():
    """Run all tests"""
    print("🩺 Breast Cancer Diagnosis App - Component Test")
    print("=" * 50)
    
    tests = [
        test_imports,
        test_model_loading,
        test_transformations
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
        print()
    
    print("=" * 50)
    print(f"📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Your app is ready to run.")
        print("\n🚀 To start the app, run:")
        print("   streamlit run app.py")
        return True
    else:
        print("❌ Some tests failed. Please check the errors above.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
