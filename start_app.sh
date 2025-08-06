#!/bin/bash
# Breast Cancer Diagnosis App Startup Script

echo "🩺 Starting Breast Cancer Diagnosis App..."
echo "=================================================="

# Navigate to the app directory
cd "$(dirname "$0")"

# Check if virtual environment exists
if [ ! -d ".venv" ]; then
    echo "❌ Virtual environment not found. Please run setup first."
    exit 1
fi

# Activate virtual environment and run tests
echo "🔍 Running pre-flight checks..."
.venv/bin/python test_app.py

if [ $? -eq 0 ]; then
    echo ""
    echo "🚀 Starting Streamlit app..."
    echo "📱 The app will open in your browser at: http://localhost:8501"
    echo "⏹️  Press Ctrl+C to stop the app"
    echo ""
    
    # Start the Streamlit app
    .venv/bin/streamlit run app.py
else
    echo "❌ Pre-flight checks failed. Please check the errors above."
    exit 1
fi
