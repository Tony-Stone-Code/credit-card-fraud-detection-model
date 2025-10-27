#!/bin/bash

# Setup script for fraud detection system
# This script sets up the complete development environment

set -e  # Exit on error

echo "=========================================="
echo "Credit Card Fraud Detection - Setup"
echo "=========================================="

# Check Python version
echo "Checking Python version..."
python_version=$(python3 --version 2>&1 | awk '{print $2}')
echo "Found Python $python_version"

# Create virtual environment
echo "Creating virtual environment..."
if [ ! -d "venv" ]; then
    python3 -m venv venv
    echo "✓ Virtual environment created"
else
    echo "✓ Virtual environment already exists"
fi

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip > /dev/null 2>&1
echo "✓ pip upgraded"

# Install dependencies
echo "Installing dependencies..."
pip install -r requirements.txt > /dev/null 2>&1
echo "✓ Core dependencies installed"

# Install production dependencies
echo "Installing production dependencies..."
pip install -r requirements-prod.txt > /dev/null 2>&1
echo "✓ Production dependencies installed"

# Create necessary directories
echo "Creating directories..."
mkdir -p models data logs
echo "✓ Directories created"

# Check for dataset
echo ""
echo "=========================================="
echo "Dataset Setup"
echo "=========================================="
if [ ! -f "data/creditcard.csv" ]; then
    echo "⚠️  Dataset not found!"
    echo ""
    echo "Please download the dataset from:"
    echo "https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud"
    echo ""
    echo "Place the downloaded 'creditcard.csv' file in the 'data/' directory"
    echo ""
else
    echo "✓ Dataset found at data/creditcard.csv"
    
    # Train model if not exists
    if [ ! -f "models/fraud_detector.joblib" ]; then
        echo ""
        read -p "Would you like to train the model now? (y/n) " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            echo "Training model... (this may take a few minutes)"
            python src/train_model.py --data data/creditcard.csv
            echo "✓ Model trained successfully"
        fi
    else
        echo "✓ Trained model found"
    fi
fi

echo ""
echo "=========================================="
echo "Setup Complete!"
echo "=========================================="
echo ""
echo "Next steps:"
echo "1. Activate virtual environment: source venv/bin/activate"
echo "2. Start API server: python src/api_server.py"
echo "3. Run demo tests: python src/demo.py --test all"
echo "4. View API docs: http://localhost:8000/docs"
echo ""
echo "For more information, see:"
echo "- README.md - Full documentation"
echo "- QUICKSTART.md - Quick reference"
echo "- DEPLOYMENT.md - Production deployment guide"
echo ""
