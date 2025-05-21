from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import numpy as np
import joblib
import os
import json
from models.model_factory import create_model, train_model, evaluate_model
from data.data_processor import load_data, preprocess_data, split_data

app = Flask(__name__)
CORS(app)

# Available diseases and their datasets
DISEASES = {
    'diabetes': {
        'filename': 'data/diabetes.csv',
        'description': 'Diabetes dataset from the Pima Indians Diabetes Database'
    },
    'brain_stroke': {
        'filename': 'data/brain_stroke.csv',
        'description': 'Brain stroke prediction dataset'
    },
    'heart_disease': {
        'filename': 'data/heart_disease.csv',
        'description': 'Heart disease dataset'
    }
}

# Available ML algorithms
ALGORITHMS = [
    'logistic_regression',
    'random_forest',
    'svm',
    'neural_network'
]

@app.route('/api/diseases', methods=['GET'])
def get_diseases():
    """Return the list of available diseases for analysis"""
    return jsonify({
        'diseases': [
            {
                'id': key,
                'name': key.replace('_', ' ').title(),
                'description': value['description']
            } for key, value in DISEASES.items()
        ]
    })

@app.route('/api/algorithms', methods=['GET'])
def get_algorithms():
    """Return the list of available ML algorithms"""
    return jsonify({
        'algorithms': [
            {
                'id': algo,
                'name': algo.replace('_', ' ').title()
            } for algo in ALGORITHMS
        ]
    })

@app.route('/api/train', methods=['POST'])
def train():
    """Train and compare multiple models on a disease dataset"""
    data = request.json
    disease = data.get('disease')
    algorithms = data.get('algorithms', ALGORITHMS)
    test_size = float(data.get('test_size', 0.2))
    random_state = int(data.get('random_state', 42))
    
    if disease not in DISEASES:
        return jsonify({'error': f'Disease {disease} not found'}), 404
    
    try:
        # Load and preprocess the data
        df = load_data(DISEASES[disease]['filename'])
        X, y = preprocess_data(df, disease)
        X_train, X_test, y_train, y_test = split_data(X, y, test_size, random_state)
        
        results = []
        
        # Train and evaluate each requested algorithm
        for algo in algorithms:
            if algo not in ALGORITHMS:
                continue
                
            model = create_model(algo, random_state)
            trained_model = train_model(model, X_train, y_train)
            
            # Evaluate the model
            metrics = evaluate_model(trained_model, X_test, y_test)
            
            # Save the model
            model_filename = f"models/{disease}_{algo}.joblib"
            joblib.dump(trained_model, model_filename)
            
            results.append({
                'algorithm': algo,
                'algorithm_name': algo.replace('_', ' ').title(),
                'metrics': metrics
            })
        
        return jsonify({
            'disease': disease,
            'disease_name': disease.replace('_', ' ').title(),
            'results': results
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/compare', methods=['POST'])
def compare():
    """Compare model performance across multiple diseases and algorithms"""
    data = request.json
    diseases = data.get('diseases', list(DISEASES.keys()))
    algorithms = data.get('algorithms', ALGORITHMS)
    test_size = float(data.get('test_size', 0.2))
    random_state = int(data.get('random_state', 42))
    
    try:
        all_results = {}
        
        for disease in diseases:
            if disease not in DISEASES:
                continue
                
            # Load and preprocess the data
            df = load_data(DISEASES[disease]['filename'])
            X, y = preprocess_data(df, disease)
            X_train, X_test, y_train, y_test = split_data(X, y, test_size, random_state)
            
            disease_results = []
            
            # Train and evaluate each requested algorithm
            for algo in algorithms:
                if algo not in ALGORITHMS:
                    continue
                    
                model = create_model(algo, random_state)
                trained_model = train_model(model, X_train, y_train)
                
                # Evaluate the model
                metrics = evaluate_model(trained_model, X_test, y_test)
                
                disease_results.append({
                    'algorithm': algo,
                    'algorithm_name': algo.replace('_', ' ').title(),
                    'metrics': metrics
                })
            
            all_results[disease] = {
                'disease_name': disease.replace('_', ' ').title(),
                'results': disease_results
            }
        
        return jsonify(all_results)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)
