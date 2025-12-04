from flask import Flask, render_template, request, jsonify
import pandas as pd
import json
import math

app = Flask(__name__)

# Load model parameters
params_df = pd.read_csv('parameters.csv')
thetas = params_df['theta'].tolist()
indicators = params_df['indicator'].tolist()

# Load normalization statistics
with open('normalization_stats.json', 'r') as f:
    norm_stats = json.load(f)


def sigmoid(z):
    """Sigmoid function with numerical stability"""
    if z >= 0:
        exp_neg = math.exp(-z)
        return 1 / (1 + exp_neg)
    else:
        exp_pos = math.exp(z)
        return exp_pos / (1 + exp_pos)


def standardize_value(value, feature_name):
    """Standardize a value using training statistics"""
    if feature_name not in norm_stats:
        return value  # Return as-is if no stats (e.g., binary features)

    mean = norm_stats[feature_name]['mean']
    std = norm_stats[feature_name]['std']

    if std == 0:
        return 0  # Avoid division by zero

    return (value - mean) / std


def compute_derived_features(inputs):
    """Compute derived features from raw inputs"""
    derived = {}

    # Ratios (avoid division by zero)
    derived['milex_ratio'] = inputs['milex_a'] / (inputs['milex_b'] + 0.0001)
    derived['tpop_ratio'] = inputs['tpop_a'] / (inputs['tpop_b'] + 0.0001)
    derived['irst_ratio'] = inputs['irst_a'] / (inputs['irst_b'] + 0.0001)

    # Total trade
    derived['smoothtotrade'] = inputs['smoothflow1'] + inputs['smoothflow2']

    return derived


@app.route('/')
def index():
    """Render the main page"""
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    """Handle prediction requests"""
    try:
        # Get input data from request
        data = request.get_json()

        # Basic validation
        required_fields = [
            'irst_a', 'milex_a', 'milper_a', 'tpop_a', 'upop_a',
            'irst_b', 'milex_b', 'milper_b', 'tpop_b', 'upop_b',
            'defense', 'neutrality', 'nonaggression', 'entente',
            'smoothflow1', 'smoothflow2', 'conttype'
        ]

        for field in required_fields:
            if field not in data:
                return jsonify({
                    'error': f'Missing required field: {field}'
                }), 400

        # Convert to float
        inputs = {k: float(v) for k, v in data.items()}

        # BACKEND LOGIC: Derive alliance and contiguity fields

        # any_alliance = 1 if ANY alliance type is checked
        inputs['any_alliance'] = 1 if (inputs['defense'] == 1 or
                                       inputs['neutrality'] == 1 or
                                       inputs['nonaggression'] == 1 or
                                       inputs['entente'] == 1) else 0

        # is_contiguous = 1 if conttype != 0
        inputs['is_contiguous'] = 1 if inputs['conttype'] != 0 else 0

        # land_contiguous = 1 if conttype == 1 (land/river border)
        inputs['land_contiguous'] = 1 if inputs['conttype'] == 1 else 0

        # Compute derived features
        derived = compute_derived_features(inputs)
        inputs.update(derived)

        # Build feature vector (x0=1, then all features in order)
        feature_vector = [1.0]  # x0 (bias)

        for i in range(1, len(indicators)):
            feature_name = indicators[i]

            if feature_name in inputs:
                raw_value = inputs[feature_name]
                # Standardize using training statistics
                standardized_value = standardize_value(raw_value, feature_name)
                feature_vector.append(standardized_value)
            else:
                # Feature not provided, use 0 (mean after standardization)
                feature_vector.append(0.0)

        # Calculate activation (dot product)
        activation = sum(theta * x for theta, x in zip(thetas, feature_vector))

        # Calculate probability
        probability = sigmoid(activation)

        # Make prediction
        prediction = 1 if probability > 0.5 else 0

        # Return results
        return jsonify({
            'prediction': int(prediction),
            'probability': float(probability),
            'escalation_risk': 'HIGH' if prediction == 1 else 'LOW',
            'confidence': f"{probability * 100:.2f}%" if prediction == 1 else f"{(1 - probability) * 100:.2f}%"
        })

    except Exception as e:
        return jsonify({
            'error': str(e)
        }), 500


@app.route('/health')
def health():
    """Health check endpoint"""
    return jsonify({'status': 'healthy'})


if __name__ == '__main__':
    # For local development
    app.run(debug=True, host='0.0.0.0', port=5000)