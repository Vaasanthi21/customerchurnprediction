from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import io
import base64
import os

app = Flask(__name__)

# Global variables for model and data
model = None
label_encoders = {}
scaler = None
feature_names = []

def create_sample_data():
    """Create sample customer churn dataset"""
    np.random.seed(42)
    n_samples = 1000
    
    data = {
        'CustomerID': range(1, n_samples + 1),
        'Gender': np.random.choice(['Male', 'Female'], n_samples),
        'SeniorCitizen': np.random.choice([0, 1], n_samples),
        'Partner': np.random.choice(['Yes', 'No'], n_samples),
        'Dependents': np.random.choice(['Yes', 'No'], n_samples),
        'Tenure': np.random.randint(1, 72, n_samples),
        'PhoneService': np.random.choice(['Yes', 'No'], n_samples),
        'MultipleLines': np.random.choice(['Yes', 'No', 'No phone service'], n_samples),
        'InternetService': np.random.choice(['DSL', 'Fiber optic', 'No'], n_samples),
        'OnlineSecurity': np.random.choice(['Yes', 'No', 'No internet service'], n_samples),
        'OnlineBackup': np.random.choice(['Yes', 'No', 'No internet service'], n_samples),
        'DeviceProtection': np.random.choice(['Yes', 'No', 'No internet service'], n_samples),
        'TechSupport': np.random.choice(['Yes', 'No', 'No internet service'], n_samples),
        'StreamingTV': np.random.choice(['Yes', 'No', 'No internet service'], n_samples),
        'StreamingMovies': np.random.choice(['Yes', 'No', 'No internet service'], n_samples),
        'Contract': np.random.choice(['Month-to-month', 'One year', 'Two year'], n_samples),
        'PaperlessBilling': np.random.choice(['Yes', 'No'], n_samples),
        'PaymentMethod': np.random.choice(['Electronic check', 'Mailed check', 'Bank transfer (automatic)', 'Credit card (automatic)'], n_samples),
        'MonthlyCharges': np.random.uniform(18, 118, n_samples).round(2),
        'TotalCharges': np.random.uniform(18, 8684, n_samples).round(2),
    }
    
    # Create churn based on features (higher tenure and contract length = lower churn)
    df = pd.DataFrame(data)
    churn_prob = 0.3 + 0.4 * (df['Contract'] == 'Month-to-month').astype(float) - 0.3 * (df['Tenure'] > 24).astype(float)
    df['Churn'] = np.random.choice(['Yes', 'No'], n_samples, p=[0.27, 0.73])
    
    return df

def train_model():
    """Train the Random Forest model"""
    global model, label_encoders, scaler, feature_names
    
    # Create or load data
    if os.path.exists('customerchurn.csv'):
        data = pd.read_csv('customerchurn.csv')
    else:
        data = create_sample_data()
        data.to_csv('customerchurn.csv', index=False)
    
    # Preprocess
    data = data.dropna()
    
    # Encode categorical columns
    categorical_columns = data.select_dtypes(include=['object']).columns
    categorical_columns = categorical_columns[categorical_columns != 'Churn']
    
    for column in categorical_columns:
        le = LabelEncoder()
        data[column] = le.fit_transform(data[column])
        label_encoders[column] = le
    
    # Scale numerical columns
    numerical_columns = data.select_dtypes(include=['int64', 'float64']).columns
    numerical_columns = numerical_columns[numerical_columns != 'Churn']
    if 'CustomerID' in numerical_columns:
        numerical_columns = numerical_columns[numerical_columns != 'CustomerID']
    
    scaler = StandardScaler()
    data[numerical_columns] = scaler.fit_transform(data[numerical_columns])
    
    # Prepare features
    X = data.drop(columns=['Churn'])
    if 'CustomerID' in X.columns:
        X = X.drop(columns=['CustomerID'])
    y = data['Churn'].map({'Yes': 1, 'No': 0})
    
    feature_names = X.columns.tolist()
    
    # Train model
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Predictions for metrics
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)
    
    return {
        'accuracy': accuracy,
        'confusion_matrix': cm,
        'classification_report': report,
        'X_test': X_test,
        'y_test': y_test,
        'y_pred': y_pred
    }

def get_feature_importance_plot():
    """Generate feature importance plot"""
    feature_importance = model.feature_importances_
    sorted_indices = np.argsort(feature_importance)[::-1]
    
    plt.figure(figsize=(12, 6))
    plt.bar(range(len(feature_importance)), feature_importance[sorted_indices], align='center', color='skyblue', edgecolor='navy')
    plt.xticks(range(len(feature_importance)), [feature_names[i] for i in sorted_indices], rotation=45, ha='right')
    plt.title('Feature Importance - Random Forest Model', fontsize=14, fontweight='bold')
    plt.xlabel('Features', fontsize=12)
    plt.ylabel('Importance', fontsize=12)
    plt.tight_layout()
    
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
    buf.seek(0)
    plot_url = base64.b64encode(buf.getvalue()).decode()
    plt.close()
    
    return plot_url

def get_confusion_matrix_plot(cm):
    """Generate confusion matrix heatmap"""
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['No Churn', 'Churn'],
                yticklabels=['No Churn', 'Churn'],
                cbar_kws={'label': 'Count'})
    plt.xlabel('Predicted', fontsize=12)
    plt.ylabel('True', fontsize=12)
    plt.title('Confusion Matrix - Random Forest', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
    buf.seek(0)
    plot_url = base64.b64encode(buf.getvalue()).decode()
    plt.close()
    
    return plot_url

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/train', methods=['POST'])
def train():
    results = train_model()
    
    return jsonify({
        'accuracy': round(results['accuracy'] * 100, 2),
        'confusion_matrix': results['confusion_matrix'].tolist(),
        'classification_report': results['classification_report'],
        'feature_importance_plot': get_feature_importance_plot(),
        'confusion_matrix_plot': get_confusion_matrix_plot(results['confusion_matrix'])
    })

@app.route('/api/predict', methods=['POST'])
def predict():
    data = request.json
    
    # Create input dataframe
    input_data = pd.DataFrame([data])
    
    # Encode categorical features
    for column in label_encoders:
        if column in input_data.columns:
            try:
                input_data[column] = label_encoders[column].transform(input_data[column])
            except ValueError:
                # Handle unseen categories
                input_data[column] = 0
    
    # Scale numerical features
    numerical_cols = input_data.select_dtypes(include=['int64', 'float64']).columns
    input_data[numerical_cols] = scaler.transform(input_data[numerical_cols])
    
    # Ensure column order matches training
    input_data = input_data[feature_names]
    
    # Predict
    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0]
    
    return jsonify({
        'prediction': 'Churn' if prediction == 1 else 'No Churn',
        'churn_probability': round(probability[1] * 100, 2),
        'retain_probability': round(probability[0] * 100, 2)
    })

if __name__ == '__main__':
    # Train model on startup
    print("Training model...")
    train_model()
    print("Model trained successfully!")
    app.run(debug=True, port=5000)
