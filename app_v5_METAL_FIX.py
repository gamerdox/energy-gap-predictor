from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import numpy as np
import joblib

app = Flask(__name__)
CORS(app)

# Load model
print("Loading ULTIMATE model...")
artifacts = joblib.load('ultimate_material_model.pkl')
models = artifacts['models']
meta_model = artifacts['meta_model']
scaler = artifacts['scaler']
imputer = artifacts['imputer']
feature_cols = artifacts['feature_cols']

# Load materials database
try:
    materials_db = pd.read_csv('materials_85k_filtered.csv')
    print(f"✅ Loaded {len(materials_db)} materials from filtered database")
except:
    try:
        materials_db = pd.read_csv('materials_100k.csv')
        print(f"✅ Loaded {len(materials_db)} materials from 100k database")
    except:
        materials_db = pd.read_csv('materials_30k.csv')
        print(f"✅ Loaded {len(materials_db)} materials from 30k database")

print(f"📊 Test MAE: {artifacts.get('test_mae', 'N/A')} eV")
print(f"📊 Test R²: {artifacts.get('test_r2', 'N/A')}")

def safe_float(value, default=0.0):
    """Safely convert to float, handling NaN"""
    try:
        val = float(value)
        return default if pd.isna(val) or np.isnan(val) or np.isinf(val) else val
    except:
        return default

def engineer_features(df):
    """Apply same feature engineering as training"""
    # Ensure numeric columns
    numeric_cols = ['Density (g/cm3)', 'Avg Atomic Rad', 'Avg Electronegativity',
                    'Avg Ionization Energy (eV)', 'Lattice Constant a (Å)',
                    'Lattice Constant b (Å)', 'Lattice Constant c (Å)',
                    'Formation Energy (eV/atom)']

    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

    if 'Density (g/cm3)' in df.columns and 'Avg Atomic Rad' in df.columns:
        df['atomic_packing'] = df['Density (g/cm3)'] * df['Avg Atomic Rad']

    if 'Avg Electronegativity' in df.columns and 'Avg Ionization Energy (eV)' in df.columns:
        df['electroneg_ionization_product'] = df['Avg Electronegativity'] * df['Avg Ionization Energy (eV)']
        df['electroneg_ionization_ratio'] = df['Avg Electronegativity'] / (df['Avg Ionization Energy (eV)'] + 0.1)

    if all(c in df.columns for c in ['Lattice Constant a (Å)', 'Lattice Constant b (Å)', 'Lattice Constant c (Å)']):
        df['lattice_volume'] = df['Lattice Constant a (Å)'] * df['Lattice Constant b (Å)'] * df['Lattice Constant c (Å)']
        df['lattice_anisotropy'] = df[['Lattice Constant a (Å)', 'Lattice Constant b (Å)', 'Lattice Constant c (Å)']].std(axis=1)
        df['density_volume_ratio'] = df['Density (g/cm3)'] / (df['lattice_volume'] + 1e-6)

    if 'Formation Energy (eV/atom)' in df.columns:
        df['formation_density_product'] = df['Formation Energy (eV/atom)'] * df['Density (g/cm3)']
        df['formation_electroneg_ratio'] = df['Formation Energy (eV/atom)'] / (df['Avg Electronegativity'] + 0.1)

    if 'Avg Atomic Rad' in df.columns and 'Density (g/cm3)' in df.columns:
        df['radius_density_ratio'] = df['Avg Atomic Rad'] / (df['Density (g/cm3)'] + 0.1)

    if 'Density (g/cm3)' in df.columns:
        df['log_density'] = np.log1p(df['Density (g/cm3)'].clip(lower=0))

    if 'Avg Ionization Energy (eV)' in df.columns:
        df['log_ionization'] = np.log1p(df['Avg Ionization Energy (eV)'].clip(lower=0))

    if 'Avg Electronegativity' in df.columns:
        df['electroneg_squared'] = df['Avg Electronegativity'] ** 2

    if 'Avg Ionization Energy (eV)' in df.columns:
        df['ionization_squared'] = df['Avg Ionization Energy (eV)'] ** 2

    df = df.fillna(0)
    return df

@app.route('/fetch_material', methods=['POST', 'OPTIONS'])
def fetch_material():
    """Fetch material from database by formula"""
    if request.method == 'OPTIONS':
        return '', 204

    try:
        data = request.json
        formula = data.get('formula', '').strip()

        if not formula:
            return jsonify({'success': False, 'error': 'No formula provided'}), 400

        # Search for material (case-insensitive)
        material = materials_db[materials_db['Formula'].str.lower() == formula.lower()]

        if material.empty:
            return jsonify({
                'success': False,
                'error': f'Material "{formula}" not found. Try: GaN, SiC, Si, GaAs, CdTe, ZnO'
            }), 404

        # Get first match
        material = material.iloc[0]

        # Extract properties with safe conversion
        properties = {
            'Lattice Constant a (Å)': safe_float(material.get('Lattice Constant a (Å)', 0)),
            'Lattice Constant b (Å)': safe_float(material.get('Lattice Constant b (Å)', 0)),
            'Lattice Constant c (Å)': safe_float(material.get('Lattice Constant c (Å)', 0)),
            'Density (g/cm3)': safe_float(material.get('Density (g/cm3)', 0)),
            'Formation Energy (eV/atom)': safe_float(material.get('Formation Energy (eV/atom)', 0), default=0),
            'Avg Electronegativity': safe_float(material.get('Avg Electronegativity', 0)),
            'Avg Atomic Rad': safe_float(material.get('Avg Atomic Rad', 0)),
            'Avg Ionization Energy (eV)': safe_float(material.get('Avg Ionization Energy (eV)', 0))
        }

        actual_gap = safe_float(material.get('Band Gap (eV)', 0))
        print(f"✅ Fetched {formula}: Band Gap = {actual_gap} eV")

        return jsonify({
            'success': True,
            'formula': str(material.get('Formula', formula)),
            'material_id': str(material.get('Material ID', 'N/A')),
            'crystal_system': str(material.get('Crystal Str', 'Unknown')),
            'actual_band_gap': actual_gap,
            'properties': properties
        })

    except Exception as e:
        print(f"❌ Error in fetch_material: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/predict', methods=['POST', 'OPTIONS'])
def predict():
    """Predict band gap from properties"""
    if request.method == 'OPTIONS':
        return '', 204

    try:
        data = request.json
        print(f"📥 Received prediction request: {data}")

        # Create DataFrame
        input_df = pd.DataFrame([data])

        # Apply feature engineering
        input_df = engineer_features(input_df)

        # Ensure all expected features exist
        for col in feature_cols:
            if col not in input_df.columns:
                input_df[col] = 0

        # Select only trained features in correct order
        X = input_df[feature_cols]

        # Check for any remaining NaN
        if X.isnull().any().any():
            print("⚠️ Warning: NaN values detected, filling with 0")
            X = X.fillna(0)

        # Impute and scale
        X_imputed = imputer.transform(X)
        X_scaled = scaler.transform(X_imputed)

        # Get predictions from all base models
        base_predictions = []
        model_results = {}

        for name, model in models.items():
            pred = model.predict(X_scaled)[0]
            base_predictions.append(pred)
            model_results[name] = float(pred)

        # Get ensemble prediction
        X_meta = np.array(base_predictions).reshape(1, -1)
        raw_prediction = meta_model.predict(X_meta)[0]

        # Ensure non-negative
        raw_prediction = max(0, raw_prediction)

        # 🔥 NEW LOGIC: Determine material type FIRST, then adjust band gap
        # Metals MUST have 0 eV band gap
        if raw_prediction < 0.1:
            # It's a metal - force band gap to exactly 0
            final_prediction = 0.0
            mat_type = 'Metal'
            type_color = '#ff5722'
        elif raw_prediction < 3.0:
            # Semiconductor - use predicted value
            final_prediction = raw_prediction
            mat_type = 'Semiconductor'
            type_color = '#2196f3'
        else:
            # Insulator - use predicted value
            final_prediction = raw_prediction
            mat_type = 'Insulator'
            type_color = '#4caf50'

        print(f"✅ Raw Prediction: {raw_prediction:.4f} eV")
        print(f"✅ Final Prediction: {final_prediction:.4f} eV ({mat_type})")

        return jsonify({
            'success': True,
            'prediction': float(final_prediction),
            'material_type': mat_type,
            'type_color': type_color,
            'model_breakdown': model_results
        })

    except Exception as e:
        print(f"❌ Error in predict: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)}), 500

if __name__ == '__main__':
    print("\n" + "="*60)
    print("🚀 Energy Gap Prediction Server v4.0")
    print("📡 Server: http://localhost:5000")
    print("✅ CORS Enabled")
    print(f"📊 Materials Database: {len(materials_db)} entries")
    print("="*60 + "\n")

    app.run(debug=True, port=5000)
