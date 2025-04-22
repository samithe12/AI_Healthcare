from pycaret.classification import *
from process_data import load_and_preprocess_target

# Step 1: Load your CSV
# Assuming load_and_preprocess_target is correctly defined in process_data.py
df = load_and_preprocess_target('cleaned_heart_final_encoded.csv', 'target')

# Step 2: Initialize PyCaret setup (binary target already set)

clf = setup(data=df, target='target', session_id=123, use_gpu=False, verbose=False)

# Step 3: Compare models and get the best one

best_model = compare_models()

if best_model is not None:
    # Step 4: Save the best model
    save_model(best_model, 'best_classification_model')
else:
    print("No best model found by compare_models.")

