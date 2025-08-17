"""
This program creates a custom framework called CognitionForge, designed to predict phone addiction levels in teenagers based on various features.
It includes a neural network model, data preprocessing, and training routines.
The framework can be used for training, evaluating, or making predictions on new data points.
"""
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import torch.optim as optim
import torch.nn as nn
import pandas as pd
import numpy as np
import argparse
import joblib
import torch
import utils  # Utility functions for loading models, setting random seeds, and multi-GPU support
import os

CONFIG = {
    "DATA_PATH": "Datasets/Teen Smartphone Usage and Addiction Impact Dataset/teen_phone_addiction_dataset.csv",  # Do not change the path...
    "MODEL_NAME": "CognitionForge_Model.pth",
    "PREPROCESSOR_NAME": "preprocessor.joblib",
    "LEARNING_RATE": 0.00001,
    "BATCH_SIZE": 64,
    "EPOCHS": 200,
    "SEED": 42,
    "TEST_SPLIT_SIZE": 0.2,
    "VALIDATION_SPLIT_SIZE": 0.1,
    "INPUT_FEATURES": 32,  # This must match the features after preprocessing - update if data changes.
}

class CognitionForge_Model(nn.Module):
    """
    A feedforward neural network regression model for predicting phone addiction level.
    """
    def __init__(self, input_size):
        super(CognitionForge_Model, self).__init__()
        self.layer_stack = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)  # Output is a single continuous value
        )

    def forward(self, x):
        return self.layer_stack(x)

def start_training():
    """
    Handles the entire data preparation and model training process.
    """
    # Start preparing the data
    print("[COGNITIONFORGE] Loading and preprocessing data for training...")
    df = pd.read_csv(CONFIG["DATA_PATH"]).drop(columns=['ID', 'Name', 'Location'])
    
    # Keep X as a DataFrame for the preprocessor
    X = df.drop(columns=['Addiction_Level'])
    y = df['Addiction_Level'].values  # Target can be a numpy array

    # Identify feature types from the DataFrame
    categorical_features = X.select_dtypes(include=['object']).columns
    numerical_features = X.select_dtypes(include=np.number).columns
    
    numerical_transformer = StandardScaler()
    categorical_transformer = OneHotEncoder(handle_unknown='ignore')
    preprocessor = ColumnTransformer(transformers=[('num', numerical_transformer, numerical_features), ('cat', categorical_transformer, categorical_features)])
    
    pipeline = Pipeline(steps=[('preprocessor', preprocessor)])
    
    # Fit and transform the DataFrame X
    X_processed = pipeline.fit_transform(X)
    
    print(f"[COGNITIONFORGE] Saving data preprocessor to {CONFIG['PREPROCESSOR_NAME']}")
    joblib.dump(pipeline, CONFIG['PREPROCESSOR_NAME'])

    if hasattr(X_processed, "toarray"): X_processed = X_processed.toarray()

    # Split data into training, validation, and test sets
    X_train_val, _, y_train_val, _ = train_test_split(X_processed, y, test_size=CONFIG["TEST_SPLIT_SIZE"], random_state=CONFIG["SEED"])
    X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=CONFIG["VALIDATION_SPLIT_SIZE"] / (1 - CONFIG["TEST_SPLIT_SIZE"]), random_state=CONFIG["SEED"])
    print(f"[COGNITIONFORGE] Data shape after processing: X_train: {X_train.shape} - Train samples: {len(X_train)}, Validation samples: {len(X_val)}")

    # Datasets and dataloaders
    train_loader = DataLoader(
        dataset=TensorDataset(torch.tensor(X_train.astype(np.float32)), torch.tensor(y_train.astype(np.float32)).view(-1, 1)),
        batch_size=CONFIG["BATCH_SIZE"],
        shuffle=True
    )
    val_loader = DataLoader(
        dataset=TensorDataset(torch.tensor(X_val.astype(np.float32)), torch.tensor(y_val.astype(np.float32)).view(-1, 1)),
        batch_size=CONFIG["BATCH_SIZE"],
        shuffle=False
    )

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[COGNITIONFORGE] Using device: {device}")

    model = CognitionForge_Model(input_size=X_processed.shape[1]).to(device)
    model, _ = utils.load_model(model, CONFIG["MODEL_NAME"])
    model = utils.model_to_multi_gpu(model)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=CONFIG["LEARNING_RATE"])
    best_validation_loss = float('inf')

    print("\n[COGNITIONFORGE] Starting model training...")
    for epoch in range(CONFIG["EPOCHS"]):

        # Training loop
        model.train()
        total_train_loss = 0
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()

        # Validation loop
        model.eval()
        total_validation_loss = 0
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                mae = torch.abs(outputs - targets).mean()
                total_validation_loss += mae.item()

        # Post-epoch processing
        average_train_loss = total_train_loss / len(train_loader)
        average_validation_loss = total_validation_loss / len(val_loader)
        print(f"Epoch [{epoch+1}/{CONFIG['EPOCHS']}], Train Loss (MSE): {average_train_loss:.4f}, Validation MAE: {average_validation_loss:.4f}")

        # Save the model if validation loss improves
        if average_validation_loss < best_validation_loss:
            best_validation_loss = average_validation_loss
            print("\033[92m" + f"[COGNITIONFORGE] New best model found! Saving to {CONFIG['MODEL_NAME']}" + "\033[0m")
            model_state = model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict()
            torch.save({'model_state_dict': model_state, 'optimizer_state_dict': optimizer.state_dict(), 'epoch': epoch, 'best_validation_loss': best_validation_loss}, CONFIG["MODEL_NAME"])

    print(f"\n[COGNITIONFORGE] Training finished.\nBest Validation MAE: {best_validation_loss:.4f}")

def evaluate_on_test_set():
    """
    Loads the trained model and preprocessor, then evaluates the model on the test set.
    """
    print("[EVAL] Starting final model evaluation...")
    
    try:
        pipeline, checkpoint = joblib.load(CONFIG['PREPROCESSOR_NAME']), torch.load(CONFIG['MODEL_NAME'])
    except FileNotFoundError as e:
        print("\033[91m" + f"Error: Missing required file: {e}. Please run the script in 'train' mode first." + "\033[0m") ; return

    df = pd.read_csv(CONFIG["DATA_PATH"]).drop(columns=['ID', 'Name', 'Location'])
    
    # Keep X as a DataFrame for the preprocessor
    X = df.drop(columns=['Addiction_Level'])
    y = df['Addiction_Level'].values
    
    # Transform the DataFrame X
    X_processed = pipeline.transform(X)
    if hasattr(X_processed, "toarray"): X_processed = X_processed.toarray()

    _, X_test, _, y_test = train_test_split(X_processed, y, test_size=CONFIG["TEST_SPLIT_SIZE"], random_state=CONFIG["SEED"])
    
    model = CognitionForge_Model(input_size=X_processed.shape[1])
    model.load_state_dict(checkpoint['model_state_dict'])
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()

    X_test_tensor = torch.tensor(X_test.astype(np.float32)).to(device)
    y_test_tensor = torch.tensor(y_test.astype(np.float32)).view(-1, 1).to(device)

    with torch.no_grad():
        predictions = model(X_test_tensor)
        test_mse_loss = nn.MSELoss()(predictions, y_test_tensor)
        test_mae_loss = torch.abs(predictions - y_test_tensor).mean()

    print("="*30 + "\nFinal Performance on Test Set:\n" + "="*30)
    print("\033[94m" + f"Mean Squared Error (MSE): {test_mse_loss.item():.4f}" + "\033[0m")
    print("\033[92m" + f"Mean Absolute Error (MAE):  {test_mae_loss.item():.4f}" + "\033[0m")
    print(f"This means the model's predictions are, on average, off by {test_mae_loss.item():.2f} points.")

def predict_single():
    """
    Loads the trained model and preprocessor, then makes a prediction for a new data point.
    """
    print("[PREDICT] Making a prediction for a new data point...")
    try:
        pipeline = joblib.load(CONFIG['PREPROCESSOR_NAME'])
        checkpoint = torch.load(CONFIG['MODEL_NAME'])
    except FileNotFoundError as e:
        print("\033[91m" + f"Error: Missing required file: {e}. Please run the script in 'train' mode first." + "\033[0m")
        return

    model = CognitionForge_Model(input_size=CONFIG['INPUT_FEATURES'])
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    # Create a random new person
    new_teen_data = {
        'Age': [16], 'Gender': ['Male'], 'School_Grade': ['10th'], 'Daily_Usage_Hours': [7.5],
        'Sleep_Hours': [5.0], 'Academic_Performance': [65], 'Social_Interactions': [2],
        'Exercise_Hours': [0.5], 'Anxiety_Level': [8], 'Depression_Level': [7], 'Self_Esteem': [3],
        'Parental_Control': [1], 'Screen_Time_Before_Bed': [2.0], 'Phone_Checks_Per_Day': [150],
        'Apps_Used_Daily': [25], 'Time_on_Social_Media': [4.0], 'Time_on_Gaming': [2.5],
        'Time_on_Education': [1.0], 'Phone_Usage_Purpose': ['Social Media'], 'Family_Communication': [3],
        'Weekend_Usage_Hours': [10.0]
    }
    new_teen_df = pd.DataFrame(new_teen_data)

    processed_data = pipeline.transform(new_teen_df)
    if hasattr(processed_data, "toarray"): processed_data = processed_data.toarray()
    
    data_tensor = torch.tensor(processed_data.astype('float32'))
    with torch.no_grad():
        prediction = model(data_tensor)
    
    predicted_level = prediction.item()
    print("\n" + "="*30)
    print("      Prediction Result")
    print("="*30)
    print(f"\033[92mPredicted Addiction Level: {predicted_level:.2f}\033[0m")
    print("="*30)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CognitionForge Pipeline: Train, Evaluate, or Predict.")
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'evaluate', 'predict'], help='Operation mode: train, evaluate, or predict.')
    args = parser.parse_args()

    # Set random seed for reproducibility
    utils.set_random_seed(CONFIG["SEED"])
    
    if args.mode == 'train':
        start_training()
    elif args.mode == 'evaluate':
        evaluate_on_test_set()
    elif args.mode == 'predict':
        predict_single()
