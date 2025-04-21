!pip install kaggle

!mkdir -p /root/.kaggle/

## Copy from your input dataset to the /root/.kaggle/ directory
!cp /kaggle/input/kaggle-api/kaggle.json /root/.kaggle/

# Set the required permissions again for this new location
!chmod 600 /root/.kaggle/kaggle.json

!ls -l /root/.kaggle/

!kaggle competitions download -c shl-intern-hiring-assessment

!mkdir /kaggle/working/data

!unzip /kaggle/working/shl-intern-hiring-assessment.zip -d /kaggle/working/data

!ls /kaggle/working/data
!ls /kaggle/working/data/Dataset/audios/

import os
import pandas as pd

# ==============================================================================
# Step: Define Paths, Columns, Load Data, Add File Paths, Define Target
# ==============================================================================

# --- Define Paths based on confirmed structure ---
BASE_PATH = '/kaggle/working/data/Dataset/'
AUDIO_DIR_PATH = os.path.join(BASE_PATH, 'audios')

# --- Use the correct folder names identified earlier ---
TRAIN_AUDIO_FOLDER_NAME = 'train'
TEST_AUDIO_FOLDER_NAME = 'test'

# --- Construct full audio paths ---
TRAIN_AUDIO_FULL_PATH = os.path.join(AUDIO_DIR_PATH, TRAIN_AUDIO_FOLDER_NAME)
TEST_AUDIO_FULL_PATH = os.path.join(AUDIO_DIR_PATH, TEST_AUDIO_FOLDER_NAME)

print(f"Full Train Audio Path: {TRAIN_AUDIO_FULL_PATH}")
print(f"Full Test Audio Path: {TEST_AUDIO_FULL_PATH}")

# Check if these full paths exist
if not os.path.exists(TRAIN_AUDIO_FULL_PATH):
    raise FileNotFoundError(f"ERROR: Full Train Audio Path '{TRAIN_AUDIO_FULL_PATH}' does not exist.")
if not os.path.exists(TEST_AUDIO_FULL_PATH):
    raise FileNotFoundError(f"ERROR: Full Test Audio Path '{TEST_AUDIO_FULL_PATH}' does not exist.")

# --- Define correct column names identified earlier ---
actual_train_filename_col = 'filename'
actual_test_filename_col = 'filename'
score_col = 'label' # This is the column containing the grammar scores

# --- Load data ---
print("\nLoading CSV data...")
try:
    train_df_orig = pd.read_csv(os.path.join(BASE_PATH, 'train.csv'))
    test_df_orig = pd.read_csv(os.path.join(BASE_PATH, 'test.csv'))
    print("CSV data loaded.")

    # Create copies
    train_df = train_df_orig.copy()
    test_df = test_df_orig.copy()

    # --- Add the 'file_path' column using the DEFINED full paths ---
    print(f"\nAdding 'file_path' to train_df using column: '{actual_train_filename_col}'")
    # Ensure TRAIN_AUDIO_FULL_PATH is defined before this line executes
    train_df['file_path'] = train_df[actual_train_filename_col].apply(lambda x: os.path.join(TRAIN_AUDIO_FULL_PATH, str(x)))
    print("Added 'file_path' to train_df.")

    print(f"Adding 'file_path' to test_df using column: '{actual_test_filename_col}'")
     # Ensure TEST_AUDIO_FULL_PATH is defined before this line executes
    test_df['file_path'] = test_df[actual_test_filename_col].apply(lambda x: os.path.join(TEST_AUDIO_FULL_PATH, str(x)))
    print("Added 'file_path' to test_df.")


    # --- Define target variable 'y' ---
    print(f"\nDefining target variable 'y' using the '{score_col}' column from train_df...")
    # Ensure train_df and score_col are defined
    y = train_df[score_col].copy()
    print(f"'y' target variable created with shape: {y.shape}")


    # --- Verification ---
    print("\n--- Verification ---")
    print("Head of train_df with 'file_path':")
    print(train_df.head(3))
    print("\nHead of test_df with 'file_path':")
    print(test_df.head(3))
    print("\nHead of target variable 'y':")
    print(y.head(3))


except FileNotFoundError as e:
    print(f"\n--- FILE NOT FOUND ERROR ---")
    print(f"Could not find a required file or folder.")
    print(f"Please double-check BASE_PATH, audio folder names, and CSV filenames.")
    print(f"Error message: {e}")
except KeyError as e:
     print(f"\n--- KEY ERROR ---")
     print(f"Could not find the column named: {e}")
     print(f"Make sure '{actual_train_filename_col}', '{actual_test_filename_col}', and '{score_col}' variables match the actual column names in your CSV files.")
except NameError as e:
     print(f"\n--- NAME ERROR ---")
     print(f"A variable was used before it was defined: {e}")
     print(f"Make sure you run the cell sections in the correct order (define paths before using them).")
except Exception as e:
    print(f"\n--- UNEXPECTED ERROR ---")
    print(e)
    raise # Re-raise unexpected errors

print("\nNext Step: Exploratory Data Analysis (EDA) on the scores.")



# ==============================================================================
# Step: Exploratory Data Analysis (EDA) - Score Distribution
# ==============================================================================
import matplotlib.pyplot as plt
import seaborn as sns

print("Starting EDA on Score Distribution...")

# Set plot style (optional)
sns.set_style("whitegrid")

# --- Plot Score Distribution ---
plt.figure(figsize=(10, 5))
# We use the 'score_col' variable defined in the previous step ('label')
sns.histplot(y, bins=15, kde=True) # Use y directly now
plt.title(f'Distribution of Grammar Scores ({score_col}) in Training Data')
plt.xlabel('Score')
plt.ylabel('Frequency')
plt.grid(True) # Add grid lines
plt.show()

# --- Plot Count Plot for Unique Values (useful if scores are discrete) ---
unique_scores = sorted(y.unique())
if len(unique_scores) < 25: # Only plot if there aren't too many unique values
    plt.figure(figsize=(10, 5))
    sns.countplot(x=y, order=unique_scores, palette='viridis') # Use y directly
    plt.title(f'Count of Each Score ({score_col}) in Training Data')
    plt.xlabel('Score')
    plt.ylabel('Count')
    plt.show()
else:
    print(f"\nSkipping count plot as there are {len(unique_scores)} unique scores.")


# --- Calculate and Print Statistics ---
print(f"\nScore ({score_col}) Statistics:")
print(y.describe()) # Use y directly


# --- Check for Missing Scores ---
missing_scores = y.isnull().sum() # Use y directly
print(f"\nNumber of missing scores in training data: {missing_scores}")
if missing_scores > 0:
    print("WARNING: Missing scores detected. These might need handling later.")

print("\nEDA on score distribution complete.")
print("\nNext Step: Setting up for Feature Extraction (Importing necessary libraries like librosa, whisper, language_tool).")


# ==============================================================================
# Step: Setup for Feature Extraction
# ==============================================================================
import librosa
import librosa.display
from tqdm.notebook import tqdm # For progress bars
import joblib # For potential parallel processing later (optional)

print("Librosa, tqdm, joblib imported.")

# --- Install and Import STT/Grammar Libraries ---
# Using try-except blocks as libraries might need re-installation after restart
try:
    import whisper
    print("Whisper library already available.")
except ImportError:
    print("Installing Whisper...")
    # Use -q for quiet installation
    !pip install -q openai-whisper
    import whisper
    print("Whisper installed and loaded.")

try:
    import language_tool_python
    print("LanguageTool library already available.")
except ImportError:
    print("Installing LanguageTool...")
    !pip install -q language_tool_python
    # LanguageTool might also need Java runtime
    import language_tool_python
    print("LanguageTool installed and loaded.")

# --- Configuration ---
TARGET_SR = 16000 # Standard sample rate for many speech models
print(f"\nTarget Sample Rate (TARGET_SR) set to: {TARGET_SR} Hz")

# --- Initialize Models/Tools ---
# Load Whisper Model (adjust size if needed: tiny, base, small, medium, large)
# Ensure Internet is ON if loading for the first time in the session
print("\nLoading Whisper model (should utilize GPU if available)...")
try:
    # Use 'base.en' for English-specific, slightly smaller than 'base'
    stt_model = whisper.load_model("base.en")
    # Verify if GPU is used (PyTorch check - optional)
    try:
        import torch
        if torch.cuda.is_available():
             print(f"Whisper model device: {stt_model.device}") # Should show 'cuda'
        else:
             print("Whisper model device: CPU (CUDA not available)")
    except ImportError:
        print("PyTorch not found, cannot confirm GPU usage directly.")
    print("Whisper 'base.en' model loaded successfully.")
except Exception as e:
    print(f"ERROR loading Whisper model: {e}")
    print("Proceeding without STT model. Text features will be limited.")
    stt_model = None

# Initialize Grammar Checker
print("\nLoading LanguageTool...")
# We expect this might fail again due to Java version on Kaggle, but we try anyway
try:
    lang_tool = language_tool_python.LanguageTool('en-US')
    print("LanguageTool loaded successfully.")
except Exception as e:
    print(f"ERROR loading LanguageTool: {e}")
    print("As expected, LanguageTool failed (likely Java version). Proceeding without grammar checking tool.")
    lang_tool = None # Explicitly set to None on error


print("\nSetup for Feature Extraction complete.")
print("\nNext Step: Define the feature extraction function.")


# ==============================================================================
# Step: Define Feature Extraction Function (with fp16 for GPU)
# ==============================================================================
import numpy as np
import librosa
import os
# whisper, language_tool_python, stt_model, lang_tool are already handled

# Define the number of MFCCs to compute
N_MFCC = 20

def extract_features(file_path):
    """
    Extracts features from a single audio file.
    Includes acoustic features and text features from STT.
    Optimized for GPU with fp16=True in transcribe.
    Handles potential errors gracefully by returning NaNs.
    """
    # Define feature names beforehand for consistent output
    feature_names = [
        'duration', 'rms_mean', 'rms_std', 'zcr_mean', 'zcr_std',
        'word_count', 'sentence_count', 'avg_sentence_length', 'grammar_errors'
    ]
    for i in range(N_MFCC): feature_names.append(f'mfcc_mean_{i}')
    for i in range(N_MFCC): feature_names.append(f'mfcc_std_{i}')

    # Initialize features dictionary with NaNs
    features = {name: np.nan for name in feature_names}
    features['grammar_errors'] = np.nan # Set default as lang_tool failed

    try:
        # 1. Load and Resample Audio
        y, sr = librosa.load(file_path, sr=TARGET_SR)
        features['duration'] = librosa.get_duration(y=y, sr=sr)

        # 2. Acoustic Features (Check for empty results)
        rms = librosa.feature.rms(y=y)[0]
        features['rms_mean'] = np.mean(rms) if len(rms) > 0 else 0.0
        features['rms_std'] = np.std(rms) if len(rms) > 0 else 0.0

        zcr = librosa.feature.zero_crossing_rate(y=y)[0]
        features['zcr_mean'] = np.mean(zcr) if len(zcr) > 0 else 0.0
        features['zcr_std'] = np.std(zcr) if len(zcr) > 0 else 0.0

        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=N_MFCC)
        if mfccs.shape[1] > 0: # Ensure MFCCs were calculated
            mfcc_mean = np.mean(mfccs, axis=1)
            mfcc_std = np.std(mfccs, axis=1)
            for i in range(N_MFCC):
                features[f'mfcc_mean_{i}'] = mfcc_mean[i]
                features[f'mfcc_std_{i}'] = mfcc_std[i]
        else:
             for i in range(N_MFCC):
                features[f'mfcc_mean_{i}'] = 0.0
                features[f'mfcc_std_{i}'] = 0.0

        # 3. STT + Text Features (Using GPU with fp16)
        transcript = ""
        if stt_model:
            try:
                # *** Use fp16=True for GPU optimization ***
                result = stt_model.transcribe(y, fp16=True)
                transcript = result['text'].strip() if result and 'text' in result else ""
                features['word_count'] = len(transcript.split()) if transcript else 0
            except Exception as stt_e:
                # Fallback to fp16=False if True causes issues (rare)
                try:
                    print(f"STT Warning (fp16=True failed for {os.path.basename(file_path)}: {stt_e}). Retrying with fp16=False.")
                    result = stt_model.transcribe(y, fp16=False)
                    transcript = result['text'].strip() if result and 'text' in result else ""
                    features['word_count'] = len(transcript.split()) if transcript else 0
                except Exception as stt_e2:
                    print(f"STT Error on {os.path.basename(file_path)} (even with fp16=False): {stt_e2}")
                    features['word_count'] = 0 # Default on error
        else:
            features['word_count'] = 0 # Default if no model

        # 4. Grammar Check (Skipped)
        # 'grammar_errors' remains NaN

        # 5. Basic Text Stats
        if transcript:
             sentences = [s.strip() for s in transcript.split('. ') if s.strip()]
             features['sentence_count'] = len(sentences)
             if features['word_count'] > 0 and features['sentence_count'] > 0:
                  features['avg_sentence_length'] = features['word_count'] / features['sentence_count']
             elif features['word_count'] > 0:
                  features['avg_sentence_length'] = features['word_count']
                  features['sentence_count'] = 1
             else:
                  features['avg_sentence_length'] = 0
        else:
             features['sentence_count'] = 0
             features['avg_sentence_length'] = 0

    except Exception as e:
        print(f"ERROR during feature extraction for file {os.path.basename(file_path)}: {e}")
        # Return the dictionary with NaNs

    return features

# --- Test the function on one file ---
print("\nTesting feature extraction function on the first training file...")
try:
    # Ensure train_df is defined from previous steps
    if 'train_df' not in locals() or train_df is None or 'file_path' not in train_df.columns:
         print("ERROR: train_df or file_path column not defined. Please re-run the data loading cell.")
    else:
        sample_file_path = train_df['file_path'].iloc[0]
        print(f"Test file: {sample_file_path}")
        sample_features = extract_features(sample_file_path)
        print("\nSample extracted features:")
        for key, value in sample_features.items():
             if isinstance(value, float): print(f"  {key}: {value:.4f}")
             else: print(f"  {key}: {value}")
except Exception as e:
    print(f"\nError testing feature extraction function: {e}")

print("\nFeature extraction function defined.")
print("\nNext Step: Apply the feature extraction function to all train and test audio files.")


# ==============================================================================
# Step: Apply Feature Extraction to All Files (Using GPU)
# ==============================================================================
from tqdm.notebook import tqdm # Ensure tqdm is imported
import pandas as pd # Ensure pandas is imported

print("Applying feature extraction to all Training data files (using GPU)...")
# Use a list comprehension with tqdm for progress bar
feature_list_train = [extract_features(fp) for fp in tqdm(train_df['file_path'], desc="Extracting Train Features")]

print("\nApplying feature extraction to all Test data files (using GPU)...")
feature_list_test = [extract_features(fp) for fp in tqdm(test_df['file_path'], desc="Extracting Test Features")]

# --- Convert lists of dictionaries to DataFrames ---
print("\nConverting results to DataFrames...")
features_train_df = pd.DataFrame(feature_list_train)
features_test_df = pd.DataFrame(feature_list_test)

# --- Define Feature Matrix X and Target y ---
# X contains the extracted features for the training set
X = features_train_df.copy()
# y (target variable) should already be defined from previous steps

# X_test contains the extracted features for the test set
X_test = features_test_df.copy()

# --- Basic Check ---
# Ensure y is defined and has the correct length
if 'y' not in locals() or len(y) != len(X):
     print("WARNING: Target variable 'y' is not defined or has incorrect length. Please re-run the cell where 'y' is defined.")
     # Attempt to redefine y just in case
     try:
         score_col = 'label' # Make sure this is the correct score column name
         y = train_df[score_col].copy()
         print(f"Re-defined 'y' using column '{score_col}'. Shape: {y.shape}")
     except Exception as e:
         print(f"Could not redefine 'y': {e}")


print("\nFeature extraction and DataFrame conversion complete.")
print(f"Shape of Training Features (X): {X.shape}")
print(f"Shape of Test Features (X_test): {X_test.shape}")
if 'y' in locals(): print(f"Shape of Target Variable (y): {y.shape}")

# --- Display Head of Feature DataFrames ---
print("\nHead of Training Features DataFrame (X):")
print(X.head())
print("\nHead of Test Features DataFrame (X_test):")
print(X_test.head())

print("\nNext Step: Preprocessing (Handling Missing Values, Aligning Columns, Scaling Features).")


# ==============================================================================
# Step: Preprocessing (Handle NaNs, Drop Zero Var, Align Cols, Scale Features) - REVISED
# ==============================================================================
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
# from sklearn.feature_selection import VarianceThreshold # Alternative way to drop

print("Starting REVISED Preprocessing...")

# --- Ensure X and X_test are DataFrames ---
if not isinstance(X, pd.DataFrame): X = pd.DataFrame(X)
if not isinstance(X_test, pd.DataFrame): X_test = pd.DataFrame(X_test)

# --- Handle Missing Values (NaNs) ---
print("\nHandling missing values (NaNs)...")
cols_with_nan_train = X.columns[X.isna().any()].tolist()

if cols_with_nan_train:
    print(f"Columns with NaNs found in Train: {cols_with_nan_train}")
    for col in cols_with_nan_train:
        # Special handling for 'grammar_errors' since the tool failed
        if col == 'grammar_errors':
            fill_value = 0 # Assume 0 errors if tool failed
            print(f"  Filling NaNs in '{col}' with specific value: {fill_value}")
        else:
            # For other columns (if any), use median
            median_val = X[col].median()
            fill_value = median_val
            print(f"  Filling NaNs in '{col}' with training median: {median_val:.4f}")

        X[col] = X[col].fillna(fill_value)
        # Fill test set NaNs using the same value/strategy
        if col in X_test.columns:
            X_test[col] = X_test[col].fillna(fill_value)

    # Final check for NaNs
    if X.isna().any().any() or X_test.isna().any().any():
         print("WARNING: NaNs still present after handling. Check logic.")
    else:
         print("NaN values handled successfully.")
else:
    print("No NaNs found in training features.")


# --- Remove Zero-Variance Features ---
# Identify columns in the training set with zero variance AFTER handling NaNs
print("\nRemoving features with zero variance...")
variances = X.var()
zero_var_cols = variances[variances == 0].index.tolist()

if zero_var_cols:
    print(f"Found columns with zero variance: {zero_var_cols}")
    # Drop these columns from both X and X_test
    original_cols = X.shape[1]
    X = X.drop(columns=zero_var_cols)
    X_test = X_test.drop(columns=zero_var_cols, errors='ignore') # errors='ignore' handles if test didn't have the col
    print(f"Dropped {len(zero_var_cols)} zero-variance columns.")
    print(f"New shape X: {X.shape}, X_test: {X_test.shape}")
else:
    print("No zero-variance columns found.")


# --- Align Columns (After potential drops) ---
print("\nAligning columns...")
train_cols = list(X.columns)
# Reindex X_test based on final training columns
X_test = X_test.reindex(columns=train_cols, fill_value=0) # Fill any newly missing cols with 0
# Verify alignment
if list(X.columns) == list(X_test.columns):
    print("Columns aligned successfully.")
    print(f"Final feature shape - X: {X.shape}, X_test: {X_test.shape}")
else:
     print("WARNING: Column alignment failed. Check feature extraction/dropping.")


# --- Feature Scaling ---
# Use StandardScaler on the cleaned and variance-filtered data
print("\nScaling features using StandardScaler...")
scaler = StandardScaler()

# Fit the scaler ONLY on the training data (X)
scaler.fit(X)

# Transform both training and test data
X_scaled = scaler.transform(X)
X_test_scaled = scaler.transform(X_test)

# Convert scaled arrays back to DataFrames
X_scaled = pd.DataFrame(X_scaled, columns=X.columns, index=X.index)
X_test_scaled = pd.DataFrame(X_test_scaled, columns=X_test.columns, index=X_test.index)

print("Scaling complete.")
print("\nHead of Scaled Training Features (X_scaled):")
print(X_scaled.head())
print("\nHead of Scaled Test Features (X_test_scaled):")
print(X_test_scaled.head())


print("\nPreprocessing complete.")
print("\nNext Step: Model Training and Validation.")



# ==============================================================================
# Step: Model Training and Validation
# ==============================================================================
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
import numpy as np

print("Starting Model Training and Selection...")

# --- Define SEED for reproducibility ---
SEED = 42
np.random.seed(SEED)

# --- Split Data for Validation ---
# Split the scaled training data for internal model evaluation
# Using 80% for training, 20% for validation
X_train_split, X_val_split, y_train_split, y_val_split = train_test_split(
    X_scaled, y, test_size=0.20, random_state=SEED, shuffle=True
)
print(f"\nData split for validation:")
print(f"  Training split shape - X: {X_train_split.shape}, y: {y_train_split.shape}")
print(f"  Validation split shape - X: {X_val_split.shape}, y: {y_val_split.shape}")

# --- Define RMSE Metric ---
def rmse(y_true, y_pred):
    """Calculates Root Mean Squared Error."""
    return np.sqrt(mean_squared_error(y_true, y_pred))

# --- Define Models to Try ---
# Using default parameters initially, or some basic ones for tree models
models = {
    "Linear Regression": LinearRegression(),
    "Ridge Regression": Ridge(random_state=SEED),
    "Support Vector Regressor (SVR)": SVR(), # Might be slow on larger datasets
    "Random Forest Regressor": RandomForestRegressor(n_estimators=100, random_state=SEED, n_jobs=-1, max_depth=10, min_samples_leaf=3),
    "Gradient Boosting Regressor": GradientBoostingRegressor(n_estimators=100, random_state=SEED, max_depth=5, learning_rate=0.1)
    # Add other models like XGBoost or LightGBM if desired (requires installation)
    # "XGBoost Regressor": xgb.XGBRegressor(random_state=SEED, n_jobs=-1),
    # "LightGBM Regressor": lgb.LGBMRegressor(random_state=SEED, n_jobs=-1)
}

results = {} # To store model performance

# --- Train and Evaluate on Validation Set ---
print("\nTraining and evaluating models on validation set...")
for name, model in models.items():
    print(f"  Training {name}...")
    # Train the model on the smaller training split
    model.fit(X_train_split, y_train_split)

    # Predict on the validation split
    y_pred_val = model.predict(X_val_split)

    # Calculate validation RMSE
    val_rmse = rmse(y_val_split, y_pred_val)
    results[name] = {'model': model, 'val_rmse': val_rmse}
    print(f"    {name} - Validation RMSE: {val_rmse:.4f}")

# --- Select Best Model based on Validation RMSE ---
best_model_name = min(results, key=lambda k: results[k]['val_rmse'])
best_val_rmse = results[best_model_name]['val_rmse']
print(f"\nBest model (based on validation RMSE): {best_model_name} with RMSE {best_val_rmse:.4f}")

# --- Final Model Training ---
# Get the best model instance identified from the validation results
final_model = results[best_model_name]['model']

# Re-train this chosen model on the ENTIRE scaled training dataset (X_scaled, y)
print(f"\nRe-training final model ({best_model_name}) on the full training dataset (X_scaled, y)...")
final_model.fit(X_scaled, y) # Fit on all available training data
print("Final model training complete.")

print("\nModel Training and Validation complete.")
print(f"The final model chosen is: {best_model_name}")
print("\nNext Step: Evaluate the final model on the full training data (as required) and visualize results.")


# ==============================================================================
# Step: Final Model Evaluation (on Training Data) & Visualization
# ==============================================================================
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error # Ensure it's imported
import numpy as np # Ensure it's imported

# --- Ensure final_model and X_scaled/y are available ---
if 'final_model' not in locals():
    print("ERROR: 'final_model' is not defined. Please re-run the previous training cell.")
elif 'X_scaled' not in locals() or 'y' not in locals():
     print("ERROR: 'X_scaled' or 'y' not defined. Please re-run relevant previous cells.")
else:
    print("Evaluating final model on the FULL Training Data...")

    # 1. --- MANDATORY: Calculate RMSE on the FULL Training Data ---
    y_pred_train_final = final_model.predict(X_scaled)

    # Define rmse function if not already defined in this session
    def rmse(y_true, y_pred):
        return np.sqrt(mean_squared_error(y_true, y_pred))

    train_rmse_final = rmse(y, y_pred_train_final)

    print("\n" + "="*60)
    print(f" FINAL MODEL EVALUATION (ON FULL TRAINING DATA)")
    print(f" Model Type: {type(final_model).__name__}")
    print(f" >>> Training RMSE: {train_rmse_final:.4f} <<< ")
    print("      (This is the value required for the assignment report)")
    print("="*60 + "\n")


    # 2. --- Evaluation Visualizations ---

    # --- Predicted vs Actual Plot (Training Data) ---
    plt.figure(figsize=(8, 8))
    plt.scatter(y, y_pred_train_final, alpha=0.6, edgecolors='k', s=50, label='Predictions')
    # Ideal line (y=x)
    min_val = min(y.min(), y_pred_train_final.min()) - 0.1
    max_val = max(y.max(), y_pred_train_final.max()) + 0.1
    plt.plot([min_val, max_val], [min_val, max_val], '--', color='red', lw=2, label='Ideal (y=x)')
    plt.xlabel("Actual Scores (Full Training Data)")
    plt.ylabel("Predicted Scores (Full Training Data)")
    plt.title(f"{type(final_model).__name__}: Predicted vs Actual (Training Data)\nTrain RMSE: {train_rmse_final:.4f}")
    plt.grid(True)
    plt.legend()
    plt.axis('equal') # Force equal scaling
    plt.xlim(min_val, max_val) # Adjust limits dynamically
    plt.ylim(min_val, max_val)
    plt.show()

    # --- Residual Plot (Training Data) ---
    # Residuals = Actual - Predicted
    residuals = y - y_pred_train_final
    plt.figure(figsize=(10, 5))
    plt.scatter(y_pred_train_final, residuals, alpha=0.6, edgecolors='k', s=50)
    plt.axhline(0, color='red', linestyle='--', lw=2)
    plt.xlabel("Predicted Scores (Full Training Data)")
    plt.ylabel("Residuals (Actual - Predicted)")
    plt.title(f"{type(final_model).__name__}: Residual Plot (Training Data)")
    plt.grid(True)
    plt.show()
    # Check if residuals have any obvious patterns (e.g., funnel shape indicates heteroscedasticity)

    # --- Feature Importance / Coefficients (Optional for SVR, more complex) ---
    # Standard SVR (especially with non-linear kernels like the default 'rbf') doesn't have direct feature_importances_
    # If the kernel was 'linear', final_model.coef_ would exist.
    if hasattr(final_model, 'coef_'):
        print("\nCalculating Coefficients (Linear SVR)...")
        # Assuming linear kernel, coef_ shape might be (1, n_features)
        coefs = final_model.coef_.flatten()
        if len(coefs) == len(X_scaled.columns):
            feature_importance_df = pd.DataFrame({'Feature': X_scaled.columns, 'Coefficient': coefs})
            feature_importance_df['Abs_Coefficient'] = np.abs(feature_importance_df['Coefficient'])
            feature_importance_df = feature_importance_df.sort_values(by='Abs_Coefficient', ascending=False)

            plt.figure(figsize=(10, max(8, len(X_scaled.columns) // 2)))
            sns.barplot(x='Coefficient', y='Feature', data=feature_importance_df.head(30), palette='coolwarm')
            plt.title(f'{type(final_model).__name__} - Top 30 Coefficients (by absolute value)')
            plt.tight_layout()
            plt.show()
        else:
            print("Could not plot coefficients: shape mismatch or not linear SVR.")
    else:
         print("\nFeature importances are not directly available for the default SVR kernel (RBF).")
         print("Consider using techniques like Permutation Importance or SHAP for SVR interpretability if needed.")


    print("\nFinal Model Evaluation Complete.")
    print("\nNext Step: Generate predictions on the test set and create the submission file.")


# ==============================================================================
# Step: Prediction and Submission File Creation
# ==============================================================================
import pandas as pd
import numpy as np
import os # <--- Added: Need os for os.path.join

print("Generating predictions on the Test Set (X_test_scaled)...")

# --- Check if final_model and X_test_scaled exist ---
if 'final_model' not in locals():
     print("ERROR: 'final_model' is not defined. Please re-run the model training cell (Step 9).")
     # Optionally add 'exit()' here if running as a script and want to stop
elif 'X_test_scaled' not in locals():
     print("ERROR: 'X_test_scaled' is not defined. Please re-run the preprocessing cell (Step 7).")
     # Optionally add 'exit()' here
else:
    # --- Predict on Test Data ---
    test_predictions = final_model.predict(X_test_scaled)
    print(f"Generated {len(test_predictions)} predictions.")

    # --- Clip Predictions (Optional but Recommended) ---
    # Ensure predictions fall within the valid score range (e.g., 1 to 5 based on rubric)
    min_score = 1.0 # Use rubric minimum
    max_score = 5.0 # Use rubric maximum
    original_test_predictions = test_predictions.copy() # Keep original for comparison
    test_predictions_clipped = np.clip(test_predictions, min_score, max_score) # Use a new variable for clipped
    print(f"Predictions clipped to range [{min_score}, {max_score}]")
    # Check if clipping changed many values
    clip_diff = np.sum(original_test_predictions != test_predictions_clipped) # Compare original to clipped
    if clip_diff > 0:
        print(f"  Note: Clipping affected {clip_diff} predictions.")


    # --- Create Submission DataFrame ---
    # We need the original test dataframe ('test_df_orig') to get the correct filenames
    # And the correct column name for the filename ('actual_test_filename_col' = 'filename')
    # And the correct column name for the score ('score_col' = 'label') expected in submission
    print("\nCreating submission DataFrame...")
    try:
        # --- FIX STARTS HERE ---
        # Ensure BASE_PATH is defined (it should be from Step 2)
        if 'BASE_PATH' not in locals():
            print("WARNING: BASE_PATH not found, attempting to redefine. Check Step 2 if this fails.")
            # Set BASE_PATH based on expected Kaggle input location
            BASE_PATH = '/kaggle/input/automatic-grammer-evaluation/'
            if not os.path.exists(BASE_PATH):
                print(f"Warning: Input path '{BASE_PATH}' not found. Trying '/kaggle/working/data/Dataset/'...")
                BASE_PATH = '/kaggle/working/data/Dataset/'
                if not os.path.exists(BASE_PATH):
                     raise FileNotFoundError("ERROR: Cannot determine BASE_PATH.")
            print(f"Redefined BASE_PATH to: {BASE_PATH}")

        # Define SAMPLE_SUB_PATH using BASE_PATH
        SAMPLE_SUB_PATH = os.path.join(BASE_PATH, 'sample_submission.csv')
        print(f"Ensured SAMPLE_SUB_PATH is defined: {SAMPLE_SUB_PATH}")
        # --- FIX ENDS HERE ---

        # Make sure other required variables are defined
        if 'test_df_orig' not in locals(): raise NameError("'test_df_orig' not defined. Check Step 2.")
        if 'actual_test_filename_col' not in locals(): raise NameError("'actual_test_filename_col' not defined. Check Step 2.")
        if 'score_col' not in locals(): raise NameError("'score_col' not defined. Check Step 2.")
        if not os.path.exists(SAMPLE_SUB_PATH): raise FileNotFoundError(f"Sample submission file not found at: {SAMPLE_SUB_PATH}")

        # Load sample submission if it wasn't loaded before or got lost
        if 'sample_submission_df' not in locals():
             print("Reloading sample_submission.csv...")
             sample_submission_df = pd.read_csv(SAMPLE_SUB_PATH)
        # --- End modification for sample_submission_df reload ---

        # Create the submission dataframe using the correct variable names
        submission_df = pd.DataFrame({
            actual_test_filename_col: test_df_orig[actual_test_filename_col], # Get filenames from original test CSV
            score_col: test_predictions_clipped # <<< Use the clipped predictions
        })

        # --- Verify Format against sample_submission.csv ---
        print("\nSample Submission Format:")
        print(sample_submission_df.head()) # Now sample_submission_df should be loaded

        print("\nGenerated Submission Format:")
        print(submission_df.head())

        # --- Save Submission File ---
        submission_filename = 'submission.csv'
        submission_output_path = os.path.join('/kaggle/working/', submission_filename) # Explicit path
        submission_df.to_csv(submission_output_path, index=False) # Save to the correct path
        print(f"\nSubmission file '{submission_filename}' created successfully in /kaggle/working/")
        # Verify file exists
        !ls /kaggle/working/ | grep submission.csv

        print("\n--- SCRIPT SECTION COMPLETE ---")
        # print("Remember to add Markdown cells for your report as the final step.") # Commented out if report is done


    except NameError as e:
        print(f"\n--- ERROR creating submission file: A required variable is not defined ---")
        print(f"Missing variable: {e}")
        print("Please ensure 'test_df_orig', 'actual_test_filename_col', and 'score_col' (and potentially BASE_PATH) are defined correctly by running previous cells (especially Step 2).")
    except FileNotFoundError as e:
        print(f"\n--- ERROR creating submission file: File not found ---")
        print(f"{e}")
        print("Please ensure BASE_PATH is correct and the 'sample_submission.csv' file exists.")
    except KeyError as e:
        print(f"\n--- ERROR creating submission file: Column not found in DataFrame ---")
        print(f"Column missing: {e}. Check column names ('{actual_test_filename_col}', '{score_col}') against test_df_orig.")
    except Exception as e:
        print(f"\n--- UNEXPECTED ERROR creating submission file ---")
        print(e)



# ==============================================================================
# Step: Calculate Correlation Scores on Validation Set
# ==============================================================================
from scipy.stats import pearsonr, spearmanr

print("Calculating Correlation Scores on Validation Set...")

# --- Get the best model identified during the validation phase ---
# Ensure the 'results' dictionary and 'best_model_name' exist from the training/validation cell
if 'results' not in locals() or 'best_model_name' not in locals():
    print("ERROR: 'results' dictionary or 'best_model_name' not found. Please re-run the model training/validation cell.")
elif 'X_val_split' not in locals() or 'y_val_split' not in locals():
     print("ERROR: Validation data (X_val_split, y_val_split) not found. Please re-run the data splitting cell.")
else:
    # Get the model instance that was trained only on the 80% split
    validation_phase_model = results[best_model_name]['model']

    # --- Generate predictions on the validation set ---
    y_pred_validation = validation_phase_model.predict(X_val_split)

    # --- Calculate Correlation Coefficients ---
    # Pearson Correlation (Linear)
    pearson_corr, pearson_p_value = pearsonr(y_val_split, y_pred_validation)

    # Spearman Correlation (Rank-based, Monotonic)
    spearman_corr, spearman_p_value = spearmanr(y_val_split, y_pred_validation)

    print("\n--- Validation Set Correlation Scores ---")
    print(f"  Pearson Correlation: {pearson_corr:.4f}")
    print(f"  Spearman Correlation: {spearman_corr:.4f}")
    print("  (Closer to 1.0 is better for correlation)")

    # --- Optional: Calculate on Full Training Set for Comparison ---
    # Using the final model trained on all data and its predictions
    if 'y_pred_train_final' in locals() and 'y' in locals():
        pearson_corr_train, _ = pearsonr(y, y_pred_train_final)
        spearman_corr_train, _ = spearmanr(y, y_pred_train_final)
        print("\n--- Full Training Set Correlation Scores (for comparison) ---")
        print(f"  Pearson Correlation (Train): {pearson_corr_train:.4f}")
        print(f"  Spearman Correlation (Train): {spearman_corr_train:.4f}")
    else:
        print("\nCould not calculate training set correlations (predictions not found).")


print("\n--- Interpretation ---")
print("The Validation Set scores (Pearson/Spearman) give an estimate of how your model might perform")
print("on the leaderboard's correlation metric using unseen data.")
print("A significant drop from Training Set Correlation to Validation Set Correlation suggests overfitting.")
