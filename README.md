
# Spoken Grammar Assessment Engine

This project aims to build a machine learning model that predicts the grammatical proficiency of spoken English from audio recordings. It takes WAV audio files (45-60 seconds long) as input and outputs a continuous grammar score between 1 and 5, based on a predefined rubric.

This solution was developed for the SHL Intern Hiring Assessment competition on Kaggle.

## Project Objective

The primary goal is to create a robust "Grammar Scoring Engine". The model should learn patterns in audio features and transcribed text that correlate with human-assigned grammar scores.

## Approach & Features

The solution employs a feature-based machine learning approach:

1.  **Audio Preprocessing:**
    *   Audio files are loaded and resampled to a standard rate (16kHz) using `librosa`.

2.  **Feature Extraction:** A combination of acoustic, prosodic, and text-based features are extracted:
    *   **Acoustic:** Mel-Frequency Cepstral Coefficients (MFCCs - mean & std), Root Mean Square energy (RMS - mean & std), Zero-Crossing Rate (ZCR - mean & std).
    *   **Prosodic:** Fundamental Frequency (F0 / Pitch - mean & std) extracted from voiced segments using `librosa.pyin`.
    *   **Speech-to-Text (STT):** OpenAI's `Whisper` model (`base.en` or configurable) is used to transcribe the audio into text.
    *   **Text-Based (from Transcript):** Word Count, Sentence Count (simple split by '.'), Average Sentence Length.
    *   **Grammar (Attempted):** `language_tool_python` was included, but typically fails in Kaggle environments due to Java dependencies. The corresponding 'grammar\_errors' feature defaults to 0.

3.  **Data Preprocessing:**
    *   **Handling Missing Values:** Features that might result in NaN during extraction (e.g., from silent audio) are imputed using the median value from the training set (`SimpleImputer`).
    *   **Scaling:** All numerical features are standardized using `StandardScaler` (mean 0, variance 1) fitted *only* on the training data.
    *   **Alignment:** Feature columns between the training and test sets are explicitly aligned to ensure consistency before scaling and prediction.

4.  **Modeling:**
    *   **Task:** Regression (predicting a continuous score).
    *   **Models Tested:** Ridge Regression, Support Vector Regressor (SVR), Random Forest Regressor, Gradient Boosting Regressor, XGBoost Regressor, LightGBM Regressor.
    *   **Validation:** 5-Fold Cross-Validation (`KFold`) is used to get a robust estimate of model performance and for hyperparameter tuning.
    *   **Hyperparameter Tuning:** `RandomizedSearchCV` explores different combinations of hyperparameters for each model, aiming to maximize the **Pearson Correlation Coefficient** on the cross-validation folds.
    *   **Model Selection:** The model type and hyperparameter combination yielding the highest average Pearson Correlation during cross-validation is selected as the final model.
    *   **Final Training:** The selected best model is re-trained on the *entire* training dataset (using the tuned hyperparameters).

## Dependencies

*   Python 3.x
*   `kaggle` (for data download)
*   `pandas`
*   `numpy`
*   `scikit-learn`
*   `librosa`
*   `soundfile` (dependency for librosa)
*   `torch` (for Whisper)
*   `openai-whisper`
*   `language_tool_python` (included but expected to fail gracefully)
*   `xgboost`
*   `lightgbm`
*   `matplotlib` & `seaborn` (for visualizations)
*   `tqdm` (for progress bars)

## How to Run

1.  **Prerequisites:**
    *   A Kaggle account.
    *   Your Kaggle API token (`kaggle.json`).
2.  **Environment:**
    *   Upload this notebook to Kaggle Notebooks.
    *   **Enable GPU Accelerator:** Go to Settings -> Accelerator -> Select GPU (e.g., T4 x2). This significantly speeds up Whisper transcription.
    *   Upload your `kaggle.json` file as a Kaggle "Input Dataset" (e.g., create a private dataset named `kaggle-api` and upload the file there).
3.  **Configure:**
    *   Verify the `KAGGLE_JSON_PATH` variable in Step 0 points to your uploaded `kaggle.json`.
    *   (Optional) Adjust `WHISPER_MODEL_SIZE` in Step 4 if desired (e.g., "small.en"), but be mindful of resource limits.
    *   (Optional) Adjust `N_SPLITS` (CV folds) or `N_ITER_SEARCH` (tuning iterations) in Step 8 based on desired thoroughness vs. runtime.
4.  **Execute:** Run all cells in the notebook sequentially.
5.  **Output:** The script will generate:
    *   `submission.csv`: A file in the `/kaggle/working/` directory ready for submission to the competition.
    *   Performance metrics printed in the output cells (including the required Training RMSE).
    *   Evaluation plots displayed within the notebook.

## Code Structure (Notebook Steps)

*   **Step 0:** Installs libraries, sets up Kaggle API credentials.
*   **Step 1:** Downloads and extracts competition data.
*   **Step 2:** Defines file paths, loads CSVs, adds full audio paths.
*   **Step 3:** Performs basic Exploratory Data Analysis (EDA) on target scores.
*   **Step 4:** Sets up feature extraction (loads Whisper model, configures parameters).
*   **Step 5:** Defines the `extract_features` function (acoustic, prosodic, text).
*   **Step 6:** Applies feature extraction to all train and test audio files.
*   **Step 7:** Preprocesses the extracted features (imputation, scaling, alignment).
*   **Step 8:** Performs model training using K-Fold CV and Randomized Search hyperparameter tuning. Selects the best model based on CV Pearson score.
*   **Step 9:** Evaluates the final retrained model on the *full training set* (calculates required Training RMSE) and generates plots.
*   **Step 10:** Generates predictions on the test set using the final model and creates the `submission.csv` file.
*   **Step 11:** Contains the text for the final report summarizing the process and results.

## Potential Improvements

*   **Advanced Feature Engineering:** Explore more sophisticated prosodic features (jitter, shimmer, speaking rate, pause analysis) using tools like Parselmouth (Praat wrapper).
*   **NLP on Transcripts:** Use more advanced NLP techniques on the Whisper transcripts (e.g., sentence embeddings via BERT, dependency parsing, actual grammar checking tools if environment issues are resolved).
*   **Larger Whisper Model:** Experiment with `small.en` for potentially better transcriptions if resources allow.
*   **Ensemble Methods:** Combine predictions from the top-performing models (e.g., averaging, stacking).
*   **Data Augmentation:** Apply audio augmentations during training to improve robustness.
*   **End-to-End Models:** Explore deep learning models (CNNs, Transformers) that process audio representations (like spectrograms) directly.
