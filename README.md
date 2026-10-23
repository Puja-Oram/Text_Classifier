# Text Classification Model Comparison

This project compares four models on two categories from the 20 Newsgroups text dataset:

- rec.sport.baseball
- sci.space

The script converts the text to TF-IDF features, makes a stratified 80/20 train/test split, trains the models, and prints accuracy, precision, recall, F1 score, classification reports, and confusion matrices. It also demonstrates predictions for two sample texts using the Linear SVM model.

# Models

- Logistic Regression
- Random Forest
- Linear SVM (LinearSVC)
- Linear Regression baseline

Linear Regression is included for comparison with the original script. It is a regression algorithm and is not generally recommended for classification; Logistic Regression, Random Forest, and Linear SVM are the classification models in this comparison.

# Requirements

- Python 3.9 or newer
- Packages listed in [requirements.txt](requirements.txt)

# Setup and run

From the project directory, create and activate a virtual environment, then install the dependencies:

--bash
python -m venv .venv
--

Windows PowerShell:

--powershell
.\.venv\Scripts\Activate.ps1
--

macOS/Linux:

--bash
source .venv/bin/activate
--

Install the packages and run the model comparison:

--bash
python -m pip install -r requirements.txt
python text_classification_models.py
--

On its first run, scikit-learn downloads the 20 Newsgroups dataset. An internet connection is needed for that download. The reported scores are calculated at run time from the fixed train/test split (random_state=42); results can vary slightly across package versions.

# Project files

--text
text_classification_models.py  # Data loading, training, evaluation, and sample predictions
requirements.txt               # Python package dependencies
README.md                      # Project overview and run instructions
.gitignore                     # Files and environments not to commit
--

