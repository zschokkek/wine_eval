import pandas as pd
import numpy as np
import re 
import string
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.svm import SVR
from sklearn.model_selection import cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.tokenize import word_tokenize
import nltk
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, explained_variance_score
import matplotlib.pyplot as plt

nltk.download('stopwords')
nltk.download('punkt_tab')
nltk.download('averaged_perceptron_tagger')
nltk.download('stopwords')
nltk.download('wordnet')



# Load the data
file_path = 'archive/winemag-data_first150k.csv'
data = pd.read_csv(file_path)

# Select relevant columns and drop rows with missing descriptions or points
df = data[['description', 'points']].dropna()

y = df['points'].values

# Initialize preprocessors
stop_words = set(stopwords.words('english'))
stemmer = PorterStemmer()
lemmatizer = WordNetLemmatizer()

# Preprocessing function
def preprocess_text(text):
    # Lowercasing
    text = text.lower()
    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    # Tokenization
    tokens = word_tokenize(text)
    # Remove stop words
    tokens = [word for word in tokens if word not in stop_words]
    # Stemming and Lemmatization
    tokens = [stemmer.stem(word) for word in tokens]  # Apply stemming
    tokens = [lemmatizer.lemmatize(word) for word in tokens]  # Apply lemmatization
    # Join tokens back to string
    return ' '.join(tokens)

# Apply preprocessing
df['cleaned_description'] = df['description'].apply(preprocess_text)

# Define the models to be used
models = {
    'Linear Regression': LinearRegression(),
    'Ridge Regression': Ridge(),
    'Lasso Regression': Lasso(),
    # 'svr' : SVR()
}

# Define the parameter grids for models with hyperparameters
param_grids = {
    'Ridge Regression': {'alpha': [.001, 0.1, 1.0, 5.0, 10.0, 100.0]},
    'Lasso Regression': {'alpha': [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]},
    # 'svr' : {
    #     'C': [0.1, 10],  # Regularization parameter
    #     'epsilon': [0.01, 0.1, 1],  # Epsilon parameter for the SVR
    #     'kernel': ['linear']  # Different kernels to try
    # }
}

# Function to perform Grid Search and return best model
def perform_grid_search(model, param_grid, X_train, y_train):
    grid_search = GridSearchCV(model, param_grid, scoring='neg_mean_squared_error', cv=3, n_jobs=-1)
    grid_search.fit(X_train, y_train)
    return grid_search.best_estimator_, grid_search.best_params_


def mean_absolute_percentage_error(y_true, y_pred):
    # Avoid division by zero and calculate MAPE
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

def symmetric_mean_absolute_percentage_error(y_true, y_pred):
    # Calculate SMAPE
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs(y_true - y_pred) / ((np.abs(y_true) + np.abs(y_pred)) / 2)) * 100

def within_range_percentage(y_true, y_pred, tolerance=2):
    # Calculate the percentage of predictions within a certain range (± tolerance)
    return np.mean(np.abs(y_true - y_pred) <= tolerance) * 100

# Function to train and evaluate models
def evaluate_models(X, y, encoding_name):
    # Split the data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    mse_scores = {}
    r2_scores = {}
    mae_scores = {}
    explained_variance_scores = {}
    cross_val_mse_scores = {}
    best_params = {}
    all_predictions = {}

    print(f"\nEvaluating models with {encoding_name} encoding:")
    
    for model_name, model in models.items():
        print(f"Training {model_name}...")
        if model_name in param_grids:
            best_model, best_param = perform_grid_search(model, param_grids[model_name], X_train, y_train)
            best_params[model_name] = best_param
        else:
            best_model = model
            best_model.fit(X_train, y_train)
        
        # Cross-validation to get a robust measure of performance
        cross_val_scores = cross_val_score(best_model, X_train, y_train, cv=5, scoring='neg_mean_squared_error')
        cross_val_mse = -np.mean(cross_val_scores)
        cross_val_mse_scores[model_name] = cross_val_mse

        # Evaluate on the test set
        y_pred = best_model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        explained_variance = explained_variance_score(y_test, y_pred)
        mape = mean_absolute_percentage_error(y_test, y_pred)
        smape = symmetric_mean_absolute_percentage_error(y_test, y_pred)
        within_range = within_range_percentage(y_test, y_pred, tolerance=2)  # ±2 points

        # Store evaluation metrics
        mse_scores[model_name] = mse
        r2_scores[model_name] = r2
        mae_scores[model_name] = mae
        explained_variance_scores[model_name] = explained_variance
        all_predictions[model_name] = y_pred

        # Print the results
        print(f'{model_name} - MSE: {mse:.2f}, R2: {r2:.2f}, MAE: {mae:.2f}, Explained_Variance: {explained_variance:.2f}')
        print(f'{model_name} - MAPE: {mape:.2f}%, SMAPE: {smape:.2f}%, Within ±2 Points: {within_range:.2f}%')
        print(f'{model_name} - Cross-Validation MSE: {cross_val_mse:.2f}, Best Params: {best_params.get(model_name, "N/A")}')
    return mse_scores, r2_scores, mae_scores, explained_variance_scores, all_predictions, X_test, y_test

# Vectorize the descriptions using TF (CountVectorizer)
tf_vectorizer = CountVectorizer(max_features=10000)
X_tf = tf_vectorizer.fit_transform(df['cleaned_description'])

# Evaluate models with TF encoding
mse_scores_tf, r2_scores_tf, mae_scores_tf, explained_variance_scores_tf, all_predictions_tf, X_test_tf, y_test_tf = evaluate_models(X_tf, y, 'TF')

# Vectorize the descriptions using TF-IDF
tfidf_vectorizer = TfidfVectorizer(max_features=10000)
X_tfidf = tfidf_vectorizer.fit_transform(df['cleaned_description'])

# Evaluate models with TF-IDF encoding
mse_scores_tfidf, r2_scores_tfidf, mae_scores_tfidf, explained_variance_scores_tfidf, all_predictions_tfidf, X_test_tfidf, y_test_tfidf = evaluate_models(X_tfidf, y, 'TF-IDF')

# Plotting the comparison of TF and TF-IDF

# Creating subplots
fig, axes = plt.subplots(2, 2, figsize=(14, 12))

# Bar plot for MSE
model_names = list(models.keys())
mse_tf_values = list(mse_scores_tf.values())
mse_tfidf_values = list(mse_scores_tfidf.values())

x = np.arange(len(model_names))
width = 0.35

fig, ax = plt.subplots(figsize=(10, 6))
ax.bar(x - width/2, mse_tf_values, width, label='TF', color='skyblue')
ax.bar(x + width/2, mse_tfidf_values, width, label='TF-IDF', color='salmon')
ax.set_ylabel('Mean Squared Error')
ax.set_title('Comparison of MSE (TF vs TF-IDF)')
ax.set_xticks(x)
ax.set_xticklabels(model_names, rotation=45, ha='right')
ax.legend()

plt.tight_layout()
plt.show()

# Bar plot for R2 Score
r2_tf_values = list(r2_scores_tf.values())
r2_tfidf_values = list(r2_scores_tfidf.values())

fig, ax = plt.subplots(figsize=(10, 6))
ax.bar(x - width/2, r2_tf_values, width, label='TF', color='skyblue')
ax.bar(x + width/2, r2_tfidf_values, width, label='TF-IDF', color='salmon')
ax.set_ylabel('R2 Score')
ax.set_title('Comparison of R2 Score (TF vs TF-IDF)')
ax.set_xticks(x)
ax.set_xticklabels(model_names, rotation=45, ha='right')
ax.legend()

plt.tight_layout()
plt.show()

# Residuals distribution for TF-IDF
fig, ax = plt.subplots(figsize=(10, 6))
for model_name in model_names:
    residuals_tfidf = y_test_tfidf - all_predictions_tfidf[model_name]
    ax.hist(residuals_tfidf, bins=30, alpha=0.5, label=model_name)
ax.set_title('Residuals Distribution (TF-IDF)')
ax.set_xlabel('Residuals')
ax.set_ylabel('Frequency')
ax.legend()

plt.tight_layout()
plt.show()

# Actual vs Predicted for TF-IDF
fig, ax = plt.subplots(figsize=(10, 6))
for model_name in model_names:
    ax.scatter(y_test_tfidf, all_predictions_tfidf[model_name], alpha=0.5, label=model_name)
ax.plot([80, 100], [80, 100], color='red', linestyle='--')  # Identity line
ax.set_title('Actual vs Predicted (TF-IDF)')
ax.set_xlabel('Actual Points')
ax.set_ylabel('Predicted Points')
ax.legend()

plt.tight_layout()
plt.show()
plt.tight_layout()
plt.show()

# Feature Importance or Coefficients Analysis for TF and TF-IDF
# For Linear Models (e.g., Ridge Regression)

def plot_top_features(vectorizer, model, model_name, top_n=10):
    feature_names = vectorizer.get_feature_names_out()
    if hasattr(model, 'coef_'):
        coefficients = model.coef_
    else:
        coefficients = model.feature_importances_
    
    coef_df = pd.DataFrame({'word': feature_names, 'coefficient': coefficients})
    coef_df['abs_coefficient'] = coef_df['coefficient'].abs()
    
    # Filter standalone numbers between 1900-2024
    mask = ~(coef_df['word'].str.match(r'^\d+$')) | (
        coef_df['word'].str.match(r'^\d+$') & 
        pd.to_numeric(coef_df['word'], errors='coerce').between(1900, 2024)
    )
    coef_df = coef_df[mask]
    
    top_positive_words = coef_df.sort_values(by='coefficient', ascending=False).head(top_n)
    top_negative_words = coef_df.sort_values(by='coefficient', ascending=True).head(top_n)

    print("Top Positive Words:")
    print(top_positive_words)
    
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.barh(top_positive_words['word'], top_positive_words['coefficient'], color='green')
    plt.xlabel('Coefficient' if hasattr(model, 'coef_') else 'Feature Importance')
    plt.title(f'Top {top_n} Positive Predictors ({model_name})')
    plt.tight_layout()
    plt.show()

    print("Top Negative Words:")
    print(top_negative_words)
    
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.barh(top_negative_words['word'], top_negative_words['coefficient'], color='red')
    plt.xlabel('Coefficient' if hasattr(model, 'coef_') else 'Feature Importance')
    plt.title(f'Top {top_n} Negative Predictors ({model_name})')
    plt.tight_layout()
    plt.show()
# Train Ridge Regression and Random Forest on TF data
ridge_tf = Ridge(alpha=1.0)
ridge_tf.fit(X_tf, y)

# Train Ridge Regression and Random Forest on TF-IDF data
ridge_tfidf = Ridge(alpha=1.0)
ridge_tfidf.fit(X_tfidf, y)

# Plot top features for Ridge Regression and Random Forest with TF data
print("Top features for Ridge Regression (TF):")
plot_top_features(tf_vectorizer, ridge_tf, 'Ridge Regression (TF)')

# Plot top features for Ridge Regression and Random Forest with TF-IDF data
print("Top features for Ridge Regression (TF-IDF):")
plot_top_features(tfidf_vectorizer, ridge_tfidf, 'Ridge Regression (TF-IDF)')
