import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, classification_report

def random_forest_for_fraud(df):
    """
    Trains and evaluates a Random Forest model for fraud detection
    using the provided columns.

    Args:
        df (pd.DataFrame): The input Pandas DataFrame containing Google review data
                           and a 'fraud' column (0 or 1).

    Returns:
        tuple: A tuple containing the trained Random Forest pipeline
               and the evaluation metrics (AUC and classification report).
    """

    # Identify categorical and numerical features
    text_feature = 'review_text'
    numerical_features = [
        'rating',
        'review_rating',
        'review_likes',
        'author_reviews_count',
        'author_ratings_count',
    ]
    categorical_features = [
        'review_questions_Positive',
        'review_questions_Negative'
    ]

    # Define the target variable
    target = 'fraud'

    if target not in df.columns:
        print(f"Error: The target column '{target}' is not found in the DataFrame.")
        return None, None

    # Separate features and target
    X = df.drop(columns=[target], errors='ignore')
    y = df[target]

    # Handle missing values in review_text: FILL WITH EMPTY STRING
    X['review_text'] = X['review_text'].fillna('')

    for col in categorical_features:
        if col in X.columns:
            X[col] = X[col].astype(str)

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Create preprocessor for different feature types
    preprocessor = ColumnTransformer(
        transformers=[
            ('text', TfidfVectorizer(max_features=5000, stop_words='english', ngram_range=(1, 2)), text_feature),
            ('num', StandardScaler(), numerical_features),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
        ],
        remainder='drop'
    )

    # Create the Random Forest pipeline
    model = Pipeline(steps=[('preprocessor', preprocessor),
                            ('classifier',
                             RandomForestClassifier(random_state=42, class_weight='balanced'))])

    # Train the model
    model.fit(X_train, y_train)

    # Make predictions on the test set
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    y_pred = model.predict(X_test)

    # Evaluate the model
    auc = roc_auc_score(y_test, y_pred_proba)
    report = classification_report(y_test, y_pred)

    print("\nRandom Forest Results:")
    print(f"AUC on the test set: {auc:.4f}")
    print("\nClassification Report on the test set:")
    print(report)

    return model, (auc, report)

# Example Usage:
# Assuming your DataFrame is named 'reviews_with_fraud_df' and has a 'fraud' column
# Make sure you have run the cleaning steps and handled missing values.

# Load your data
reviews_with_fraud_df = pd.read_csv('Data/combined_with_reviews.csv')

# Clean the data (as per your previous steps)
reviews_with_fraud_cleaned_df = reviews_with_fraud_df

# Handle missing values in numerical columns
for col in ['rating', 'review_rating', 'review_likes', 'author_reviews_count', 'author_ratings_count', 'review_questions_Positive', 'review_questions_Negative']:
    if col in reviews_with_fraud_cleaned_df.columns:
        reviews_with_fraud_cleaned_df[col] = reviews_with_fraud_cleaned_df[col].fillna(0)

# Convert categorical features to string
for col in ['review_questions_Positive', 'review_questions_Negative']:
    if col in reviews_with_fraud_cleaned_df.columns:
        reviews_with_fraud_cleaned_df[col] = reviews_with_fraud_cleaned_df[col].astype(str)

# Convert numerical columns to numeric
for col in ['rating', 'review_rating', 'review_likes', 'author_reviews_count', 'author_ratings_count']:
    if col in reviews_with_fraud_cleaned_df.columns:
        reviews_with_fraud_cleaned_df[col] = pd.to_numeric(reviews_with_fraud_cleaned_df[col], errors='coerce').fillna(0)

# Run the Random Forest model
trained_rf_model, rf_evaluation_metrics = random_forest_for_fraud(reviews_with_fraud_cleaned_df.copy())

if trained_rf_model:
    print("\nTrained Random Forest Pipeline:", trained_rf_model)
    rf_auc, rf_report = rf_evaluation_metrics