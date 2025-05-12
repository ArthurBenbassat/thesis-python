import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix

def svm_for_fraud(df):
    """
    Trains and evaluates a Support Vector Machine (SVM) model for fraud detection at the company level.
    """
    text_feature = 'review_text'
    numerical_features = [
        'rating', 'review_rating', 'review_likes',
        'author_reviews_count', 'author_ratings_count'
    ]
    categorical_features = [
        'review_questions_Positive', 'review_questions_Negative'
    ]
    target = 'fraud'

    X = df.drop(columns=[target], errors='ignore')
    y = df[target]
    X['review_text'] = X['review_text'].fillna('')

    for col in categorical_features:
        X[col] = X[col].astype(str)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ('text', TfidfVectorizer(max_features=5000, stop_words='english', ngram_range=(1, 2)), text_feature),
            ('num', StandardScaler(), numerical_features),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
        ]
    )

    model = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', SVC(probability=True, class_weight='balanced', random_state=42))
    ])

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]

    auc = roc_auc_score(y_test, y_pred_proba)
    report = classification_report(y_test, y_pred)

    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Purples',
                xticklabels=['Non-Fraud', 'Fraud'],
                yticklabels=['Non-Fraud', 'Fraud'])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('SVM Confusion Matrix')
    plt.show()

    print("\nSupport Vector Machine (SVM) Results:")
    print(f"AUC on the test set: {auc:.4f}")
    print("\nClassification Report on the test set:")
    print(report)

    return model, (auc, report)

# === PREPROCESSING & BALANCING ON COMPANY LEVEL ===

# Load data
df_raw = pd.read_csv('Data/combined_with_reviews.csv')

# Combine reviews per company
def combine_reviews(reviews, n=15):
    return " ".join(reviews.dropna().tolist()[:n])

grouped_df = df_raw.groupby('place_id').agg({
    'fraud': 'max',
    'review_text': lambda x: combine_reviews(x, n=15),
    'rating': 'mean',
    'review_rating': 'mean',
    'review_likes': 'mean',
    'author_reviews_count': 'mean',
    'author_ratings_count': 'mean',
    'review_questions_Positive': lambda x: x.mode().iloc[0] if not x.mode().empty else 'unknown',
    'review_questions_Negative': lambda x: x.mode().iloc[0] if not x.mode().empty else 'unknown'
}).reset_index()

# Balance per company
df_fraud = grouped_df[grouped_df['fraud'] == 1]
df_nonfraud = grouped_df[grouped_df['fraud'] == 0]
n_samples = min(len(df_fraud), len(df_nonfraud))

df_fraud_sampled = df_fraud.sample(n=n_samples, random_state=42)
df_nonfraud_sampled = df_nonfraud.sample(n=n_samples, random_state=42)

balanced_df = pd.concat([df_fraud_sampled, df_nonfraud_sampled]).sample(frac=1, random_state=42).reset_index(drop=True)

# Fill and convert types
for col in ['rating', 'review_rating', 'review_likes', 'author_reviews_count', 'author_ratings_count']:
    balanced_df[col] = pd.to_numeric(balanced_df[col], errors='coerce').fillna(0)

for col in ['review_questions_Positive', 'review_questions_Negative']:
    balanced_df[col] = balanced_df[col].fillna('unknown').astype(str)

# Train model
trained_svm_model, svm_metrics = svm_for_fraud(balanced_df)

if trained_svm_model:
    print("\nTrained SVM Pipeline:", trained_svm_model)
