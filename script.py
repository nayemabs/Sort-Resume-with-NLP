import logging
import os
import sys
import pandas as pd
import joblib
import string
import re
import fitz  # PyMuPDF
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, \
    confusion_matrix
from sklearn.ensemble import RandomForestClassifier
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
import shutil

nltk.download('wordnet', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('punkt', quiet=True)

# Paths
RESUME_CSV_PATH = 'dataset/Resume/Resume.csv'
MODEL_PATH = 'resume_categorizer_model.pkl'

# Configure logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    handlers=[logging.StreamHandler(),
                              logging.FileHandler('model_training.log')])

# Initialize Lemmatizer
lemmatizer = WordNetLemmatizer()

def preprocess_text(text):
    text = text.lower().translate(str.maketrans('', '', string.punctuation))
    text = clean_resume(text)
    words = word_tokenize(text)
    return ' '.join(lemmatizer.lemmatize(word) for word in words if word not in stopwords.words('english'))

def clean_resume(text):
    return re.sub(r'http\S+|RT|cc|#\S+|@\S+|[!"#$%&\'()*+,-./:;<=>?@\[\\\]^_`{|}~]|\s+', ' ', text)

def train_model():
    df = pd.read_csv(RESUME_CSV_PATH)
    logging.info("Dataset loaded.")

    df['Resume_str'] = df['Resume_str'].apply(preprocess_text)

    X_train, X_test, y_train, y_test = train_test_split(df['Resume_str'], df['Category'], test_size=0.2,
                                                        random_state=42)

    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(stop_words='english', max_features=5000)),
        ('clf', RandomForestClassifier(class_weight='balanced', random_state=42)),
    ])

    best_params = {
        'tfidf__ngram_range': (1, 2),
        'tfidf__max_df': 0.75,
        'tfidf__min_df': 5,
        'clf__n_estimators': 300,
        'clf__max_depth': None,
        'clf__min_samples_split': 10
    }

    pipeline.set_params(**best_params)
    pipeline.fit(X_train, y_train)
    logging.info("Model training completed.")

    y_pred = pipeline.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
    recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)

    logging.info(f"Model accuracy: {accuracy:.2f}")
    logging.info(f"Model precision: {precision:.2f}")
    logging.info(f"Model recall: {recall:.2f}")
    logging.info(f"Model F1 score: {f1:.2f}")
    logging.info("Classification Report:\n" + classification_report(y_test, y_pred, zero_division=0))
    logging.info(f"Confusion Matrix:\n{confusion_matrix(y_test, y_pred)}")

    joblib.dump(pipeline, MODEL_PATH)
    logging.info(f"Model saved to {MODEL_PATH}")

def extract_text_from_pdf(pdf_path):
    text = ''
    try:
        with fitz.open(pdf_path) as doc:
            for page in doc:
                text += page.get_text()
    except Exception as e:
        logging.error(f"Error reading {pdf_path}: {e}")
    return text

def move_resume_to_category_folder(filename, category, output_dir):
    category_folder = os.path.join(output_dir, category)
    os.makedirs(category_folder, exist_ok=True)

    src_path = os.path.join(output_dir, filename)
    dst_path = os.path.join(category_folder, filename)

    shutil.move(src_path, dst_path)
    logging.info(f"Moved {filename} to {category_folder}")

def categorize_resumes(output_dir):
    if not os.path.exists(MODEL_PATH):
        logging.error("Model not found. Train the model first.")
        return

    model = joblib.load(MODEL_PATH)
    logging.info("Model loaded.")

    new_resumes = []
    for filename in os.listdir(output_dir):
        if filename.endswith(".pdf"):
            file_path = os.path.join(output_dir, filename)
            resume_text = preprocess_text(extract_text_from_pdf(file_path))
            new_resumes.append((filename, resume_text))

    if not new_resumes:
        logging.info("No new resumes found.")
        return

    filenames, texts = zip(*new_resumes)
    predicted_categories = model.predict(texts)

    categorized_data = {'Filename': filenames, 'Category': predicted_categories}

    for filename, category in zip(filenames, predicted_categories):
        move_resume_to_category_folder(filename, category, output_dir)

    categorized_df = pd.DataFrame(categorized_data)
    categorized_csv_path = os.path.join(output_dir, 'categorized_resumes.csv')
    categorized_df.to_csv(categorized_csv_path, index=False)
    logging.info(f"Categorized resumes saved to {categorized_csv_path}")

if __name__ == "__main__":
    if len(sys.argv) == 1:
        train_model()
    elif len(sys.argv) == 2:
        output_directory = sys.argv[1]
        os.makedirs(output_directory, exist_ok=True)
        categorize_resumes(output_directory)
    else:
        logging.error("Usage: python script.py [path/to/dir]")
        sys.exit(1)
