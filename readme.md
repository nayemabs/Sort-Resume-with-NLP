# Resume Categorization Project

This project is designed to automatically categorize resumes into different domains (e.g., Sales, Marketing) using machine learning. The model is trained on a dataset of resumes and is capable of categorizing new resumes by predicting their respective categories.

## Table of Contents

- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Preprocessing](#preprocessing)
- [Training the Model](#training-the-model)
- [Categorizing New Resumes](#categorizing-new-resumes)
- [Logging](#logging)

## Project Structure
.
├── Data_Exploration.ipynb
├── dataset
│   ├── data
│   │   └── data
│   │       ├── ACCOUNTANT
│   │       ├── ADVOCATE
│   │       ├── AGRICULTURE
│   │       ├── APPAREL
│   │       ├── ARTS
│   │       ├── AUTOMOBILE
│   │       ├── AVIATION
│   │       ├── BANKING
│   │       ├── BPO
│   │       ├── BUSINESS-DEVELOPMENT
│   │       ├── CHEF
│   │       ├── CONSTRUCTION
│   │       ├── CONSULTANT
│   │       ├── DESIGNER
│   │       ├── DIGITAL-MEDIA
│   │       ├── ENGINEERING
│   │       ├── FINANCE
│   │       ├── FITNESS
│   │       ├── HEALTHCARE
│   │       ├── HR
│   │       ├── INFORMATION-TECHNOLOGY
│   │       ├── PUBLIC-RELATIONS
│   │       ├── SALES
│   │       └── TEACHER
│   ├── Resume
│   │   ├── AGRICULTURE
│   │   ├── ARTS
│   │   ├── ENGINEERING
│   │   ├── HR
│   │   ├── INFORMATION-TECHNOLOGY
│   │   └── TEACHER
│   └── Test
│       ├── AGRICULTURE
│       ├── ARTS
│       ├── ENGINEERING
│       ├── HR
│       ├── INFORMATION-TECHNOLOGY
│       └── TEACHER
├── dataset.zip
├── ML Engineer Task.pdf
├── model_training.log
├── readme.md
├── requirements.txt
├── results
│   ├── AGRICULTURE
│   ├── ARTS
│   ├── ENGINEERING
│   ├── HR
│   ├── INFORMATION-TECHNOLOGY
│   └── TEACHER
├── resume_categorizer_model.pkl
└── script.py

- `dataset/Resume/Resume.csv`: The dataset containing resumes used for training the model.
- `script.py`: The main script that trains the model and categorizes new resumes.
- `model_training.log`: Log file for recording the training process.
- `resume_categorizer_model.pkl`: Saved model after training.
- `Data_Exploration.ipynb`: Data exploration and initial works

## Installation

1. Clone the repository:
    ```bash
    git clone https://github.com/your-repo/resume-categorization.git
    cd resume-categorization
    ```

2. Install the required Python packages:
    ```bash
    pip install -r requirements.txt
    ```

   The project has the following python package requirements:
    ```
    pandas
    scikit-learn
    joblib
    nltk
    PyMuPDF
    ```

3. Download and unzip the dataset:
    - Download the dataset from [this link](https://drive.google.com/file/d/1S_QL3ELp1scyBIxGg52iuxBjeO1UAyRV/view).
    - Unzip the downloaded file and place the contents in the `dataset/` directory.

4. Download NLTK data (this is done automatically in the script, but can be run manually if needed):
    ```python
    import nltk
    nltk.download('wordnet')
    nltk.download('stopwords')
    nltk.download('punkt')
    ```

## Usage

### Training the Model

1. Prepare your dataset:
   - Ensure that your `Resume.csv` file is placed in the `dataset/Resume/` directory.
   - The CSV file should have the following columns: `['ID', 'Resume_str', 'Resume_html', 'Category']`, where:
     - `Resume_str`: The text content of the resume.
     - `Category`: The domain to which the resume belongs (e.g., Sales, Marketing).

2. Train the model by running the script with no arguments:
    ```bash
    python script.py
    ```
   - The model will be trained using the dataset and saved as `resume_categorizer_model.pkl`.
   - Log information will be saved in `model_training.log`.

### Categorizing New Resumes

1. Place new resumes in PDF format in the directory specified when running the script.

2. Run the script to categorize the new resumes:
    ```bash
    python script.py path/to/dir
    ```
   - The script will load the saved model, categorize the new resumes, and move them to subfolders corresponding to their predicted categories.
   - A new CSV file named `categorized_resumes.csv` will be created in the specified directory containing the filenames and their predicted categories.

## Preprocessing

- Text preprocessing includes converting text to lowercase, removing punctuation, tokenizing, and lemmatizing words. Stopwords are also removed during this process.

- Resumes are cleaned using regular expressions to remove unwanted characters such as URLs, special symbols, and extra spaces.

## Model Selection

### Chosen Model: RandomForestClassifier

#### Rationale

**1. Versatility and Robustness:**
The `RandomForestClassifier` is a versatile and robust ensemble learning model that combines multiple decision trees to make more accurate and stable predictions. It is particularly well-suited for classification tasks with structured data, such as categorizing resumes into different domains.

**2. Handling of Imbalanced Data:**
In the context of resume categorization, there might be imbalances in the number of samples for different categories. The `RandomForestClassifier` can handle class imbalances effectively by using class weights. In our case, we use `class_weight='balanced'` to adjust weights inversely proportional to class frequencies, ensuring that minority classes are not overshadowed by majority classes.

**3. Feature Importance:**
Random forests provide insights into feature importance, which can help understand which features (in this case, terms and phrases extracted from resumes) are most influential in making predictions. This can be valuable for feature engineering and model interpretation.

**4. Reduced Overfitting:**
By averaging predictions from multiple decision trees, the `RandomForestClassifier` reduces the risk of overfitting compared to a single decision tree. This helps in generalizing better on unseen data, which is crucial for reliable resume categorization.

**5. Hyperparameter Tuning:**
The `RandomForestClassifier` offers several hyperparameters (e.g., `n_estimators`, `max_depth`, `min_samples_split`) that can be tuned to optimize model performance. The selected hyperparameters in our project (e.g., `n_estimators=300`, `max_depth=None`, `min_samples_split=10`) were chosen based on grid search optimization, balancing between model complexity and performance.

Overall, the `RandomForestClassifier` was chosen for its ability to handle complex classification tasks with imbalanced data, provide feature importance insights, and reduce overfitting, making it a strong candidate for the resume categorization project.

**Note**: The transformer-based and more complex models require a lot of graphical memory, so those were avoided. RNN-based models performed worse than the ML-based models. In my experience, Naive Bayes (`sklearn.naive_bayes.MultinomialNB`) performed quite well, but the `RandomForestClassifier` showed the best performance with tuned parameters.

## Training the Model

- The model uses a `RandomForestClassifier` wrapped in a `Pipeline` with `TfidfVectorizer` for feature extraction.

- The best hyperparameters are predefined in the script.

- The model's performance is evaluated using accuracy, precision, recall, F1 score, and confusion matrix.

## Categorizing New Resumes

- The script extracts text from PDF resumes using `PyMuPDF`, preprocesses the text, and predicts the category using the trained model.

- Resumes are moved to subfolders corresponding to their predicted categories within the specified output directory.

- The results are saved to a CSV file named `categorized_resumes.csv` in the specified directory.

## Logging

- The script logs important events, such as data loading, model training, and file movement, into `model_training.log`.

- The log file is helpful for debugging and understanding the flow of the script.
