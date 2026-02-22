import os
import pandas as pd
import PyPDF2
import re
import nltk
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Download NLTK stopwords
nltk.download('stopwords')
from nltk.corpus import stopwords

# Define folder paths
input_folder = r"C:\Users\User\Desktop\ML input"
output_folder = r"C:\Users\User\Desktop\output"
csv_path = r"C:\Users\User\Desktop\ML input\labels.csv"

# Load labels
df = pd.read_csv(csv_path)

# Clean and convert PDFs to text
def extract_and_clean_text(pdf_path):
    with open(pdf_path, 'rb') as file:
        reader = PyPDF2.PdfReader(file)
        raw_text = ''
        for page in reader.pages:
            raw_text += page.extract_text() or ''
    
    # Clean text: remove page numbers, headers, footers
    cleaned_text = re.sub(r'\s*\n\s*\d+\s*\n', ' ', raw_text)  # remove page numbers
    cleaned_text = re.sub(r'\n+', '\n', cleaned_text)
    cleaned_text = re.sub(r'[^\w\s]', '', cleaned_text)  # remove punctuation
    cleaned_text = cleaned_text.lower()
    
    stop_words = set(stopwords.words('english'))
    words = cleaned_text.split()
    words = [word for word in words if word not in stop_words]
    
    return ' '.join(words)

texts = []
labels = []

for index, row in df.iterrows():
    file_name = row['filename']
    label = row['label']
    
    input_path = os.path.join(input_folder, file_name)
    output_path = os.path.join(output_folder, file_name.replace('.pdf', '.txt'))
    
    if os.path.exists(input_path):
        text = extract_and_clean_text(input_path)
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(text)
        
        texts.append(text)
        labels.append(label)
    else:
        print(f"File not found: {file_name}")

# Vectorize and classify
vectorizer = TfidfVectorizer(max_features=3000)
X = vectorizer.fit_transform(texts)
y = labels

# Stratified split to maintain class distribution
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Train Random Forest classifier
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

# Evaluation
print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred, labels=clf.classes_)
sns.heatmap(cm, annot=True, xticklabels=clf.classes_, yticklabels=clf.classes_, cmap='Blues', fmt='g')
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()
