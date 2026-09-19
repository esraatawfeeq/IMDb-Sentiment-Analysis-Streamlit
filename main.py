
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
import re
import string

# NLTK setup
import nltk
from nltk.corpus import stopwords
nltk.download('stopwords')

# Scikit-Learn
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# ML Models
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier

# Keras / TensorFlow Models
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Bidirectional, Flatten, Conv1D, GlobalMaxPooling1D

# ==========================================
# 1. Load and Clean Dataset
# ==========================================
print("--- 1. Loading and Cleaning Data ---")
df = pd.read_csv('IMDB Dataset.csv')

# Sample 8000 rows for faster training
df = df.sample(n=8000, random_state=42).dropna()

stop_words = set(stopwords.words("english"))

def clean_text(text):
    text = text.lower()  # lowercase
    text = re.sub(r'\d+', '', text)  # remove numbers
    text = text.translate(str.maketrans('', '', string.punctuation))  # remove punctuation
    text = " ".join([word for word in text.split() if word not in stop_words])  # remove stopwords
    return text

df["review"] = df["review"].apply(clean_text)

texts = df["review"].astype(str).tolist()
labels = df["sentiment"].tolist()

# Encode Labels (Positive/Negative -> 1/0)
le = LabelEncoder()
y_encoded = le.fit_transform(labels)

# ==========================================
# 2. TF-IDF & Machine Learning Models
# ==========================================
print("\n--- 2. TF-IDF & ML Models Training ---")
tfidf = TfidfVectorizer(max_features=5000)
X_tfidf = tfidf.fit_transform(texts)

X_train, X_test, y_train, y_test = train_test_split(
    X_tfidf, y_encoded, test_size=0.2, random_state=42
)

models = {
    "Logistic Regression": LogisticRegression(max_iter=200),
    "Naive Bayes": MultinomialNB(),
    "SVM": LinearSVC(),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "Gradient Boosting": GradientBoostingClassifier(random_state=42),
    "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric="logloss", random_state=42)
}

accuracies = {}
reports = {}

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    acc = accuracy_score(y_test, y_pred)
    accuracies[name] = acc
    reports[name] = classification_report(y_test, y_pred, output_dict=True)
    
    print(f"\n🔹 {name} Accuracy: {acc:.4f}")

# Plot ML Model Comparison
plt.figure(figsize=(8, 5))
sns.barplot(x=list(accuracies.keys()), y=list(accuracies.values()))
plt.xticks(rotation=45)
plt.ylabel("Accuracy")
plt.title("ML Models Comparison")
plt.show()

# Best ML Model evaluation
best_model_name = max(accuracies, key=accuracies.get)
best_model = models[best_model_name]
print(f"\n✅ Best ML Model: {best_model_name} with Accuracy {accuracies[best_model_name]:.4f}")

y_pred_best = best_model.predict(X_test)

# Confusion Matrix for Best Model
cm = confusion_matrix(y_test, y_pred_best)
plt.figure(figsize=(5, 4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Negative", "Positive"], yticklabels=["Negative", "Positive"])
plt.title(f"Confusion Matrix - {best_model_name}")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# Save ML Models & Vectorizer
for name, model in models.items():
    filename = f"{name.lower().replace(' ', '_')}.pkl"
    with open(filename, "wb") as f:
        pickle.dump(model, f)

with open("tfidf_vectorizer.pkl", "wb") as f:
    pickle.dump(tfidf, f)

print("✅ Saved all ML models and TF-IDF Vectorizer successfully!")

# ==========================================
# 3. Deep Learning Preparation (Tokenizer & Padding)
# ==========================================
print("\n--- 3. Deep Learning Setup ---")
max_words = 5000
tokenizer = Tokenizer(num_words=max_words)
tokenizer.fit_on_texts(texts)

sequences = tokenizer.texts_to_sequences(texts)
max_sequence_length = max([len(seq) for seq in sequences])

X_padded = pad_sequences(sequences, maxlen=max_sequence_length, padding="post", truncating="post")

X_train_pad, X_test_pad, y_train_dl, y_test_dl = train_test_split(
    X_padded, y_encoded, test_size=0.2, random_state=42
)

epochs = 10
batch_size = 32

# --- Model 1: Bi-LSTM ---
print("\nTraining Bi-LSTM...")
lstm_model = Sequential([
    Embedding(input_dim=max_words, output_dim=100, input_length=max_sequence_length),
    Bidirectional(LSTM(128)),
    Dense(1, activation="sigmoid")
])
lstm_model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
lstm_model.fit(X_train_pad, y_train_dl, epochs=epochs, batch_size=batch_size, validation_data=(X_test_pad, y_test_dl), verbose=1)

# --- Model 2: Dense / Flatten Embedding ---
print("\nTraining Dense Embedding Model...")
embedding_model = Sequential([
    Embedding(input_dim=max_words, output_dim=100, input_length=max_sequence_length),
    Flatten(),
    Dense(1, activation='sigmoid')
])
embedding_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
embedding_model.fit(X_train_pad, y_train_dl, epochs=epochs, batch_size=batch_size, validation_data=(X_test_pad, y_test_dl), verbose=1)

# --- Model 3: CNN ---
print("\nTraining CNN Model...")
cnn_model = Sequential([
    Embedding(input_dim=max_words, output_dim=100, input_length=max_sequence_length),
    Conv1D(128, 5, activation="relu"),
    GlobalMaxPooling1D(),
    Dense(10, activation="relu"),
    Dense(1, activation="sigmoid")
])
cnn_model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
cnn_model.fit(X_train_pad, y_train_dl, epochs=epochs, batch_size=batch_size, validation_data=(X_test_pad, y_test_dl), verbose=1)

# ==========================================
# 4. Final Models Evaluation & Comparison
# ==========================================
print("\n--- 4. Final Model Comparison (ML + DL) ---")

ml_metrics = {}
for name, report in reports.items():
    ml_metrics[name] = {
        'Accuracy': report['accuracy'],
        'Precision': report['weighted avg']['precision'],
        'Recall': report['weighted avg']['recall'],
        'F1-score': report['weighted avg']['f1-score']
    }

def get_dl_metrics(model, X_test, y_test):
    loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
    y_pred_prob = model.predict(X_test)
    y_pred = (y_pred_prob > 0.5).astype("int")
    report = classification_report(y_test, y_pred, output_dict=True)
    return {
        'Accuracy': accuracy,
        'Precision': report['weighted avg']['precision'],
        'Recall': report['weighted avg']['recall'],
        'F1-score': report['weighted avg']['f1-score']
    }

lstm_metrics = get_dl_metrics(lstm_model, X_test_pad, y_test_dl)
embedding_metrics = get_dl_metrics(embedding_model, X_test_pad, y_test_dl)
cnn_metrics = get_dl_metrics(cnn_model, X_test_pad, y_test_dl)

all_metrics = {
    **ml_metrics,
    "LSTM": lstm_metrics,
    "Embedding": embedding_metrics,
    "CNN": cnn_metrics
}

metrics_df = pd.DataFrame(all_metrics).T
print(metrics_df)

# Plot All Models Performance
metrics_df.plot(kind='bar', figsize=(12, 6))
plt.title('All Models Performance Comparison')
plt.ylabel('Score')
plt.xticks(rotation=45, ha='right')
plt.legend(loc='lower right')
plt.tight_layout()
plt.show()