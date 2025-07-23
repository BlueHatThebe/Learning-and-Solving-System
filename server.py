import json
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
import torch.optim as optim
from collections import Counter
import re
import math
from flask import Flask, request, jsonify, send_from_directory
from flask_sqlalchemy import SQLAlchemy
import sys

# --- Verify Dependencies ---
def check_dependencies():
    required = {'pandas': 'pandas', 'numpy': 'numpy', 'torch': 'torch', 'flask': 'flask', 'flask_sqlalchemy': 'flask-sqlalchemy'}
    for module, pkg in required.items():
        try:
            __import__(module)
        except ImportError:
            print(f"Error: {pkg} is not installed. Run 'pip install {pkg}'.")
            sys.exit(1)
check_dependencies()

app = Flask(__name__, static_folder='static')
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///chat_history.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)

# --- Database Model ---
class ChatMessage(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    question = db.Column(db.String(500), nullable=False)
    answer = db.Column(db.String(1000), nullable=False)
    timestamp = db.Column(db.DateTime, default=db.func.current_timestamp())

# --- Create Database ---
with app.app_context():
    db.create_all()

# --- Chatbot Backend ---
try:
    with open('faq_dataset.json', 'r') as f:
        data = json.load(f)
except FileNotFoundError:
    print("Error: faq_dataset.json not found. Ensure the file exists in the working directory.")
    sys.exit(1)

rows = []
for category, items in data.items():
    for item in items:
        question = item.get('question', '').strip().lower()
        answer = item.get('answer', '')
        if question and answer:
            rows.append({
                'category': category,
                'question': question,
                'answer': answer
            })

df = pd.DataFrame(rows)
df = df.drop_duplicates(subset=['question']).dropna()

if df.empty:
    print("Error: No valid data after cleaning. Check the JSON dataset.")
    sys.exit(1)

train_df = df.sample(frac=0.8, random_state=42)
val_df = df.drop(train_df.index)

def build_vocab_and_idf(texts):
    word_counts = Counter()
    doc_counts = Counter()
    n_docs = len(texts)
    for text in texts:
        words = set(re.findall(r'\w+', text))
        word_counts.update(words)
        doc_counts.update(words)
    vocab = {word: i for i, (word, _) in enumerate(word_counts.most_common())}
    idf = {word: math.log(n_docs / (1 + count)) for word, count in doc_counts.items()}
    return vocab, idf

def tfidf_vectorize(text, vocab, idf):
    words = re.findall(r'\w+', text)
    if not words:
        return torch.zeros(len(vocab))
    tf = Counter(words)
    vector = torch.zeros(len(vocab))
    for word in tf:
        if word in vocab:
            vector[vocab[word]] = (tf[word] / len(words)) * idf.get(word, 1.0)
    return vector

vocab, idf = build_vocab_and_idf(df['question'])
X_train = torch.stack([tfidf_vectorize(q, vocab, idf) for q in train_df['question']])
X_val = torch.stack([tfidf_vectorize(q, vocab, idf) for q in val_df['question']])
X_all = torch.stack([tfidf_vectorize(q, vocab, idf) for q in df['question']])

category_map = {'html': 0, 'css': 1, 'javascript': 2}
y_train = torch.tensor(train_df['category'].map(category_map).values, dtype=torch.long)
y_val = torch.tensor(val_df['category'].map(category_map).values, dtype=torch.long)

category_indices = {cat: df[df['category'] == cat].index.tolist() for cat in df['category'].unique()}

class CategoryClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(CategoryClassifier, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.dropout = nn.Dropout(0.3)
        self.fc2 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

class LinearClassifier(nn.Module):
    def __init__(self, input_size, num_classes):
        super(LinearClassifier, self).__init__()
        self.linear = nn.Linear(input_size, num_classes)

    def forward(self, x):
        return self.linear(x)

input_size = len(vocab)
hidden_size = 128
num_classes = 3

pytorch_model = CategoryClassifier(input_size, hidden_size, num_classes)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(pytorch_model.parameters(), lr=0.001)

train_dataset = TensorDataset(X_train, y_train)
val_dataset = TensorDataset(X_val, y_val)
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=16)

num_epochs = 10
for epoch in range(num_epochs):
    pytorch_model.train()
    for X_batch, y_batch in train_loader:
        optimizer.zero_grad()
        outputs = pytorch_model(X_batch)
        loss = criterion(outputs, y_batch)
        loss.backward()
        optimizer.step()
    
    pytorch_model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for X_batch, y_batch in val_loader:
            outputs = pytorch_model(X_batch)
            _, predicted = torch.max(outputs, 1)
            total += y_batch.size(0)
            correct += (predicted == y_batch).sum().item()
    accuracy = correct / total if total > 0 else 0
    print(f'PyTorch Epoch {epoch+1}, Validation Accuracy: {accuracy:.4f}')

linear_model = LinearClassifier(input_size, num_classes)
linear_optimizer = optim.Adam(linear_model.parameters(), lr=0.001)

for epoch in range(num_epochs):
    linear_model.train()
    for X_batch, y_batch in train_loader:
        linear_optimizer.zero_grad()
        outputs = linear_model(X_batch)
        loss = criterion(outputs, y_batch)
        loss.backward()
        linear_optimizer.step()
    
    linear_model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for X_batch, y_batch in val_loader:
            outputs = linear_model(X_batch)
            _, predicted = torch.max(outputs, 1)
            total += y_batch.size(0)
            correct += (predicted == y_batch).sum().item()
    linear_accuracy = correct / total if total > 0 else 0
    print(f'Linear Model Epoch {epoch+1}, Validation Accuracy: {linear_accuracy:.4f}')

best_model = pytorch_model if accuracy > linear_accuracy else linear_model
use_pytorch = accuracy > linear_accuracy

def knn_predict_category(question_vec, X_train, y_train, k=3):
    similarities = F.cosine_similarity(question_vec, X_train)
    top_k_indices = torch.topk(similarities, k).indices
    top_k_categories = y_train[top_k_indices]
    category_counts = Counter(top_k_categories.numpy())
    most_common = category_counts.most_common(1)
    return most_common[0][0] if most_common else 0

def predict_category(question, model, vocab, idf, X_train, y_train, use_pytorch=True):
    question_clean = question.strip().lower()
    question_vec = tfidf_vectorize(question_clean, vocab, idf).unsqueeze(0)
    if use_pytorch:
        model.eval()
        with torch.no_grad():
            output = model(question_vec)
            _, predicted = torch.max(output, 1)
        category_id = predicted.item()
    else:
        category_id = knn_predict_category(question_vec, X_train, y_train, k=3)
    return [k for k, v in category_map.items() if v == category_id][0]

def find_most_similar_question(question_vec, category, df, X_all, category_indices):
    indices = category_indices.get(category, [])
    if not indices:
        return "I don't know."
    category_vectors = X_all[indices]
    similarities = F.cosine_similarity(question_vec, category_vectors)
    max_index = similarities.argmax().item()
    if similarities[max_index] > 0.5:
        return df.iloc[indices[max_index]]['answer']
    return "I don't know."

def chatbot(question, model, vocab, idf, df, X_all, X_train, y_train, category_indices):
    try:
        category = predict_category(question, model, vocab, idf, X_train, y_train, use_pytorch=True)
        question_clean = question.strip().lower()
        question_vec = tfidf_vectorize(question_clean, vocab, idf).unsqueeze(0)
        answer = find_most_similar_question(question_vec, category, df, X_all, category_indices)
        return answer
    except Exception as e:
        return f"Error processing question: {str(e)}"

# --- Flask Routes ---
@app.route('/')
def serve_index():
    return send_from_directory('static', 'index.html')

@app.route('/static/<path:path>')
def serve_static(path):
    return send_from_directory('static', path)

@app.route('/ask', methods=['POST'])
def ask():
    data = request.get_json()
    question = data.get('question', '')
    if not question:
        return jsonify({'answer': 'Please provide a question.'}), 400
    answer = chatbot(question, best_model, vocab, idf, df, X_all, X_train, y_train, category_indices)
    with app.app_context():
        message = ChatMessage(question=question, answer=answer)
        db.session.add(message)
        db.session.commit()
    return jsonify({'answer': answer})

@app.route('/history', methods=['GET'])
def history():
    with app.app_context():
        messages = ChatMessage.query.order_by(ChatMessage.timestamp.asc()).all()
        return jsonify([
            {'question': msg.question, 'answer': msg.answer, 'timestamp': msg.timestamp.isoformat()}
            for msg in messages
        ])

if __name__ == '__main__':
    app.run(debug=True)