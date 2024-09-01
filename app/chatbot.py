from flask import Flask, request, jsonify
from .embeddings import load_embeddings
from sentence_transformers import SentenceTransformer
from .data_loader import load_data

app = Flask(__name__)

# Load data and embeddings
url = "https://brainlox.com/courses/category/technical"
documents = load_data(url)
index = load_embeddings()
model = SentenceTransformer('all-MiniLM-L6-v2')

@app.route('/chat', methods=['POST'])
def chat():
    user_input = request.json.get('message')
    if not user_input:
        return jsonify({'error': 'No message provided'}), 400

    # Create embedding for the user input
    user_embedding = model.encode([user_input])

    # Search for the closest documents
    distances, indices = index.search(user_embedding, k=5)

    # Retrieve the closest documents
    response_docs = [documents[i] for i in indices[0]]

    return jsonify({'response': response_docs})

if __name__ == '__main__':
    app.run(debug=True)
