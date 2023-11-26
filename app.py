from flask import Flask, render_template, request
from src.tfidf_cos_dist import calculate_tfidf_similarity

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/calculate_similarity', methods=['GET','POST'])
def calculate_similarity():
    input_choice = request.form['inputChoice']

    if input_choice == 'text':
        content1 = request.form['content1']
        content2 = request.form['content2']
    elif input_choice == 'file':
        file1 = request.files['file1']
        file2 = request.files['file2']
        content1 = file1.read().decode("utf-8")
        content2 = file2.read().decode("utf-8")

    # Calculate TF-IDF vectors and similarity percentage using the function from tfidf_cos_dist.py
    tfidf_vectors, tfidf_similarity_percentage = calculate_tfidf_similarity(content1, content2)

    return render_template('result.html', tfidf_vectors=tfidf_vectors, similarity_percentage=tfidf_similarity_percentage)

if __name__ == '__main__':
    app.run(debug=True)
