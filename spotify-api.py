import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from flask import Flask, request, jsonify, render_template_string
# ---- Data Loading & Preprocessing ----
df = pd.read_csv('spotify_dataset.csv', on_bad_lines='skip')
df.columns = df.columns.str.strip().str.replace('"', '')
df = df.drop_duplicates()
df = df.dropna(subset=['artistname', 'trackname', 'playlistname'])
df = df.reset_index(drop=True)
df['combined_features'] = df['trackname'] + ' ' + df['artistname'] + ' ' + df['playlistname']
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(df['combined_features'])
def recommend_tracks_tfidf(seed_track, n=10):
    if seed_track not in df['trackname'].values:
        return pd.DataFrame(columns=['trackname', 'artistname', 'playlistname'])
    idx = df[df['trackname'] == seed_track].index[0]
    cosine_scores = cosine_similarity(tfidf_matrix[idx], tfidf_matrix).flatten()
    similar_indices = np.argsort(cosine_scores)[::-1][1:n+1]
    recommended = df.iloc[similar_indices][['trackname', 'artistname', 'playlistname']]
    return recommended
app = Flask(__name__)
# HTML template string for the dashboard
HTML_DASHBOARD = """
<!DOCTYPE html>
<html>
<head>
    <title>Spotify Recommendation Dashboard</title>
    <style>
        body { font-family: Arial, sans-serif; margin:40px; }
        table { border-collapse: collapse; width: 60%; margin-top:20px;}
        th, td { border: 1px solid #ccc; padding: 8px; text-align: left;}
        th { background-color: #f4f4f4; }
        input[type=text], input[type=number] { padding:4px; width:180px; }
        .error { color: red; margin-top: 10px;}
    </style>
</head>
<body>
    <h1>Spotify Discover Weekly Recommendation Dashboard</h1>
    <form method="get" action="/">
        <label for="seed">Seed Track Name:</label>
        <input type="text" id="seed" name="seed" value="{{seed}}">
        <label for="n">Number of Recommendations:</label>
        <input type="number" id="n" name="n" min="1" max="20" value="{{n}}">
        <input type="submit" value="Get Recommendations">
    </form>
    {% if error %}
    <div class="error">{{error}}</div>
    {% endif %}
    {% if recommendations %}
    <table>
        <thead>
            <tr>
                <th>Track Name</th>
                <th>Artist Name</th>
                <th>Playlist Name</th>
            </tr>
        </thead>
        <tbody>
        {% for rec in recommendations %}
            <tr>
                <td>{{rec['trackname']}}</td>
                <td>{{rec['artistname']}}</td>
                <td>{{rec['playlistname']}}</td>
            </tr>
        {% endfor %}
        </tbody>
    </table>
    {% endif %}
</body>
</html>
"""
@app.route('/', methods=['GET'])
def dashboard():
    seed = request.args.get('seed', '')
    n = int(request.args.get('n', 10))
    recommendations = []
    error = None
    if seed:
        recs = recommend_tracks_tfidf(seed, n)
        if recs.empty:
            error = "No recommendations found. Try another seed track name."
        else:
            recommendations = recs.to_dict('records')
    return render_template_string(HTML_DASHBOARD, seed=seed, n=n, recommendations=recommendations, error=error)
@app.route('/recommend', methods=['GET'])
def recommend():
    seed = request.args.get('seed', '')
    n = int(request.args.get('n', 10))
    recs = recommend_tracks_tfidf(seed, n)
    return jsonify(recs.to_dict(orient='records'))
if __name__ == "__main__":
    app.run(debug=True, port=8000)