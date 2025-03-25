from flask import Flask, request, jsonify
from src.recommender import MusicRecommender
from src.feedback import FeedbackManager
from flask_cors import CORS

app = Flask(__name__)
CORS(app)  # Enable CORS for frontend integration

# Initialize core components
recommender = MusicRecommender()
feedback_manager = FeedbackManager()

@app.route('/test', methods=['GET'])
def test():
    """Simple test endpoint to verify the API is running."""
    return jsonify({"status": "API is working!"})

@app.route('/recommend', methods=['GET'])
def recommend():
    """Get recommendations based on a song."""
    song_name = request.args.get('song')
    top_n = int(request.args.get('top_n', 5))
    
    if not song_name:
        return jsonify({"error": "Missing 'song' parameter"}), 400

    results = recommender.recommend(song_name, top_n=top_n)
    return jsonify(results.to_dict('records'))

@app.route('/recommend/user/<user_id>', methods=['GET'])
def recommend_for_user(user_id):
    """Get personalized recommendations for a user."""
    top_n = int(request.args.get('top_n', 5))
    results = feedback_manager.get_personalized_recommendations(user_id, top_n)
    return jsonify(results.to_dict('records'))

@app.route('/feedback', methods=['POST'])
def log_feedback():
    """Log user feedback."""
    data = request.json
    required = ['user_id', 'song', 'feedback_type']
    
    if not all(key in data for key in required):
        return jsonify({"error": "Missing required parameters"}), 400

    feedback_manager.log_feedback(
        data['user_id'],
        data['song'],
        data['feedback_type']
    )
    return jsonify({"status": "success"})

if __name__ == '__main__':
    # Run the API on host 0.0.0.0 and port 5000 in debug mode.
    app.run(debug=True, host='0.0.0.0', port=5000)
