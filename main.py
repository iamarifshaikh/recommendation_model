from flask import Flask, jsonify, request
from flask_restful import Resource, Api
from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi
from recommendation import calculate_similarity, recommend_clubs, recommend_mentors
from train_model import encode_mentors, encode_interests, encode_clubs, align_features
import pandas as pd

app = Flask(__name__)
api = Api(app)

uri = "mongodb+srv://theshaikhasif03:fPQSb56RBLe2lG84@cluster1.o65jh.mongodb.net/?retryWrites=true&w=majority&appName=Cluster1"
# Create a new client and connect to the server
client = MongoClient(uri, server_api=ServerApi("1"))

# Send a ping to confirm a successful connection
try:
    client.admin.command("ping")
    print("Pinged your deployment. You successfully connected to MongoDB!")
except Exception as e:
    print(e)

# MongoDB setup
db = client["test"]

class Recommend(Resource):
    def post(self):
        data = request.json
        name = data.get("name")
        interest_1 = data.get("interest_1")
        interest_2 = data.get("interest_2")

        # Querying the students, clubs, and mentors collection from MongoDB
        students_df = pd.DataFrame(list(db.students.find()))
        clubs_df = pd.DataFrame(list(db.clubs.find()))
        mentors_df = pd.DataFrame(list(db.mentors.find()))

        # Encoding the features
        students_df = encode_interests(students_df)
        clubs_df = encode_clubs(clubs_df)
        mentors_df = encode_mentors(mentors_df)

        # Align features for similarity calculation
        students_df, clubs_df, mentors_df = align_features(
            students_df, clubs_df, mentors_df
        )

        # Creating user input DataFrame
        user_input = pd.DataFrame(
            {"name": [name], "interest_1": [interest_1], "interest_2": [interest_2]}
        )

        # Process user input for encoding
        user_encoded = encode_interests(user_input)

        # Prepare features for similarity calculation
        user_features = user_encoded.drop(columns=["name", "interest_1", "interest_2"])
        clubs_features = clubs_df.drop(
            columns=["club_id", "club_name", "category_1", "category_2", "category_3"]
        )
        mentors_features = mentors_df.drop(
            columns=[
                "mentor_id",
                "mentor_name",
                "category_1",
                "category_2",
                "category_3",
            ]
        )

        # Calculate similarity
        similarity_clubs = calculate_similarity(
            user_features, clubs_features, user_encoded, clubs_df
        )
        similarity_mentors = calculate_similarity(
            user_features, mentors_features, user_encoded, mentors_df
        )

        # Get recommendations
        club_recommendations = recommend_clubs(similarity_clubs, name, top_n=10)
        mentor_recommendations = recommend_mentors(similarity_mentors, name, top_n=5)

        # Return recommendations as JSON
        return jsonify(
            {
                "clubs": club_recommendations.to_dict(),
                "mentors": mentor_recommendations.to_dict(),
            }
        )


# Register the resource for the Flask app
api.add_resource(Recommend, "/recommend")

if __name__ == "__main__":
    app.run(debug=True)
