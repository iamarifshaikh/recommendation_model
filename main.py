from recommendation import calculate_similarity, recommend_clubs,recommend_mentors
from train_model import (
    load_students_data,
    load_clubs_data,
    load_mentors_data,
    encode_mentors,
    encode_interests,
    encode_clubs,
    align_features)

import pandas as pd


def get_user_input():
    name = input("Enter your name: ")
    interest_1 = input("Enter your first interest: ")
    interest_2 = input("Enter your second interest: ")
    return pd.DataFrame(
        {"name": [name], "interest_1": [interest_1], "interest_2": [interest_2]}
    )


def process_user_input(user_input, students_df, clubs_df):
    # Encode user interests
    user_encoded = encode_interests(user_input)

    # Create a DataFrame with all feature columns, initialized to 0
    all_features = students_df.columns.drop(
        ["student_id", "name", "interest_1", "interest_2"]
    )
    missing_features = [
        feature for feature in all_features if feature not in user_encoded.columns
    ]

    # Initialize missing columns with 0
    for feature in missing_features:
        user_encoded[feature] = 0

    # Ensure user_encoded has the same columns as students_df (except student_id)
    user_encoded = user_encoded.reindex(columns=students_df.columns.drop("student_id"))

    return user_encoded

def main():
    # Load the student, club data and mentors data
    students_df = load_students_data()
    clubs_df = load_clubs_data()
    mentors_df = load_mentors_data()

    # Encode student interests
    students_df = encode_interests(students_df)

    # Encode club categories
    clubs_df = encode_clubs(clubs_df)

    # Encode mentors
    mentors_df = encode_mentors(mentors_df)  

    # Align the features of both students and clubs to have the same set of columns
    students_df, clubs_df, mentors_df = align_features(students_df, clubs_df, mentors_df)

    # Get user input
    user_input = get_user_input()

    # Process user inputstudents_df, clubs_df,  mentors_df = align_features(students_df, clubs_df, mentors_df)
    user_encoded = process_user_input(user_input
    , students_df, clubs_df)

    # Prepare features for similarity calculation
    clubs_features = clubs_df.drop(
        columns=["club_id", "club_name", "category_1", "category_2", "category_3"]
    )
    mentors_features = mentors_df.drop(
        columns=["mentor_id", "mentor_name", "category_1", "category_2", "category_3"]
    )
    user_features = user_encoded.drop(columns=["name", "interest_1", "interest_2"])

    # Calculate similarity for clubs and mentors
    similarity_clubs = calculate_similarity(
        user_features, clubs_features, user_encoded, clubs_df
    )
    similarity_mentors = calculate_similarity(
        user_features, mentors_features, user_encoded, mentors_df
    )

    # Get recommendations
    club_recommendations = recommend_clubs(
        similarity_clubs, user_input["name"].iloc[0], top_n=10
    )
    mentor_recommendations = recommend_mentors(
        similarity_mentors, user_input["name"].iloc[0], top_n=5
    )

    # Print recommendations
    print("\nRecommended clubs for you:")
    for club, score in club_recommendations.items():
        print(f"{club}: {score:.2f}")

    print("\nRecommended mentors for you:")
    for mentor, score in mentor_recommendations.items():
        print(f"{mentor}: {score:.2f}")


if __name__ == "__main__":
    main()
