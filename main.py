from recommendation import calculate_similarity, recommend_clubs
from train_model import (
    load_students_data,
    load_clubs_data,
    encode_interests,
    encode_clubs,
    align_features,
)

# Load the student and club data
students_df = load_students_data()
clubs_df = load_clubs_data()

# Encode student interests
students_df = encode_interests(students_df)

# Encode club categories
clubs_df = encode_clubs(clubs_df)


# Align the features of both students and clubs to have the same set of columns
students_df, clubs_df = align_features(students_df, clubs_df)

# Print column names before dropping
print("Students DataFrame columns before dropping:", students_df.columns.tolist())

# Select features for similarity calculation
columns_to_drop = ["student_id", "name", "interest_1", "interest_2"]
columns_to_drop = [col for col in columns_to_drop if col in students_df.columns]
students_features = students_df.drop(columns=columns_to_drop)

columns_to_drop = ["club_id", "club_name", "category_1", "category_2", "category_3"]
columns_to_drop = [col for col in columns_to_drop if col in clubs_df.columns]
clubs_features = clubs_df.drop(columns=columns_to_drop)

# Debugging: Print shapes and columns to identify any mismatch
print("Students Features Shape:", students_features.shape)
print("Clubs Features Shape:", clubs_features.shape)
print("Students Features Columns:", students_features.columns.tolist())
print("Clubs Features Columns:", clubs_features.columns.tolist())
# Calculate the similarity matrix
similarity_df = calculate_similarity(
    students_features, clubs_features, students_df, clubs_df
)
