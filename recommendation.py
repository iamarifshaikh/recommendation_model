from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd

def calculate_similarity(students_features, clubs_features, students_data, clubs_data):
    # Compute cosine similarity
    similarity_matrix = cosine_similarity(students_features, clubs_features)

    # Create DataFrame for easier access
    similarity_df = pd.DataFrame(
        similarity_matrix, index=students_data["name"], columns=clubs_data["club_name"]
    )

    return similarity_df


def recommend_clubs(similarity_df, student_name, top_n=5):
    if student_name in similarity_df.index:
        sorted_similarities = similarity_df.loc[student_name].sort_values(
            ascending=False
        )
        return sorted_similarities.head(top_n)
    else:
        raise ValueError(f"Student {student_name} not found.")