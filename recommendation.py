from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd


def calculate_similarity(user_features, target_features, user_data, target_data):
    # Compute cosine similarity
    similarity_matrix = cosine_similarity(user_features, target_features)

    # Create DataFrame for easier access
    if "club_name" in target_data.columns:
        name_column = "club_name"
    elif "mentor_name" in target_data.columns:
        name_column = "mentor_name"
    else:
        raise ValueError(
            "Target data must have either 'club_name' or 'mentor_name' column"
        )

    similarity_df = pd.DataFrame(
        similarity_matrix, index=user_data["name"], columns=target_data[name_column]
    )

    return similarity_df


def recommend_items(similarity_df, student_name, top_n=5):
    if student_name in similarity_df.index:
        sorted_similarities = similarity_df.loc[student_name].sort_values(
            ascending=False
        )
        return sorted_similarities.head(top_n)
    else:
        raise ValueError(f"Student {student_name} not found.")


# You can keep the original recommend_clubs and recommend_mentors functions,
# but they will now both use the recommend_items function:


def recommend_clubs(similarity_df, student_name, top_n=5):
    return recommend_items(similarity_df, student_name, top_n)


def recommend_mentors(similarity_df, student_name, top_n=10):
    return recommend_items(similarity_df, student_name, top_n)


# def recommend_clubs(similarity_df, student_name, top_n=5):
#     if student_name in similarity_df.index:
#         sorted_similarities = similarity_df.loc[student_name].sort_values(
#             ascending=False
#         )
#         return sorted_similarities.head(top_n)
#     else:
#         raise ValueError(f"Student {student_name} not found.")


# def recommend_mentors(similarity_df, student_name, top_n=10):
#     if student_name in similarity_df.index:
#         sorted_similarities = similarity_df.loc[student_name].sort_values(
#             ascending=False
#         )
#         return sorted_similarities.head(top_n)
#     else:
#         raise ValueError(f"Student {student_name} not found.")
