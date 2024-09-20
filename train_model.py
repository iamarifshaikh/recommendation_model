import pandas as pd

# Load and preprocess student data from MongoDB
def load_students_data(db):
    students_df = pd.DataFrame(list(db.students.find()))
    if students_df.empty:
        print("No student data found.")
    return students_df


# Load and preprocess clubs data from MongoDB
def load_clubs_data(db):
    clubs_df = pd.DataFrame(list(db.clubs.find()))
    if clubs_df.empty:
        print("No club data found.")
    return clubs_df


# Load and preprocess mentors data from MongoDB
def load_mentors_data(db):
    mentors_df = pd.DataFrame(list(db.mentors.find()))
    if mentors_df.empty:
        print("No mentor data found.")
    return mentors_df

def encode_mentors(mentors_df):
    mentor_columns = ["category_1", "category_2", "category_3"]
    encoded_mentors = (
        pd.get_dummies(mentors_df[mentor_columns].stack()).groupby(level=0).sum()
    )
    encoded_mentors.columns = encoded_mentors.columns.str.strip()
    mentors_df = pd.concat([mentors_df, encoded_mentors], axis=1)
    return mentors_df

# Function to encode interests into dummy variables
def encode_interests(students_df):
    print("Before encoding - Students DataFrame columns:", students_df.columns.tolist())
    interests_columns = ["interest_1", "interest_2"]
    encoded_interests = (pd.get_dummies(students_df[interests_columns].stack()).groupby(level=0).sum())
    
    # Strip spaces from encoded column names
    encoded_interests.columns = encoded_interests.columns.str.strip()  
    
    students_df = pd.concat([students_df, encoded_interests], axis=1)
    print("After encoding - Students DataFrame columns:", students_df.columns.tolist())
    return students_df


# Function to encode club categories into dummy variables
def encode_clubs(clubs_df):
    print("Before encoding - Clubs DataFrame columns:", clubs_df.columns.tolist())
    club_columns = ["category_1", "category_2", "category_3"]
    encoded_clubs = (pd.get_dummies(clubs_df[club_columns].stack()).groupby(level=0).sum())
    # Strip spaces from encoded column names
    encoded_clubs.columns = (encoded_clubs.columns.str.strip())  
    clubs_df = pd.concat([clubs_df, encoded_clubs], axis=1)
    print("After encoding - Clubs DataFrame columns:", clubs_df.columns.tolist())
    return clubs_df


def align_features(students_df, clubs_df,mentors_df):
    # Get all unique columns from both dataframes (only dummy-encoded ones)
    student_feature_columns = students_df.columns.difference(
        ["student_id", "name", "interest_1", "interest_2"]
    )
    club_feature_columns = clubs_df.columns.difference(
        ["club_id", "club_name", "category_1", "category_2", "category_3"]
    )
    mentor_feature_columns = mentors_df.columns.difference(["mentor_id", "mentor_name", "category_1", "category_2", "category_3"])

    all_columns = student_feature_columns.union(club_feature_columns).union(mentor_feature_columns)

    # Reindex both dataframes to ensure they have the same columns
    students_df = students_df.reindex(
        columns=["student_id", "name", "interest_1", "interest_2"] + list(all_columns),
        fill_value=0,
    )
    clubs_df = clubs_df.reindex(
        columns=["club_id", "club_name", "category_1", "category_2", "category_3"]
        + list(all_columns),
        fill_value=0,
    )

    mentors_df = mentors_df.reindex(
        columns=["mentor_id", "mentor_name", "category_1", "category_2", "category_3"]
        + list(all_columns),
        fill_value=0,
    )
    return students_df, clubs_df, mentors_df
