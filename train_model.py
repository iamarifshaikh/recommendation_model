import pandas as pd

# Load and preprocess student data
def load_students_data(): 
    students_df = pd.read_csv("./students.csv")
    # Remove extra spaces from column names
    students_df.columns = (students_df.columns.str.strip())  
    # Strip spaces from column names
    students_df = students_df.rename(columns=lambda x: x.strip()) 
    # Data Cleaning: Check for missing values
    
    if students_df.isnull().values.any():
        print("Warning: Missing values found in student data. Cleaning...")
        students_df = students_df.dropna()  # Drop rows with missing values

    # Data Cleaning: Check for duplicate entries
    if students_df.duplicated().any():
        print("Warning: Duplicate entries found in student data. Removing duplicates...")
        students_df = students_df.drop_duplicates()

    print("Students data cleaned and loaded successfully.")
    return students_df

# Load and preprocess clubs data
def load_clubs_data():
    clubs_df = pd.read_csv("./clubs.csv")
    
    # Remove extra spaces from column names
    clubs_df.columns = (clubs_df.columns.str.strip())  

    # Strip spaces from column names
    clubs_df = clubs_df.rename(columns=lambda x: x.strip())  

    # Data Cleaning: Check for missing values
    if clubs_df.isnull().values.any():
        print("Warning: Missing values found in club data. Cleaning...")
        clubs_df = clubs_df.dropna()  # Drop rows with missing values

    # Data Cleaning: Check for duplicate entries
    if clubs_df.duplicated().any():
        print("Warning: Duplicate entries found in club data. Removing duplicates...")
        clubs_df = clubs_df.drop_duplicates()

    print("Clubs data cleaned and loaded successfully.")
    return clubs_df


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


def align_features(students_df, clubs_df):
    # Get all unique columns from both dataframes (only dummy-encoded ones)
    student_feature_columns = students_df.columns.difference(
        ["student_id", "name", "interest_1", "interest_2"]
    )
    club_feature_columns = clubs_df.columns.difference(
        ["club_id", "club_name", "category_1", "category_2", "category_3"]
    )
    all_columns = student_feature_columns.union(club_feature_columns)

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
    return students_df, clubs_df
