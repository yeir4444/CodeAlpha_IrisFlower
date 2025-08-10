import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

def load_and_split_data(path="../data/iris.csv", test_size=0.2, random_state=42):
    df = pd.read_csv(path)
    
    # Drop Id column since it's not a feature
    if 'Id' in df.columns:
        df = df.drop('Id', axis=1)
    
    # Remove 'Iris-' prefix from Species
    df['Species'] = df['Species'].str.replace('Iris-', '', regex=False)

    X = df.drop("Species", axis=1)
    y = df["Species"]

    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    target_names = le.classes_

    split = train_test_split(X, y_encoded, test_size=test_size, random_state=random_state)
    return split, target_names
