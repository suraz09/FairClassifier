import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
create_gif = False

"""
DataProcessor class contains the methods to load, preprocess and split the data into train and test set.
"""
class DataProcessor:

    def __init__(self, file):
        self.path = file

    """
    Pulling the data in raw format found here: 
    DataProcessing done as:
    1. Identifying the sensitive attribute of the data
    2. Dropping the sensitive attribute from the dataFrame
    """

    def loadProcessData(self):
        raw_input_data = pd.read_csv(self.path)

        df = pd.DataFrame(raw_input_data)

        sensitive_attribs = ['race']
        Z = (df.loc[:, sensitive_attribs].assign(race=lambda df: (df['race'] == 'African-American').astype(int)))

        y = (df['score_text'] == 'High').astype(int)

        X = (df.drop(columns=['race', 'score_text']).fillna('Unknown')
             .pipe(pd.get_dummies, drop_first=True))

        return X, y, Z

    """
    Split the data into train and test set.
    """
    def split_data(self,X, y, Z):
        X_train, X_test, y_train, y_test, Z_train, Z_test = train_test_split(X, y, Z, test_size=0.3, stratify=y,
                                                                             random_state=7)
        # standerize the data
        scaler = StandardScaler().fit(X_train)
        scale_df = lambda df, scaler: pd.DataFrame(scaler.transform(df), columns=df.columns, index=df.index)
        X_train = X_train.pipe(scale_df, scaler)
        X_test = X_test.pipe(scale_df, scaler)

        return X_train, X_test, y_train, y_test, Z_train, Z_test




