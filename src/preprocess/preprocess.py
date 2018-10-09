import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

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

    def loadData(self, sensitive_attribute, attribute, predictionValue, prediction_column):
        input_data = pd.read_csv(self.path)
        df = pd.DataFrame(input_data)
        """ 
            Perform the same preprocessing as the original analysis by Pro-Publica
            https://github.com/propublica/compas-analysis/blob/master/Compas%20Analysis.ipynb
        """
        df = df[(df.days_b_screening_arrest <= 30)
                & (df.days_b_screening_arrest >= -30)
                & (df.is_recid != -1)
                & (df.c_charge_degree != 'O')
                & (df.score_text != 'N/A')]
        sensitive_attribs = [sensitive_attribute]
        Z = self.split_columns(df, sensitive_attribs, sensitive_attribute, attribute)
        y = (df[prediction_column] == predictionValue).astype(int)
        X = (df.drop(columns=['race', 'score_text']).fillna('Unknown').pipe(pd.get_dummies, drop_first=True))

        return X, y, Z

    """
    Split the sensitive attribute column so that this is not part of training set
    """
    def split_columns(self, df, sensitive_attribs, sensitive_attribute, attribute):
        Z = (df.loc[:, sensitive_attribs].assign(
            new_column=lambda df: (df[sensitive_attribute] == attribute).astype(int)))
        Z.drop(columns=[sensitive_attribute], inplace=True)
        Z.rename(columns={'new_column': sensitive_attribute}, inplace=True)
        return Z


    """
    Split the data into train and test set.
    """
    def split_data(self, X, y, Z):
        X_train, X_test, y_train, y_test, Z_train, Z_test = train_test_split(X, y, Z, test_size=0.3, stratify=y)
        # standardize the data
        scaler = StandardScaler().fit(X_train)
        scale_df = lambda df, scaler: pd.DataFrame(scaler.transform(df), columns=df.columns, index=df.index)
        X_train = X_train.pipe(scale_df, scaler)
        X_test = X_test.pipe(scale_df, scaler)

        return X_train, X_test, y_train, y_test, Z_train, Z_test




