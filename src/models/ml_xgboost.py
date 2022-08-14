import numpy as np
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from xgboost import XGBClassifier
from sklearn.impute import SimpleImputer


class MLXGBoostClassifier:
    """
    Gradient Boosting Decision Tree Classifier
    """
    def __init__(self):
        """
        The constructor will initialize variables.
        """
        self.clf = XGBClassifier(objective='multi:softprob')
        self.is_model_trained = False
        self.target_le = LabelEncoder()
        self.categorical_cols = ['Day', 'Venue']
        self.one_hot_encoder = OneHotEncoder(handle_unknown='ignore')
        self.num_imputer = SimpleImputer(strategy='median')

    def _fit_preprocessor(self, train_df):
        """
        Tranforms the data frame from categorical to number(one-hot-encoder)
        Param train_df: Panda dataframe is expected
        Return: a pandas df
        """
        transformed = self.one_hot_encoder.fit_transform(train_df[self.categorical_cols])
        train_df[np.concatenate(self.one_hot_encoder.categories_).ravel()] = transformed.toarray()
        train_df['Poss'] = self.num_imputer.fit_transform(train_df[['Poss']])
        train_df = train_df.drop(columns=self.categorical_cols)
        return train_df

    def _transform_preprocessor(self, test_df):
        """
        Transform the testing Data into numerical values
        Param test_df: Part of the Data Frame for the training
        Return: Result of the transformation of the data frame
        """
        transformed = self.one_hot_encoder.transform(test_df[self.categorical_cols])
        test_df[np.concatenate(self.one_hot_encoder.categories_).ravel()] = transformed.toarray()
        test_df['Poss'] = self.num_imputer.transform(test_df[['Poss']])
        test_df = test_df.drop(columns=self.categorical_cols)
        return test_df

    def train(self, X_train, y_train):
        """
        Trains the model
        Param X_train: Part of the dataset that includes the features and every
        single row is input into the model
        Param y_train:Part of the data set that we compare
        An iterable containing the observed outputs (each 0 or 1)
        Return: A trained model
        """
        X_train = self._fit_preprocessor(X_train)
        y_train_enc = self.target_le.fit_transform(y_train)
        self.clf.fit(X_train, y_train_enc)
        self.is_model_trained = True
        return self.is_model_trained

    def predict(self, X_test):
        """
        Runs model on test data
        Param X_test: Partial data that was not used on the training
        Return: Test the accuracy of the training
        """
        X_test = self._transform_preprocessor(X_test)
        if not self.is_model_trained:
            print("Please train model before trying to predict")
            return
        predictions = self.clf.predict(X_test)
        return self.target_le.inverse_transform(predictions)
