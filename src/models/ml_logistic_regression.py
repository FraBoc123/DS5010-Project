import numpy as np

from sklearn.linear_model import LogisticRegression

from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer


class MLModelLogisticRegression:

    def __init__(self):
        self.clf = LogisticRegression(random_state=10, multi_class='ovr')
        self.is_model_trained = False
        self.categorical_cols = ['Day', 'Venue']
        self.one_hot_encoder = OneHotEncoder(handle_unknown='ignore')
        self.std_scaler = StandardScaler()
        self.num_imputer = SimpleImputer(strategy='median')

    def _fit_preprocessor(self, train_df):
        transformed = self.one_hot_encoder.fit_transform(train_df[self.categorical_cols])
        train_df[np.concatenate(self.one_hot_encoder.categories_).ravel()] = transformed.toarray()
        train_df = train_df.drop(columns=self.categorical_cols)
        train_df['Poss'] = self.num_imputer.fit_transform(train_df[['Poss']])
        train_df['Poss'] = self.std_scaler.fit_transform(train_df[['Poss']])
        return train_df

    def _transform_preprocessor(self, test_df):
        transformed = self.one_hot_encoder.transform(test_df[self.categorical_cols])
        test_df[np.concatenate(self.one_hot_encoder.categories_).ravel()] = transformed.toarray()
        test_df = test_df.drop(columns=self.categorical_cols)
        test_df['Poss'] = self.num_imputer.transform(test_df[['Poss']])
        test_df['Poss'] = self.std_scaler.transform(test_df[['Poss']])
        return test_df

    def train(self, X_train, y_train):
        X_train = self._fit_preprocessor(X_train)
        self.clf.fit(X_train, y_train)
        self.is_model_trained = True
        return self.is_model_trained

    def predict(self, X_test):
        X_test = self._transform_preprocessor(X_test)
        if not self.is_model_trained:
            print("Please train model before trying to predict")
            return
        return self.clf.predict(X_test)
