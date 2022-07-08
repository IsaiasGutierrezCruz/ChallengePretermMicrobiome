# from sklearn.base import BaseEstimator, TransformerMixin
# from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler


class PipelineChallenge:
    def __init__(self, data, output_name, test_size=0.2):
        self.data = data
        self.output_name = output_name
        self.test_size = test_size

    def feature_dropper(self, list_feat):
        return self.data.copy().drop(list_feat, axis=1)

    def split_data(self, data) -> tuple:
        strati_train_set = None
        strati_test_set = None
        split = StratifiedShuffleSplit(n_splits=10, test_size=self.test_size)
        for train_indices, test_indices in split.split(data, data[self.output_name]):
            strati_train_set = data.loc[train_indices]
            strati_test_set = data.loc[test_indices]
        return strati_train_set, strati_test_set

    def standard_data(self, strati_train_set, strati_test_set) -> tuple:
        x = strati_train_set.drop([self.output_name], axis=1)
        y = strati_train_set[self.output_name]

        scaler = StandardScaler()
        x_train = scaler.fit_transform(x)
        y_train = y.to_numpy()

        x_test = strati_test_set.drop([self.output_name], axis=1)
        y_test = strati_test_set[self.output_name]

        x_test = scaler.transform(x_test)
        y_test = y_test.to_numpy()

        return x_train, y_train, x_test, y_test
