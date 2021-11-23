import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List


class Model:
    def __init__(self):
        self.model_collection = {
            2: np.array([[3112.17201423],
                         [-4.50613225]]),
            3: np.array(
                [[3114.7716564436582], [114.5365002969627], [-123.82672040039442]]),
            4: np.array(
                [[3112.31261905]
                 [128.51500931]
                    [-21.74180792]
                    [-120.29047432]])

        }

    def predict(self, df_features_test_list):
        df_feature = np.array(df_features_test_list)
        num_features = df_feature.shape[1]
        beta = self.model_collection[num_features+1]  # accounting for beta 0

        df_feature = self.normalize_z(df_feature)
        X = self.prepare_feature(df_feature)

        return self.predict_norm(X, beta)

    def predict_norm(self, X, beta):
        return np.matmul(X, beta)

    def normalize_z(self, dfin):

        dfout = (dfin - dfin.mean(axis=0))/dfin.std(axis=0)

        return dfout

    def get_features_targets(self, df, feature_names, target_names):
        df_feature = df.loc[:, feature_names]
        df_target = df.loc[:, target_names]

        return df_feature, df_target

    def prepare_feature(self, df_feature):
        n = df_feature.shape[0]
        ones = np.ones(n).reshape(n, 1)

        return np.concatenate((ones, df_feature), axis=1)

    def prepare_target(self, df_feature):
        return df_feature.to_numpy()
