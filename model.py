import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List
import logging

logger = logging.getLogger('werkzeug')  # grabs underlying WSGI logger
handler = logging.FileHandler('record.log')  # creates handler for the log file
logger.addHandler(handler)  # adds handler to the werkzeug WSGI logger


class Model:
    def __init__(self):
        self.model_collection = {
            2: np.array([[3112.17201423],
                         [-4.50613225]]),
            3: np.array(
                [[3114.7716564436582], [114.5365002969627], [-123.82672040039442]]),
            4: np.array(
                [[3112.31261905],
                 [128.51500931],
                    [-21.74180792],
                    [-120.29047432]]),
            6: np.array(
                [[3.11481098e+03],
                 [2.26040217e+00],
                    [-1.95620273e+00],
                    [7.50681636e+01],
                    [-3.56648783e+01],
                    [-3.54075348e+01]]),
            7: np.array([[3.11502364e+03],
                         [9.88427416e-01],
                         [6.76987097e-01],
                         [7.26205565e+01],
                         [-4.03252348e+01],
                         [-5.74715052e+00],
                         [-3.45302625e+01],
                         [-2.39080728e+00]])

        }

    def predict(self, df_features_test_list):
        df = pd.read_csv('./data/df_feature1.csv')
        df.loc[df.shape[0], :] = df_features_test_list
        df_z = self.normalize_z(df)
        line = df_z.tail(1)

        beta = self.model_collection[6]  # accounting for beta 0

        X = self.prepare_feature(line)
        logger.info("X : ", X)

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
