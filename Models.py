import numpy as np
from collections import Counter
import pandas as pd


def load_csv(gender):
    if gender.lower() == 'male':
        male_df_knn = pd.read_csv('male_knn_X_df.csv')
        X_male = np.array(male_df_knn[['stature', 'weightkg']])
        y_male = np.array(male_df_knn['t_shirt_size'])
        return X_male, y_male

    elif gender.lower() == 'female':
        female_df_knn = pd.read_csv('female_knn_X_df.csv')
        X_female = np.array(female_df_knn[['stature', 'weightkg']])
        y_female = np.array(female_df_knn['t_shirt_size'])
        return X_female, y_female


def euclidean_dist(dataset, user_input):
    x1, y1 = dataset
    x2, y2 = user_input
    delta_X = x1-x2
    delta_y = y1-y2
    return np.sqrt((delta_X**2) + (delta_y**2))


class KNN:
    def __init__(self, k=3):
        self.k = k

    def fit(self, X, y):
        self.X_train = X
        self.y_train = y

    def predict(self, X):
        predicted_labels = [self._predict(x) for x in X]
        return np.array(predicted_labels)

    def _predict(self, x):
        distances = [euclidean_dist(x, x_train) for x_train in self.X_train]
        k_indices = np.argsort(distances)[:self.k]
        k_nearest_labels = [self.y_train[i] for i in k_indices]
        most_common = Counter(k_nearest_labels).most_common(1)
        return most_common[0][0]



