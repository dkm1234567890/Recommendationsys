import pandas as pd
import numpy as np
from surprise import KNNWithMeans, SVD, Dataset, Reader
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler

# Load the datasets
movies = pd.read_csv('movies.csv')
ratings = pd.read_csv('ratings.csv')

# Create a reader for the ratings data
reader = Reader(rating_scale=(1, 5))

# Build the surprise dataset
data = Dataset.load_from_df(ratings, reader)

# Build the user feature matrix
user_features = pd.get_dummies(movies['userId'], drop_first=True)
user_features['age'] = movies['age']
user_features['gender'] = movies['gender']
user_features['occupation'] = movies['occupation']

# Build the movie feature matrix
movie_features = TfidfVectorizer(max_features=5000).fit_transform(movies['genres'])
movie_features = pd.DataFrame(movie_features.toarray(), columns=['genre_' + str(i) for i in range(5000)])
movie_features['director'] = movies['director']
movie_features['actor'] = movies['actor']
movie_features['release_year'] = movies['release_year']
movie_features['average_rating'] = movies['average_rating']

# Build the interaction feature matrix
interaction_features = pd.pivot_table(ratings, values='rating', index='userId', columns='movieId')

# Define the hybrid model
class HybridModel:
    def __init__(self, num_factors=50, num_neighbors=50):
        self.mf_model = SVD(n_factors=num_factors)
        self.cbf_model = KNNWithMeans(k=num_neighbors, sim_options={'name': 'pearson_baseline', 'user_based': False})
        self.cf_model = KNNWithMeans(k=num_neighbors, sim_options={'name': 'pearson_baseline', 'user_based': True})

    def fit(self, trainset):
        self.mf_model.fit(trainset)
        self.cbf_model.fit(trainset)
        self.cf_model.fit(trainset)

    def predict(self, testset):
        mf_pred = self.mf_model.test(testset)
        cbf_pred = self.cbf_model.test(testset)
        cf_pred = self.cf_model.test(testset)
        return [(mf_pred[i] + cbf_pred[i] + cf_pred[i]) / 3 for i in range(len(testset))]

# Train the hybrid model
hybrid_model = HybridModel(num_factors=50, num_neighbors=50)
hybrid_model.fit(data.build_full_trainset())

# Evaluate the hybrid model
testset = data.build_anti_testset()
predictions = hybrid_model.predict(testset)
accuracy.rmse(predictions, verbose=True)

# Make recommendations for a user
def get_recommendations(user_id, num_recommendations=5):
    user_inner_uid = data.trainset.to_inner_uid(user_id)
    user_ratings = data.trainset.ur[user_inner_uid]
    recommended_movies = []
    for movie_id in range(data.trainset.n_items):
        if not user_ratings[movie_id]:
            predicted_rating = hybrid_model.predict(user_id, movie_id)
            recommended_movies.append((movie_id, predicted_rating))
    recommended_movies.sort(key=lambda x: x[1], reverse=True)
    return [data.trainset.to_raw_iid(movie_id) for movie_id, _ in recommended_movies[:num_recommendations]]

# Example usage:
user_id = 1
recommended_movies = get_recommendations(user_id)
print("Recommended movies for user", user_id, ":", recommended_movies)