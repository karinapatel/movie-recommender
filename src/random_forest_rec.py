from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, cross_val_score

import numpy as np
import pandas as pd
import math

import rf_cleaning as clean


def predict(movies):


	print("Movies:", movies)
	single_movies = movie_data.drop(['title'], axis=1).drop_duplicates()

	use = [(m in movies) for m in single_movies.movie.values]
	single_movies = single_movies.loc[use,:]
	single_movies.drop(['movie'], axis=1, inplace=True)

	to_fill = movie_rating.loc[use, :].drop(['rating', 
											'title', 
											'timestamp', 
											'user', 
											'movie'], 
											axis=1).drop_duplicates()

	y_train = movie_rating.rating.values

	movies = movie_rating.movie
	movie_rating.drop(['rating', 'title', 'timestamp', 'user', 'movie'], axis=1, inplace=True)

	X_train = movie_rating.values

	# ######### RANDOM FORREST #########
	print("Fitting")
	rf = RandomForestRegressor(n_estimators=15, n_jobs=-1, random_state=1)
	rf.fit(X_train, y_train)
	print("Fit")

	return rf.predict(single_movies)


def avg(df):
	df['avg'] = df.apply(lambda x: x['rating'] 
						if math.isnan(x['avg_rating']) 
						else x['avg'], axis=1)
	return df

def rat(df):
	df['n_ratings'] = df.apply(lambda x: 1
								if math.isnan(x['n_ratings'])
								else x['n_ratings'], axis=1)
	return df

## Getting all the data frames bby calling the cleaning function
frames = clean.get_frames("../data/training.csv")

## Grabbing individual data frames from frames dict
movie_rating = frames['movie_rating']
movie_data = frames['movie_data']
ratings = frames['total_frame']

## Getting test data and unique movie list
requests = pd.read_csv('../data/requests.csv')
movies = list(set(requests.movie.values))

## Getting the movie predicitons, making it a datframe
all_movie_preds = predict(movies)
preds_df = pd.DataFrame({'movie':movies, 'rating':all_movie_preds})

## Used for avg rating and rating count
X = ratings[['user','movie','rating']]

## Getting mean user rating data frame
user_means = X.groupby('user')['rating'].mean().reset_index()
user_means['avg_rating'] = user_means['rating']
user_means.drop('rating', axis=1, inplace=True)

## Getting the number of user ratings as dataframe
user_ns = X.groupby('user')['rating'].count().reset_index()
user_ns['n_ratings'] = user_ns['rating']
user_ns.drop('rating', axis=1, inplace=True)

## Merging the different frames
final = pd.merge(requests, preds_df, how='left')
final = pd.merge(final, user_means, how='left')
final = pd.merge(final, user_ns, how='left')


final = rat(final)
logged = np.log(final['n_ratings'].values)
w = logged/logged.sum()
weights = w

final['avg'] = ((final['rating']+
				.5*final['avg_rating'])
				/(1.5))
df = avg(final)


df.drop(['rating', 'avg_rating', 'n_ratings'], axis=1, inplace=True)
df['rating'] = df['avg']
df.drop(['avg'],axis=1, inplace=True)

df.to_csv('../data/sub.csv', index=False)





