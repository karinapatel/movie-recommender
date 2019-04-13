import logging
import numpy as np
import pandas as pd
from pyspark.sql.types import *
from pyspark.sql import functions as F
from pyspark.ml.recommendation import ALS
import matplotlib.pyplot as plt
import pyspark as ps
from sklearn.model_selection import train_test_split
from ALS_cleaning import get_frames1
from collections import defaultdict

class MovieRecommender():
    """Template class for a Movie Recommender system."""

    def __init__(self):
        """Constructs a MovieRecommender"""
        self.logger = logging.getLogger('reco-cs')
        # ...
        self.model=None

        self.spark = (ps.sql.SparkSession.builder 
        .master("local[4]") 
        .appName("sparkSQL exercise") 
        .getOrCreate()
        )
        self.sc = self.spark.sparkContext


    def fit(self, ratings, mov = final_dict["movie_data"]):
        """
        Trains the recommender on a given set of ratings.

        Parameters
        ----------
        ratings : pandas dataframe, shape = (n_ratings, 4)
                  with columns 'user', 'movie', 'rating', 'timestamp'

        Returns
        -------
        self : object
            Returns self.
        """
        self.logger.debug("starting fit")

        '''
        spark = (ps.sql.SparkSession.builder 
        .master("local[4]") 
        .appName("sparkSQL exercise") 
        .getOrCreate()
        )
        sc = spark.sparkContext
        '''
        self.ratings=ratings
        self.mov=mov
        X = ratings[['user','movie','rating']]

        self.training_means = X['rating'].mean()

        self.movie_means = X.groupby('movie')['rating'].mean()
        self.user_means = X.groupby('user')['rating'].mean()
        
        self.r1 = ratings.groupby(["genre","user"]).mean()['rating']
        self.r2 = ratings.groupby('movie').mean()['rating']
        self.r3 = ratings.groupby("user").mean()['rating']
        self.r4 = ratings.rating.mean()
                
        spark_df = self.spark.createDataFrame(X)

        #alter regularization and rank(k)
        als_model = ALS(
        itemCol='movie',
        userCol='user',
        ratingCol='rating',
        nonnegative=True,    
        regParam=0.05,
        rank=15)

        self.model = als_model.fit(spark_df)
        # ...

        self.logger.debug("finishing fit")
        return(self)

    def nulls(self,n):
        if n.user not in self.ratings.user.unique() or n.movie not in self.mov.movie:
            return r4
        elif n.user in r1[self.mov.genre[n.movie]]:
            return r1[self.mov.genre[n.movie]][n.user]
        elif n.movie in r2:
            return r2[n.movie]
        elif n.movie in r3:
            return r3[n.movie]
        else:
            return r4

    def transform(self, requests):
        """
        Predicts the ratings for a given set of requests.

        Parameters
        ----------
        requests : pandas dataframe, shape = (n_ratings, 2)
                  with columns 'user', 'movie'

        Returns
        -------
        dataframe : a pandas dataframe with columns 'user', 'movie', 'rating'
                    column 'rating' containing the predicted rating
        """
        self.logger.debug("starting predict")
        self.logger.debug("request count: {}".format(requests.shape[0]))
        X = requests[['user','movie']]
        spark_df = self.spark.createDataFrame(X)

        y_pred = self.model.transform(spark_df)
        pd_y_pred = y_pred.toPandas()
        pd_y_pred['rating']=pd_y_pred['prediction']
        pd_y_pred=pd_y_pred.drop('prediction',axis=1)
        #modify this to fill according to the RF
        test = pd_y_pred.apply(lambda x: self.nulls(x) if pd.isnull(x.prediction) else x['prediction'], axis = 1)
        pd_y_pred['rating'] = test 
        #pd_y_pred['rating'] = pd_y_pred.apply(lambda x: self.fill_na(x) if (pd.isnull(x.rating)) else x['rating'], axis = 1)


        #pd_y_pred = pd_y_pred.fillna(3.2)

        self.logger.debug("finishing predict")
        return(pd_y_pred)


if __name__ == "__main__":
    logger = logging.getLogger('reco-cs')
    logger.critical('you should use run.py instead')
    '''
    spark = (ps.sql.SparkSession.builder 
        .master("local[4]") 
        .appName("sparkSQL exercise") 
        .getOrCreate()
        )
    sc = spark.sparkContext
    '''

    final_dict = get_frames1('../data/training',test_file=False)
    ratings = final_dict['total_frame']

    final_test = get_frames1('../data/requests.csv',test_file=False)
    requests = final_dict['total_frame']

    mov = final_dict["movie_data"]
