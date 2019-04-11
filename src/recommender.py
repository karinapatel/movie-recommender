import logging
import numpy as np
import pandas as pd
from pyspark.sql.types import IntegerType
from pyspark.ml.recommendation import ALS
import matplotlib.pyplot as plt
import pyspark as ps
from sklearn.model_selection import train_test_split
from noah_cleaning import get_frames

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
        #self.train_users = set()
        #self.train_movies = set()

    def fit(self, ratings):
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

        X = ratings[['user','movie','rating']]

        self.training_means = X['rating'].mean()

        self.movie_means = X.groupby('movie')['rating'].mean()
        self.user_means = X.groupby('user')['rating'].mean()


        
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

    def fill_na(self,n):
        if (n.user not in self.user_means) and (n.movie not in self.movie_means):
            final = self.training_means
        elif n.movie not in self.user_means:
            final = self.user_means.loc[n.user]
        else:
            final = self.movie_means.loc[n.movie]
        return final

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
        pd_y_pred['rating'] = pd_y_pred.apply(lambda x: self.fill_na(x) if (pd.isnull(x.rating)) else x['rating'], axis = 1)


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

    final_dict = get_frames('../data/training')
    ratings = final_dict['total_frame']

    final_test = get_frames('../data/requests.csv')
    requests = final_dict['total_frame']

    #we get ratings df from noahs script, then:
    #full df = ratings
    #train_df, test_df = spark_df.randomSplit([0.8, 0.2], seed=427471138)
