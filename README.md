# Movie Recommender Case Study

We built a recommendation system based off data from the
[MovieLens dataset](http://grouplens.org/datasets/movielens/). It includes movie
information, user information, and the users' ratings. The goal was to build a
recommendation system and to suggest movies to users!

The **movies data** and **user data** are in `data/movies.dat` and `data/users.dat`.

The **ratings data** can be found in `data/training.csv`. The users' ratings have been broken into a training and test set to obtain the testing set, we  split the 20% of **the most recent** ratings.


## Mission 

The **request** file in `data/requests.csv` contains a list of `user,movie` pairs.

Provide a rating for each of those `user,movie` pairs.

The **score** was measured based on how well we predicted the ratings for the users' ratings compared to the test set. 


## How to implement the recommender

The file `src/recommender.py` is the main template for creating our recommender.


## How to run your recommender

`src/run.py` has been prepared for convenience (doesn't need modification). By executing it, an instance is created of a `MovieRecommender` class (see file `src/recommender.py`), feeds it with the training data and outputs the results in a file.

It outputs a _properly formatted_ file of recommendations!

  Here's how to use this script:
  ```bash
  usage: run.py [-h] [--train TRAIN] [--requests REQUESTS] [--silent] outputfile

  positional arguments:
    outputfile           output file (where predictions are stored)

  optional arguments:
    -h, --help           show this help message and exit
    --train TRAIN        path to training ratings file (to fit)
    --requests REQUESTS  path to the input requests (to predict)
    --silent             deactivate debug output
  ```

When running this script, **you need to** specify your prediction output file as an argument (the one you will submit).

**Try now** to create a random prediction file by typing:

```bash
python src/run.py data/sample_submission.csv
```

## Evaluation: how the score is computed

For each user, the scoring metric will select the 5% of movies you thought would be most highly rated by that user. It then looks at the actual ratings (in the test data) that the user gave those movies.  The score is the average of those ratings.

Thus, for an algorithm to score well, it only needs to identify which movies a user is likely to rate most highly (so the absolute accuracy of ratings is less important than the rank ordering).


## Note on running the script with Spark

Since the `recommender.py` script relies on spark, we use the `run_on_spark.sh` to execute the code.

In a terminal, use: `bash run_on_spark.sh src/run.py` with arguments to run the recommender.

### `run_on_spark.sh` hello world

After cloning this repo to your machine, `cd` to this repo and start a `sparkbook` container.

```bash
$ cd /path/to/dsi-recommender-case-study
$ docker run --name sparkbook -p 8881:8888 -v "$PWD":/home/jovyan/work jupyter/pyspark-notebook start.sh jupyter lab --LabApp.token=''
```

Go to different terminal tab and `docker exec` into the container, then run `run.py` using `run_on_spark.sh`.

```bash
$ docker exec -it sparkbook bash
jovyan@3b34208f7e10:~$ cd work
jovyan@3b34208f7e10:~/work$ bash run_on_spark.sh src/run.py data/foobar.csv
jovyan@3b34208f7e10:~/work$ exit
$ head /path/to/dsi-recommender-case-study/data/foobar.csv
user,movie,rating
4958,1924,4
4958,3264,4
4958,2634,4
4958,1407,1
```

## Results
We were able to get a score of 4.32 by using an ALS model. The logic behind this was to fill the null values with a bit more logic than randomly guessing the mean. In our case, we tried to fill with the user's average rating, the movie's average rating, and if both of those failed, then just the average rating overall.