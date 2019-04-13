# Movie Recommender Case Study

The goal of this case study was to build a recommendation system based off data from the
[MovieLens dataset](http://grouplens.org/datasets/movielens/). It includes movie
information, user information, and the users' ratings. This project uses recommendation systems to suggest movies to users!

The **movies data** and **user data** are in `data/movies.dat` and `data/users.dat`.

The users' ratings have been broken into a training and test set to obtain the testing set, we  split the 20% of **the most recent** ratings.


## Mission 


Provide a rating for each of those `user,movie` pairs.

The **score** was measured based on how well we predicted the ratings for the users' ratings compared to the test set. 


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

## Modeling and Results
The initial plan was to build an ALS model to predict the ratings: fill Nan any values with 3.2

- We first merged all the data on the ID columns in the case that we wanted to use some of these features to help fill in Nan’s (cold call problem)
- If a movie had never been seen: fill with that user’s average rating
- If a user was new: fill with the movie’s average rating across the rest of the users

Upon testing this, we found that the model unfortunately did not perform much stronger than guessing the average. 

The next steps involved strategically filling the Nan values and also incorporatiing data on movies.
By using a Random Forest model on movie genres and year released averaged with average past rating for each user, we filled any new users with the random forest model prediction to achieve a final score of 4.216.

Additionally, we were able to improve our ALS model to get a score of 4.3. The logic behind this was to fill the null values with a bit more logic than randomly guessing the mean. In our case, we tried to fill with the user's average rating, the movie's average rating, and if both of those failed, then just the average rating overall.

## Future Steps
- Incorporate user features into RF model
- Consider different weighting options when combining RF predictions with past user ratings
- Some users’ past ratings was based on more information than other
- Figure out a way to combine matrix factorization approach with RF model (use movie topic latent factors into RF model)


