def get_frames1(filename,test_file = False):

    ## Reading in the data
    ratings_data = pd.read_csv(filename)

    movie_data = pd.read_csv("../data/movies.dat",
                            delimiter = "::",
                            names=["movie","title","genre"])

    user_data = pd.read_csv("../data/users.dat",
                            delimiter = "::",
                            names=["user","gender","age","occupation","zipcode"])


    ## Adding Movie Genre Dummy Cols
    dummy_cols = movie_data.genre.str.get_dummies()
    movie_data = pd.concat((movie_data,dummy_cols),axis = 1)
    #movie_data.drop("genre",axis=1, inplace=True)


    ## Creating seperate year column and title column
    movie_data["year"]=movie_data["title"].apply(lambda x: x[-5:-1])
    movie_data["title"] = movie_data["title"].apply(lambda x: x[:-7])

    ## Mapping M and F in user data to 1 and 0
    user_data["gender"] = user_data["gender"].map({"M":1,"F":0})


    ###################################
    ####### MERGES ###################

    ## DF with movie rating and the movie info
    movie_rating = pd.merge(ratings_data,
                            movie_data,
                            how="left",
                            left_on ="movie",
                            right_on="movie")

    ## DF with movie rating and the user info
    user_rating = pd.merge(ratings_data,
                           user_data,
                           how="left",
                           left_on ="user",
                           right_on="user")

    ## Final DF with both movie info and user info
    if test_file == False:
        final_train = pd.merge(movie_rating,
                               user_rating,
                               on=["user","movie","rating","timestamp"])
    else:
        final_train = pd.merge(movie_rating,
                               user_rating,
                               on=["user","movie"])


    ## Returning frames as dictionary
    frames = {"ratings_data": ratings_data,
              "movie_data": movie_data,
              "user_data": user_data,
              "movie_rating": movie_rating,
              "user_rating": user_rating,
              "total_frame": final_train}
    print("Name of Frames for reference")
    print("ratings_data, movie_data, user_data, movie_rating, user_rating, total_frame")
    return frames