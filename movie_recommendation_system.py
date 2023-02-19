
import spark as sc
from dotenv import load_dotenv
import os 
from pyspark.mllib.recommendation import ALS
import math
from time import time


load_dotenv()


storage_account_name = 'movie_recommendation_system'
storage_account_access_key = os.getenv('STORAGE_AC_KEY')
sc.conf.set('fs.azure.account.key.' + storage_account_name + '.blob.core.windows.net', storage_account_access_key)

blob_container = 'mv_recommendation_system'
filePath = "wasbs://" + blob_container + "@" + storage_account_name + ".blob.core.windows.net/"

#loading ratings dataset for all the movies
small_ratings_file = sc.read.format("csv").load(filePath+"small/ratings.csv", inferSchema = True, header = True)

small_ratings_raw_data = sc.textFile(small_ratings_file)
small_ratings_raw_data_header = small_ratings_raw_data.take(1)[0]

#parse the raw data to a new RDD
small_ratings_data = small_ratings_raw_data.filter(lambda line: line!=small_ratings_raw_data_header)\
.map(lambda line: line.split(",")).map(lambda tokens: (tokens[0],tokens[1],tokens[2])).cache()

small_ratings_data.take(3)


#loading movies dataset into new RDD
small_movies_file = sc.read.format("csv").load(filePath+"small/movies.csv", inferSchema = True, header = True)

small_movies_raw_data = sc.textFile(small_movies_file)
small_movies_raw_data_header = small_movies_raw_data.take(1)[0]

small_movies_data = small_movies_raw_data.filter(lambda line: line!=small_movies_raw_data_header)\
    .map(lambda line: line.split(",")).map(lambda tokens: (tokens[0],tokens[1])).cache()
    
small_movies_data.take(3)

#spliting dataset for training and testing
training_RDD, validation_RDD, test_RDD = small_ratings_data.randomSplit([6, 2, 2], seed=0)
validation_for_predict_RDD = validation_RDD.map(lambda x: (x[0], x[1]))
test_for_predict_RDD = test_RDD.map(lambda x: (x[0], x[1]))

seed = 5
iterations = 10
regularization_parameter = 0.1
ranks = [4, 8, 12]
errors = [0, 0, 0]
err = 0
tolerance = 0.02

min_error = float('inf')
best_rank = -1
best_iteration = -1
for rank in ranks:
    model = ALS.train(training_RDD, rank, seed=seed, iterations=iterations,
                      lambda_=regularization_parameter)
    predictions = model.predictAll(validation_for_predict_RDD).map(lambda r: ((r[0], r[1]), r[2]))
    rates_and_preds = validation_RDD.map(lambda r: ((int(r[0]), int(r[1])), float(r[2]))).join(predictions)
    error = math.sqrt(rates_and_preds.map(lambda r: (r[1][0] - r[1][1])**2).mean())
    errors[err] = error
    err += 1
    print('For rank {0} the RMSE is {1}'.format(rank, error))
    if error < min_error:
        min_error = error
        best_rank = rank

print('The best model was trained with rank {}'.format(best_rank))

predictions.take(3)
rates_and_preds.take(3)


#testing the model
model = ALS.train(test_RDD, best_rank, seed=seed, iterations=iterations,
                      lambda_=regularization_parameter)
predictions = model.predictAll(test_for_predict_RDD).map(lambda r: ((r[0], r[1]), r[2]))
rates_and_preds = test_RDD.map(lambda r: ((int(r[0]), int(r[1])), float(r[2]))).join(predictions)
error = math.sqrt(rates_and_preds.map(lambda r: (r[1][0] - r[1][1])**2).mean())
    
print('For testing data the RMSE is {}'.format(error))

# Load the big dataset file
big_ratings =sc.read.format("csv").load(filePath+"big/ratings.csv", inferSchema = True, header = True)
big_ratings_raw_data = sc.textFile(big_ratings)
big_ratings_raw_data_header = big_ratings_raw_data.take(1)[0]

# Parse into RDD
big_ratings_data = big_ratings_raw_data.filter(lambda line: line!=big_ratings_raw_data_header)\
    .map(lambda line: line.split(",")).map(lambda tokens: (int(tokens[0]),int(tokens[1]),float(tokens[2]))).cache()
    
print("There are {} recommendations in the complete dataset".format(big_ratings_data.count()))

training_RDD, test_RDD = big_ratings_data.randomSplit([7, 3], seed=0)

big_model = ALS.train(training_RDD, best_rank, seed=seed,terations=iterations, lambda_=regularization_parameter)

test_for_predict_RDD = test_RDD.map(lambda x: (x[0], x[1]))

predictions = big_model.predictAll(test_for_predict_RDD).map(lambda r: ((r[0], r[1]), r[2]))
rates_and_preds = test_RDD.map(lambda r: ((int(r[0]), int(r[1])), float(r[2]))).join(predictions)
error = math.sqrt(rates_and_preds.map(lambda r: (r[1][0] - r[1][1])**2).mean())
    
print('For testing data the RMSE is {}'.format(error))


#movies
big_movies_file = sc.read.format("csv").load(filePath+"big/movies.csv", inferSchema = True, header = True)
big_movies_raw_data = sc.textFile(big_movies_file)
big_movies_raw_data_header = big_movies_raw_data.take(1)[0]

# Parse
big_movies_data = big_movies_raw_data.filter(lambda line: line!=big_movies_raw_data_header)\
    .map(lambda line: line.split(",")).map(lambda tokens: (int(tokens[0]),tokens[1],tokens[2])).cache()

big_movies_titles = big_movies_data.map(lambda x: (int(x[0]),x[1]))
    
print("There are {} movies in the complete dataset".format(big_movies_titles.count()))


def get_counts_and_averages(ID_and_ratings_tuple):
    nratings = len(ID_and_ratings_tuple[1])
    return ID_and_ratings_tuple[0], (nratings, float(sum(x for x in ID_and_ratings_tuple[1]))/nratings)

movie_ID_with_ratings_RDD = (big_ratings_data.map(lambda x: (x[1], x[2])).groupByKey())
movie_ID_with_avg_ratings_RDD = movie_ID_with_ratings_RDD.map(get_counts_and_averages)
movie_rating_counts_RDD = movie_ID_with_avg_ratings_RDD.map(lambda x: (x[0], x[1][0]))




#adding new user
new_user_ID = 0

# The format of each line is (userID, movieID, rating)
new_user_ratings = [
     (0,260,4), # Star Wars (1977)
     (0,1,3), # Toy Story (1995)
     (0,16,3), # Casino (1995)
     (0,25,4), # Leaving Las Vegas (1995)
     (0,32,4), # Twelve Monkeys (a.k.a. 12 Monkeys) (1995)
     (0,335,1), # Flintstones, The (1994)
     (0,379,1), # Timecop (1994)
     (0,296,3), # Pulp Fiction (1994)
     (0,858,5) , # Godfather, The (1972)
     (0,50,4) # Usual Suspects, The (1995)
    ]
new_user_ratings_RDD = sc.parallelize(new_user_ratings)
print('New user ratings: {}'.format(new_user_ratings_RDD.take(10)))

complete_data_with_new_ratings_RDD = big_ratings_data.union(new_user_ratings_RDD)



t0 = time()
new_ratings_model = ALS.train(complete_data_with_new_ratings_RDD, best_rank, seed=seed, iterations=iterations, lambda_=regularization_parameter)
tt = time() - t0

print("New model trained in {} seconds".format(round(tt,3)))

new_user_ratings_ids = map(lambda x: x[1], new_user_ratings) # get movie IDs

new_user_unrated_movies_RDD = (big_movies_data.filter(lambda x: x[0] not in new_user_ratings_ids).map(lambda x: (new_user_ID, x[0])))

# Use the input RDD, new_user_unrated_movies_RDD, with new_ratings_model.predictAll() to predict new ratings for the movies
new_user_recommendations_RDD = new_ratings_model.predictAll(new_user_unrated_movies_RDD)

# Transform new_user_recommendations_RDD into pairs of the form (Movie ID, Predicted Rating)
new_user_recommendations_rating_RDD = new_user_recommendations_RDD.map(lambda x: (x.product, x.rating))
new_user_recommendations_rating_title_and_count_RDD = \
    new_user_recommendations_rating_RDD.join(big_movies_titles).join(movie_rating_counts_RDD)
new_user_recommendations_rating_title_and_count_RDD.take(3)

# Transforming into (Title, Ratings, Ratings count) 
new_user_recommendations_rating_title_and_count_RDD = \
    new_user_recommendations_rating_title_and_count_RDD.map(lambda r: (r[1][0][1], r[1][0][0], r[1][1]))
    
top_movies = new_user_recommendations_rating_title_and_count_RDD.filter(lambda r: r[2]>=25).takeOrdered(25, key=lambda x: -x[1])

print ('TOP recommended movies (with more than 25 reviews):\n{}'.format('\n'.join(map(str, top_movies))))


#individual recommendation
my_movie = sc.parallelize([(0, 500)]) # Quiz Show (1994)
individual_movie_rating_RDD = new_ratings_model.predictAll(new_user_unrated_movies_RDD)
individual_movie_rating_RDD.take(1)
