import numpy as np
import pandas as pd
import copy
import random
import collections
from scipy import linalg, spatial
import math
from collections import defaultdict
from scipy.stats import pearsonr
from prettytable import PrettyTable
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import argparse
from tqdm import tqdm

def weighted_tags_improvement(df):
    df_tmp = df.copy(deep=True)
    df_tags = pd.read_csv('./data/tags.csv')

    # start sentiment analysis
    analyzer = SentimentIntensityAnalyzer()

    # add a sentiment column to the tags utility_matrixframe and drop the timestamp column
    df_tags['sentiment'] = [analyzer.polarity_scores(x)['compound'] for x in df_tags['tag']]
    df_tags.drop('timestamp', inplace=True, axis=1)

    # aggregate the multiple tag sentiments for each user-movie pair using mean
    df_agg = df_tags.groupby(['userId', 'movieId']).agg('mean')

    df_agg = df_agg.reset_index()

    # convert single to multi-index
    df_tmp.set_index(['userId', 'movieId'], inplace=True)

    df_agg.set_index(['userId', 'movieId'], inplace=True)
    df_agg.rename({"sentiment": "rating"}, axis='columns', inplace=True)

    # replace the common indices between df_agg and df_tmp with the weighted average of rating and
    # mean tag semantic score
    i = df_agg.index.intersection(df_tmp.index)
    df_combined = df_tmp.copy(deep=True)
    df_combined.loc[i, 'rating'] = df_tmp.loc[i, 'rating'] * 0.7  + df_agg.loc[i, 'rating'] * (0.3)

    # reconvert the multi-index to single index
    df_combined.reset_index(inplace=True)

    return df_combined


def calc_weight(df): #Calculate the inverse user frequency for each movie
    n = len(df) # n = total number of users
    weights = []
    for i in range(df.shape[1]):
        num_valid = len(df[df[:,i]>0]) # nt = number of users rating a particular movie

        if num_valid > 0:
            weights.append(math.log(n/num_valid)) # take log(n/nt)
        else:
            weights.append(0)
    return weights

def case_amplification(weight): #give a greater weight to those users who have similarity closer to 1
    res = weight
    if weight >= 0: 
        res = math.pow(weight,2.5) #raise the weights to power 2.5
    elif weight<=0:
        res = -1* (math.pow(-1*weight,2.5))
    return res

def weighted_pearson_corated(vec1, vec2, weight): #reduce the PCC linearly for number of corated movies < 15
    h = 15    # threshold for linear scaling. Beyond 15 movies, PCC will not be scaled
    count = 0
    res = weight

    for i in range(len(vec1)):
        if vec1[i] > 0 and vec2[i] > 0:
            count = count + 1
    
    if count <= h:
        res = res * (count/h)

    return res

def scale_rating(vec, weights): #scale user ratings by their inverse user frequency
    res = []
    for i in range(len(vec)):   
        res.append(weights[i] * vec[i]) # multiply user rating by the weight of that movie
    return res


#K nearest neighbors based on pearson correlation.
def get_k_neighbors(r, utility_matrix, K):
    neighbors = []
    count=0
    for u in range(len(utility_matrix)):
        if utility_matrix[u,r]>0 and count<K:
            neighbors.append(utility_matrix[u])   
            count +=1 
        elif count==K:
            break
    return np.array(neighbors)


#Resnick Prediction Function   
def calculate_rating(user_ratings, r, neighbors):
    rating = 0.
    den = 0.
    for j in range(len(neighbors)):
        rating += neighbors[j][-1] * float(neighbors[j][r] - neighbors[j][neighbors[j] > 0][:-1].mean())
        den += abs(neighbors[j][-1])
    if den > 0:
        rating = np.round(user_ratings[user_ratings > 0].mean()+(rating/den), 5)
    else:
        rating = np.round(user_ratings[user_ratings > 0].mean(), 5)
    if rating > 5:
        return 5.
    elif rating < 0:
        return 0.
    return rating 


def collaborative_filtering(user_ratings, K, utility_matrix, case=0):

    #add similarity col
    utility_matrix = utility_matrix.astype(float) 
    r = len(utility_matrix)      #No of rows
    c = len(utility_matrix[0])   #No of Columns
    sim_matrix = np.zeros((r,c+1)) # 2D matrix

    #filling all rows and columns except the last column
    sim_matrix[:,:-1] = utility_matrix 
    
    wts = None
    if case == 2: 
        wts = calc_weight(utility_matrix)
    #calculating similarities for each user:
    for u in range(r):

        # sim_matrix and user_ratings are unequal
        if np.array_equal(sim_matrix[u,:-1],user_ratings)==False:
            # Only PCC used
            if case == 0:  
                sim_matrix[u,c] =  pearsonr(sim_matrix[u,:-1],user_ratings)[0]
            
            # Weighted_PCC
            elif case == 1: 
                sim_matrix[u,c] =  weighted_pearson_corated(sim_matrix[u,:-1],user_ratings, pearsonr(sim_matrix[u,:-1],user_ratings)[0])
            
            #Inverse User Frequency
            elif case == 2: 
                sim_matrix[u,c] =  pearsonr(scale_rating(sim_matrix[u,:-1], wts), scale_rating(user_ratings, wts))[0]
            
            #Cosine Similarity
            else: 
                sim_matrix[u,c] =  1 - spatial.distance.cosine(sim_matrix[u,:-1],user_ratings)
        else:
            sim_matrix[u,c] = 0.
            
    # order by similarity:
    sim_matrix =sim_matrix[sim_matrix[:,c].argsort()][::-1]

    # find the K users for each item not rated:
    user_predictions = np.zeros(len(user_ratings))
    for r in range(c):
        if user_ratings[r]==0:
            neighbors = get_k_neighbors(r,sim_matrix,K)
           # calc the predicted rating
            user_predictions[r] = calculate_rating(user_ratings,r,neighbors)
   
    return user_predictions


def cross_validation(df, k):
    fold_size = int(len(df)/float(k))
    # print(fold_size)
    df_train = []
    df_validation = []

    for i in range(k):
        df_train.append(pd.concat([df[: i * fold_size],df[i * fold_size + fold_size:]]))  # Will be using df_train we predict for df_validation
        df_validation.append(df[i * fold_size:i*fold_size + fold_size])
        
    return df_train, df_validation


def hide_random_ratings(user_ratings, ratio_hide=0.5):
    random.seed(42)
    user_visible = np.zeros(len(user_ratings))
    user_hidden = np.zeros(len(user_ratings))
    count = 0
    nratings = len(user_ratings[user_ratings>0])
    
    for i in range(len(user_ratings)):
        if user_ratings[i]>0:        
            if bool(random.getrandbits(1)) or count >= int(nratings*ratio_hide):
                user_visible[i]=user_ratings[i]

            #random choice to hide the rating:
            else:
                count +=1
                user_hidden[i]=user_ratings[i]
    # user_hidden stores the values to predict , user_visible contains ratings for getting the neighbors
    return user_visible, user_hidden  

#Used to compute MAE , returns error between predicted and actual value. And count of users  
def calculate_error(user_predictions, user_hidden):
    n = len(user_hidden)
    error = 0.
    count = 0
    for i in range(n):
        if user_hidden[i]>0:
            error +=  abs(user_hidden[i]-user_predictions[i])
            count += 1
    return error,count

def runner(df, neighbors=10, case=0):
    df_train,df_validation = cross_validation(df,5) #5 folds , df_validation contains the test set

    num_folds = 5

    num_movies = len(df_validation[0].values[:,1:][0])
    hidden_folds = []
    visible_folds = []
    for i in tqdm(range(num_folds)):
        user_ratings_list = df_validation[i].values[:,1:]
        user_visible_list = np.empty((0,num_movies),float)
        user_hidden_list = np.empty((0,num_movies),float)
        for user_ratings in user_ratings_list:

            #Hiding half of validation set utility_matrix , so that real value can be predicted
            user_visible,user_hidden = hide_random_ratings(user_ratings)
            user_hidden_list = np.vstack([user_hidden_list,user_hidden])
            user_visible_list = np.vstack([user_visible_list,user_visible])
        hidden_folds.append(user_hidden_list)
        visible_folds.append(user_visible_list)

    average_mae = 0
    res=[]
    for i in tqdm(range(num_folds)):
        error = 0.                    #Used in calculating MAE for each fold
        count = 0
        utility_matrix = df_train[i].values[:,1:]   
        print ('fold: ',i+1)
        user_hidden_list = hidden_folds[i]
        user_visible_list = visible_folds[i]
        
        #Predicting user rating for each fold
        for j in tqdm(range(len(user_hidden_list))):
            user_hidden = user_hidden_list[j]
            user_visible = user_visible_list[j]
            #cf_userbased
            user_predictions = collaborative_filtering(user_visible,neighbors,utility_matrix, case)
            e,c = calculate_error(user_predictions,user_hidden)
            error +=e
            count +=c

        mae = (error / float(count))           #calculating MAE  = sigma(e)/N
        print (' : MAE:',mae,'--',count)
        res.append(mae)
        average_mae += mae
    average_mae = average_mae / num_folds
    res.append(average_mae)
    print('Average MAE: ',average_mae)
    return pd.DataFrame([res], columns=['Fold 1 MAE','Fold 2 MAE','Fold 3 MAE','Fold 4 MAE','Fold 5 MAE','Average MAE'])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--input")
    parser.add_argument("--output")


    args = parser.parse_args()

    df1 = pd.read_csv(args.input)
    df1.drop('timestamp', inplace=True, axis=1)
    df1.head()

    df2 = df1.copy(deep=True)
    df1 = weighted_tags_improvement(df1)

    # improvement: tag semantics
    df6 = pd.pivot_table(df1, index='userId', columns='movieId', values ='rating', fill_value=0)
    
    # save utility matrix as a csv file for use in further steps
    df6.to_csv('utilitymatrix.csv')

    # original case
    df7 = pd.pivot_table(df2, index='userId', columns='movieId', values ='rating', fill_value=0)
    df7.to_csv('utilitymatrix2.csv')
    # print(df6) 

    sim_matrix1 = []

    print('User-based collaborative filtering of the MovieLens Dataset:')
    print('We perform our experiments in the following order:')
    print('1. Basic Implementation of User-based Collaborative Filtering (Part A)')
    print('2. Improvement 1 : Combining with sentiment analysis of tags.csv')
    print('3. Improvement 2: Using Weighted Pearson Correlated Coefficients')
    print('4. Improvement 3: Increasing the number of correlated neighbours considered for predictions')
    print('5. Improvement 4: Combining with Inverse User frequency')
    print('6. Improvement 5: Using Cosine Similarity as a measure')

    print()
    print()

    results = pd.DataFrame(columns = ['Fold 1 MAE','Fold 2 MAE','Fold 3 MAE','Fold 4 MAE','Fold 5 MAE','Average MAE'])
    
    print('1. Basic Implementation of User-based Collaborative Filtering (Part A)')
    print('Executing...')
    #call here
    results = results.append(runner(df=df7,case=0))

    print('2. Improvement 1 : Combining with sentiment analysis of tags.csv')
    print('Executing...')
    #call here
    results = results.append(runner(df=df6, neighbors=10, case=0))

    print('3. Improvement 2: Using Weighted Pearson Correlated Coefficients')
    print('Executing...')
    #call here
    results = results.append(runner(df=df7, neighbors=10, case=1))

    print('4. Improvement 3: Increasing the number of correlated neighbors considered for predictions')
    print('Executing...')
    #call here
    results = results.append(runner(df=df7,neighbors=30,case=0))

    print('5. Improvement 4: Combining with Inverse User frequency')
    print('Executing...')
    #call here
    results = results.append(runner(df=df7,case=2))

    print('6. Improvement 5: Using Cosine Similarity as a measure')
    print('Executing...')
    #call here
    results = results.append(runner(df=df7,case=3))

    imple = ['Basic (Part A)','Improvement 1','Improvement 2','Improvement 3','Improvement 4','Improvement 5']
    results.insert(loc=0, column='Implementation', value=imple)

    #writing all results to eval.csv
    results.to_csv(args.output)
