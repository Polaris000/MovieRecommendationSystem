import csv
import numpy as np
import pandas as pd
import copy
import random
import collections
from scipy import linalg, spatial
import math
from collections import defaultdict
from scipy.stats import pearsonr
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import argparse

# ## Predict Top 5 Movies 


#K nearest neighbors based on pearson correlation.
def get_k_neighbors(r, utility_matrix, K):
    neighbors = []
    count = 0
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

    # print(user_ratings)
    for j in range(len(neighbors)):
        rating += neighbors[j][-1] * float(neighbors[j][r] - neighbors[j][neighbors[j] > 0][:-1].mean())
        den += abs(neighbors[j][-1])
    if den > 0:
        rating = np.round(user_ratings[user_ratings > 0].mean()+(rating/den), 5)
    else:
        rating = np.round(user_ratings[user_ratings > 0].mean(), 5)
    # print(rating)
    if rating > 5:
        print(rating)
        return 5.
    elif rating < 0:
        return 0.
    return rating 


def get_sim_matrix(user_ratings, utility_matrix):
    #add similarity col
    utility_matrix = utility_matrix.astype(float)
    r = len(utility_matrix)      #No of rows
    c = len(utility_matrix[0])   #No of Columns
    sim_matrix = np.zeros((r,c+1)) # 2D matrix
    #filling all rows and columns except the last column
    sim_matrix[:,:-1] = utility_matrix 

    #calculating similarities for each user:
    for u in range(r):

        # sim_matrix and user_ratings are unequal
        if np.array_equal(sim_matrix[u,:-1],user_ratings) == False:
            sim_matrix[u,c] =  pearsonr(sim_matrix[u,:-1],user_ratings)[0]   
        else:
            sim_matrix[u,c] = 0.

    #order by similarity:
    sim_matrix = sim_matrix[sim_matrix[:,c].argsort()][::-1]
    return sim_matrix

# returns ratings predicted for each movie
def calculate_top_five(usr):
    sim_matrix = get_sim_matrix(usr, utility_matrix)
    #find the K users for each item not rated:
    u_rec = np.zeros(len(usr))
    c = len(utility_matrix[0])
    for r in range(c):
        if usr.iloc[r]==0: # user has not seen the movie 
            neighbors = get_k_neighbors(r, sim_matrix, 30)
           #calc the predicted rating
            u_rec[r] = calculate_rating(usr[1:], r, neighbors)
    return u_rec

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("--input")
    parser.add_argument("--output")
    args = parser.parse_args()

    print('Recommending movies for 10 random users:')
    print('Executing....')

    # Reading from the test users file
    myLst = []
    with open(args.input, "r") as f:
        for l in f:
            mystr = l
            mystr = mystr[1:]
            mystr = mystr[:-1]
            myLst = mystr.split(',')

    # Changing the elements into int
    for i,l in enumerate(myLst):
        myLst[i] = int(l)

    # final user list
    rand_users = myLst 

    df6 = pd.read_csv("utilitymatrix2.csv")

    user_vec = []
    for r in rand_users:
        user_vec.append(df6.iloc[r,:])

    movieIdLst = list(df6.columns.values)

    #dropping random users so they aren't neighbors of each other
    df6 = df6.drop(df6.index[rand_users])
    df6.to_csv('utilitymatrix3.csv')

    df = pd.read_csv('utilitymatrix3.csv')
    df6_new = []
    df6_new.append(df)
    utility_matrix = df6_new[0].values[:,1:]


    pd.options.display.float_format = '{:.5f}'.format

    results = pd.DataFrame(columns=['User_ID', 'Predicted Movie ID', 'Predicted Rating', 'Past Movie ID', 'Past Rating'])
    for k,u in enumerate(user_vec):#Find top 5 rated movies
        print("UserID: ", rand_users[k])
        
        myLst = np.array(calculate_top_five(u))
        myLst2 = np.argsort(myLst)[::-1]
        myLst3 = np.argsort(u)[::-1]
        myLst3 = myLst3.tolist()
        myLst3 = myLst3[1:]

        for i in range(5):
            row = [rand_users[k], movieIdLst[myLst2[i]], myLst[myLst2[i]], movieIdLst[myLst3[i]], user_vec[k].iloc[myLst3[i]]]
            results = results.append(pd.DataFrame([row], columns=['User_ID', 'Predicted Movie ID', 'Predicted Rating', 'Past Movie ID', 'Past Rating']))

    

    results.to_csv(args.output)
    print("Execution Successfully completed!")
