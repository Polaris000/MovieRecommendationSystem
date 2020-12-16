# MovieRecommendationSystem

## Abstract
Recommender Systems are used almost everywhere in today’s world, from e-commerce websites,
streaming services to various social media websites. These systems have become an integral part
of our day to day life that we may not even realize that we are using one. In this project we are
implementing collaborative filtering based recommender systems. There are two main approaches
of collaborative filtering:
- User-based    
- Item-based    

We explore several user-based collaborative filtering techniques on MovieLens Dataset, and try
to predict the top 5 similar movies for a given target user. The approaches used in this project have
been novel and are able to give lower MAE values as compared to the standard implementations.


## Project Implementation
The general idea of the project is simple: implement a naive user-vased recommendation system and build upon that to get better results.

- Naive
User based collaborative filtering works based on the assumption that the users who have liked
similar movies in the past will tend to like similar movies in the future. Hence, the first step in
recommending movies to target users is to find similar neighbours of the given target user. The
similarity metric used in our study is Pearson Correlation. After calculating the similarity score of
target user with every other user, we consider top 10 most similar users to the target users in our
further calculations. Predictions are made using the Resnick prediction formula.

- Improvements made:
    - Using tag data (unlike the naive approach, which used only rating data)
    - Giving importance to the number of co-rated movies
    - Increasing the number of neighbours
    - Giving more importance to rare movies (with a lesser number of movies)
    - Using cosine similarity instead of Pearson Correlation
    
- Testing: MAE loss was used along with 5-fold cross-validation.

    
    
## Results
- Naive Implementation  
![](/results/naive.png)

- Improvement 1:  
![](/results/imp1.png)

- Improvement 2:  
![](/results/imp2.png)

- Improvement 3:  
![](/results/imp3.png)

- Improvement 4:  
![](/results/imp4.png)

- Improvement 5:  
![](/results/imp5.png)

## Usage

### Libraries

```
numpy==1.19.4
pandas==1.1.4
scipy==1.5.4
prettytable==2.0.0
vaderSentiment==3.3.2
tqdm==4.51.0
```

To install libraries:  
```
$ pip3 install -r requirements.txt
```

### Data
MovieLens Dataset
#### Files
- ratings.csv
- tags.csv
- movies.csv
- test_user.txt: random users to make predictions on

### Input Files
--------
- to RS_main.py:  
    - ratings.csv  
    - tags.csv  

- to test.py:  
    - test_user.txt    
    - utilitymatrix2.csv (generated from RS_main.py)  

### Execution  
RS_main.py:  

Running this file executes the recommender system including prediction and and performance evaluation for the basic implementation and its 5 improvements.   

    $ python3 RS_main.py --input ./data/ratings.csv --output eval.csv


test.py:  
  
Executing this file lists the top-5 recommended movies along with previously seen movies for the 10 random users using our best performing improvement to the recommender system.  
  
    $ python3 test.py --input ./data/test_user.txt --output output.csv

### Output files

- RS_main.py:  
  - utilitymatrix.csv  
  - utilitymatrix2.csv (main utility matrix)  
  - eval.csv: MAE values for each implementation
  
- test.py:  
    - output.csv: final predictions
    
## Acknowledgements
This project is a team effort. Contributions were made by:
- Aniruddha Karajgi
- Rohit K Bharadwaj
- Jhaveri Ayush Rajesh
- Rahul Jha
- Pranay Khariwal


For more information regarding this project, have a look at our [report](MovieRecommenderSystem.pdf).
  
  
