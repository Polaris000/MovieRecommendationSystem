## MovieRecommendationSystem



## About



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

This file will be main file to run recommender system including prediction and and performance evaluation for the basic implementation and its 5 improvements.   

    $ python3 RS_main.py --input ratings.csv --output eval.csv


test.py:  
  
This file lists the top-5 recommended movies along with previously seen movies for the 10 random users using our best performing improvement to the recommender system.  
  
    $ python3 test.py --input test_user.txt --output output.csv

### Output files  

RS_main.py:  
-----------
  utilitymatrix.csv  
  utilitymatrix2.csv (main utility matrix)  
  eval.csv  
  
test.py:  
    output.csv   
  
  
