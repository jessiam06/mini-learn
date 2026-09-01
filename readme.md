# Mini-Learn: A minimal from-scratch machine learning library
## Overview
I've been teaching myself machine learning from scratch, deriving all the mathematics behind every model, and implementing them using only **python** and **numpy**.
All of my handwritten learning notes are visible in the "Notes" folder. So far I've learnt the theory behind:

- Linear Regression
- Logistic Regression
- Decision Trees
- KMeans Clustering
- Gaussian Mixture Models
- DBSCAN
- Naïve Bayes
- Principal Component Analysis
- Random Forests and Gradient Boosted Methods
- Support Vector Machines
- Neural Networks (separate repo)

With the first half of those being currently implemented in code, while I continue to work on the rest. 

## Design
I chose to take an object oriented approach, and use the scikit learn architecture as a rough framework for how I implement my own. Thus, each model is its own class, with `fit()`, `predict()` and `score()` methods. I've also written a file to hold all of the scoring methods separately, so each class is concerned only with taking in and fitting data, everything else is done through external function calls.

I always do my best to keep my code well commented and readable, but I'll let you be the judge of that ;)
