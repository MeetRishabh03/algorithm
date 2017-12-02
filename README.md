# algorithm
step by step guide for machine learning
# algorithms
includes all necessary algorithms you should know ...before doing actual machine learning

I. Introduction

sub-field of machine learning / artificial intelligence has increasingly gained more popularity 
in the past couple of years.
machine learning is incredibly powerful to make predictions
or calculated suggestions based on large amounts of data.

for example: 1. netflix algorithmn-
                for movie recommendation,based upon your previously watched movies
             
             2. Amazon's book recommendation-
                based on previously bought books
             
Machine learning algorithmn can broadly classified into 3 categories 

1.Supervised Learning
2.Unsupervised learning
3.Reinforcement learning


1.Supervised Learning:

supervised learning is useful in the cases where a property(also termed as label)
is available for certain dataset (training set),but is missing and needs to be predicted 
for the other instances.

    a.Linear Regression:
    in ML, we have a set of input variables (x) that are used to determine the output variable (y). A relationship exists      
    between the input variables and the output variable. The goal of ML is to quantify this relationship.
    In Linear Regression, the relationship between the input variables (x) and output variable (y) is expressed as an equation  
    of the form y = ax +b. Thus, the goal of linear regression is to find out the values of coefficients a and b. Here, a is the 
    intercept and b is the slope of the line.

    b.Logistic Regression:
    Linear regression predictions are continuous values (rainfall in cm),logistic regression predictions are discrete values 
    (whether a student passed/failed) after applying a transformation function.

    Logistic regression is best suited for binary classification (datasets where y = 0 or 1, where 1 denotes the default class. 
    Example: In predicting whether an event will occur or not, the event that it occurs is classified as 1. In predicting 
    whether a person will be sick or not, the sick instances are denoted as 1). It is named after the transformation function 
    used in it, called the logistic function h(x)= 1/ (1 + e^x), which is an S-shaped curve.

    In logistic regression, the output is in the form of probabilities of the default class (unlike linear regression, where the 
    output is directly produced). As it is a probability, the output lies in the range of 0-1. The output (y-value) is generated 
    by log transforming the x-value, using the logistic function h(x)= 1/ (1 + e^ -x) . A threshold is then applied to force 
    this probability into a binary classification.

     3. CART

    Classification and Regression Trees (CART) is an implementation of Decision Trees, among others such as ID3, C4.5.

    The non-terminal nodes are the root node and the internal node. The terminal nodes are the leaf nodes. Each non-terminal 
    node represents a single input variable (x) and a splitting point on that variable; the leaf nodes represent the output 
    variable (y). The model is used as follows to make predictions: walk the splits of the tree to arrive at a leaf node and 
    output the value present at the leaf node.
     
    4. Naïve Bayes

    To calculate the probability that an event will occur, given that another event has already occurred, we use Bayes’ Theorem. 
    To calculate the probability of an outcome given the value of some variable, that is, to calculate the probability of a 
    hypothesis(h) being true, given our prior knowledge(d), we use Bayes’ Theorem as follows:

    P(h|d)= (P(d|h) * P(h)) / P(d)

    where

    P(h|d) = Posterior probability. The probability of hypothesis h being true, given the data d, where P(h|d)= P(d1| h)* P(d2| 
    h)*....*P(dn| h)* P(d)
    P(d|h) = Likelihood. The probability of data d given that the hypothesis h was true.
    P(h) = Class prior probability. The probability of hypothesis h being true (irrespective of the data)
    P(d) = Predictor prior probability. Probability of the data (irrespective of the hypothesis) 

    This algorithm is called ‘naive’ because it assumes that all the variables are independent of each other, which is a naive a
    ssumption to make in real-world examples. 


     5. KNN

    The k-nearest neighbours algorithm uses the entire dataset as the training set, rather than splitting the dataset into a 
    trainingset and testset.

    When an outcome is required for a new data instance, the KNN algorithm goes through the entire dataset to find the k-nearest 
    instances to the new instance, or the k number of instances most similar to the new record, and then outputs the mean of the 
    
    The similarity between instances is calculated using measures such as Euclidean distance and Hamming distance.

2. Unsupervised learning algorithms:

 
     6. Apriori

     The Apriori algorithm is used in a transactional database to mine frequent itemsets and then generate association rules. It    
     is popularly used in market basket analysis, where one checks for combinations of products that frequently co-occur in the 
     database. In general, we write the association rule for ‘if a person purchases item X, then he purchases item Y’ as : X -> 
     Y.

     Example: if a person purchases milk and sugar, then he is likely to purchase coffee powder. This could be written in the 
     form of an association rule as: {milk,sugar} -> coffee powder. Association rules are generated after crossing the threshold 
     for support and confidence.


     The Support measure helps prune the number of candidate itemsets to be considered during frequent itemset generation. This 
     support measure is guided by the Apriori principle. The Apriori principle states that if an itemset is frequent, then all 
     of its subsets must also be frequent.

     7. K-means

     K-means is an iterative algorithm that groups similar data into clusters.It calculates the centroids of k clusters and 
     assigns a data point to that cluster having least distance between its centroid and the data point.

     Step 1: k-means initialization:

     a) Choose a value of k. Here, let us take k=3.

     b) Randomly assign each data point to any of the 3 clusters.

     c) Compute cluster centroid for each of the clusters. The red, blue and green stars denote the centroids for each of the 3   
     clusters.

     Step 2: Associating each observation to a cluster:

     Reassign each point to the closest cluster centroid. Here, the upper 5 points got assigned to the cluster with the blue 
     colour centroid. Follow the same procedure to assign points to the clusters containing the red and green colour centroid.

     Step 3: Recalculating the centroids:

     Calculate the centroids for the new clusters. The old centroids are shown by gray stars while the new centroids are the 
     red, green and blue stars.

     Step 4: Iterate, then exit if unchanged.

     Repeat steps 2-3 until there is no switching of points from one cluster to another. Once there is no switching for 2 
     consecutive steps, exit the k-means algorithm.

     8. PCA

     Principal Component Analysis (PCA) is used to make data easy to explore and visualize by reducing the number of variables. 
     This is done by capturing the maximum variance in the data into a new co-ordinate system with axes called ‘principal 
     components’. Each component is a linear combination of the original variables and is orthogonal to one another. 
     Orthogonality between components indicates that the correlation between these components is zero.

      The first principal component captures the direction of the maximum variability in the data. The second principal 
      component captures the remaining variance in the data but has variables uncorrelated with the first component. Similarly, 
      all successive principal components (PC3, PC4 and so on) capture the remaining variance while being uncorrelated with the 
      previous component.

3. Ensemble learning techniques:

 
     Ensembling means combining the results of multiple learners (classifiers) for improved results, by voting or averaging. 
     Voting is used during classification and averaging is used during regression. The idea is that ensembles of learners 
     perform 
     better than single learners.

     There are 3 types of ensembling algorithms: Bagging, Boosting and Stacking. We are not going to cover ‘stacking’ here, but 
     
     if you’d like a detailed explanation of it, let me know in the comments section below, and I can write a separate blog on 
     it.

     9. Bagging with Random Forests

       Random Forest (multiple learners) is an improvement over bagged decision trees (a single learner).

       Bagging: The first step in bagging is to create multiple models with datasets created using the Bootstrap Sampling 
       method. In Bootstrap Sampling, each generated trainingset is composed of random subsamples from the original dataset. 
       Each of 
       these trainingsets is of the same size as the original dataset, but some records repeat multiple times and some records 
       do not 
       appear at all. Then, the entire original dataset is used as the testset. Thus, if the size of the original dataset is N, 
       then 
       the size of each generated trainingset is also N, with the number of unique records being about (2N/3); the size of the 
       testset 
       is also N.

       The second step in bagging is to create multiple models by using the same algorithm on the different generated 
       trainingsets. In this case, let us discuss Random Forest. Unlike a decision tree, where each node is split on the best 
       feature 
       that minimizes error, in random forests, we choose a random selection of features for constructing the best split. The 
       reason 
       for randomness is: even with bagging, when decision trees choose a best feature to split on, they end up with similar 
       structure 
       and correlated predictions. But bagging after splitting on a random subset of features means less correlation among 
       predictions 
       from subtrees.

       The number of features to be searched at each split point is specified as a parameter to the random forest algorithm.

       Thus, in bagging with Random Forest, each tree is constructed using a random sample of records and each split is 
       constructed using a random sample of predictors.

       10. Boosting with AdaBoost

       a) Bagging is a parallel ensemble because each model is built independently. On the other hand, boosting is a sequential 
       ensemble where each model is built based on correcting the misclassifications of the previous model.

       b) Bagging mostly involves ‘simple voting’, where each classifier votes to obtain a final outcome– one that is determined 
       by the majority of the parallel models; boosting involves ‘weighted voting’, where each classifier votes to obtain a 
       final outcome which is determined by the majority– but the sequential models were built by assigning greater weights to 
       misclassified instances of the previous models.

Conclusion:

 
To recap, we have learnt:

  1.  5 supervised learning techniques- Linear Regression, Logistic Regression, CART, Naïve Bayes, KNN.
  2.  3 unsupervised learning techniques- Apriori, K-means, PCA.
  3.  2 ensembling techniques- Bagging with Random Forests, Boosting with XGBoost. 
