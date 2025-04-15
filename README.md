# Does Parental Influence Matter For Marijuana Use: An Experiment


### Introduction

The goal of this study is to predict marijuana use among youths (ages 12–17) based on parental and household factors. I used data from the 2020 National Survey on Drug Use and Health (NSDUH), a nationally representative survey conducted by the Substance Abuse and Mental Health Services Administration (SAMHSA). The NSDUH provides extensive information on adolescents’ substance use and related attitudes and behaviors. 

In this analysis, the target variable is whether a youth has ever used marijuana (a binary indicator -> MRJFLAG: 0 = never used, 1 =  used), and how many days in a year did a youth use marijuana (an encoded indicator -> MRJYDAYS: 1 = 1-11 days, 2 = 12-49 days, 3 = 50-99 days, 4 = 100-299 days, 5 = 300-365 days, 6 = Non User or No Past Year Use). The features used as predictors are selected parental factors, including measures of parental monitoring, support, communication, and family structure. 

For example, the features used are the following: 
1. Whether parents tell the youth they are proud of them (PRPROUD2, coded 1 if parents “always or sometimes” expressed pride vs. 2 if “seldom or never”), 
2. Whether parents limit the youth’s TV watching (PRLMTTV2, 1 = always/sometimes vs. 2 = seldom/never), 
3. Whether the youth talked with a parent about the dangers of drug use (PRTALK3, 1 = yes, 2 = no​),
4. Frequency of serious arguments with a parent (ARGUPAR, 1 = 9 or fewer fights in past year, 2 = 10 or more​), 
5. Presence of a mother/father in the household (IMOTHER/IFATHER), 
6. Family socioeconomic status indicators (household income and poverty) as contextual factors,
7. And several others related to parental involvement (e.g. checking homework, helping with homework, enforcing curfews). 


I employed four modeling techniques to predict youth marijuana use: (1) a single decision tree classifier, (2) a bagging ensemble of trees, (3) a random forest, and (4) a gradient boosting model. These models were chosen to compare simple interpretable modeling (decision tree) with more complex ensemble methods that often yield higher predictive accuracy. The models are evaluated and compared based on their predictive performance (primarily classification accuracy, confusion matrices, and MSE on a held out test set), and their interpretability (e.g. feature importance). 

The central research question is: How well can we predict whether a given youth has used marijuana, based on parental influence factors, and which parental factors are most predictive?


### Theoretical Background


**Decision Trees:** 
A decision tree classifier is a flowchart-like predictive model that splits the data based on feature values to classify individuals. Each internal node of the tree applies a decision rule (e.g. do the parents limit the amount of TV or not?) that partitions the data, and the leaf nodes represent final predictions (classifying a youth as a user or non-user of marijuana). The tree is learned by recursively choosing the feature and split that best separates the classes, using criteria like Gini impurity or entropy. 

Decision trees are appealing for their interpretability--a path from the root to a leaf to understand a particular prediction can be acquired--but they tend to overfit if grown too deep. Pruning or setting limits (like a maximum depth) is used to prevent overfitting at the cost of some accuracy. In the study, the tree’s are pruned (e.g. limiting the number of leaf nodes) using a grid search to find the optimal depth via cross-validation. 


**Bagging (Bootstrap Aggregating) with Trees:**
Bagging is an ensemble technique aimed at reducing the variance of unstable models like decision trees. In bagging, many trees are trained on bootstrapping  subsets of the training data and average their predictions (for classification, by majority vote). Each tree is grown fully (or to a set maximum depth) on a different random subset, which helps smooth out the importances of any one tree. The result is usually improved accuracy and more stable predictions compared to a single tree. However, because all features are considered for each split in classic bagging, the trees in the ensemble can still be correlated if one or a few features are very dominant. Bagging models are usually pruned by the number of trees and the depth of each tree. 

Interpretation of a bagged model is more challenging, since we have many trees instead of one--the neat set of rules is typically lost (unless there is a highly correlated feature with the target). As a result, resorting to measures like variable importance (e.g. averaged reduction in impurity) to understand the decisions the model makes is important.


**Random Forest:**
 A random forest is an improvement over bagging that further decorrelates the ensemble trees by introducing randomness in feature selection. In a random forest, each tree is trained on a bootstrap sample and at each split, the algorithm is restricted to choose the best split among a random subset of features (instead of all features). This random feature selection prevents the ensemble from always splitting on the same dominant predictors, thus diversifying the trees and potentially doing away with high correlations. The result is often better generalization performance than bagging, especially when some features are very strong predictors (the high correlations). 
 
Important hyperparameters for random forests include the number of trees, the number of features considered at each split (denoted mtry in R, and max_features in Python), and tree depth/leaf size. In the experiment, the number of features, tree depth, and the number of estimators are tuned via cross-validation using grid search. 
Random forests typically achieve high accuracy and can handle large feature sets well, at the expense of interpretability because, just like bagging, one must rely on aggregate importances. They also tend to be more computationally intensive. And so I use random forests to see if the added randomization yields better prediction of marijuana use and to identify which parental factors show up consistently as important across the trees.


**Gradient Boosting:**
Gradient boosting takes a different approach by building an ensemble of trees sequentially, where each new tree corrects the errors of the previous ensemble. The algorithm starts with an initial prediction (e.g. assuming everyone is a non-user) and then fits a small decision tree to the residuals (the current mistakes), thus adding this tree’s predictions to the model. This process is repeated for many trees, which leads to a gradual improvement of the fit. The important tuning parameters include the number of trees, the learning rate (how much each tree contributes), and tree depth.

Gradient boosting often achieves higher accuracy than bagging or random forests, but it is prone to overfitting if not carefully regularized (i.e., using a low learning rate and/or limiting tree depth), so it typically requires more tuning. It also sacrifices some interpretability, although one can examine the feature importances or partial dependence plots. 

In this study, gradient boosting is used to attempt to maximize predictive performance for identifying youth marijuana users. A grid search is used to tune the parameters (number of trees and learning rate, as well as tree depth) and evaluate whether this more complex model yields gains in accuracy because boosting might capture subtle combinations of parental factors that the other methods miss. But caution must be taken about overfitting.


### Methodology

**Data Preprocessing/Cleaning:** I begun by checking the missing values in the dataset. About 20% of the dataset after cleaning contained NaNs and so these were dropped. They were dropped because the proportion of the target variable largely remains the same after dropping the NaNs (85/15 before dropping and 84/16 after dropping), so the proportions remain representative of the dataset. I then constructed a feature matrix of parental influence variables, followed by checking for high correlations between the features and the target variables. No high correlations were found, leading me to proceed to splitting the data for training and testing in preparation for model fitting.
For the regression models, I created a range mapper that decodes the encoded numbers to their ranges denoted in the codebook by randomly assigning an integer value between the range. For instance, for the target variable MRJYDAYS, having an encoded value of 1 denotes the youth using marijuana 1-11 days in a year. The range mapper will therefore assign a random value between 1 and 11 if it sees an encoded value of 1. This function does this for each encoded value (e.g., 1 to 6 with their associated ranges). 

**Model Training and Tuning:** I split the data into a training set (80%) and a test set (20%), using stratified sampling to preserve the approximate 1:5 ratio of users to non-users in each split. The training set was used for model fitting and hyperparameter tuning via cross-validation. Four models were trained: 

1. *Decision Tree:* The scikit-learn DecisionTreeClassifier with Gini impurity was used. Initially, a tree without a grid search was performed and evaluated with the accuracy score. Then to find the optimal model, a grid search over the maximum number of leaf nodes with cross-validation (10-fold) on the training set suggested an optimal pruned size between 5 and 9 leaf nodes. I therefore pruned the tree to at most 9 terminal nodes for maximum interpretability without losing accuracy.

2. *Bagging Ensemble:* A bagging model was implemented using scikit-learn’s RandomForestRegressor with max_features set to the total number of features to eliminate randomness. A grid search was performed over the number of trees in the ensemble (evaluating 500, 1000, and 2000 trees) and the maximum depth of each tree (e.g. depth 3, 5, 7). The scoring metric for cross-validation was the negative MSE. The best bagging model used 1000 trees of depth 5. 

3. *Random Forest:* When using RandomForestClassifier,  a grid search was performed over the number of trees in the ensemble (evaluating 500, 1000, and 2000 trees), the maximum depth of each tree (3, 5, and 7), and the maximum amount of features used at each split (3, 6 and 9). 10 fold cross-validation indicated a depth of 3, using 3 features at each split small, and a length of 500 trees. These settings basically encouraged each tree to be quite simple. Furthermore, when using  RandomForestRegressor, another grid search was performed with the same parameters as the classifier, but the results were different. The best parameters found was having a maximum depth of 5, the maximum number of features used at each split of 9, and a length of 1000 trees. These setting show a big difference from the use of the classifier. 

4. *Gradient Boosting:* When using GradientBoostingClassifier, a grid search was performed with the number of boosting iterations (up to 200), learning rate (0.1 vs 0.01), and tree depth (3 or 4). The best boosting model had a learning rate of 0.01, 200 trees, and max depth = 4 per tree. This relatively low learning rate and moderate number of trees indicates the model benefited from slow learning to generalize better. We used 10-fold CV accuracy to select these parameters. Notably, we did not apply explicit class weighting or oversampling, so the boosting model’s objective was still dominated by majority class accuracy.

In summary, all model selection (except for a basic decision tree) was done using grid search with cross-validation on the training set. After tuning, we refit each model on the entire training data with the best hyperparameters and evaluated performance on the test set. The evaluation metrics were overall accuracy and confusion matrix for the classification models, and the mean squared error for the regression models. These are used to identify actual marijuana users vs. non-users. I also examined the feature importances for the decision tree models (using Gini-based importance scores) and saved the final pruned decision tree diagram for interpretation.




