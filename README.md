# Hate-Nonhate
## Problem Overview: 
In this exercise, I am creating a classifier that determines whether a piece of content is hate speech. I assumed that the data I used for this exercise, Twitter hate speech data from Kaggle (https://www.kaggle.com/vkrahul/twitter-hate-speech?select=train_E6oV3lV.csv), was labeled according to the collected ground truth.

## Solution approach
### Data processing
After cleaning up the tweets, I removed stopwords and applied lemmatization and stemming to generate the root form of the words. The dataset included an unequal proportion of hate tweets and non-hate tweets. To account for this imbalance, I downsampled non-hate tweets (because up-sampling hate tweets resulted in runtime that was too long later when vectorizing the data).

### Vectorization
Tf-idf vectorization was used to score the relative importance of words.

### Model-building
I built classifiers using the following methods: logistic regression, naive bayes, random forest, decision tree, and gradient boosting classifier. Depending on the data, I would choose the model with the highest AUC.

### Other ideas for future work
I would explore XLNet as I found from researching that it often performs better than BERT. This is because it does not assume that the masked words are independent from each other. Another model I would have explored is the multi-view SVM from a study (https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0221152#pone.0221152.ref021).

### Libraries/packages 
pandas, numpy, sklearn, seaborn, matplotlib, re, nltk, collections, wordcloud

### How to run code 
There are 2 iPython notebooks: Data_Preprocessing_Visualization.ipynb and hate-nonhate.ipynb. The pre-processing notebook should be run first.
