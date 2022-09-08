#!/usr/bin/env python
# coding: utf-8

# # Example Flow to Modeling

# ## Import modules

# In[5]:


import pandas as pd
#import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import random


# In[6]:


from PipelineChallenge import split_data_stratified, standard_data, evaluate_model, plot_confusion_matrix, plot_roc_auc


# In[7]:


data = pd.read_csv('/home/cerverat/Dropbox/Shared/rana/training_data_2022-07-20/transformadas/stats_phylotypes_1e0___transformed.csv',
                  index_col=0)


# In[8]:


data.head()


# In[9]:


metadata = pd.read_csv('/home/cerverat/Dropbox/Shared/rana/training_data_2022-07-20/metadata/metadata.csv')


# In[10]:


metadata.head()


# In[11]:


all_participants = sorted(metadata[metadata.collect_wk <= 32].participant_id.unique())


# In[12]:


len(all_participants)


# In[13]:


all_participants[0]


# In[14]:


metadata.info()


# ## Prepare data

# In[15]:


random.seed(12345)
train_participants = random.sample(all_participants, int(0.7*len(all_participants)))
test_participants = [p for p in all_participants if p not in train_participants]
len(test_participants)


# In[16]:


len(train_participants)


# In[17]:


train_specimens = list(metadata[
        metadata.participant_id.apply(lambda p: p in train_participants) &
        (metadata.collect_wk <= 32)
    ].specimen)
test_specimens = list(metadata[
        metadata.participant_id.apply(lambda p: p in test_participants) &
        (metadata.collect_wk <= 32)
    ].specimen)
print(len(train_specimens),
     len(test_specimens),
     train_specimens[0],
     test_specimens[0])


# In[18]:


# get list of labels for each class for train and test participants
preterm_tf = {
        sp: bool(t)
        for (sp, t) in
        zip(
            metadata[metadata.participant_id.apply(lambda p: p in all_participants)].specimen,
            metadata[metadata.participant_id.apply(lambda p: p in all_participants)].was_preterm
        )
    }
preterm_tf_train = [
        preterm_tf.get(p)
        for p in train_specimens
    ]
preterm_tf_test = [
        preterm_tf.get(p)
        for p in test_specimens
    ]

# In[19]:


print(preterm_tf_train[1])


# In[20]:


strat_train = data.loc[train_specimens]
strat_test = data.loc[test_specimens]


# In[21]:


strat_train.iloc[0:3]


# In[53]:


lenTest=len(strat_test)
lenTrain=len(strat_train)
print("Train data preterm == True:",len(strat_train[preterm_tf_train]),
      "\nTrain data preterm == False:",lenTrain-len(strat_train[preterm_tf_train]), 
     "\nTest data preterm == True:",len(strat_test[preterm_tf_test]),
     "\nTest data preterm == False:",lenTest-len(strat_test[preterm_tf_test]))


# ## Models

# In[54]:


print(strat_train.iloc[0:3])


# In[24]:


preterm_tf_train[0:10]


# In[55]:


X = preterm_tf_train
Y = []
for x in X:
#  print ("x = ", x)
  y = int(x == True)
#  print ("y = ", y)
  Y.append(y)
  
X_train = strat_train
y_train = Y
print(X_train.iloc[0:3])
print(y_train[0:7])
print(preterm_tf_train[0:7])

X = preterm_tf_test
Y = []
for x in X:
#  print ("x = ", x)
  y = int(x == True)
#  print ("y = ", y)
  Y.append(y)
  
X_test = strat_test
y_test = Y 
print(X_test.iloc[0:3])
print(y_test[0:7])
print(preterm_tf_test[0:7])
# In[56]:



# ### Random forest

# In[27]:


from sklearn.ensemble import RandomForestClassifier
clf = RandomForestClassifier()
#clf.fit(x_train, y_train)
clf.fit(X_train, y_train)


# In[28]:


clf.score(X_test, y_test)


# ### XGBoost

# In[57]:


from xgboost import XGBClassifier
import matplotlib
from matplotlib import pyplot


# In[58]:


xgb =  XGBClassifier(nthread=-1)


# In[59]:


xgb.fit(X_train, y_train)


# In[60]:


xgb.score(X_test, y_test)


# ## Hyperparameter tunning

# ### Random forest

# In[61]:


from sklearn.model_selection import GridSearchCV 

# search space
n_estimators = [10, 100, 200, 500]
max_depth = [None, 5, 10]
min_samples_split = [2, 3, 4]

# param_grid = [
#     {'n_estimators': [10, 100, 200, 500], "max_depth": [None, 5, 10], "min_samples_split": [2, 3, 4]}
# ]
param_grid = dict(n_estimators=n_estimators, max_depth = max_depth, min_samples_split=min_samples_split)
grid_search = GridSearchCV(clf, param_grid, cv=9, scoring="accuracy", return_train_score=True,  n_jobs=-1)
grid_search.fit(X_train, y_train)


# In[63]:


final_clf = grid_search.best_estimator_


# In[64]:


print("Best: %f using %s" % (grid_search.best_score_, grid_search.best_params_))
means = grid_search.cv_results_['mean_test_score']
stds = grid_search.cv_results_['std_test_score']
params = grid_search.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
    print("%f (%f) with: %r" % (mean, stdev, param))


# ### XGBoost

# #### Random search
# We will be able to shrink the hyperparameter space where to search

# In[ ]:


from sklearn.model_selection import RandomizedSearchCV


n_estimators = [50, 100, 150, 200]
max_depth = [2, 4, 6, 8]
learning_rate = [0.0001, 0.001, 0.01, 0.1, 0.2, 0.3]

random_grid = dict(n_estimators=n_estimators, max_depth=max_depth, learning_rate=learning_rate)

grid_random_search = RandomizedSearchCV(estimator = xgb, param_distributions = random_grid, 
                        n_iter = 96, cv = 9, verbose=2, random_state=42, n_jobs = -1)

random_grid_result = grid_random_search.fit(X_train, y_train)


# In[ ]:


print("Best: %f using %s" % (random_grid_result.best_score_, random_grid_result.best_params_))


# In[59]:


n_estimators = [50, 100, 150, 200]
max_depth = [2, 4, 6, 8]
learning_rate = [0.0001, 0.001, 0.01, 0.1]#, 0.2, 0.3]
param_grid = dict(max_depth=max_depth, n_estimators=n_estimators, learning_rate=learning_rate)
#kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=7)
grid_search = GridSearchCV(xgb, param_grid, scoring="neg_log_loss", n_jobs=-1, cv=9,
verbose=1)
grid_result = grid_search.fit(X_train, y_train)


# In[60]:


final_xgb = grid_search.best_estimator_


# In[62]:


print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
    print("%f (%f) with: %r" % (mean, stdev, param))


# In[77]:


# scores = np.array(means).reshape(len(max_depth), len(n_estimators))
# for i, value in enumerate(max_depth):
#     pyplot.plot(n_estimators, scores[i], label='depth: ' + str(value))
# pyplot.legend()
# pyplot.xlabel('n_estimators')
# pyplot.ylabel('Log Loss')
# pyplot.savefig('n_estimators_vs_max_depth.png')


# ### Store the best models

# In[79]:


import joblib


# In[84]:


joblib.dump(final_clf, "rf_joblib.dat")
joblib.dump(final_xgb, "xgb_joblib.dat")


# ##### Example of load the models

# In[81]:


loaded_rf = joblib.load("rf_joblib.dat")


# ## model evaluation

# In[64]:

#aqui hacer un merge metadata (specimen y was_term) y data y luego quitar las mayores a 32 semanas
df=data_32.reset_index(drop=True, inplace=False)
predicted_target, actual_target = evaluate_model(df, model=final_clf, output_name="preterm")#logreg, output_name)


# ### Confusion matrix

# In[65]:


fig_conf_mat = plot_confusion_matrix(actual_target, predicted_target, classes = "preterm")


# ### ROC curve

# In[66]:


fig_roc_auc = plot_roc_auc(actual_target, predicted_target)


# ## Explainability

# ### Shap values
# [Documentation](https://shap.readthedocs.io/en/latest/index.html)

# In[ ]:


import shap


# In[ ]:


explainer = shap.TreeExplainer(
    final_clf, 
    X_train,
    model_output='probability'
)


# In[ ]:


shap_values = explainer.shap_values(X_train)


# In[ ]:


shap.summary_plot(shap_values, plot_type="bar")


# In[70]:


shap.summary_plot(shap_values, X_train)


# In[71]:


shap.plots._waterfall.waterfall_legacy(explainer.expected_value, shap_values[0])

