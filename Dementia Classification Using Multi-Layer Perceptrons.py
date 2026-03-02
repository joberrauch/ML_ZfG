#!/usr/bin/env python
# coding: utf-8

# # Dementia Classification Using Multi-Layer Perceptrons
# *Examination Assignment* | *Studium Integrale: Hands-On Data Science* | *Institute for Biomedical Informatics Cologne* | *Winter Semester 2024/25*
# 
# *Author*: Johannes Oberrauch 7411765
# 
# *Submission*: 27.01.2025 

# ## 1. Introduction
# 
# Dementia is an increasingly relevant disease in an aging population. As the exact causes of dementia are still debated, so are markers for a definite diagnosis. Thus, diagnosis usually consists of interviews, cognitive tests and MRI scans, which are then evaluated by a doctor. Testing whether machine-learning models can predict the diagnosis can give important insights on what information seem to matter.

# ## 2. Materials and Methods
# 
# ### 2.1 Materials
# The dataset used was found on Kaggle and was published by multiple universities such as Washington University. It contains 373 datapoints of 150 subjects collected on multiple visits. Important features recorded are e.g. estimated total intracranial volume (eTIV), Mini-Mental State Examination (MMSE), Age and Magnetic Resonance Delay.
# Further information about the dataset can be found on Kaggle at: https://www.kaggle.com/datasets/jboysen/mri-and-alzheimers
# 
# ### 2.2. Methods
# The main libraries used are Pandas and scikitlearn.
# 
# At first, I used mostly Pandas to clean up the dataframe. I chose to remove the columns 'MRI ID', as it contains the same information as 'Visit', and 'Hand', as all the subjects are right-handed. I also removed all the rows that were 'Converted' in the 'Group' column since we're just focusing on dementia or not. Then I renamed some columns to be more fitting and filled in NaN values in the 'SES' column, keeping the distribution as it was before. OneHotEncoding was used to transform the 'sex' column and then I trained three MLP's on a 80/20-train-test split, evaluating them afterwards with the scikit-learn method .classification_report() and visualising them with sns.barplot().

# ## 3. Results

# In[349]:


#import libraries

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn import model_selection, neighbors, preprocessing, neural_network, linear_model, multiclass, metrics
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report


# In[350]:


#import the used dataset and display it
df = pd.read_csv("oasis_longitudinal.csv")
df.head()


# In[351]:


df.info()


# In[352]:


df.describe(include = 'all')


# In[353]:


#drop the columns 'MRI ID' and 'Hand' as they are not relevant
df = df.drop(['MRI ID', 'Hand'], axis = 1)
df


# In[354]:


#drop the 'Converted' subjects
df = df[df.Group != 'Converted']
df


# In[355]:


#rename certain features to be more easily typed

df = df.rename(
        columns={
            'Subject ID': 'subject_id', 
            'Group': 'group', 
            'Visit': 'visit',
            'MR Delay': 'mr_delay',
            'M/F': 'sex',
            'Age': 'age','EDUC': 'education'
        }
)

df.head()


# In[356]:


#check where and how many NaN values exist in the dataset. We see that the majority of our NaN values are in the 'SES' column
#which is a factor that is not likely to change in between different visits, hence we'll try to keep these subjects by filling
#the NaN values. 'MMSE' however, is a test directly trying to measure cognitive ability which we would expect to worsen between
#visits so we can't just fill it and will therefore remove those rows

df.isna().sum()


# In[357]:


#drop the rows with NaN values in the 'MMSE' column
df = df[df['MMSE'].isnull() == False]
df.isna().sum()


# In[358]:


#grab only the subset of the dataset containing 'subject_id' and 'SES' to further examine the types of NaN
df_replace_SES = df[['subject_id', 'SES']]


#check whether the NaN values are from unique subjects or accumulated within fewer subjects due to multiple MRI scans
df_replace_unique_SES = df_replace_SES.drop_duplicates(subset='subject_id')
df_replace_unique_SES.isna().sum()


# In[359]:


#now we know there's only 8 subjects with NaN values; we don't want to drop those subjects since we have only 150 total and
#'SES' is presumably not the most important feature of this dataset. Thus we assume that it's okay to fill these NaN values
#by checking the distribution among 'SES' and then assigning our 8 NaN subjects a 'SES' rating based on the distribution

df_replace_unique_formatted_SES = df_replace_unique_SES.groupby('SES').count()
df_replace_unique_formatted_SES


# In[360]:


#to visualize the distribution of 'SES' in our dataset we display it on a piechart

df_replace_unique_formatted_SES.plot(
    y = 'subject_id', 
    figsize = (5,5), 
    kind = 'pie', 
    legend = False, 
    autopct = '%1.1f%%', 
    colormap = 'summer'
)


# In[361]:


#now we use this while-loop to fill the NaN values by assigning one of the 4 'SES' ratings to each 2 subjects to keep the 
#distribution of 'SES' in our dataset

item = df_replace_unique_SES
values = 0
while df_replace_unique_SES['SES'].isna().sum() > 0:
    values = values + 1
    df_replace_unique_SES.fillna(value = values, limit = 2, inplace = True)

df_replace_unique_SES


# In[362]:


#we filled the NaN values for every individual but since there's multiple visits we have to first update our original subset of 
#the dataframe and then forward fill the remaining NaN values

#update original subset of our dataframe
df_replace_SES.update(df_replace_unique_SES)

#forward fill the remaining NaN values which are within subjects now
df_replace_SES.ffill(inplace = True)

#check if there are any NaN values left
df_replace_SES.isna().sum()


# In[363]:


#visualize the 'SES' distribution after filling our NaN values and we can see that the distribution hasn't changed significantly
df_replace_SES.groupby('SES').count().plot(
    y = 'subject_id', 
    figsize = (5,5), 
    kind = 'pie', 
    legend = False, 
    autopct = '%1.1f%%', 
    colormap = 'summer'
)


# In[364]:


#update original dataframe with the filled in values
df.update(df_replace_SES)
print(df.isna().sum())
df


# In[365]:


#drop the 'subject_id' column now that we're done filling in NaN values
df = df.drop(['subject_id'], axis = 1)
df.head()


# In[366]:


#check dtypes to see which columns are numerical and which still need to be transformed
df.dtypes


# In[367]:


#use one hot encoding to transform cat to num
ohe = preprocessing.OneHotEncoder()


# In[368]:


#ohe for our 'sex' classification
ohe_sex = ohe.fit_transform(X = df[['sex']])


# In[369]:


#add the encoded columns to our dataframe
df[ohe.categories_[0]] = ohe_sex.toarray()
df


# In[370]:


#now we drop the original columns and only keep the encoded ones
df.drop(columns = ['sex'], inplace = True)
df


# In[371]:


#check the types again
df.dtypes


# In[372]:


#our goal is to classify based on MRI scans whether a person is demented or not. Here we investigate the distribution in our
#target variable
df.group.value_counts()


# In[373]:


#split the dataset into feature matrix X and response vector y
X = df.drop(columns=['group'])
y = df.group.to_numpy()
print(X.head())
print(y[:10])


# In[374]:


#split into training and testing sets (80/20)
X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.2, random_state=2)


# In[375]:


#training the first model
model = neural_network.MLPClassifier(solver='lbfgs', 
                                     activation='logistic', 
                                     hidden_layer_sizes=(3,), 
                                     random_state=24
                                    )

#fit the model
nn = model.fit(X_train, y_train)

#predict with the first model
pred_y_1 = nn.predict(X_test)


# In[376]:


#training the second model
model = neural_network.MLPClassifier(solver='lbfgs', 
                                     activation='logistic', 
                                     hidden_layer_sizes=(3,2), 
                                     random_state=24
                                    )

#fit the model
nn = model.fit(X_train, y_train)

#predict with the second model
pred_y_2 = nn.predict(X_test)


# In[377]:


#training the third model
model = neural_network.MLPClassifier(solver='lbfgs', 
                                     activation='logistic', 
                                     hidden_layer_sizes=(3,), 
                                     alpha=0.7, 
                                     random_state=24
                                    )

#fit the model
nn = model.fit(X_train, y_train)

#predict with the third model
pred_y_3 = nn.predict(X_test)


# In[378]:


#determine the accuracy of the three models on the test dataset
acc_1 = metrics.accuracy_score(y_test, pred_y_1)
acc_2 = metrics.accuracy_score(y_test, pred_y_2)
acc_3 = metrics.accuracy_score(y_test, pred_y_3)
print(acc_1)
print(acc_2)
print(acc_3)


# In[379]:


#now we standardize our values to increase our accuracy
X = StandardScaler().fit_transform(X)


# In[380]:


#split into training and testing sets (80/20)
X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.2, random_state=2)


# In[381]:


#training the first model
model = neural_network.MLPClassifier(solver='lbfgs', 
                                     activation='logistic', 
                                     hidden_layer_sizes=(2,), 
                                     random_state=24
                                    )

#fit the model
nn = model.fit(X_train, y_train)

#predict with the first model
pred_y_1 = nn.predict(X_test)


# In[382]:


#training the second model
model = neural_network.MLPClassifier(solver='lbfgs', 
                                     activation='logistic', 
                                     hidden_layer_sizes=(6,5), 
                                     random_state=24
                                    )

#fit the model
nn = model.fit(X_train, y_train)

#predict with the second model
pred_y_2 = nn.predict(X_test)


# In[383]:


#training the third model
model = neural_network.MLPClassifier(solver='lbfgs', 
                                     activation='logistic', 
                                     hidden_layer_sizes=(2,), 
                                     alpha=0.6, 
                                     random_state=24
                                    )

#fit the model
nn = model.fit(X_train, y_train)

#predict with the third model
pred_y_3 = nn.predict(X_test)


# In[384]:


#determine the accuracy of the three models on the test dataset
acc_1 = metrics.accuracy_score(y_test, pred_y_1)
acc_2 = metrics.accuracy_score(y_test, pred_y_2)
acc_3 = metrics.accuracy_score(y_test, pred_y_3)
print(acc_1)
print(acc_2)
print(acc_3)


# In[385]:


#now we evaluate the performance of our models
model_1 = classification_report(y_test, pred_y_1, labels=['Demented', 'Nondemented'], output_dict = True)

model_2 = classification_report(y_test, pred_y_2, labels=['Demented', 'Nondemented'], output_dict = True)

model_3 = classification_report(y_test, pred_y_3, labels=['Demented', 'Nondemented'], output_dict = True)


# In[386]:


#create a dataframe from our dictionary and add column 'model'
mlp1 = pd.DataFrame.from_dict(model_1)
mlp1['model'] = 'MLP1'
mlp1


# In[387]:


#drop the 'support' column
mlp1 = mlp1.drop(['Demented','Nondemented'], axis = 1).drop(['support'])
mlp1


# In[388]:


#use melt to transform the dataframe to long format 
mlp1 = mlp1.melt(id_vars = 'model', var_name = 'metric')
mlp1 = mlp1.drop([0,1])
mlp1


# In[389]:


#now we use .loc[] to rename the entries 
mlp1.loc[3,'metric'] = 'precision macro'
mlp1.loc[4,'metric'] = 'sensitivity macro'
mlp1.loc[5,'metric'] = 'f1 macro'
mlp1.loc[6,'metric'] = 'precision weighted'
mlp1.loc[7,'metric'] = 'recall weighted'
mlp1.loc[8,'metric'] = 'f1 weighted'

mlp1


# In[390]:


#now the same for the second and third model
mlp2 = pd.DataFrame.from_dict(model_2)
mlp2['model'] = 'MLP2'


# In[391]:


mlp2 = mlp2.drop(['Demented','Nondemented'], axis = 1).drop(['support'])


# In[392]:


mlp2 = mlp2.melt(id_vars = 'model', var_name = 'metric').drop_duplicates()


# In[393]:


mlp2.loc[3,'metric'] = 'precision macro'
mlp2.loc[4,'metric'] = 'sensitivity macro'
mlp2.loc[5,'metric'] = 'f1 macro'
mlp2.loc[6,'metric'] = 'precision weighted'
mlp2.loc[7,'metric'] = 'recall weighted'
mlp2.loc[8,'metric'] = 'f1 weighted'

mlp2


# In[394]:


mlp3 = pd.DataFrame.from_dict(model_3)
mlp3['model'] = 'MLP3'


# In[395]:


mlp3 = mlp3.drop(['Demented','Nondemented'], axis = 1).drop(['support'])


# In[396]:


mlp3 = mlp3.melt(id_vars = 'model', var_name = 'metric').drop_duplicates()


# In[397]:


mlp3.loc[3,'metric'] = 'precision macro'
mlp3.loc[4,'metric'] = 'sensitivity macro'
mlp3.loc[5,'metric'] = 'f1 macro'
mlp3.loc[6,'metric'] = 'precision weighted'
mlp3.loc[7,'metric'] = 'recall weighted'
mlp3.loc[8,'metric'] = 'f1 weighted'

mlp3


# In[398]:


#now we merge those 3 dataframes with pd.merge() and then we sort the dataframe by the column 'metric'
merged1 = mlp1.merge(mlp2, how = 'outer')
merged_all = merged1.merge(mlp3, how = 'outer')
merged_all = merged_all.sort_values('metric')
merged_all


# In[399]:


#finally, we plot the resulting dataframe on a barplot to compare the performance of the three models
sns.barplot(x = merged_all.metric, y = merged_all.value, hue = merged_all.model)
plt.legend(loc = 'center left', bbox_to_anchor = (1.0, 0.5))
plt.xticks(rotation = 90)
plt.show()


# ## 4. Discussion
# This was a very limited analysis in a very controlled environment to demonstrate mainly the understanding of basic data science methods. Still, the performance of the models suggests that a classification of dementia is possible given extensive data on a subject.
# A much more detailed and professional analysis of this dataset was published by a group of researches in the Journal of Clinical Medicine and can be found via this link: https://www.mdpi.com/2077-0383/9/7/2146

# ## 5. Conclusion
# We used MLP to classify whether a subject is demented or not. Thanks to the extensive features available it seems a reliable classification is possible in this case. However, it's important to keep in mind that real-world situations can differ vastly and machine learning algorithms might not always be as reliable as in this example.

# ## References
# 
# [1] https://www.kaggle.com/datasets/jboysen/mri-and-alzheimers
# 
# [2] https://scikit-learn.org/1.5/user_guide.html
# 
# [3] https://pandas.pydata.org/docs/reference/index.html
# 
# [4] https://stackoverflow.com/
