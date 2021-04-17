#!/usr/bin/env python
# coding: utf-8

# # Problem Statement
# We need to predict the percentage scores of a student based on the number of hours he/she studies.

# ## Task
# Also, we need to tell what will be predicted score if a student studies for 9.25 hrs/ day?

# ### Identifying the Problem Nature
# There are only two variables and we have to predict the relationship between the dependent and independent variable. So we will first have a look at the data available. We will use Python's Pandas and matplotlib module for that.

# In[2]:


import pandas as pd
import numpy as np  
import matplotlib.pyplot as plt


# In[3]:


# Reading the data
student_data = pd.read_csv("C:/Users/91816/Downloads/student_scores.csv")

# Total Data items
print("Student Data has " + str(len(student_data)) + " rows.")  

# Display student_data
student_data.head(5)


# In[5]:


get_ipython().run_line_magic('matplotlib', 'inline')
student_data.plot(x='Hours', y='Scores', style='c^')  
plt.title('Hours vs Percentage')  
plt.xlabel('Study Hour')  
plt.ylabel('Percentage Score')
plt.show()

Visualising the data provide us the insight that the two variables are linearly related in a positive way. So, we can use Simple Linear Regression to design a solution for this regression problem as Simple linear regression is used to estimate the relationship between two quantitative variables.
# ### Train/Test Data Preparation
# For training and testing our Linear Regression model, we need to divide the data into "attributes" (inputs) and "labels" (outputs).

# In[7]:


# Storing Study Hours ('Input/Attributes') Values in np Array
X = student_data['Hours'].values

# Storing Scores(Labels) Values in np Array
y = student_data['Scores'].values
print("Before reshaping, X is "+str(X.ndim)+"-D array.")

# Reshaping Array from 1-D to 2-D for training ahead

X = X.reshape(len(student_data), 1)
print("After reshaping, X is "+str(X.ndim)+"-D array.")


# In[8]:


# Spliting data into training and test sets using Scikit-Learn Module

from sklearn.model_selection import train_test_split  

X_train, X_test, y_train, y_test = train_test_split(X, y, 
                            test_size=0.2, random_state=0)


# In[9]:


# Training our model using sklearn and training data
from sklearn.linear_model import LinearRegression  

reg = LinearRegression()  
reg.fit(X_train, y_train) 

print("Model Trained Successfully.")


# In[11]:


m = reg.coef_
b = reg.intercept_

# Equation of Line
line = m*X+b

# Plotting for the test data

plt.xlabel('Hours',fontsize=18)
plt.ylabel('Scores',fontsize=18)
plt.title('Regression Line',fontsize=18)
plt.scatter(X, y,color='cyan')
plt.plot(X, line,color='red')
plt.show()


# #### Testing Linear Regression Model
# 

# In[12]:


# Testing data - In Hours
print('Study Hours in Test Data\n')
print(X_test) 

# Predicting the scores
y_pred = reg.predict(X_test)


# In[13]:


# Comparing Actual vs Predicted

com_df = pd.DataFrame({'Actual Scores': y_test, 'Predicted Scores': y_pred})  
com_df


# ### Accuracy of the model
# To compare how well different algorithms perform on a particular dataset, sklearn provide different metrics for accuracy and errors. Here, we have chosen the Mean Square Error.

# In[14]:


from sklearn import metrics

print('Mean Absolute Error:', 
      metrics.mean_absolute_error(y_test, y_pred))

print("R2 score :", 
      round(metrics.r2_score(y_test, y_pred), 2))


# ### Predicting Output Scores where Student Study hours = 9.25
# 

# In[15]:



hours = np.array([[9.25]])
output = reg.predict(hours)

print("Number of Study Hours = {}".format(hours[0,0]))
print("Predicted Score = {}".format(output[0]))


# ### According to the regression model if a student studies for 9.25 hours a day he/she is likely to score around 94 marks.

# # THANK YOU FOR WATCHING !
