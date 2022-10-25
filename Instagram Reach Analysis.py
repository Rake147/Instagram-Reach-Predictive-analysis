#!/usr/bin/env python
# coding: utf-8

# In[32]:


import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
from sklearn.model_selection import train_test_split
from sklearn.linear_model import PassiveAggressiveRegressor


# In[33]:


data=pd.read_csv('C:/Users/Rakesh/Datasets/Instagram.csv',encoding = 'latin1')
data.head()


# In[34]:


data.isnull().sum()


# In[35]:


data=data.dropna()


# In[36]:


data.isnull().sum()


# In[37]:


# Analyzing Instagram Reach
plt.figure(figsize=(10,8))
plt.style.use('fivethirtyeight')
plt.title('Distribution of Impressions from Home')
sns.distplot(data['From Home'])
plt.show()


# In[38]:


plt.figure(figsize=(10,8))
plt.title('Distribution of Impressions from Home')
sns.distplot(data['From Hashtags'])
plt.show()


# In[39]:


plt.figure(figsize=(10,8))
plt.title('Distribution of Impressions from Home')
sns.distplot(data['From Explore'])
plt.show()


# In[40]:


#Analyzing in Pie
home=data['From Home'].sum()
hashtags=data['From Hashtags'].sum()
explore=data['From Explore'].sum()
other=data['From Other'].sum()

labels=['From Home','From Hashtags','From Explore','From Other']
values=[home,hashtags,explore,other]

fig=px.pie(data,values=values, names=labels, title='Impression on Instagram posts from various sources', hole=0.5)
fig.show()


# ## Analyze the content of instagram posts

# In[41]:


text="".join(i for i in data.Caption)
stopwords=set(STOPWORDS)
wordcloud=WordCloud(stopwords=stopwords, background_color='white').generate(text)
plt.style.use('classic')
plt.figure(figsize=(12,10))
plt.imshow(wordcloud,interpolation='bilinear')
plt.axis("off")
plt.show()


# In[42]:


text="".join(i for i in data.Hashtags)
stopwords=set(STOPWORDS)
wordcloud=WordCloud(stopwords=stopwords, background_color='black').generate(text)
plt.style.use('classic')
plt.figure(figsize=(12,10))
plt.imshow(wordcloud,interpolation='bilinear')
plt.axis("off")
plt.show()


# In[43]:


#Analyzing Relationships
figure=px.scatter(data,x="Impressions",y="Likes",size="Likes",trendline="ols",title="Relationship between likes and impressions")
figure.show()


# In[44]:


#Analyzing Relationships
figure=px.scatter(data,x="Impressions",y="Comments",size="Comments",trendline="ols",title="Relationship between comments and impressions")
figure.show()


# In[45]:


#Analyzing Relationships
figure=px.scatter(data,x="Impressions",y="Shares",size="Shares",trendline="ols",title="Relationship between shares and impressions")
figure.show()


# In[46]:


#Analyzing Relationships
figure=px.scatter(data,x="Impressions",y="Saves",size="Saves",trendline="ols",title="Relationship between Saves and impressions")
figure.show()


# In[47]:


correlation=data.corr()


# In[48]:


print(correlation["Impressions"].sort_values(ascending=False))


# ## Analyzing Conversion rate

# In[49]:


conversion_rate=(data['Follows'].sum()/data["Profile Visits"].sum())*100
print(conversion_rate)


# In[51]:


figure=px.scatter(data,x="Profile Visits",y="Follows",size="Follows",trendline="ols", title="Relationship between profile visits and followers gained")
figure.show()


# ## Instagram Reach Prediction Model

# In[54]:


x=np.array(data[['Likes','Saves','Comments','Shares','Profile Visits', 'Follows']])


# In[55]:


y=np.array(data[['Impressions']])


# In[56]:


xtrain,xtest,ytrain,ytest= train_test_split(x,y,test_size=0.2,random_state=42)


# In[58]:


model=PassiveAggressiveRegressor()


# In[59]:


model.fit(xtrain,ytrain)


# In[60]:


model.score(xtest,ytest)


# In[61]:


features = np.array([[282.0, 233.0, 4.0, 9.0, 165.0, 54.0]])


# In[63]:


model.predict(features)


# In[ ]:




