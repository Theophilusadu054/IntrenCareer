#!/usr/bin/env python
# coding: utf-8

# <center><h1>TASK-1</h1></center>

# ### <center>DATE: DECEMBER 7, 2023</center>
# 
# <h4>NAME : THEOPHILUS ADU ACHIDO</h4>

# <h4>EMAIL: theophilusadu054@gmail.com</h4>

# <center><h2> Task 1: YouTube Streamer Analysis</h2></center>

# <h4>Dataset : Top 1000 Youtubers statistics</h4>

# <p style= "text-align : justify">Description: This dataset contains valuable information about the top YouTube streamers, including their ranking, categories, subscribers, country, visits, likes, comments, and more. The task is to perform a comprehensive analysis of the dataset to extract insights about the top YouTube content creators.</p>

# #### GUIDELINES FOR TASK 1

# 1. Data Exploration:
# - Start by exploring the dataset to understand its structure and identify key variables.
# - Check for missing data and outliers.
# 2. Trend Analysis:
# - Identify trends among the top YouTube streamers. Which categories are the most popular?
# - Is there a correlation between the number of subscribers and the number of likes or comments?
# 3. Audience Study:
# - Analyze the distribution of streamers'audiences by country. Are there regional preferences for specific content categories?
# 4. Performance Metrics:
# - Calculate and visualize the average number of subscribers, visits, likes, and comments.
# - Are there patterns or anomalies in these metrics?
# 5. Content Categories:
# - Explore the distribution of content categories. Which categories have the highest number of streamers?
# - Are there specific categories with exceptional performance metrics?
# 6. Brands and Collaborations:
# - Analyze whether streamers with high performance metrics receive more brand collaborations and marketing campaigns.
# 7. Benchmarking:
# - Identify streamers with above-average performance in terms of subscribers, visits, likes, and comments.
# - Who are the top-performing content creators?
# 8. Content Recommendations:
# - Propose a system for enhancing content recommendations to YouTube users based on streamers

# ##### Libraries

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity


# ##### 1. Data Exploration

# In[2]:


# Load the dataset
data = pd.read_csv("Downloads/youtubers_df.csv")

# Check the structure
data.head()  # Display the first few rows


# In[3]:


data.info()  # Get information about the dataset


# In[4]:


# Check for missing data
data.isnull().sum()  # Check missing values


# In[5]:


data["Categories"].unique()


# In[6]:


# Handle missing data
data.fillna("Unknown", inplace=True)
# Checking output
data.isnull().sum()


# In[7]:


# Correcting the column name Suscribers to Subscribers
data.rename(columns={"Suscribers":"Subscribers"},inplace=True)
# Checking the columns of the DataFrame 
data.columns


# In[8]:


# checking for duplicates
data.duplicated().sum()


# ##### Observation from the info about the dataset
The provided DataFrame contains data related to youtubers on Youtube. Here are some initial observations:
1. Number of Entries: The data contains 1000 entries(Rows) with 0 to 999.
2. Columns: There are 9 columns in the dataset with the following information:
   . Rank        - Which is the rank of the youtuber(int64) with no missing value
   . Username    - Username of the youtuber(object) with no missing value
   . Categories  - Categories of the youtuber(object) with 306 missing value but I replaced with unknown
   . Subscribers - Number subscribers of the youtuber(was float64 but I changed to int64) was misspelled 
     like suscribers with no missing value
   . Country     - Country of the youtubers(object) with no missing value
   . Visits      - Number of visits to the channel of the youtuber(was float64 but I changed to int64) 
     with no missing value
   . Likes       - Number of likes of the youtuber(was float64 but I changed to int64) with no missing value
   . Comments    - Number of comment of the youtuber(was float64 but I changed to int64) with no missing value
   . Links       - Youtube link of the youtuber         
# In[9]:


# First 10 set of the data
data.head(10)


# In[10]:


# Display summary statistics for numeric columns
data.describe()


# In[11]:


# Display summary statistics for object (string) columns
data.describe(include = ["O"])


# ##### 2. Trend Analysis

# In[12]:


# Popular Categories
category_counts = data["Categories"].value_counts()
category_counts


# In[13]:


# Graphical representation of the categories
plt.figure(figsize = (18,10))
sns.barplot(x=category_counts.index,y = category_counts.values, color = "red")
plt.title("Distribution of Categories")
plt.xlabel("Categories")
plt.ylabel("Number of Streamers")
plt.xticks(rotation = 90)
plt.show()


# In[14]:


data_sub = data.drop(columns=["Rank","Visits"])
# Correlation Analysis
sns.heatmap(data_sub.corr(), annot=True)


# ##### 3. Audience Study

# In[15]:


# Country
country_counts = data["Country"].value_counts()
country_counts


# In[16]:


# Geographical Distribution
sns.countplot(x="Country", data=data)
plt.title("Distribution of Country")
plt.ylabel("Number of Streamers")
plt.xticks(rotation = 90)
plt.show()


# In[17]:


category_counts = data.groupby(["Country","Categories"])["Username"].count().unstack().fillna(0)
category_counts


# In[18]:


# Assuming you already have the category_counts DataFrame
plt.figure(figsize=(20, 10))
sns.barplot(data=category_counts, palette='viridis')  # Assuming you have the category_counts DataFrame ready
plt.title('Country Preferences for Content Categories')
plt.xlabel('Content Categories')
plt.ylabel('Count')
plt.xticks(rotation=90)
plt.tight_layout()
plt.show()


# In[19]:


# Reshaping the DataFrame for plotting
category_counts = category_counts.transpose()  # Transposing for better visualization

plt.figure(figsize=(20, 10))
sns.barplot(data=category_counts, palette='viridis')
plt.title('Country Preferences for Content Categories')
plt.xlabel('Country')
plt.ylabel('Count')
plt.xticks(rotation=90)
plt.tight_layout()
plt.show()


# #### 4. Performance Metrics

# In[20]:


# Average Metrics Calculation
avg_subscribers = data["Subscribers"].mean()
avg_visits = data["Visits"].mean()
avg_likes = data["Likes"].mean()
avg_comment = data["Comments"].mean()
# Output
print("The average numbers of Subscriber is: ",avg_subscribers)
print("The average numbers of Visit is:      ",avg_visits)
print("The average numbers of Likes is:      ",avg_likes)
print("The average numbers of Comments is:   ",avg_comment)

# Visualization
sns.barplot(x=["Subscribers", "Visits", "Likes", "Comments"], y=[avg_subscribers, avg_visits, avg_likes, avg_comment])
plt.ylabel("Average [Mean]")
plt.xlabel("Metrics")


# #### 5. Content Categories

# In[21]:


# Value count of the content categories
category_counts = data["Categories"].value_counts()
print(category_counts)
# Graphical representation of value count of the content categories
plt.figure(figsize = (18,10))
sns.barplot(x=category_counts.index,y = category_counts.values, color = "red") # Barplot
plt.title("Distribution of Categories") # Title of the barplot
plt.xlabel("Categories") # X-axis label
plt.ylabel("Number of Streamers") # Y-axis label
plt.xticks(rotation = 90) # Rotation of the index
plt.show() # Output of the graph


# In[22]:


category_count = data["Categories"].value_counts()
categories_with_streamers_above_30 = category_count[category_count > 30]
print("Categories with streamers above 30")
print(categories_with_streamers_above_30)


# In[23]:


plt.figure(figsize = (18,10))
sns.barplot(x=categories_with_streamers_above_30.index,y = categories_with_streamers_above_30.values) # Barplot
plt.title("Categories with streamers above 30") # Title of the barplot
plt.xlabel("Categories") # X-axis label
plt.ylabel("Number of Streamers") # Y-axis label
plt.xticks(rotation = 120) # Rotation of the index
plt.show() # Output of the graph


# In[24]:


performance_metrics = ["Subscribers", "Visits", "Likes", "Comments"]
exceptional_categories_by_metric = {metric: [] for metric in performance_metrics}

for metric in performance_metrics:
    z_scores = (data[metric] - data[metric].mean()) / data[metric].std()
    exceptional_categories = data[z_scores.abs() > 2]["Categories"].unique()
    exceptional_categories_by_metric[metric] = exceptional_categories

# Find categories exceptional across multiple metrics
exceptional_categories_across_metrics = set()
for metric, categories in exceptional_categories_by_metric.items():
    exceptional_categories_across_metrics.update(categories)

print(f'Categories exceptional across metrics: {", ".join(exceptional_categories_across_metrics)}')


# #### 6. Brands and Collabrations

# In[25]:


# Define thresholds for high and low performance based on quartiles
high_subscriber = data["Subscribers"].quantile(0.75, interpolation = "nearest")
high_likes = data["Likes"].quantile(0.75, interpolation = "nearest")
high_comments = data["Comments"].quantile(0.75, interpolation = "nearest")

low_subscriber = data["Subscribers"].quantile(0.25, interpolation = "nearest")
low_likes = data["Likes"].quantile(0.25, interpolation = "nearest")
low_comments = data["Comments"].quantile(0.25, interpolation = "nearest")

# High Performance and Low Performance
data["HighPerformance"] = ((data["Subscribers"] >= high_subscriber) & 
                          (data["Likes"] >= high_likes) &
                           (data["Comments"] >= high_comments))
data["LowPerformance"] = ((data["Subscribers"] >= low_subscriber) & 
                          (data["Likes"] >= low_likes) &
                           (data["Comments"] >= low_comments))
# Engagement based on likes and comment
data["Engagement"] = data["Likes"] + data["Comments"]
high_performance = data[data["HighPerformance"]]["Engagement"].sum()
low_performance = data[data["LowPerformance"]]["Engagement"].sum()

# barplot of High Performance and Low Performance
plt.figure(figsize = (10,5))
sns.barplot(x = ["High Performance","Low Performance"], y = [high_performance,low_performance])
plt.title("Comparison of Engagement between High and Low Performing Streamers")
plt.xlabel("Performance")
plt.ylabel("Total Engagement")
plt.show()


# In[26]:


data.head()


# In[27]:


# Removing Links column from the dataset
data.drop(columns=["Links"], inplace=True)
data.head()


# #### 7. Benchmarking

# In[28]:


# Metrics above their Average
data["Above_Avg_Subscribers"] = data["Subscribers"] > avg_subscribers
data["Above_Avg_Visits"] = data["Visits"] > avg_visits
data["Above_Avg_Likes"] = data["Likes"] > avg_likes
data["Above_Avg_Comments"] = data["Comments"] > avg_comment

#Top performancing streamers interm of metrics above average
top_performing_steamers = data[data["Above_Avg_Subscribers"] &
                               data["Above_Avg_Visits"] &
                               data["Above_Avg_Likes"] &
                               data["Above_Avg_Comments"]]
print("Top-performing Creators: The identification of the top performers based on subscribers, visits, likes, and comments.")
# Output of the result
top_performing_steamers


# #### 8. Content Recommendations

# In[29]:


# Creating the user-item matrix based on subscribers in different content categories
user_item_matrix = data.pivot_table(index="Username", columns="Categories", values="Subscribers", fill_value=0)

# Displaying the user-item matrix
user_item_matrix


# In[30]:


# Assuming user_item_matrix is created correctly
cosine_sim = cosine_similarity(user_item_matrix)

def get_recommendations(username, cosine_sim=cosine_sim):
    idx = user_item_matrix.index.get_loc(username)
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:21]  # Considering top 20 similar streamers
    streamer_indices = [i[0] for i in sim_scores]
    return user_item_matrix.index[streamer_indices]

recommend_streamers = get_recommendations("tseries")
recommend_streamers


# In[ ]:





# In[ ]:




