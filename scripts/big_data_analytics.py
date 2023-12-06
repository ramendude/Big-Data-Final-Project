import sys
from awsglue.dynamicframe import DynamicFrame
from awsglue.transforms import *
from awsglue.utils import getResolvedOptions
from pyspark.context import SparkContext
from pyspark.sql.functions import *
from awsglue.context import GlueContext
from awsglue.job import Job

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline

sc = SparkContext.getOrCreate()
glueContext = GlueContext(sc)
spark = glueContext.spark_session
job = Job(glueContext)

import warnings
warnings.filterwarnings('ignore')

spark.conf.set("spark.sql.execution.arrow.pyspark.enabled", "true")

# s3 buckets for data preprocessing
masterpath = 's3://finalproject-analytics/Master_dataset.parquet/'
gmpath = 's3://finalproject-analytics/GrandMaster_dataset.parquet/'
cpath = 's3://finalproject-analytics/Challenger_dataset.parquet/'

# creating pandas dataframe
mdata = pd.read_parquet(masterpath, engine='pyarrow')
gmdata = pd.read_parquet(gmpath, engine='pyarrow')
cdata = pd.read_parquet(cpath,engine='pyarrow')

# assigning a new column to each dataframe with their respective Rank
mdata = mdata.assign(Rank = 'Master')
cdata = cdata.assign(Rank = 'Challenger')
gmdata = gmdata.assign(Rank = 'Grandmaster')

# merging all the dataframes into one
merged_data = pd.concat([mdata,gmdata,cdata])

merged_data

# checking if data has successfully merged
# by checking the location of the
# second gameId of each Rank
merged_data.loc[1,:]

# doubling checking with the length = 199925
len(merged_data.index)

# Creating a list of Ranks to filter the gamesId
ranks = ['Challenger', 'Grandmaster', 'Master']
rank_counts = merged_data['Rank'].value_counts()
rank_counts

# Pie chart of total ranked games split by ranks
rank_counts = merged_data['Rank'].value_counts()

lighter_colors = {
    'Challenger': '#ADD8E6',  # Blue
    'Grandmaster': '#FFA07A',  # Red
    'Master': '#D8BFD8'  # Purple
}

plt.figure(figsize=(8, 8))
plt.pie(rank_counts, labels=[f"{rank} ({count})" for rank, count in zip(rank_counts.index, rank_counts)],
        autopct='%1.1f%%', colors=[lighter_colors[rank] for rank in rank_counts.index], startangle=140)
plt.title("Distribution of Total Games by Rank")
plt.text(0, -1.2, f"Total number of games: {rank_counts.sum()}", ha='center', fontsize=12, color='black')
plt.show()

#  Win Percentage by Side
labels = ['Blue', 'Red']
sizes = [merged_data['blueWins'].mean(),  merged_data['redWins'].mean()]
colors = ['#63D1F4', '#EE6363']

fig1, ax1 = plt.subplots()
ax1.pie(sizes, labels=labels, autopct='%1.1f%%',
        colors=colors, shadow=True, startangle=140)
plt.title("Win Percentage by Side")
plt.show()

# Average Game Length
print("Average game length: {:.2f} minutes".format(merged_data['gameDuraton'].mean()/60))
plt.figure(figsize=(20,10))
sns.distplot(merged_data['gameDuraton']/60, hist=True, kde=False)
sns.set(font_scale = 2)
plt.xlabel('Duration (min)')
plt.ylabel('Number of Games')

plt.show()

# Blue Side vs Red Side Data Analytics
categories = ['Wins', 'FirstBlood', 'FirstTower',
       'FirstBaron', 'FirstDragon', 'FirstInhibitor',
       'DragonKills', 'BaronKills', 'TowerKills',
       'InhibitorKills', 'WardPlaced', 'Wardkills', 'Kills',
       'Death', 'Assist', 'ChampionDamageDealt', 'TotalGold',
       'TotalMinionKills', 'TotalLevel', 'AvgLevel',
       'JungleMinionKills', 'KillingSpree', 'TotalHeal',
       'ObjectDamageDealt']

blue_percentage_data = {}

for category in categories:
    blue_total = merged_data['blue'+category].sum()
    red_total = merged_data['red'+category].sum()
    total = (blue_total + red_total).sum()

    blue_percent = blue_total/total
    red_percent = red_total/total

    blue_percentage_data[category] = blue_percent

blue_over = {k:v for k,v in blue_percentage_data.items() if (abs(v-0.5)>0.01) or k=="Wins"}
red_over = {k:1-v for k,v in blue_over.items()}

# Plotting Blue Side vs Red Side
y = range(len(blue_over))
plt.figure(figsize=(20,10))
barWidth = 0.9
# blue
plt.bar(y, list(blue_over.values()), color='#0EBFE9', edgecolor='black', width=barWidth)
# red
plt.bar(y, list(red_over.values()), bottom=list(blue_over.values()), color='#cc0000', edgecolor='black', width=barWidth)

# Custom x axis
plt.xticks(y, blue_over.keys())
plt.ylim((0.45,0.55))
plt.ylabel("Percentage Taken")
plt.title("Differences in sides for values over 1%")
# Show graphic
plt.show()

# Win Correlations
# pearson method normalizes values for me - Kevin
blue_corr = merged_data.drop('Rank', axis=1).corr()['blueWins'][:].sort_values(axis=0, ascending=False)
blue_corr.head()

red_corr = merged_data.drop('Rank', axis=1).corr()['redWins'][:].sort_values(axis=0, ascending=False)
red_corr.head()

# get correlations of 0.5 or more
corr_cols = [prop for prop, corr in blue_corr.items() if abs(corr) > 0.5]
plt.figure(figsize=(12, 12))
sns.set(font_scale=1)
sns.heatmap(merged_data[corr_cols].corr(), annot=True)

# Blue wins to any correlation above 0.5
plt.figure(figsize=(2,7))
sns.heatmap(blue_corr[corr_cols].to_frame(), annot=True, cbar=False)

# Red wins to any correlation above 0.5
plt.figure(figsize=(2,7))
sns.heatmap(red_corr[corr_cols].to_frame(), annot=True, cbar=False)

f, axes = plt.subplots(1, 2, figsize=(10, 7))

# Plot for 'red' correlations
plt.figure(figsize=(2, 7))
red_corra = [prop for prop, corr in blue_corr.items() if 'red' in prop]
sns.heatmap(blue_corr[red_corra].sort_values(axis=0, ascending=False).to_frame(), annot=True, cbar=False, ax=axes[0], cmap='Reds')

# Plot for 'blue' correlations
plt.figure(figsize=(2, 7))
blue_corra = [prop for prop, corr in blue_corr.items() if 'blue' in prop]
sns.heatmap(blue_corr[blue_corra].to_frame(), annot=True, cbar=False, ax=axes[1], cmap='Blues')

f.tight_layout(w_pad=8)

plt.show()