# -*- coding: utf-8 -*-
"""
Created on Tue Mar 24 16:18:15 2015

@author: jchen
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import psycopg2 as pg

pd.set_option('display.max_columns', None)

# connection to database
conn = pg.connect("dbname=jchen user=postgres")

# First, we'll pull in only the tweets (geo-spatial data included)
mmtd_query='''select * from mmtd'''

mmtd = pd.read_sql(mmtd_query, conn)
conn.close()

##################
# DATA EXPLORATION
##################

# Explore the data and pull some summary stats
print mmtd.describe()
print mmtd.info()
print mmtd.tweet_datetime.describe()
# just over a million tweets
# from 11/09/2011 to 4/30/2013
# most columns non-null except geo-location info

print mmtd.track_title.describe()
print mmtd.artist_name.describe()
# 89k unique tracks by 24k unique artists
# Song with most tweets: Someone Like You (Adele)
# Artist with most tweets: Rihanna

print mmtd.country.describe()
# 202 countries represented, the most frequent being US

# Tweets by continent
print mmtd.groupby('continentName').tweet_id.count()

# Check out the top 20 countries by tweets
tweets_by_country = mmtd.groupby('countryName').tweet_id.count()
tweets_by_country.sort(ascending=False)
print tweets_by_country[0:20]
# store the top countries by tweet volume
top_countries_by_tweet=tweets_by_country.index[0:20].values

# Top cities by tweets
tweets_by_city = mmtd.groupby(['countryName','city']).tweet_id.count()
tweets_by_city.sort(ascending=False)
print tweets_by_city[0:25]
# Interesting, most of these cities are not in the US

# Get count of unique tweets, twitter users, artists, songs by country
grouped = mmtd[mmtd.countryName.isin(top_countries_by_tweet)].groupby('countryName')
count_df = pd.DataFrame(index=top_countries_by_tweet)
cols = ['tweet_id','tweet_userId', 'artist_name', 'track_title']
for i in cols:
    counts = grouped[i].nunique()
    count_df[i]=counts

# Most popular song, artist in each country (top 25 countries)


########################
# GEO-SPATIAL CLUSTERING 
########################
from sklearn import preprocessing
from sklearn.cluster import DBSCAN

# increase the figure size of plots for better readabaility.
from pylab import rcParams
rcParams['figure.figsize'] = 10, 5

# pull out only the lat-long coordinates for custering
all_lat_long=mmtd[['tweet_longitude', 'tweet_latitude']]

# Create a function for scaling data
def scale(df):
    scaler = preprocessing.StandardScaler()
    scaler.fit(df)
    df=pd.DataFrame(scaler.transform(df), columns=df.columns)
    return df

# Scale all data
all_lat_long_scaled=scale(all_lat_long)
# plot the data
plt.figure(figsize=(10, 6), dpi=100)
all_scatter = plt.scatter(all_lat_long_scaled.tweet_longitude, all_lat_long_scaled.tweet_latitude, c='b', edgecolor='', s=15, alpha=0.3)

########
# DBSCAN
########

from time import time

# Bundle up DBSCAN and plot into function that takes df, epsilon, min_samples, filename for plot
def dbscan_plot(df, e=0.5, n=5, filename='dbscan_plot_default.pdf'): 
    start_time = time()
    
    db = DBSCAN(eps=e, min_samples=n).fit(np.array(df))
    core_samples = db.core_sample_indices_
    core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
    core_samples_mask[core_samples] = True
    labels = db.labels_
    
    # number of clusters in labels, ignoring noise if present
    num_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    print('Estimated number of clusters: %d' % num_clusters)
    print 'process took %s seconds' % round(time() - start_time, 2)
    
    # Plot results of dbscan
    unique_labels = set(labels)
    colors = plt.cm.Spectral(np.linspace(0, 1, len(unique_labels)))
    
    for k, color in zip(unique_labels, colors):
        
        class_member_mask = (labels == k)        
        
        if k == -1: # plot noise smaller and lighter
            xy = np.array(df)[class_member_mask & ~core_samples_mask]
            plt.plot(xy[:,0], xy[:,1], 'o', markerfacecolor='1',
                   alpha = .2, markersize=1) 
                   
        if k<> -1:
            # in-cluster members but not core
            xy = np.array(df)[class_member_mask & ~core_samples_mask]
            plt.plot(xy[:,0], xy[:,1], 'o', markerfacecolor=color,
                   markeredgecolor=color, markersize=4) 
               
            xy = np.array(df)[class_member_mask & core_samples_mask]
            plt.plot(xy[:,0], xy[:,1], 'o', markerfacecolor=color,
                   markeredgecolor='k', markersize=8)  # plot core samples larger      
                   
    plt.title('Estimated number of clusters: %d' % num_clusters + ' (eps: %d' % e + ', min. pts: %d' % n +')')
    plt.savefig(filename)
    plt.show()
# Run DBSCAN on all tweet lat long data
dbscan_plot(all_lat_long, 1, 50, 'dbscan_cluster_all.pdf')

# Estimated number of clusters: 209
# process took 12572.35 seconds
# A very long process
# Not a terrible job, but highly imbalanced clusters
# e.g. multiple clusters for islands of Hawaii, yet one large cluster for mainland US

##################

# Let's look at US only tweets
mmtd_us=mmtd[mmtd.country=='US']
us_lat_long=mmtd_us[['tweet_longitude', 'tweet_latitude']]
plt.figure(figsize=(10, 6), dpi=100)
all_scatter = plt.scatter(us_lat_long.tweet_longitude, us_lat_long.tweet_latitude, c='b', edgecolor='', s=15, alpha=0.3)
# Can make out the shape of the US but
# pretty heavily concentrated on eastern half 

# Run DBSCAN on US tweets
dbscan_plot(us_lat_long, 'dbscan_cluster_us.pdf')
# Estimated number of clusters: 18
# process took 235.78 seconds
# Not terribly successful for the US - entire eastern half in one cluster


# Let's try on Europe
mmtd_eu=mmtd[mmtd.continent=='EU']
eu_lat_long=mmtd_eu[['tweet_longitude', 'tweet_latitude']]
plt.figure(figsize=(10, 6), dpi=100)
all_scatter = plt.scatter(eu_lat_long.tweet_longitude, eu_lat_long.tweet_latitude, c='b', edgecolor='', s=15, alpha=0.3)
# no discernable figures when plotted

# Run DBSCAN on EU tweets
dbscan_plot(eu_lat_long, 'dbscan_cluster_eu.pdf')
# Estimated number of clusters: 52
# process took 1572.52 seconds
# Not terribly successful for Europe either
# we can see that there is one very dominant cluster in eastern/central Europe

#########
# K-MEANS
#########

# Use k-means clustering on data subsets to see if we can get better results on more localized data
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist

# Function that runs kmeans with default 8 clusters, plots and saves image as pdf
def kmeans_default_plot(df, title, filename):
    data = df.values
    kmeans_est = KMeans()
    kmeans_est.fit(data)
    labels = kmeans_est.labels_
    plt.scatter(data[:,0], data[:,1], s=60, c=labels)
    plt.title(title)
    plt.savefig(filename)

# Function to iterate to find an optimal number of clusters
def kmeans_cluster_iter(range_start, range_end, df, title, filename):
    data = df.values
    k_range = range(range_start, range_end)
    k_means_var = [KMeans(n_clusters=k).fit(data) for k in k_range]
    centroids = [x.cluster_centers_ for x in k_means_var]
    # calaulate euclidean distance from each point to cluster centroid
    k_euclid = [cdist(data, cent, 'euclidean') for cent in centroids]
    dist = [np.min(ke, axis=1) for ke in k_euclid]
    avg_wcss = [sum(d)/data.shape[0] for d in dist]  # avg within-cluster sum of squares
    plt.plot(k_range, avg_wcss)
    plt.title(title)
    plt.ylabel('Average Within-Cluster Sum of Squares')
    plt.xlabel('Num. of clusters')
    plt.savefig(filename)
    
# Run k-means on US tweets
kmeans_default_plot(us_lat_long, 'Default K-Means on US tweets', 'kmeans_default_us.pdf')   
# Produces pretty generic clusters that could be mapped to US regions
kmeans_cluster_iter(3, 30, us_lat_long,'K-Means cluster performance on US tweets', 'kmeans_n_clusters_us.pdf' )

# Now run k-means with 25 clusters, since that looks like a good plateau point
kmeans_est=KMeans(n_clusters=25).fit(us_lat_long.values)
labels = kmeans_est.labels_
plt.scatter(us_lat_long.values[:,0], us_lat_long.values[:,1], s=30, c=labels)
# A bit more believable

############
# GENRE DATA
############

# Query to pull tweets with associated lastfm tags matched on song title and artist
# only top 25 default Lastfm tags from homepage included 
# Note that we are exlucding tags that may have been mismatched according to Lastfm 
mmtd_tag_query ='''
with lastfm_track_tags as
		(select tid.tid,
			tags.tag,
			tid_tag.val
		from tid_tag
		left join tags on tid_tag.tag=tags.id
		left join tid on tid_tag.tid=tid.id
		where tid.tid not in (select * from tid_mismatches)
			and tags.tag in ('acoustic', 'ambient', 'blues', 'classical', 'country', 'electronic',
                   'emo', 'folk', 'hardcore', 'hip hop', 'indie', 'jazz', 'latin', 'metal',
                   'pop', 'pop punk', 'punk', 'reggae', 'rnb', 'rock', 'soul', 'world', 
                   '60s', '70s', '80s', '90s')
		group by 1,2,3)
select a.*,
	lastfm_track_tags.tag,
	lastfm_track_tags.val as tag_score
from
(select mmtd.*,
	all_tracks.track_id as lastfm_tid
from mmtd
left join (select * from lastfm_all_tracks where track_id not in (select * from tid_mismatches)) as all_tracks on lower(all_tracks.artist)=lower(mmtd.artist_name)
	and lower(all_tracks.title)=lower(mmtd.track_title)) as a
left join lastfm_track_tags on a.lastfm_tid=lastfm_track_tags.tid
order by artist_name, track_title
'''
conn = pg.connect("dbname=jchen user=postgres")
mmtd_tags = pd.read_sql(mmtd_tag_query, conn)
conn.close()

# Check out data
print mmtd_tags.tag.describe()

tags = ['acoustic', 'ambient', 'blues', 'classical', 'country', 'electronic',
       'emo', 'folk', 'hardcore', 'hip hop', 'indie', 'jazz', 'latin', 'metal',
       'pop', 'pop punk', 'punk', 'reggae', 'rnb', 'rock', 'soul', 'world', 
       '60s', '70s', '80s', '90s']

# Tweet count by genre

# Tweets by genre in each country for top 25 countries
country_genre_grouped = mmtd_tags[mmtd_tags.countryName.isin(top_countries_by_tweet)].groupby(['countryName','tag'])
country_genre_count = country_genre_grouped['tweet_id'].count().reset_index()


#############3#####
# CLUSTERING ON MAP
###################
from mpl_toolkits.basemap import Basemap

# redefine dbscan function to plot on top of world map
def dbscan_world_map_plot(df, e=0.5, n=5, filename='dbscan_map_plot_default.pdf'): 
    start_time = time()
    
    db = DBSCAN(eps=e, min_samples=n).fit(np.array(df))
    core_samples = db.core_sample_indices_
    core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
    core_samples_mask[core_samples] = True
    labels = db.labels_
    
    # number of clusters in labels, ignoring noise if present
    num_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    print('Estimated number of clusters: %d' % num_clusters)
    print 'process took %s seconds' % round(time() - start_time, 2)
    
    # Plot results of dbscan
    unique_labels = set(labels)
    colors = plt.cm.Spectral(np.linspace(0, 1, len(unique_labels)))
    
    # Equidistant Cylindrical Projection
    m = Basemap(projection='cyl',llcrnrlat=-65,urcrnrlat=90,\
                llcrnrlon=-180,urcrnrlon=180,resolution='c')
    m.drawcoastlines()
    m.drawcountries()
            
    for k, color in zip(unique_labels, colors):
        
        class_member_mask = (labels == k)        
        
        if k == -1: # plot noise smaller and lighter
            xy = np.array(df)[class_member_mask & ~core_samples_mask]
            xpt, ypt = m(xy[:,0], xy[:,1]) # convert to map coordinates
            m.plot(xpt, ypt, 'o', markerfacecolor='1',
                   alpha = .2, markersize=1) 
                   
        if k<> -1:
            # in-cluster members but not core
            xy = np.array(df)[class_member_mask & ~core_samples_mask]
            xpt, ypt = m(xy[:,0], xy[:,1]) 
            m.plot(xpt, ypt, 'o', markerfacecolor=color,
                   markeredgecolor=color, markersize=4) 
               
            xy = np.array(df)[class_member_mask & core_samples_mask]
            xpt, ypt = m(xy[:,0], xy[:,1])   
            m.plot(xpt, ypt, 'o', markerfacecolor=color,
                   markeredgecolor='k', markersize=8)  # plot core samples larger      
                   
    plt.title('Estimated number of clusters: %d' % num_clusters + ' (eps: %.1f' % e + ', min. pts: {:,}'.format(n) +')')
    plt.savefig(filename)
    plt.show()


# Run and plot DBSCAN for each genre in our top 25 to find global clusters
for tag in tags:
    data=mmtd_tags[mmtd_tags.tag==tag][['tweet_longitude', 'tweet_latitude']]
    if len(data)>0:
        print 'working on ' + tag
        dbscan_world_map_plot(data, 2, 1000, 'dbscan_'+tag+'_ww.pdf')


# Focus on a smaller geographical area to get more specificity
# Let's look
