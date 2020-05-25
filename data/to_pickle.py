"""Script data converts the StockTweets data to dictionary with the structure:
- info: dictionary{'label' : []}
- texts: list
- val_id: list
- test_id: list
- train_id: list
"""

import pandas as pd
import numpy as np
import pytz
import datetime
import pickle
from sklearn.model_selection import train_test_split
import random

# Set seed:
random.seed(11)

# Directories with the data:
dir_raw_data = "C:/Users/danie/Documents/Research Data/Project data/Data Project Sentiment Race/00_raw/"
dir_original_data = 'C:/Users/danie/Documents/Research Data/Original data/StockTwits SP500/'

data_issue = pd.read_csv(
    dir_raw_data + 'data_issue_info.tsv',
    delimiter='\t')

to_be_excluded = data_issue.loc[data_issue['exclude'] == 1, 'rpid'].values

company_mapping = pd.read_csv(
    dir_raw_data + "SP500_Company_Mapping.tsv",
    delimiter="\t")

company_mapping['taq_ticker'] = company_mapping['taq_ticker'].map(lambda ticker: ticker.lower())
company_mapping['original_name'] = company_mapping['original_name'].map(lambda name: name.lower())
company_mapping['cleaned_name'] = company_mapping['cleaned_name'].map(lambda name: name.lower())

to_remove = company_mapping['rpid'].map(lambda x: x in to_be_excluded)
company_mapping = company_mapping.loc[~to_remove, ]


data_tweets = pd.DataFrame()
for rpid_i in company_mapping['rpid'].unique():
    data_i = pd.read_csv(
            dir_original_data + rpid_i + '_tweets.tsv',
            encoding="ANSI", quotechar='"', delimiter="\t", engine='python')
    # Keep only observations which have been classified by users:
    data_i = data_i.loc[data_i['StockTwits_sentiment'] != 'None', ['StockTwits_sentiment', 'text', 'tweet_datetime']]
    # Add RavenPack ID:
    data_i['rpid'] = rpid_i
    # Append data:
    data_tweets = data_tweets.append(data_i, ignore_index=True)

tz_utc = pytz.timezone('UTC')
tz_NY = pytz.timezone('America/New_York')
data_tweets['tweet_datetime'] = data_tweets['tweet_datetime'].map(lambda x: pd.Timestamp(x, tz=tz_utc))
data_tweets['tweet_datetime_ET'] = data_tweets['tweet_datetime'].map(lambda x: x.astimezone(tz_NY))
data_tweets['tweet_date_ET'] = data_tweets['tweet_datetime_ET'].dt.date

training_date_range = pd.date_range(start=datetime.datetime(2013, 6, 1), end=datetime.datetime(2014, 8, 31))

data_tweets['for_train_val'] = data_tweets['tweet_date_ET'].map(lambda x: x in training_date_range)

in_range_bearish = data_tweets.apply(lambda x: x['StockTwits_sentiment'] == 'Bearish' and x['for_train_val'], axis=1)
in_range_bullish = data_tweets.apply(lambda x: x['StockTwits_sentiment'] == 'Bullish' and x['for_train_val'], axis=1)

num_bearish = np.sum(in_range_bearish)
num_bullish = np.sum(in_range_bullish)

ind_train_val_bullish = [i for i in range(len(in_range_bullish)) if in_range_bullish[i]]
ind_to_test_bullish = random.sample(ind_train_val_bullish, num_bullish-num_bearish)
data_tweets.loc[ind_to_test_bullish, 'for_train_val'] = False

ind_train_val = [i for i in range(len(data_tweets['for_train_val'])) if data_tweets['for_train_val'][i]]
ind_train, ind_val = train_test_split(ind_train_val, test_size=0.3)
ind_test = [i for i in range(len(data_tweets['for_train_val'])) if not data_tweets['for_train_val'][i]]

label = [{'label': label == 'Bullish'} for label in list(data_tweets['StockTwits_sentiment'])]

texts = list(data_tweets['text'])

data_to_pickle = {'info': label,
                  'texts': texts,
                  'val_ind': ind_val,
                  'test_ind': ind_test,
                  'train_ind': ind_train}

with open(dir_raw_data + 'DataDeepMojiStockTwits_finetuning.pickle', 'wb') as handle:
    pickle.dump(data_to_pickle, handle, protocol=2)

