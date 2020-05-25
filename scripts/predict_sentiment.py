from __future__ import print_function
import scripts_helper
import json
from keras.models import model_from_json
from deepmoji.model_def import deepmoji_architecture, load_specific_weights
from deepmoji.global_variables import RAW_DATA_PATH, MODEL_DATA_PATH, \
    STOCKTWITS_DATA_PATH, ORIGINAL_DATA_PATH, PROCESSED_DATA_PATH, TWITTER_DATA_PATH
from deepmoji.sentence_tokenizer import SentenceTokenizer
import pandas as pd
import re
import numpy as np
import pytz
import datetime


def bull(x):
    ratio = float(1 + np.sum(x > 0)) / float(1 + np.sum(x < 0))
    return np.log(ratio)


"""
Aggregation method: define how the tweets are aggregated
"""
# Define whether the aggregation takes place close-to-close or open-to-open:
aggregation_tweets = 'c2c'
if aggregation_tweets == 'c2c':
    hours_shift = 8
    minutes_shift = 0
elif aggregation_tweets == 'o2o':
    hours_shift = -9
    minutes_shift = -30
else:
    hours_shift = 0
    minutes_shift = 0

"""
Filtering: define how the tweets should be filtered, i.e. which tweets should be removed
"""
# Keep only messages which contain the company's cashtag:
has_cashtag = True
# Keep only messages which only contain the compnay's cashtag:
unique_cashtag = True
# Unique cashtags makes only sense if we keep only tweets which mention the company's cashtag:
unique_cashtag = unique_cashtag and has_cashtag

""" 
Other settings: define time zones, location of data, ...
"""
# Define time zones:
tz_utc_in_stocktwits = pytz.timezone('UTC')
tz_zurich_in_twitter = pytz.timezone('Europe/Zurich')
tz_est_out = pytz.timezone('America/New_York')
# Define location of the relevant data:
dataset_path = '{}/DataDeepMojiStockTwits_finetuning.pickle'.format(RAW_DATA_PATH)
model_path = '{}/DeepMojiStockTwits_model.json'.format(MODEL_DATA_PATH)
specs_path = '{}/DeepMojiStockTwits_specs.json'.format(MODEL_DATA_PATH)
weights_path = '{}/DeepMojiStockTwits_weights.h5'.format(MODEL_DATA_PATH)
vocab_path = '{}/DeepMojiStockTwits_vocab.json'.format(MODEL_DATA_PATH)
# File specification:
file_specifications = ''
file_specifications = file_specifications + aggregation_tweets
if has_cashtag:
    file_specifications = file_specifications + '_cashtag_only'
if unique_cashtag:
    file_specifications = file_specifications + '_unique'

"""
Model: define and load all model-related things
"""
# Specify number of classes used for the classification:
nb_classes = 2
# Load the vocabulary:
with open(vocab_path, 'r') as f:
    vocab = json.load(f)
# Load model specifications:
with open(specs_path, 'r') as f:
    model_specs = json.load(f)
# Define the sentence tokenizer:
st = SentenceTokenizer(vocab, model_specs['maxlen'])
# Define architecture of the model:
model = deepmoji_architecture(nb_classes=nb_classes,
                              nb_tokens=len(vocab),
                              maxlen=model_specs['maxlen'],
                              embed_dropout_rate=0.25,
                              final_dropout_rate=0.5,
                              embed_l2=1E-6)
# Load weights of the model:
load_specific_weights(model=model, weight_path=weights_path)
# Load information about stocks for which we have data issues:
data_issue = pd.read_csv(
    RAW_DATA_PATH + '/data_issue_info.tsv',
    delimiter='\t')
# Load mapping file for the companies:
company_mapping = pd.read_csv(
    RAW_DATA_PATH + "/SP500_Company_Mapping.tsv",
    delimiter="\t")

"""
Information: load information about the stocks and the stock market
"""
# Exclude stocks for which we have data issues:
to_be_excluded = data_issue.loc[data_issue['exclude'] == 1, 'rpid'].values
to_remove = company_mapping['rpid'].map(lambda x: x in to_be_excluded)
company_mapping = company_mapping.loc[~to_remove, ]
# Load data with information about closing times/days of the NYSE:
holidays = pd.read_csv(ORIGINAL_DATA_PATH + '/Miscellaneous/NYSE_closing_days.tsv', delimiter='\t')
# Change column names:
holidays.columns = ['Date', 'Time', 'Holiday']
# Remove the time column:
holidays = holidays.drop('Time', axis=1)
# Define whether a date is a holiday:
holidays['Holiday'] = holidays['Holiday'].map(lambda x: x == 1)
# Transform the date-strings into datetime objects:
holidays['Date'] = holidays['Date'].map(lambda x: pd.Timestamp(x))
# Define a data-frame with all dates in our sample
closing_info = pd.DataFrame({'Date': pd.date_range(start=datetime.datetime(2010, 1, 1), end=datetime.datetime(2019, 1, 1))})
# Define a column that indicates all weekends:
closing_info['Weekend'] = closing_info['Date'].map(lambda x: x.weekday() in [5, 6])
# Merge the new data-frame with the information about holidays
closing_info = closing_info.merge(holidays, how='left', on='Date')
closing_info['Holiday'] = closing_info['Holiday'].fillna(False)
# Mark all closed days (either holiday or weekend)
closing_info['Closed'] = closing_info.apply(lambda x: x['Weekend'] or x['Holiday'], axis=1)
# Ensure that dates are a datetime object
closing_info.Date = closing_info.Date.dt.date
# Remove columns that are no longer needed:
closing_info = closing_info.drop(['Weekend', 'Holiday'], axis=1)


"""
Predict and aggregate sentiment of StockTwits
"""
sentiment_stocktwits = pd.DataFrame()
for rpid_i in company_mapping['rpid'].unique():
    # Load data:
    data_i = pd.read_csv(
                STOCKTWITS_DATA_PATH + '/' + rpid_i + '_tweets.tsv',
                encoding="mbcs",
                quotechar='"', delimiter="\t", engine='python')
    # Keep only relevant columns:
    data_i = data_i[['text', 'tweet_datetime']]
    # Remove empty messages:
    data_i = data_i.loc[data_i['text'].map(lambda x: x is not None), :]
    # Define regular expression for the company's cashtag:
    cashtag_regex_i = '|'.join(r'([$]{1}\b' + company_mapping.loc[company_mapping['rpid'] == rpid_i, 'taq_ticker'] + r'\b)')
    # Count number of company cashtags:
    data_i['num_companycashtag'] = data_i['text'].map(lambda x: len(re.findall(cashtag_regex_i, x)))
    # Count the number of all cashtags:
    data_i['num_cashtag'] = data_i['text'].map(lambda x: len(re.findall(r'[$]\b[a-zA-z]+\b', x)))
    # If wanted, remove tweets that do not mention the company's cashtag:
    if has_cashtag:
        data_i = data_i.loc[data_i['num_companycashtag'] > 0]
    # If wanted, remove tweets that mention other cashtags:
    if unique_cashtag:
        data_i = data_i.loc[data_i['num_cashtag'] == data_i['num_companycashtag']]
    # Transform strings to timestamps:
    data_i['tweet_datetime'] = data_i['tweet_datetime'].map(lambda x: pd.Timestamp(x, tz=tz_utc_in_stocktwits))
    # Change timezone to Eastern Time:
    data_i['tweet_datetime_ET'] = data_i['tweet_datetime'].map(lambda x: x.astimezone(tz_est_out))
    # Shift time depending on the aggregation scheme chosen previously:
    data_i['tweet_datetime_ET_shifted'] = \
        data_i['tweet_datetime_ET'].map(lambda x: x + datetime.timedelta(hours=hours_shift, minutes=minutes_shift))
    # Define date based on the shifted ET timestamp:
    data_i['Date'] = data_i['tweet_datetime_ET_shifted'].dt.date

    try:
        texts = [unicode(x) for x in data_i['text']]
    except UnicodeDecodeError:
        texts = [x.decode('utf-8') for x in data_i['text']]
    X = st.get_test_sentences(texts)
    pred_sentiment = model.predict(X, model_specs['batch_size'])
    data_i['DeepMoji'] = [(int(x > 0.5) - 0.5)*2 for x in pred_sentiment.flatten()]
    # For the aggregation, we shift the date of messages posted during holidays or weekends to the next trading day:
    data_i = data_i.merge(closing_info, how='left', on='Date')[['Date', 'Closed', 'DeepMoji']]
    while any(data_i.Closed):
        data_i['Date'] = data_i.apply(lambda x: x['Date'] + datetime.timedelta(days=1) if x['Closed'] else x['Date'], axis=1)
        data_i = data_i.drop('Closed', axis=1).merge(closing_info, how='left', on='Date')
    # Aggregate sentiments on a daily basis:
    sentiment_i = data_i.drop('Closed', axis=1).groupby('Date').aggregate({'DeepMoji': [bull, np.mean]})
    # Delete the raw data:
    del data_i
    # Transform multi-index column names to single level:
    sentiment_i.columns = ['_'.join(col).strip() for col in sentiment_i.columns.values]
    # Date (which acts as an index) to a column:
    sentiment_i.reset_index(level=0, inplace=True)
    # Add information about RavenPack ID:
    sentiment_i['rpid'] = rpid_i
    # Append data:
    sentiment_stocktwits = sentiment_stocktwits.append(sentiment_i, ignore_index=True)
    # Remove the sentiment data:
    del sentiment_i

# Save aggregated StockTwits sentiment:
sentiment_stocktwits.to_csv(PROCESSED_DATA_PATH + '/StockTwits_DeepMoji_daily_' + file_specifications + '.csv')

"""
Predict and aggregate sentiment of Twitter
"""
sentiment_twitter = pd.DataFrame()
for rpid_i in company_mapping['rpid'].unique():
    # Load data:
    data_i = pd.read_csv(
                TWITTER_DATA_PATH + '/' + rpid_i + '_tweets.tsv',
                encoding="mbcs",
                quotechar='"', delimiter="\t", engine='python')
    # Keep only relevant columns:
    data_i = data_i[['text', 'datetime']]
    # Define regular expression for the company's cashtag:
    cashtag_regex_i = '|'.join(r'([$]{1}\b' + company_mapping.loc[company_mapping['rpid'] == rpid_i, 'taq_ticker'] + r'\b)')
    # Count number of company cashtags:
    data_i['num_companycashtag'] = data_i['text'].map(lambda x: len(re.findall(cashtag_regex_i, x)))
    # Count the number of all cashtags:
    data_i['num_cashtag'] = data_i['text'].map(lambda x: len(re.findall(r'[$]\b[a-zA-z]+\b', x)))
    # If wanted, remove tweets that do not mention the company's cashtag:
    if has_cashtag:
        data_i = data_i.loc[data_i['num_companycashtag'] > 0]
    # If wanted, remove tweets that mention other cashtags:
    if unique_cashtag:
        data_i = data_i.loc[data_i['num_cashtag'] == data_i['num_companycashtag']]
    # Transform strings to timestamps:
    data_i['datetime'] = \
        data_i['datetime'].map(lambda x: pd.Timestamp(x).tz_localize(tz=tz_zurich_in_twitter, ambiguous=True))
    # Change timezone to Eastern Time:
    data_i['datetime_ET'] = data_i['datetime'].map(lambda x: x.astimezone(tz_est_out))
    # Shift time depending on the aggregation scheme chosen previously:
    data_i['datetime_ET_shifted'] = \
        data_i['datetime_ET'].map(lambda x: x + datetime.timedelta(hours=hours_shift, minutes=minutes_shift))
    # Define date based on the shifted ET timestamp:
    data_i['Date'] = data_i['datetime_ET_shifted'].dt.date
    # Encode text data:
    try:
        texts = [unicode(x) for x in data_i['text']]
    except UnicodeDecodeError:
        texts = [x.decode('utf-8') for x in data_i['text']]
    X = st.get_test_sentences(texts)
    pred_sentiment = model.predict(X, model_specs['batch_size'])
    data_i['DeepMoji'] = [(int(x > 0.5) - 0.5)*2 for x in pred_sentiment.flatten()]
    # For the aggregation, we shift the date of messages posted during holidays or weekends to the next trading day:
    data_i = data_i.merge(closing_info, how='left', on='Date')[['Date', 'Closed', 'DeepMoji']]
    while any(data_i.Closed):
        data_i['Date'] = data_i.apply(lambda x: x['Date'] + datetime.timedelta(days=1) if x['Closed'] else x['Date'], axis=1)
        data_i = data_i.drop('Closed', axis=1).merge(closing_info, how='left', on='Date')
    # Aggregate sentiments on a daily basis:
    sentiment_i = data_i.drop('Closed', axis=1).groupby('Date').aggregate({'DeepMoji': [bull, np.mean, len]})
    # Delete the raw data:
    del data_i
    # Transform multi-index column names to single level:
    sentiment_i.columns = ['_'.join(col).strip() for col in sentiment_i.columns.values]
    # Date (which acts as an index) to a column:
    sentiment_i.reset_index(level=0, inplace=True)
    # Add information about RavenPack ID:
    sentiment_i['rpid'] = rpid_i
    # Append data:
    sentiment_twitter = sentiment_twitter.append(sentiment_i, ignore_index=True)
    # Remove the sentiment data:
    del sentiment_i

# Save aggregated Twitter sentiment:
sentiment_twitter.to_csv(PROCESSED_DATA_PATH + '/Twitter_DeepMoji_daily_' + file_specifications + '.csv')