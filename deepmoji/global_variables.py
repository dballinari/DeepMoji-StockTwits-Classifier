""" Global variables.
"""
import tempfile
from os.path import abspath, dirname

# The ordering of these special tokens matter
# blank tokens can be used for new purposes
# Tokenizer should be updated if special token prefix is changed
SPECIAL_PREFIX = 'CUSTOM_'
SPECIAL_TOKENS = ['CUSTOM_MASK',
                  'CUSTOM_UNKNOWN',
                  'CUSTOM_AT',
                  'CUSTOM_URL',
                  'CUSTOM_NUMBER',
                  'CUSTOM_BREAK']
SPECIAL_TOKENS.extend(['{}BLANK_{}'.format(SPECIAL_PREFIX, i) for i in range(6, 10)])

ROOT_PATH = dirname(dirname(abspath(__file__)))
VOCAB_PATH = '{}/model/vocabulary.json'.format(ROOT_PATH)
PRETRAINED_PATH = '{}/model/deepmoji_weights.hdf5'.format(ROOT_PATH)

WEIGHTS_DIR = tempfile.mkdtemp()

NB_TOKENS = 50000
NB_EMOJI_CLASSES = 64
FINETUNING_METHODS = ['last', 'full', 'new', 'chain-thaw']
FINETUNING_METRICS = ['acc', 'weighted_f1']

# PROJECT_DATA_PATH = 'C:/Users/danie/Documents/Research Data/Project data/Data Project Sentiment Race'
PROJECT_DATA_PATH = 'D:/Research Data/Project data/Data Project Sentiment Race'
RAW_DATA_PATH = '{}/00_raw'.format(PROJECT_DATA_PATH)
PROCESSED_DATA_PATH = '{}/01_processed'.format(PROJECT_DATA_PATH)
MODEL_DATA_PATH = '{}/02_models'.format(PROJECT_DATA_PATH)

# ORIGINAL_DATA_PATH = 'C:/Users/danie/Documents/Research Data/Original data/'
ORIGINAL_DATA_PATH = 'D:/Research Data/Original data/'
STOCKTWITS_DATA_PATH = '{}/StockTwits SP500'.format(ORIGINAL_DATA_PATH)
TWITTER_DATA_PATH = '{}/Twitter SP500'.format(ORIGINAL_DATA_PATH)

