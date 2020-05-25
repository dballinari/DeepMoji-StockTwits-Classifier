from __future__ import print_function
import scripts_helper
import json
from keras.models import model_from_json
from deepmoji.model_def import deepmoji_transfer
from deepmoji.global_variables import PRETRAINED_PATH, RAW_DATA_PATH, MODEL_DATA_PATH
from deepmoji.finetuning import (
    load_benchmark,
    finetune)

dataset_path = '{}/DataDeepMojiStockTwits_finetuning.pickle'.format(RAW_DATA_PATH)
model_path = '{}/DeepMojiStockTwits_model.json'.format(MODEL_DATA_PATH)
specs_path = '{}/DeepMojiStockTwits_specs.json'.format(MODEL_DATA_PATH)
vocab_path = '{}/DeepMojiStockTwits_vocab.json'.format(MODEL_DATA_PATH)
weights_path = '{}/DeepMojiStockTwits_weights.h5'.format(MODEL_DATA_PATH)
nb_classes = 2

with open('../model/vocabulary.json', 'r') as f:
    vocab = json.load(f)

# Load dataset.
data = load_benchmark(dataset_path, vocab, extend_with=10000, save_vocab=True, path_new_vocab=vocab_path)

# Set up model and finetune
model = deepmoji_transfer(nb_classes, data['maxlen'], PRETRAINED_PATH, extend_embedding=data['added'])
model.summary()
model, acc = finetune(model, data['texts'], data['labels'], nb_classes,
                      data['batch_size'], method='chain-thaw', verbose=3, metric='acc')
print('Acc: {}'.format(acc))
# Save model weights:
model.save_weights(weights_path)
# Save model specifications:
model_specs = {'maxlen': data['maxlen'], 'batch_size': data['batch_size']}
with open(specs_path, 'w') as f:
    json.dump(model_specs, f)

