import os
import numpy as np

from create_model import create_model
from fit_model import fit_model
from plot_examples import plot_examples
from create_submission import create_submission
from create_train_data import create_train_data
from process_test_data import process_test_data

TRAIN_DIR = 'F:\Studia\Semestr 4\Python\Cats vs Dogs recognition/train'
TEST_DIR = 'F:\Studia\Semestr 4\Python\Cats vs Dogs recognition/test'
IMG_SIZE = 50
LR = 5e-4
EPOCH_NR = 5
RETRAIN_FLAG = 0

MODEL_NAME = 'dogsvscats-{}-{}.model'.format(LR, '2conv-basic') # just so we remember which saved model is which, sizes must match

train_data = create_train_data(TRAIN_DIR, IMG_SIZE)
# If you have already created the dataset:
#train_data = np.load('train_data.npy')

model = create_model(IMG_SIZE, LR)

# use pre-trained model or train one it if it is new
if os.path.exists('{}.meta'.format(MODEL_NAME)) and RETRAIN_FLAG == 0:
    model.load(MODEL_NAME)
    print('model loaded!')
else:
    fit_model(model, EPOCH_NR, MODEL_NAME, train_data, IMG_SIZE)
    model.save(MODEL_NAME)

# if you need to create the data:
test_data = process_test_data(TEST_DIR, IMG_SIZE)
# if you already have some saved:
#test_data = np.load('test_data.npy')

plot_examples(test_data, IMG_SIZE, model)

create_submission(test_data, IMG_SIZE, model)