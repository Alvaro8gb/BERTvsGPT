import sys
import os
import utils as util
import spacy
from sklearn.model_selection import KFold

__K = int(sys.argv[1])
__MODEL_PATH = "es_core_news_lg"
__DATASET_PATH = "breast_oswaldo.spacy"

print("__K=", str(__K))

kf = KFold(n_splits=__K, random_state=8, shuffle=True)

print("k-fold info: " + str(kf))

folds_path = "folds_" + str(__K)
os.mkdir(folds_path)

# Load the model and obtain a list of docs from the .spacy file
model = spacy.load(__MODEL_PATH)
docs = util.load_dataset(__DATASET_PATH, model)

i = 1
for train_i, test_i in kf.split(docs):
    train = docs[train_i]
    test = docs[test_i]

    # Obtain statistics for the training docs:
    train_stats = util.get_train_stats(train)

    # Obtain statistics for the test docs:
    test_stats = util.get_test_stats(test)

    # Create the i-th fold:
    fold_name = "fold_" + str(i) + "/"
    fold_path = folds_path + "/" + fold_name
    os.mkdir(fold_path)

    # Add the 4 files (train, test, train_stats, test_stats) to the i-th fold:
    util.write_train_test(fold_path, train, test, train_stats, test_stats)

    i += 1
