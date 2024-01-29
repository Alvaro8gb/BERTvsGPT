import os
import sys

__FOLDS_PATH = str(sys.argv[1]) 
__MODEL_PATH = str(sys.argv[2])

# Comprobar o crear directorios

config =  __MODEL_PATH + "/config.cfg"

for fold in os.listdir(__FOLDS_PATH):

    fold_path = __FOLDS_PATH + "/" + fold
    model_path = __MODEL_PATH + "/model_" + fold

    try:
        os.mkdir(model_path)
        output = " --output " + model_path
        train = " --paths.train " + fold_path + "/train.spacy"
        dev = " --paths.dev " + fold_path + "/dev.spacy"
        args = "-g 0 " + config + output + train + dev
        print(args)
        os.system("python -m spacy train " + args)

    except FileExistsError:
        print(model_path, " this fold exists, no overwrtie")