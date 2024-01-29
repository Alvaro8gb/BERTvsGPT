from spacy.tokens import DocBin
import numpy as np
import json

def load_dataset(path, model):
    doc_bin = DocBin().from_disk(path)
    docs = list(doc_bin.get_docs(model.vocab))
    return np.array(docs, dtype=object)

def write_train_test(path, train, test, train_stats, test_stats):
    doc_bin = DocBin(docs=train)
    doc_bin.to_disk(path + "train.spacy")
    doc_bin = DocBin(docs=test)
    doc_bin.to_disk(path +"dev.spacy")

    trs_f = open(path + "train_stats.json", "w")
    trs_f.write(json.dumps(train_stats, indent=4))
    trs_f.close()

    tes_f = open(path + "test_stats.json", "w")
    tes_f.write(json.dumps(test_stats, indent=4))
    tes_f.close()

def get_train_stats(train):
    train_stats = {"train_stats": {}}
    n_ents_train = 0
    for doc in train:
        for ent in doc.ents:
            if train_stats["train_stats"].get(ent.label_) == None:
                train_stats["train_stats"][ent.label_] = 1
            else:
                train_stats["train_stats"][ent.label_] += 1
            n_ents_train += 1

    for key in train_stats["train_stats"].keys():
        perc = (train_stats["train_stats"][key] / n_ents_train) * 100
        train_stats["train_stats"][key] = (train_stats["train_stats"][key], str(round(perc, ndigits=2)) + "%")

    return {"train_stats": {key: value for key, value in sorted(train_stats["train_stats"].items())}}

def get_test_stats(test):
    test_stats = {"test_stats": {}}
    n_ents_test = 0
    for doc in test:
        for ent in doc.ents:
            if test_stats["test_stats"].get(ent.label_) == None:
                test_stats["test_stats"][ent.label_] = 1
            else:
                test_stats["test_stats"][ent.label_] += 1
            n_ents_test += 1

    for key in test_stats["test_stats"].keys():
        perc = (test_stats["test_stats"][key] / n_ents_test) * 100
        test_stats["test_stats"][key] = (test_stats["test_stats"][key], str(round(perc, ndigits=2)) + "%")

    return {"test_stats": {key: value for key, value in sorted(test_stats["test_stats"].items())}}
