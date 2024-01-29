import json
import sys
import statistics as st

__BASE_PATH = str(sys.argv[1]) 
__MODEL_NAME = __BASE_PATH.split("/")[-1]
__STATS_PATH = __BASE_PATH +"/stats/" + __MODEL_NAME

performances = []
labels = []

for i in range(1,10):
    meta_file = open(__BASE_PATH + "/model_fold_"+str(i)+"/model-best/meta.json", "r")
    meta = json.loads(meta_file.read())
    meta_file.close()

    performances.append(dict(meta["performance"]))
    labels = list(meta["labels"]["ner"])

print(labels)
data_general = { i : [] for i in ["ents_f", "ents_p", "ents_r"] }
data_per_tag = { i : {"p" :[], "r":[], "f":[]} for i in labels}


for p in performances:
    for metric in ["ents_f", "ents_p", "ents_r"]:
            data_general[metric].append(p[metric])

for p in performances:
    for tag in p["ents_per_type"].keys():
        for metric in ["p", "r", "f"]:
            data_per_tag[tag][metric].append(p["ents_per_type"][tag][metric])


stats_general = { i : 0 for i in ["ents_f", "ents_p", "ents_r"] }
stats_per_tag = { i : {"p" : 0, "r": 0, "f": 0} for i in labels}

for metric in data_general.keys():
    stats_general[metric] =  ( st.mean(data_general[metric]), st.pstdev(data_general[metric]))

for tag in data_per_tag:
    for metric in ["p", "r", "f"]:
        stats_per_tag[tag][metric] = ( st.mean(data_per_tag[tag][metric]), st.pstdev(data_per_tag[tag][metric]))

print(stats_general)
print(stats_per_tag)

with open(__STATS_PATH+"_stats_general.json", 'w') as outfile:
    json.dump(stats_general, outfile, indent=4)

with open(__STATS_PATH+"_stats_per_tag.json", 'w') as outfile:
    json.dump(stats_per_tag, outfile, indent=4)

