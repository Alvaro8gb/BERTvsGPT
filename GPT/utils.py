import random
import json

from spacy.tokens import Doc

import os
from tqdm import tqdm
from GPT.models import Message, EvalInput, SetEvalDocs, Promt, Sample

from GPT.gptFormat import decoder
from GPT.database import dump_dataset, load_dataset
from GPT.main import LargeLanguageModel, log_ner


### CORPUS SPLITING

def get_number_tokens(docs: list[Doc]):
    """
    Get number of tokens of a list of documents using the tokenizer used to create this model.
    """
    return sum([len(d) for d in docs])


def get_tokens_ents(docs: list[Doc], entity: str) -> (list, list):
    num_tokens_note = []
    num_labels_note = []

    for doc in tqdm(docs, total=len(docs), desc="Procesing Docs"):

        num_tokens = len([t for t in doc])
        num_labels = sum(1 for ent in doc.ents if ent.label_ == entity)

        num_tokens_note.append(num_tokens)
        num_labels_note.append(num_labels)

    return num_tokens_note, num_labels_note


def calculate_and_categorize_entities(docs_good, ENTS):
    ents_counter = {ent: get_tokens_ents(docs_good, ent) for ent in ENTS}

    notes_with_1_ents = {ent: [] for ent in ents_counter}
    notes_with_morethan1_ents = {ent: [] for ent in ents_counter}
    notes_without_ents = {ent: [] for ent in ents_counter}

    for ent in ents_counter:
        tokens, ents = ents_counter[ent]
        notes_with_1_ents[ent] = [i for i, n in enumerate(ents) if n == 1]
        notes_with_morethan1_ents[ent] = [i for i, n in enumerate(ents) if n > 1]
        notes_without_ents[ent] = [i for i, n in enumerate(ents) if n == 0]

    return notes_with_1_ents, notes_with_morethan1_ents, notes_without_ents

def select_entity_results(docs_good, LABEL, notes_with_1_ents, notes_with_morethan1_ents, notes_without_ents):
    ent_0 = [docs_good[index] for index in notes_without_ents[LABEL]]
    ent_1 = [docs_good[index] for index in notes_with_1_ents[LABEL]]
    ent_morethan1 = [docs_good[index] for index in notes_with_morethan1_ents[LABEL]]

    return ent_0, ent_1, ent_morethan1


# CORPUS LOAD AND DUMP 

folder_name = lambda path, label: path + os.path.sep + label + os.path.sep

def save_datasets(docs_dic:dict, path, label):
    path_final = folder_name(path, label)

    if not os.path.exists(path_final):
        os.makedirs(path_final)

        for name, list_docs in docs_dic.items():
            dump_dataset(list_docs, f"{path_final + name}.spacy")
    else:
        print("Folder alredy exist, no overwriting")

def load_datasets(model, path, label):

    path_final = folder_name(path, label)
    files = []
    docs_dic = {}

    files = [(file.replace(".spacy", ""), os.path.join(path_final, file))
         for file in os.listdir(path_final) if file.endswith(".spacy")]

    for name, path in files:
        docs_dic[name] = load_dataset(model, path)
        
    return docs_dic


# GPT Validation
def extract_y_true(doc, label: str):
    y_true = []
    for ent in doc.ents:
        if ent.label_ == label:
            span = doc[ent.start:ent.end]
            start_token = len(list(doc[:ent.start].text.split()))
            end_token = start_token + len(list(span.text.split()))
            y_true.append((ent.text, start_token, end_token - 1))
    return y_true


def generate_ner_gpt(docs: list[Doc], label: str):

    return [EvalInput(input_task=Message(role="user", content=doc.text), y_true=extract_y_true(doc, label))
            for doc in tqdm(docs, total=len(docs), desc="Creating promts")]


def srs(population: list, sample_size: int, seed=8):
    """
    Simple Random Sample
    """
    random.seed(seed)

    sample = random.sample(population, sample_size)

    return sample


def select_notes(corpus: list[Doc], n: int, label: str):

    if n > len(corpus):
        print(
            f"There are not enough notes in the corpus. Found {len(corpus)} notes.")
        return None

    eligible_docs = [note for note in corpus if any(
        entity.label_ == label for entity in note.ents)]

    if n > len(eligible_docs):
        print(
            f"There are not enough notes with the specified entity label. Found {len(eligible_docs)} eligible notes.")
        return None

    selected_docs = srs(eligible_docs, n)
    corpus = [note for note in corpus if note not in selected_docs]

    return selected_docs, corpus


def query_openai(lm: LargeLanguageModel, eval_promts: list[EvalInput], behave: Message, samples: list[Sample] = []):
    money = 0
    results = []  # [(y_true, y_pred)]
    for promt in tqdm(eval_promts, desc="Quering OpenAI"):

        result = log_ner(lm, behave, promt.input_task, samples)

        if result: # If dont ERRROR
            y_pred = decoder(result.answer)
            y_true = promt.y_true
            results.append([y_pred, y_true])
            money += result.price_total

    return results, money


def eval_pipline(llms: list[LargeLanguageModel], set_evals: list[SetEvalDocs], promts: list[Promt], label: str):

    pipline_results = {m.model.name: {c["name"]: {
        p.name: None for p in promts} for c in set_evals} for m in llms}
    print("Print starting test for:", pipline_results)

    for model in llms:
        model_name = model.model.name
        print("Quering with ", model_name)
        for c in set_evals:
            print("Using", c["name"])
            eval_promts = generate_ner_gpt(c["docs"], label)

            for p in promts:
                print("With promt", p.name)

                results, money = query_openai(
                    model, eval_promts, p.behave, p.samples)
                pipline_results[model_name][c["name"]][p.name] = {
                    "money": money, "results": results}
                

    return pipline_results


## LOAD AND DUMP RESULTS

def dump_results(results, path):
    with open(path, 'w') as archivo:
        json.dump(results, archivo)


def load_results(path):
    results = None
    list2tuple = lambda list: tuple(list)

    with open(path, 'r') as archivo:
        results = json.load(archivo)

    for m, sets in results.items():
        for s, types_promt in sets.items():
            for promt, info in types_promt.items():
                casted_results = [ [list(map(list2tuple, y_pred)), list(map(list2tuple, y_true)) ]for y_pred, y_true  in info["results"] ]
                info["results"] = casted_results

    return results
