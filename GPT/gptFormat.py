import re
from spacy.tokens import Doc

def insert_around(string, start_index, end_index, offset=0):
    start = start_index + offset
    string = string[:start] + '@@' + string[start:]
    end = end_index  + offset + 2 

    string = string[:end] + '##' + string[end:]

    return string

def encoder(doc:Doc, label)-> str:
    
    text = str(doc)
    offset = 0
    
    for ent in doc.ents:
        if ent.label_ == label:
            text = insert_around(text, ent.start_char, ent.end_char, offset)
            offset+=4

    return text


def delete_format(word):
    """
    The regular expression matches any character that is not a word character (w) or a blank space (s).
    """
    word = re.sub(r'[^\w\s]', '', word)

    return word

def decoder(text_formated:str):
    tokens = text_formated.strip().split() ## the tokenizer split in white spaces only
    
    y_true = []
    
    start = None
    ent =  []
    flag = False

    for i, t in enumerate(tokens):
        if bool(re.search("@@", t)):
            if bool(re.search("##", t)):
                y_true.append((delete_format(t), i, i))
            else:
                flag = True
                start = i
                ent = [delete_format(t)]
  
        elif bool(re.search("##", t)):
            ent.append(delete_format(t))
            y_true.append((" ".join(ent), start, i))
            flag = False

        elif flag:
            ent.append(t)

    return y_true

def normalizer(text:str)->list[str]:
    return text.strip().split()
