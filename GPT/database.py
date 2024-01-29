import pandas as pd
from GPT.models import Message

from pydantic import BaseModel
from typing import List

from spacy.language import Language
from spacy.tokens import DocBin, Doc

class BasePromt(BaseModel):
    id: int
    msg: Message
    description: str
    n_tokens:int


def load_promts(path: str) -> List[BasePromt]:

    df = pd.read_excel(path)

    dict_prompts = {
        row.id: BasePromt(
            id=row.id,
            description=row.description,
            msg=Message(role="system", content=row.prompt),
            n_tokens=len(row.prompt.split())
        )
        for row in df.itertuples(index=False)
    }
    
    return dict_prompts


def load_dataset(model:Language, path:str):
    try:
        doc_bin = DocBin().from_disk(path)
        docs = list(doc_bin.get_docs(model.vocab))
        return docs
    except FileNotFoundError:
        print(f"The file {path} was not found.")
        return None
    
def dump_dataset(docs:list[Doc], path:str):
    doc_bin = DocBin()

    for doc in docs:
        doc_bin.add(doc)

    with open(path, "wb") as archivo:
        archivo.write(doc_bin.to_bytes())


if __name__ == "__main__":
    dict_promts = load_promts("promts.xlsx")
    print(dict_promts)
