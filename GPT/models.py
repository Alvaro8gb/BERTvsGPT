from typing import List, Literal, Optional
from typing_extensions import TypedDict
from pydantic import BaseModel
from spacy.tokens import Doc


Role = Literal["system", "user", "assistant"]


class Message(TypedDict):
    role: Role
    content: str


Dialog = List[Message]


class ResultOpenAI(BaseModel):
    price_total: float
    n_tokens_total: int
    price_input: float
    n_tokens_input: int
    answer: str


class Sample(BaseModel):
    user: Message
    agent: Message


class MetricsPromt(BaseModel):
    price: float
    promt_n_tokens: int
    score: float


class EvalInput(BaseModel):
    input_task: Message
    y_true: list


class Promt(BaseModel):
    behave: Message
    name: str
    samples: Optional[list[Sample]] = []


class SetEvalDocs(TypedDict):
    name: str
    docs: List