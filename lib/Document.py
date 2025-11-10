from pydantic import BaseModel, ConfigDict

from utils import get_dataset
import os


class Document(BaseModel):
    id:str
    contents:str

def get_corpus():
    return get_dataset("corpus_clean")

