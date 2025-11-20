from pydantic import BaseModel, ConfigDict

from src.utils import get_dataset


class Document(BaseModel):
    model_config = ConfigDict(strict=True)
    
    id:str
    contents:str

def get_corpus():
    return get_dataset("corpus_clean")

