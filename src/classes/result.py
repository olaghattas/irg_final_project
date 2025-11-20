from pydantic import BaseModel, ConfigDict

class Result(BaseModel):
    docid: str
    queryid: str
    score: float
    method: str