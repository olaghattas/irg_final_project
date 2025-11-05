from pydantic import BaseModel, ConfigDict

class Document(BaseModel):
    id:str
    contents:str
