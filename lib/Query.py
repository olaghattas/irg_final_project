from pydantic import BaseModel, ConfigDict

class Query(BaseModel):
    contents: str    