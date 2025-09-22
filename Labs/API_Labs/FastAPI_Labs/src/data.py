from pydantic import BaseModel


class IrisData(BaseModel):
    petal_length: float
    sepal_length: float
    petal_width: float
    sepal_width: float


class IrisResponse(BaseModel):
    response: int
    class_name: str
    proba: float
