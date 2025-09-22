from fastapi import FastAPI, HTTPException
from .data import IrisData, IrisResponse
from .predict import load_model, make_features, predict_class

app = FastAPI(title="API Labs - Section 1")


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/predict", response_model=IrisResponse)
async def predict(data: IrisData):
    try:
        model = load_model()
    except FileNotFoundError as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    features = make_features(
        petal_length=data.petal_length,
        sepal_length=data.sepal_length,
        petal_width=data.petal_width,
        sepal_width=data.sepal_width,
    )

    response = predict_class(model, features)
    return IrisResponse(response=response)
