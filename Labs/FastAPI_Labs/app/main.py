from fastapi import FastAPI, HTTPException
from src.data import IrisData, IrisResponse
from src.predict import load_model, make_features, predict_class
from monitoring import add_monitoring_middleware, add_health_endpoints

app = FastAPI(title="FastAPI Labs")

# Add monitoring
add_monitoring_middleware(app)
add_health_endpoints(app)


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

    predicted_class, class_name, proba = predict_class(model, features)
    return IrisResponse(response=predicted_class, class_name=class_name, proba=proba)
