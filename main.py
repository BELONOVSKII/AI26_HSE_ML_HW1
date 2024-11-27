from fastapi import FastAPI, UploadFile, File, HTTPException, Depends
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, ValidationError, field_validator, Field
from typing import List, Optional, Union
import csv
from io import StringIO

import joblib
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import Ridge

from feature_processing_basic import FeaturePreprocessorBasic

app = FastAPI()


class Item(BaseModel):
    name: str
    year: int
    selling_price: int
    km_driven: int
    fuel: str
    seller_type: str
    transmission: str
    owner: str
    mileage: Union[str, float] = Field(np.nan)
    engine: Union[str, float] = Field(np.nan)
    max_power: Union[str, float] = Field(np.nan)
    torque: Union[str, float] = Field(np.nan)
    seats: Union[float] = Field(np.nan)

    @field_validator("*", mode="before")
    def ignore_empty_strings(cls, value):
        if isinstance(value, str) and value.strip() == "":
            return np.nan
        return value


@app.post("/predict_item")
def predict_item(item: Item) -> float:
    # load preprocessings and model
    preproc = joblib.load("data/preproc.joblib")
    model = joblib.load("data/model.joblib")

    x = pd.DataFrame.from_dict(item.model_dump(), orient="index").T
    y = np.exp(model.predict(preproc.transform(x)))
    return y


@app.get("/")
def read_root() -> str:
    return "App is working"


def parse_csv(file: UploadFile) -> List[Item]:

    content = file.file.read()
    decoded_content = content.decode("utf-8")
    reader = csv.DictReader(StringIO(decoded_content))

    items = []

    for row in reader:
        try:
            valid_row = Item.model_validate(row)
            items.append(valid_row.model_dump())
        except ValidationError as e:
            raise HTTPException(
                status_code=400,
                detail=f"Validation error in row: {row}. Errors: {e.errors()}",
            )
    return items


@app.post("/predict_items")
def predict_items(items: List[Item] = Depends(parse_csv)) -> StreamingResponse:
    # load preprocessings and model
    preproc = joblib.load("data/preproc.joblib")
    model = joblib.load("data/model.joblib")

    items = pd.DataFrame(items)
    y = np.exp(model.predict(preproc.transform(items)))
    items["pred"] = y

    # return the files
    csv_buffer = StringIO()
    items.to_csv(csv_buffer, index=False)
    csv_buffer.seek(0)
    # Stream the buffer as a CSV file response
    return StreamingResponse(
        csv_buffer,
        media_type="data/csv",
        headers={"Content-Disposition": "attachment; filename=dataframe.csv"},
    )
