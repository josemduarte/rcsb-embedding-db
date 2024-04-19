import os
import random

import pandas as pd
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

from src.db import init_db_collection
from src.utils import img_url, alignment_url

embedding_path = os.environ.get("RCSB_EMBEDDING_PATH")
if not embedding_path or not os.path.isdir(embedding_path):
    raise Exception(f"Embedding path {embedding_path} is not available")

collection = init_db_collection(embedding_path)
app = FastAPI()
templates = Jinja2Templates(directory="src/templates")


@app.get("/search_chains/{asym_id}", response_class=HTMLResponse)
async def search_chain(request: Request, asym_id: str, n_results: int = 100):

    if not os.path.isfile(f"{embedding_path}/{asym_id}.csv"):
        random_id = ".".join(random.choice(os.listdir(embedding_path)).split(".")[0:2])
        context = {"asym_id": asym_id, "search_id": random_id}
        return templates.TemplateResponse(
            request=request, name="null-instance.html.jinja", context=context
        )

    ch_embedding = list(pd.read_csv(f"{embedding_path}/{asym_id}.csv").iloc[:, 0].values)
    result = collection.query(
        query_embeddings=[ch_embedding],
        n_results=int(n_results)
    )
    results = [
        {
            "index": idx,
            "instance_id": x,
            "img_url": img_url(x),
            "alignment_url": alignment_url(asym_id, x),
            "score": y
        } for idx, (x, y) in enumerate(zip(result['ids'][0], result['distances'][0]))
    ]
    context = {"search_id": asym_id, "results": results}
    return templates.TemplateResponse(
        request=request, name="search.html.jinja", context=context
    )


@app.get("/", response_class=HTMLResponse)
@app.get("/search_chains", response_class=HTMLResponse)
async def form(request: Request):
    random_id = ".".join(random.choice(os.listdir(embedding_path)).split(".")[0:2])
    context = {"search_id": random_id}
    return templates.TemplateResponse(
        request=request, name="index.html.jinja", context=context
    )
