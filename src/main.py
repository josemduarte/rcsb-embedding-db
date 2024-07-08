import os
import random

import pandas as pd
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

from src.db import init_db_collection
from src.utils import img_url, alignment_url

embedding_path = os.environ.get("RCSB_EMBEDDING_PATH")
assembly_path = os.environ.get("RCSB_ASSEMBLY_PATH")
if not embedding_path or not os.path.isdir(embedding_path):
    raise Exception(f"Embedding path {embedding_path} is not available")
if not assembly_path or not os.path.isdir(assembly_path):
    raise Exception(f"Embedding path {assembly_path} is not available")

chain_collection, assembly_collection = init_db_collection(embedding_path, assembly_path)
app = FastAPI()
templates = Jinja2Templates(directory="src/templates")

_ef_search = 10000


@app.get("/embedding_search/{rcsb_id}", response_class=HTMLResponse)
async def search_chain(request: Request, rcsb_id: str, granularity: str = "chain", n_results: int = 100):
    if not os.path.isfile(f"{embedding_path}/{rcsb_id}.csv") and not os.path.isfile(f"{assembly_path}/{rcsb_id}.csv"):
        random_id = ".".join(random.choice(os.listdir(embedding_path)).split(".")[0:2])
        context = {"rcsb_id": rcsb_id, "search_id": random_id, "request": request}
        return templates.TemplateResponse(
            name="null-instance.html.jinja", context=context
        )

    rcsb_embedding = list(pd.read_csv(f"{embedding_path}/{rcsb_id}.csv").iloc[:, 0].values) \
        if os.path.isfile(f"{embedding_path}/{rcsb_id}.csv") \
        else list(pd.read_csv(f"{assembly_path}/{rcsb_id}.csv").iloc[:, 0].values)

    result = (chain_collection if granularity != "assembly" else assembly_collection).query(
        query_embeddings=[rcsb_embedding],
        n_results=n_results if n_results > _ef_search else _ef_search
    )
    results = [
        {
            "index": idx,
            "instance_id": x,
            "img_url": img_url(x),
            "alignment_url": alignment_url(rcsb_id, x),
            "score": y
        } for idx, (x, y) in enumerate(zip(result['ids'][0], result['distances'][0]))
    ]
    if n_results < _ef_search:
        results = results[0:n_results]

    context = {"search_id": rcsb_id, "results": results, "request": request}
    return templates.TemplateResponse(
        name="search.html.jinja", context=context
    )


@app.get("/", response_class=HTMLResponse)
@app.get("/embedding_search", response_class=HTMLResponse)
async def form(request: Request):
    random_id = ".".join(random.choice(os.listdir(embedding_path)).split(".")[0:2])
    context = {"search_id": random_id, "request": request}
    return templates.TemplateResponse(
        name="index.html.jinja", context=context
    )
