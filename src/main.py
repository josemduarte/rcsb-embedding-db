import os
import random

import pandas as pd
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates

from src.db import init_db_collection
from src.utils import img_url, alignment_url, arches_score

embedding_path = os.environ.get("RCSB_EMBEDDING_PATH")
csm_path = os.environ.get("CSM_EMBEDDING_PATH")
assembly_path = os.environ.get("RCSB_ASSEMBLY_PATH")

if not embedding_path or not os.path.isdir(embedding_path):
    raise Exception(f"Embedding path {embedding_path} is not available")

if not csm_path or not os.path.isdir(csm_path):
    raise Exception(f"Embedding path {csm_path} is not available")

if not assembly_path or not os.path.isdir(assembly_path):
    raise Exception(f"Embedding path {assembly_path} is not available")

chain_collection, csm_collection, assembly_collection = init_db_collection(embedding_path, csm_path, assembly_path)
app = FastAPI()
templates = Jinja2Templates(directory="src/templates")

_ef_search = 10000


@app.get("/embedding_search/{rcsb_id}", response_class=HTMLResponse)
async def search_chain(request: Request, rcsb_id: str, granularity: str = "chain", n_results: int = 100, include_csm: bool = False):
    if not os.path.isfile(f"{embedding_path}/{rcsb_id}.csv") and not os.path.isfile(f"{assembly_path}/{rcsb_id}.csv"):
        random_id = ".".join(random.choice(os.listdir(embedding_path)).split(".")[0:2])
        context = {"rcsb_id": rcsb_id, "search_id": random_id, "request": request}
        return templates.TemplateResponse(
            name="null-instance.html.jinja", context=context
        )

    rcsb_embedding = list(pd.read_csv(f"{embedding_path}/{rcsb_id}.csv").iloc[:, 0].values) \
        if os.path.isfile(f"{embedding_path}/{rcsb_id}.csv") \
        else list(pd.read_csv(f"{assembly_path}/{rcsb_id}.csv").iloc[:, 0].values)

    result = (assembly_collection if granularity == "assembly" else (csm_collection if include_csm else chain_collection)).query(
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

    context = {
        "search_id": rcsb_id,
        "results": results,
        "request": request,
        "granularity": granularity,
        "n_results": n_results,
        "include_csm": include_csm
    }

    return templates.TemplateResponse(
        name="search.html.jinja",
        context=context
    )


@app.get("/", response_class=HTMLResponse)
@app.get("/embedding_search", response_class=HTMLResponse)
async def form(request: Request):
    random_id = ".".join(random.choice(os.listdir(embedding_path)).split(".")[0:2])
    context = {"search_id": random_id, "request": request}
    return templates.TemplateResponse(
        name="index.html.jinja", context=context
    )


@app.get("/search/chain/{entry_id}/{asym_id}", response_class=JSONResponse)
async def search_chain(request: Request, entry_id: str, asym_id: str, tm_threshold: float = 80):
    rcsb_id = f"{entry_id}.{asym_id}"
    if not os.path.isfile(f"{embedding_path}/{rcsb_id}.csv"):
        return []
    rcsb_embedding = list(pd.read_csv(f"{embedding_path}/{rcsb_id}.csv").iloc[:, 0].values)
    result = chain_collection.query(
        query_embeddings=[rcsb_embedding],
        n_results=10000
    )
    return [
        {
            "geometry_score": arches_score(y),
            "total_score": arches_score(y),
            "rcsb_shape_container_identifiers": {
                "entry_id": x.split(".")[0],
                "asym_id": x.split(".")[1]
            },
        } for idx, (x, y) in enumerate(zip(result['ids'][0], result['distances'][0])) if arches_score(y) >= tm_threshold
    ]


@app.get("/search/assembly/{entry_id}/{assembly_id}", response_class=JSONResponse)
async def search_assembly(request: Request, entry_id: str, assembly_id: str, tm_threshold: float = 80):
    rcsb_id = f"{entry_id}-{assembly_id}"
    if not os.path.isfile(f"{assembly_path}/{rcsb_id}.csv"):
        return []
    rcsb_embedding = list(pd.read_csv(f"{assembly_path}/{rcsb_id}.csv").iloc[:, 0].values)
    result = assembly_collection.query(
        query_embeddings=[rcsb_embedding],
        n_results=10000
    )
    return [
        {
            "geometry_score": arches_score(y),
            "total_score": arches_score(y),
            "rcsb_shape_container_identifiers": {
                "entry_id": x.split("-")[0],
                "assembly_id": x.split("-")[1]
            },
        } for idx, (x, y) in enumerate(zip(result['ids'][0], result['distances'][0])) if arches_score(y) >= tm_threshold
    ]


def ready_results(results, threshold_set):
    if len(results) == 0:
        return False
    if results[len(results)-1]['distances'] < threshold_set:
        return False
    return True
