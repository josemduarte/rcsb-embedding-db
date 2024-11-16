import os
import random

import pandas as pd
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates

from embedding_provider import EmbeddingProvider
from src.utils import img_url, alignment_url, arches_score


app = FastAPI()
collection_name = 'af_embeddings'
templates = Jinja2Templates(directory="src/templates")

embedding_provider = EmbeddingProvider(collection_name)


@app.get("/embedding_search/{rcsb_id}", response_class=HTMLResponse)
async def search_chain(
        request: Request,
        rcsb_id: str,
        granularity: str = "chain",
        n_results: int = 100,
        include_csm: bool = False
):

    rcsb_embedding = embedding_provider.get_by_id(rcsb_id)
    result = {}
    results = [
        {
            "index": idx,
            "instance_id": x,
            "img_url": img_url(x),
            "alignment_url": alignment_url(rcsb_id, x),
            "score": y
        } for idx, (x, y) in enumerate(zip(result['ids'][0], result['distances'][0]))
    ]

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
