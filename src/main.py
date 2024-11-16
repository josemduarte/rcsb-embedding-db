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
    if not rcsb_embedding:
        rcsb_embedding = embedding_provider.get_by_id("AF_AFA0A009E3R5F1")

    search_result = embedding_provider.get_by_embedding(rcsb_embedding)
    results = [
        {
            "index": idx,
            "instance_id": r.id,
            "img_url": img_url(r.id),
            "alignment_url": alignment_url(rcsb_id, r.id),
            "score": r.distance
        } for idx, r in enumerate(search_result[0])
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
    random_id = "AF_AFA0A009E3R5F1"
    context = {"search_id": random_id, "request": request}
    return templates.TemplateResponse(
        name="index.html.jinja", context=context
    )


def ready_results(results, threshold_set):
    if len(results) == 0:
        return False
    if results[len(results)-1]['distances'] < threshold_set:
        return False
    return True
