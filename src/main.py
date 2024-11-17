import os
import random

import pandas as pd
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates

from src.embedding_provider import EmbeddingProvider
from src.utils import img_url, alignment_url, arches_score


app = FastAPI()
templates = Jinja2Templates(directory="src/templates")


instance_collection = "instance_embeddings"
assembly_collection = "assembly_embeddings"
embedding_path = "/mnt/vdb1/embedding-34466681"


@app.get("/embedding_search/{rcsb_id}", response_class=HTMLResponse)
async def search_chain(
        request: Request,
        rcsb_id: str,
        granularity: str = "chain",
        n_results: int = 100,
        include_csm: bool = False
):

    collection_name = assembly_collection if granularity == "assembly" else instance_collection
    embedding_provider = EmbeddingProvider(collection_name)
    rcsb_embedding = embedding_provider.get_by_id(rcsb_id)
    if not rcsb_embedding:
        random_id = ".".join(random.choice(os.listdir(embedding_path)).split(".")[0:2])
        context = {"rcsb_id": rcsb_id, "search_id": random_id, "request": request}
        return templates.TemplateResponse(
            name="null-instance.html.jinja", context=context
        )

    search_result = embedding_provider.get_by_embedding(rcsb_embedding)
    results = [
        {
            "index": idx,
            "instance_id": r.id,
            "alignment_url": alignment_url(rcsb_id, r.id),
            "img_url": img_url(r.id),
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
    random_id = ".".join(random.choice(os.listdir(embedding_path)).split(".")[0:2])
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
