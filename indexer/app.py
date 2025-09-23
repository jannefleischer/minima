import nltk
import logging
import asyncio
from indexer import Indexer
from pydantic import BaseModel
from storage import MinimaStore
from async_queue import AsyncQueue
from fastapi import FastAPI, APIRouter
from contextlib import asynccontextmanager
from fastapi_utilities import repeat_every
from async_loop import index_loop, crawl_loop
from typing import Optional

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

indexer = Indexer()
router = APIRouter()
async_queue = AsyncQueue()
MinimaStore.create_db_and_tables()

def init_loader_dependencies():
    nltk.download('punkt')
    nltk.download('punkt_tab')
    nltk.download('wordnet')
    nltk.download('omw-1.4')
    nltk.download('punkt')
    nltk.download('averaged_perceptron_tagger_eng')

init_loader_dependencies()

class Query(BaseModel):
    query: str

class QueryWithFilters(BaseModel):
    query: str
    filters: dict | None = None
    limit: int | None = 10


@router.post(
    "/query", 
    response_description='Query local data storage',
)
async def query(request: Query):
    logger.info(f"Received query: {request.query}")
    try:
        result = indexer.find(request.query)
        logger.info(f"Found {len(result.get('results', [])) if isinstance(result, dict) else 0} results for query: {request.query}")
        logger.info(f"Results: {result}")
        return {"result": result}
    except Exception as e:
        logger.error(f"Error in processing query: {e}")
        return {"error": str(e)}


@router.post(
    "/queryid", 
    response_description='Query local data storage',
)
async def queryid(request: QueryWithFilters):
    logger.info(f"Received query: {request.query}")
    try:
        result = indexer.find_with_id(request.query, filters=request.filters, limit=(request.limit or 10))
        logger.info(f"Found {len(result.get('results', [])) if isinstance(result, dict) else 0} results for query: {request.query}")
        logger.info(f"Results: {result}")
        return {"result": result}
    except Exception as e:
        logger.error(f"Error in processing query: {e}")
        return {"error": str(e)}

@router.post(
    "/embedding", 
    response_description='Get embedding for a query',
)
async def embedding(request: Query):
    logger.info(f"Received embedding request: {request}")
    try:
        result = indexer.embed(request.query)
        logger.info(f"Found {len(result)} results for query: {request.query}")
        return {"result": result}
    except Exception as e:
        logger.error(f"Error in processing embedding: {e}")
        return {"error": str(e)}    


@router.get(
    "/document/{doc_id}",
    response_description='Get document by id'
)
async def get_document(doc_id: str):
    logger.info(f"Received document fetch for id: {doc_id}")
    try:
        result = indexer.get_document(doc_id)
        if 'error' in result:
            return {"error": result['error']}
        return {"result": result}
    except Exception as e:
        logger.error(f"Error in fetching document {doc_id}: {e}")
        return {"error": str(e)}


@router.get(
    "/filenames",
    response_description='List distinct filenames in index'
)
async def list_filenames(prefix: Optional[str] = None, limit: int = 1000):
    try:
        names = indexer.list_filenames(limit=limit, prefix=prefix)
        return {"result": names}
    except Exception as e:
        logger.error(f"Error listing filenames: {e}")
        return {"error": str(e)}

@asynccontextmanager
async def lifespan(app: FastAPI):
    tasks = [
        asyncio.create_task(crawl_loop(async_queue)),
        asyncio.create_task(index_loop(async_queue, indexer))
    ]
    await schedule_reindexing()
    try:
        yield
    finally:
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)


def create_app() -> FastAPI:
    app = FastAPI(
        openapi_url="/indexer/openapi.json",
        docs_url="/indexer/docs",
        lifespan=lifespan
    )
    app.include_router(router)
    return app

async def trigger_re_indexer():
    logger.info("Reindexing triggered")
    try:
        await asyncio.gather(
            crawl_loop(async_queue),
            index_loop(async_queue, indexer)
        )
        logger.info("reindexing finished")
    except Exception as e:
        logger.error(f"error in scheduled reindexing {e}")


@repeat_every(seconds=60*20)
async def schedule_reindexing():
    await trigger_re_indexer()

app = create_app()