import os
import fnmatch
import uuid
import torch
import logging
import time
from dataclasses import dataclass
from typing import List, Dict, Optional
from pathlib import Path

from qdrant_client import QdrantClient
from langchain_qdrant import QdrantVectorStore
from langchain_huggingface import HuggingFaceEmbeddings
from qdrant_client.http.models import Distance, VectorParams, Filter, FieldCondition, MatchValue
from langchain.text_splitter import RecursiveCharacterTextSplitter

from langchain_community.document_loaders import (
    TextLoader,
    CSVLoader,
    Docx2txtLoader,
    UnstructuredExcelLoader,
    PyMuPDFLoader,
    UnstructuredPowerPointLoader,
)

from storage import MinimaStore, IndexingStatus

logger = logging.getLogger(__name__)


@dataclass
class Config:
    EXTENSIONS_TO_LOADERS = {
        ".pdf": PyMuPDFLoader,
        ".pptx": UnstructuredPowerPointLoader,
        ".ppt": UnstructuredPowerPointLoader,
        ".xls": UnstructuredExcelLoader,
        ".xlsx": UnstructuredExcelLoader,
        ".docx": Docx2txtLoader,
        ".doc": Docx2txtLoader,
        ".txt": TextLoader,
        ".md": TextLoader,
        ".csv": CSVLoader,
    }
    
    DEVICE = torch.device(
        "mps" if torch.backends.mps.is_available() else
        "cuda" if torch.cuda.is_available() else
        "cpu"
    )
    
    START_INDEXING = os.environ.get("START_INDEXING")
    LOCAL_FILES_PATH = os.environ.get("LOCAL_FILES_PATH")
    CONTAINER_PATH = os.environ.get("CONTAINER_PATH")
    QDRANT_COLLECTION = "mnm_storage"
    QDRANT_BOOTSTRAP = "qdrant"
    EMBEDDING_MODEL_ID = os.environ.get("EMBEDDING_MODEL_ID")
    EMBEDDING_SIZE = os.environ.get("EMBEDDING_SIZE")
    
    CHUNK_SIZE = 500
    CHUNK_OVERLAP = 200

class Indexer:
    def __init__(self):
        self.config = Config()
        self.qdrant = self._initialize_qdrant()
        self.embed_model = self._initialize_embeddings()
        self.document_store = self._setup_collection()
        self.text_splitter = self._initialize_text_splitter()

    # def _container_to_local(self, path: str) -> str:
    #     try:
    #         container_prefix = (self.config.CONTAINER_PATH or "").rstrip('/') + '/'
    #         local_prefix = (self.config.LOCAL_FILES_PATH or "").rstrip('/') + '/'
    #         if path.startswith(container_prefix):
    #             return path.replace(container_prefix, local_prefix, 1)
    #         return path
    #     except Exception:
    #         return path

    def _initialize_qdrant(self) -> QdrantClient:
        return QdrantClient(host=self.config.QDRANT_BOOTSTRAP)

    def _initialize_embeddings(self) -> HuggingFaceEmbeddings:
        return HuggingFaceEmbeddings(
            model_name=self.config.EMBEDDING_MODEL_ID,
            model_kwargs={'device': self.config.DEVICE},
            encode_kwargs={'normalize_embeddings': False}
        )

    def _initialize_text_splitter(self) -> RecursiveCharacterTextSplitter:
        return RecursiveCharacterTextSplitter(
            chunk_size=self.config.CHUNK_SIZE,
            chunk_overlap=self.config.CHUNK_OVERLAP
        )

    def _setup_collection(self) -> QdrantVectorStore:
        if not self.qdrant.collection_exists(self.config.QDRANT_COLLECTION):
            self.qdrant.create_collection(
                collection_name=self.config.QDRANT_COLLECTION,
                vectors_config=VectorParams(
                    size=self.config.EMBEDDING_SIZE,
                    distance=Distance.COSINE
                ),
            )
        self.qdrant.create_payload_index(
            collection_name=self.config.QDRANT_COLLECTION,
            field_name="file_path",
            field_schema="keyword"
        )
        # try:
        #     self.qdrant.create_payload_index(
        #         collection_name=self.config.QDRANT_COLLECTION,
        #         field_name="metadata.file_path",
        #         field_schema="keyword"
        #     )
        # except Exception as e:
        #     logger.debug(f"create_payload_index(metadata.file_path) skipped/failed: {e}")
        self.qdrant.create_payload_index(
            collection_name=self.config.QDRANT_COLLECTION,
            field_name="file_name",
            field_schema="keyword"
        # )
        # # Top-level filename index for direct filtering (e.g., ChatGPT curl)
        # try:
        #     self.qdrant.create_payload_index(
        #         collection_name=self.config.QDRANT_COLLECTION,
        #         field_name="file_name",
        #         field_schema="keyword"
        #     )
        # except Exception as e:
        #     logger.debug(f"create_payload_index(file_name) skipped/failed: {e}")
        return QdrantVectorStore(
            client=self.qdrant,
            collection_name=self.config.QDRANT_COLLECTION,
            embedding=self.embed_model,
        )

    def _create_loader(self, file_path: str):
        file_extension = Path(file_path).suffix.lower()
        loader_class = self.config.EXTENSIONS_TO_LOADERS.get(file_extension)
        
        if not loader_class:
            raise ValueError(f"Unsupported file type: {file_extension}")
        
        return loader_class(file_path=file_path)

    def _process_file(self, loader) -> List[str]:
        try:
            documents = loader.load_and_split(self.text_splitter)
            if not documents:
                logger.warning(f"No documents loaded from {loader.file_path}")
                return []

            for doc in documents:
                doc.metadata['file_path'] = loader.file_path
                try:
                    file_name = Path(loader.file_path).name
                    doc.metadata['file_name'] = file_name
                except Exception:
                    pass

            uuids = [str(uuid.uuid4()) for _ in range(len(documents))]
            ids = self.document_store.add_documents(documents=documents, ids=uuids)
            
            logger.info(f"Successfully processed {len(ids)} documents from {loader.file_path}")
            # Also set top-level payload keys so external tools can filter by 'file_name' directly
            try:
                file_name = Path(loader.file_path).name
                top_payload = {
                    "file_path": loader.file_path,
                    "file_name": file_name,
                }
                self.qdrant.set_payload(
                    collection_name=self.config.QDRANT_COLLECTION,
                    payload=top_payload,
                    points=ids,
                )
            except Exception as e:
                logger.warning(f"Failed to set top-level payload keys for {loader.file_path}: {e}")
            return ids
            
        except Exception as e:
            logger.error(f"Error processing file {loader.file_path}: {str(e)}")
            return []

    def index(self, message: Dict[str, any]) -> None:
        start = time.time()
        path, file_id, last_updated_seconds = message["path"], message["file_id"], message["last_updated_seconds"]
        logger.info(f"Processing file: {path} (ID: {file_id})")
        indexing_status: IndexingStatus = MinimaStore.check_needs_indexing(fpath=path, last_updated_seconds=last_updated_seconds)
        if indexing_status != IndexingStatus.no_need_reindexing:
            logger.info(f"Indexing needed for {path} with status: {indexing_status}")
            try:
                if indexing_status == IndexingStatus.need_reindexing:
                    logger.info(f"Removing {path} from index storage for reindexing")
                    self.remove_from_storage(files_to_remove=[path])
                loader = self._create_loader(path)
                ids = self._process_file(loader)
                if ids:
                    logger.info(f"Successfully indexed {path} with IDs: {ids}")
            except Exception as e:
                logger.error(f"Failed to index file {path}: {str(e)}")
        else:
            logger.info(f"Skipping {path}, no indexing required. timestamp didn't change")
        end = time.time()
        logger.info(f"Processing took {end - start} seconds for file {path}")

    def purge(self, message: Dict[str, any]) -> None:
        existing_file_paths: list[str] = message["existing_file_paths"]
        files_to_remove = MinimaStore.find_removed_files(existing_file_paths=set(existing_file_paths))
        if len(files_to_remove) > 0:
            logger.info(f"purge processing removing old files {files_to_remove}")
            self.remove_from_storage(files_to_remove)
        else:
            logger.info("Nothing to purge")

    def remove_from_storage(self, files_to_remove: list[str]):
        filter_conditions = Filter(
            must=[
                FieldCondition(
                    key="file_path",
                    match=MatchValue(value=fpath)
                )
                for fpath in files_to_remove
            ]
        )
        response = self.qdrant.delete(
            collection_name=self.config.QDRANT_COLLECTION,
            points_selector=filter_conditions,
            wait=True
        )
        logger.info(f"Delete response for {len(files_to_remove)} for files: {files_to_remove} is: {response}")
    
    def find(self, query: str) -> Dict[str, any]:
        try:
            logger.info(f"Searching for: {query}")
            found = self.document_store.search(query, search_type="similarity")
            
            if not found:
                logger.info("No results found")
                return {"links": [], "output": ""}

            links = set()
            results = []
            
            for item in found:
                path = item.get("file_path")
                links.add(f"file://{path}")
                results.append(item.page_content)

            output = {
                "links": list(links),
                "output": ". ".join(results)
            }
            
            logger.info(f"Found {len(found)} results")
            return output
            
        except Exception as e:
            logger.error(f"Search failed: {str(e)}")
            return {"error": "Unable to find anything for the given query"}

    def find_with_id(self, query: str, filters: Optional[Dict[str, any]] = None, limit: int = 10) -> Dict[str, any]:
        try:
            logger.info(f"Searching for: {query}")
            # Build an optional Qdrant filter from provided filters
            q_filter = None
            if isinstance(filters, dict):
                must_conditions = []
                should_conditions = []
                fname = filters.get('file_name')
                fpath = filters.get('file_path')
                wildcard_mode = isinstance(fname, str) and any(ch in fname for ch in ('*', '?'))
                if fname:
                    try:
                        should_conditions.append(
                            FieldCondition(key="file_name", match=MatchValue(value=str(fname)))
                        )
                    except Exception:
                        pass
                if fpath:
                    must_conditions.append(FieldCondition(key="file_path", match=MatchValue(value=fpath)))
                if must_conditions or should_conditions:
                    q_filter = Filter(must=must_conditions or None, should=should_conditions or None)

            found = []
            # If filename or filepath filter provided without a meaningful query, prefer scroll listing
            only_filtering = (q_filter is not None) and (
                not query or Path(str(filters.get('file_name', ''))).suffix.lower() in ['.pdf','.docx','.txt','.md','.csv','.pptx','.xlsx']
            )
            if isinstance(filters, dict) and isinstance(filters.get('file_name'), str) and any(ch in filters.get('file_name') for ch in ('*','?')):
                pattern = str(filters.get('file_name'))
                scanned = 0
                next_page = None
                max_scan = int(os.environ.get('MAX_WILDCARD_SCAN', '50000'))
                while True and len(found) < limit and scanned < max_scan:
                    points_batch, next_page = self.qdrant.scroll(
                        collection_name=self.config.QDRANT_COLLECTION,
                        with_payload=True,
                        with_vectors=False,
                        limit=min(512, max(1, limit * 20)),
                        offset=next_page,
                    )
                    if not points_batch:
                        break
                    for pt in points_batch:
                        scanned += 1
                        payload = getattr(pt, 'payload', {}) or {}
                        name_top = payload.get('file_name')
                        if isinstance(name_top, str) and fnmatch.fnmatch(name_top.lower(), pattern.lower()):
                            found.append(pt)
                            if len(found) >= limit:
                                break
                    if not next_page or len(found) >= limit or scanned >= max_scan:
                        break
            elif only_filtering:
                next_page = None
                while True and len(found) < limit:
                    points_batch, next_page = self.qdrant.scroll(
                        collection_name=self.config.QDRANT_COLLECTION,
                        scroll_filter=q_filter,
                        with_payload=True,
                        with_vectors=False,
                        limit=min(256, max(1, limit - len(found))),
                        offset=next_page,
                    )
                    found.extend(points_batch or [])
                    if not next_page or len(found) >= limit:
                        break
            else:
                # Vector search, optionally narrowed by filter
                emb = self.embed_model.embed_query(query)
                found = self.qdrant.search(
                    collection_name=self.config.QDRANT_COLLECTION,
                    query_vector=emb,
                    limit=limit,
                    with_payload=True,
                    with_vectors=False,
                    query_filter=q_filter,
                )

            if not found:
                logger.info("No results found")
                return {"links": [], "output": "", "results": []}

            links = set()
            results = []

            for hit in found:
                # hit.id and hit.payload come from qdrant
                hit_id = str(getattr(hit, 'id', getattr(hit, 'point_id', None)))
                payload = getattr(hit, 'payload', {}) or {}
                # tolerate different payload key names
                meta = payload.get('metadata', {}) or {}
                file_path = (
                    payload.get('file_path')
                )
                page_content = payload.get('page_content') or payload.get('text') or ''

                if file_path:
                    # path = self._container_to_local(file_path)
                    links.add(f"file://{file_path}")

                # Build result item with proper URL using LOCAL_FILES_PATH mapping
                url = None
                if file_path:
                    try:
                        local_path = file_path.replace(self.config.CONTAINER_PATH, self.config.LOCAL_FILES_PATH)
                        url = f"file://{local_path}"
                    except Exception:
                        url = f"file://{file_path}"

                results.append({
                    "id": hit_id,
                    "text": page_content,
                    "url": url,
                    "metadata": payload,
                })

            #chatGPT Connectors needs full text or a working chunks/paging solution.
            try:
                output_text = self.get_document(hit_id).get(text) #is hit_id the uuid?
            except Exception as e:
                #shortcut: Stiching it together might currently be more stable (and faster?)
                output_text = ". ".join([r.get('text', '') for r in results if r.get('text')])

            output = {
                "links": list(links),
                "output": output_text,
                "results": results,
            }

            logger.info(f"Found {len(found)} results")
            return output
            
        except Exception as e:
            logger.error(f"Search failed: {str(e)}")
            return {"error": "Unable to find anything for the given query"}

    def embed(self, query: str):
        return self.embed_model.embed_query(query)

    def list_filenames(self, limit: int = 1000, prefix: Optional[str] = None) -> List[str]:
        """Collect distinct file names from Qdrant payloads (file_name) via scroll.
        """
        try:
            names: set[str] = set()
            next_page = None
            normalized_prefix = str(prefix).lower() if prefix else None
            while len(names) < limit:
                points_batch, next_page = self.qdrant.scroll(
                    collection_name=self.config.QDRANT_COLLECTION,
                    with_payload=True,
                    with_vectors=False,
                    limit=512,
                    offset=next_page,
                )
                if not points_batch:
                    break
                for pt in points_batch:
                    payload = getattr(pt, 'payload', {}) or {}
                    meta = payload.get('metadata', {}) or {}
                    file_name = payload.get('file_name')
                    if not file_name:
                        continue
                    if normalized_prefix:
                        if str(file_name).lower().startswith(normalized_prefix):
                            names.add(str(file_name))
                    else:
                        names.add(str(file_name))
                    if len(names) >= limit:
                        break
                if not next_page or len(names) >= limit:
                    break
            return sorted(names)
        except Exception as e:
            logger.error(f"Failed to list filenames: {e}")
            return []

    def get_document(self, doc_id: str) -> Dict[str, any]:
        """
        Retrieve a single document by its Qdrant point id (as string).
        Returns the full file content (not just a single chunk) by re-loading
        the original file; falls back to aggregating all chunks from Qdrant.
        Returns a dict with id, title, text, url and metadata or an error key.
        """
        try:
            # Qdrant retrieve expects ids in a list
            points = self.qdrant.retrieve(
                collection_name=self.config.QDRANT_COLLECTION,
                ids=[doc_id],
                with_payload=True,
                with_vectors=False,
            )

            if not points:
                return {"error": "not_found"}

            p = points[0]
            payload = getattr(p, 'payload', {})
            meta = payload.get('metadata', {})
            file_path = payload.get('file_path')
            # default to the chunk text if reconstruction fails
            chunk_text = payload.get('page_content') or payload.get('text') or ''

            full_text = chunk_text
            if file_path:
                # Try to re-load the full source file via the appropriate loader
                try:
                    loader = self._create_loader(file_path)
                    docs = loader.load()  # loaders often return one per page (e.g., PDF)
                    full_text = "\n\n".join([
                        d.page_content for d in docs if getattr(d, "page_content", None)
                    ])
                except Exception as e:
                    logger.warning(
                        f"Loader-based full text reconstruction failed for {file_path}: {e}. "
                        "Falling back to Qdrant payload aggregation."
                    )
                    # Fallback: aggregate all chunks in Qdrant with the same file_path
                    try:
                        flt = Filter(
                            must=[
                                FieldCondition(
                                    key="file_path",
                                    match=MatchValue(value=file_path),
                                )
                            ]
                        )
                        # Scroll in pages to collect all chunks
                        all_parts: list[str] = []
                        next_page = None
                        while True:
                            points_batch, next_page = self.qdrant.scroll(
                                collection_name=self.config.QDRANT_COLLECTION,
                                scroll_filter=flt,
                                with_payload=True,
                                with_vectors=False,
                                limit=1024,
                                offset=next_page,
                            )
                            for pt in points_batch or []:
                                pp = getattr(pt, "payload", {}) or {}
                                txt = pp.get("page_content") or pp.get("text") or ""
                                if txt:
                                    all_parts.append(txt)
                            if not next_page:
                                break
                        if all_parts:
                            full_text = "\n\n".join(all_parts)
                    except Exception as ee:
                        logger.error(f"Failed to aggregate chunks from Qdrant for {file_path}: {ee}")

            url = None
            if file_path:
                # local_path = self._container_to_local(file_path)
                url = f"file://{file_path}"

            return {
                "id": str(getattr(p, 'id', getattr(p, 'point_id', None))),
                "title": Path(file_path).name if file_path else payload.get('title', ''),
                "text": full_text,
                "url": url,
                "metadata": payload,
            }

        except Exception as e:
            logger.error(f"Error retrieving document {doc_id}: {e}")
            return {"error": str(e)}