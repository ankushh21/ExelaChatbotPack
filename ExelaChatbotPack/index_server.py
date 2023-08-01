import os
import shutil

from llama_index import SimpleDirectoryReader
from dotenv import load_dotenv

load_dotenv('.env')
from multiprocessing import Lock
from typing import List
from langchain.document_loaders import (
    OnlinePDFLoader,
    CSVLoader,
    EverNoteLoader,
    TextLoader,
    UnstructuredEmailLoader,
    UnstructuredEPubLoader,
    UnstructuredHTMLLoader,
    UnstructuredMarkdownLoader,
    UnstructuredPowerPointLoader,
    UnstructuredWordDocumentLoader
)
from langchain.text_splitter import CharacterTextSplitter
from langchain.docstore.document import Document
from .chroma_service import ChromaService

index = None
lock = Lock()

chroma_service_obj = ChromaService()


# Custom document loaders
class MyElmLoader(UnstructuredEmailLoader):
    """Wrapper to fallback to text/plain when default does not work"""

    def load(self) -> List[Document]:
        """Wrapper adding fallback for elm without html"""
        try:
            try:
                doc = UnstructuredEmailLoader.load(self)
            except ValueError as e:
                if 'text/html content not found in email' in str(e):
                    # Try plain text
                    self.unstructured_kwargs["content_source"] = "text/plain"
                    doc = UnstructuredEmailLoader.load(self)
                else:
                    raise
        except Exception as e:
            # Add file_path to exception message
            raise type(e)(f"{self.file_path}: {e}") from e

        return doc


# Map file extensions to document loaders and their arguments
LOADER_MAPPING = {
    ".csv": (CSVLoader, {}),
    ".doc": (UnstructuredWordDocumentLoader, {}),
    ".docx": (UnstructuredWordDocumentLoader, {}),
    ".enex": (EverNoteLoader, {}),
    ".eml": (MyElmLoader, {}),
    ".epub": (UnstructuredEPubLoader, {}),
    ".html": (UnstructuredHTMLLoader, {}),
    ".md": (UnstructuredMarkdownLoader, {}),
    ".pdf": (OnlinePDFLoader, {}),
    ".ppt": (UnstructuredPowerPointLoader, {}),
    ".pptx": (UnstructuredPowerPointLoader, {}),
    ".txt": (TextLoader, {"encoding": "utf8"}),
}


def query_index(query, chat_history=[]):
    try:
        return chroma_service_obj.query_index(query, chat_history)
    except Exception as ex:
        return "Internal Error Occurred"


def delete_index():
    try:
        # shutil.rmtree(os.environ.get("PERSIST_DIRECTORY"), ignore_errors=True)
        # shutil.rmtree(os.environ.get("PERSIST_DIRECTORY"), ignore_errors=True)
        for root, dirs, files in os.walk(os.environ.get("PERSIST_DIRECTORY"), topdown=False):
            for name in files:
                os.remove(os.path.join(root, name))
            for name in dirs:
                os.rmdir(os.path.join(root, name))
        chroma_service_obj.reset_vector_store_object()
        return True
    except OSError as error:
        return False


def insert_into_index(doc_file_path, doc_id=None):
    try:
        document = SimpleDirectoryReader(input_files=[doc_file_path]).load_data()[0]
        if doc_id is not None:
            document.doc_id = doc_id
        ext = "." + doc_file_path.rsplit(".", 1)[-1]
        if ext.lower() not in LOADER_MAPPING:
            raise Exception(f"Unsupported file extension '{ext.lower()}'")
        else:
            loader_class, loader_args = LOADER_MAPPING[ext.lower()]
            loader = loader_class(doc_file_path, **loader_args)
            documents = loader.load()
            text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0, separator='\n')
            texts = text_splitter.split_documents(documents)
        with lock:
            chroma_service_obj.insert_into_index(texts)
        return True
    except Exception as ex:
        return False
