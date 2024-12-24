from langchain.document_loaders import PyPDFLoader,DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.docstore.document import Document


def load_pdf_file(data):
    loader = DirectoryLoader(data,glob="*.pdf",loader_cls=PyPDFLoader)
    documents = loader.load()

    return documents


def text_split(extracted):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap = 40)
    doc_chunks = text_splitter.split_documents(extracted)
    return doc_chunks


def download_hugging_face_embeddings():
    embeddings = HuggingFaceEmbeddings(model_name = 'all-MiniLM-L6-v2')
    return embeddings