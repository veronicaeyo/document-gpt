import os
import hashlib
from typing import List
from decouple import config
from tempfile import _TemporaryFileWrapper
from domdf_python_tools.typing import PathLike

import warnings

warnings.simplefilter("ignore")

from result_info import ResultInfo

from langchain.docstore.document import Document
from langchain.embeddings import CohereEmbeddings
from langchain.vectorstores import Chroma
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
from langchain.chains import ConversationalRetrievalChain
from langchain.retrievers.self_query.base import SelfQueryRetriever
from langchain.chat_models import ChatOpenAI
from langchain.llms import OpenAI
from langchain.chains.query_constructor.base import AttributeInfo


llm_name: str = "gpt-3.5-turbo"


metadata_field_info: List[AttributeInfo] = [
    AttributeInfo(
        name="page",
        description="The page from the document",
        type="integer",
    ),
]

document_content_description: str = ""
base_llm = OpenAI(temperature=0, openai_api_key=config("OPENAI_API_KEY"))


embeddings = CohereEmbeddings(cohere_api_key=config("COHERE_API_KEY"))


def hash_file(file: _TemporaryFileWrapper):
    file_name = file.name
    unique_id = hashlib.sha256(file_name.encode()).hexdigest()
    return unique_id


def load_db(
    file: _TemporaryFileWrapper,
    document_content_description: str,
    k: int = 4,
    chain_type: str = "stuff",
):
    file_hash: str = hash_file(file)
    file_name: str = file.name
    persist_directory: PathLike = f"chroma/{file_hash}"

    if os.path.exists(persist_directory):
        db = Chroma(embedding_function=embeddings, persist_directory=persist_directory)
        print(f"Folder {persist_directory} exists!!!")
    else:
        # load documents
        loader = PyPDFLoader(file_name)
        documents = loader.load()
        # split documents
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, chunk_overlap=150
        )
        docs = text_splitter.split_documents(documents)

        # create vector database from data
        db = Chroma.from_documents(
            docs, embeddings, persist_directory=persist_directory
        )
        print(f"Folder {persist_directory} created!!!")

    retriever = SelfQueryRetriever.from_llm(
        base_llm,
        db,
        document_content_description,
        metadata_field_info,
        verbose=True,
        enable_limit=True,
        search_kwargs={"k": k},
    )
    # create a chatbot chain. Memory is managed externally.
    qa = ConversationalRetrievalChain.from_llm(
        llm=ChatOpenAI(
            openai_api_key=config("OPENAI_API_KEY"), model_name=llm_name, temperature=0
        ),
        chain_type=chain_type,
        retriever=retriever,
        return_source_documents=True,
        return_generated_question=True,
        get_chat_history=lambda h: h,
    )

    return qa


def get_result(
    file: _TemporaryFileWrapper,
    history: List[list[str]],
    document_content_description: str,
    k: int,
) -> ResultInfo:
    qa = load_db(file, document_content_description, k=k)
    result = qa({"question": history[-1][0], "chat_history": history[:-1]})
    return result


def get_combined_result(
    file: _TemporaryFileWrapper,
    history: List[list[str]],
    document_content_description: str,
    k: int,
):
    result = get_result(file, history, document_content_description, k=k)

    for character in result["answer"]:
        history[-1][1] += character

    return history, fmt_search(result["source_documents"])


def fmt_search(docs: List[Document]) -> str:
    result = []
    for i, d in enumerate(docs):
        result.append(
            f"Document {i+1}:\n\n{d.page_content}\n\nMetadata {i+1}:\n\n{str(d.metadata)}"
        )
    return f"\n{'-' * 100}\n".join(result)
