from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from typing import List
from langchain.schema import Document


#Extract Data From the PDF File
def load_pdf_file(data):
    loader= DirectoryLoader(data,
                            glob="*.pdf",
                            loader_cls=PyPDFLoader)

    documents=loader.load()

    return documents



def filter_to_minimal_docs(docs: List[Document]) -> List[Document]:
    """
    Given a list of Document objects, return a new list of Document objects
    containing only 'source' in metadata and the original page_content.
    """
    minimal_docs: List[Document] = []
    for doc in docs:
        src = doc.metadata.get("source")
        minimal_docs.append(
            Document(
                page_content=doc.page_content,
                metadata={"source": src}
            )
        )
    return minimal_docs



#Split the Data into Text Chunks
def text_split(extracted_data):
    text_splitter=RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=20)
    text_chunks=text_splitter.split_documents(extracted_data)
    return text_chunks



#Download the Embeddings from HuggingFace 
def download_hugging_face_embeddings():
    embeddings=HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')  #this model return 384 dimensions
    return embeddings

def ask(query, rag_chain, retriever):
    try:
        result = rag_chain.invoke({"input": query})
        if isinstance(result, dict):
            return result.get("answer", result)
        content = getattr(result, "content", None)
        return content if content is not None else result
    except Exception:
        docs = retriever.invoke(query)
        snippets = []
        for d in docs:
            try:
                src = d.metadata.get("source")
            except Exception:
                src = None
            text = d.page_content if hasattr(d, "page_content") else str(d)
            snippets.append(f"- {text[:500]}... (source: {src})")
        return "Model unavailable; showing retrieved context:\n" + "\n".join(snippets)
