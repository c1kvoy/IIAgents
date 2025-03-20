import re
from langchain.vectorstores import FAISS
from langchain_community.document_loaders import RecursiveUrlLoader
from bs4 import BeautifulSoup
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings


def bs4_extractor(html: str) -> str:
    soup = BeautifulSoup(html, "lxml")
    return re.sub(r"\n\n+", "\n\n", soup.text).strip()

def get_splitter(docs):
    text_contents = [doc.page_content for doc in docs]
    splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=100,
    length_function=len,
    )
    split = splitter.create_documents(text_contents)
    return split

# векторизатор, после векторизации файл сразу сохраняется
def get_vector(split, emb_model, name_folder):
    vector = FAISS.from_documents(split, emb_model)
    vector.save_local(f"/content/drive/MyDrive/AI/{name_folder}")

loader_sklearn = RecursiveUrlLoader("https://scikit-learn.org/stable/", extractor=bs4_extractor)
docs_sklearn = loader_sklearn.load()
split_sklearn = get_splitter(docs_sklearn)

loader_EDA = TextLoader("Прикладной анализ данных.txt")
docs_EDA = loader_EDA.load()
split_EDA = get_splitter(docs_EDA)

loader_ML = TextLoader("Машинное обучение.txt")
docs_ML = loader_ML.load()
split_ML = get_splitter(docs_ML)

emb_model = HuggingFaceEmbeddings(model_name="intfloat/multilingual-e5-large-instruct")

get_vector(split_sklearn, emb_model, "sklearn")
get_vector(split_ML, emb_model, "ML")
get_vector(split_EDA, emb_model, "EDA")