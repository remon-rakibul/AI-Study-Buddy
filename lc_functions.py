from langchain.document_loaders import PyPDFLoader
from PyPDF2 import PdfReader
from langchain.text_splitter import TokenTextSplitter
from langchain.vectorstores import Chroma
from langchain.chat_models import ChatOpenAI
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chains import RetrievalQA
from langchain.chains.summarize import load_summarize_chain
from langchain.docstore.document import Document
from prompts import PROMPT_QUESTIONS, REFINE_PROMPT_QUESTIONS

def load_data(uploaded_file):
    '''
    loads uploaded file and
    returns text
    '''
    pdf_reader = PdfReader(uploaded_file)

    text =""

    for page in pdf_reader.pages:
        text += page.extract_text()
    return text

def split_text(text, chunk_size, chunk_overlap):
    '''
    splits texts accroding to chunk size and overlap
    returns list of splitted texts
    '''
    text_splitter = TokenTextSplitter(model_name='gpt-3.5-turbo', chunk_size=chunk_size, chunk_overlap=chunk_overlap)

    text_chunk = text_splitter.split_text(text)

    document = [Document(page_content=t) for t in text_chunk]
    return document

def initialize_llm(model, temperature):
    llm = ChatOpenAI(model=model, temperature=temperature)

    return llm

def generate_questions(llm, chain_type, documents):
    qa_chain = load_summarize_chain(llm=llm, chain_type=chain_type, question_prompt=PROMPT_QUESTIONS, refine_prompt=REFINE_PROMPT_QUESTIONS, verbose=True)

    questions = qa_chain.run(documents)

    return questions

def create_retrieval_qa_chain(documents, llm):
    embeddings = OpenAIEmbeddings()

    vector_database = Chroma.from_documents(documents=documents, embedding=embeddings)

    retrieval_qa_chain = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=vector_database.as_retriever())

    return retrieval_qa_chain