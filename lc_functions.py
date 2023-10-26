from langchain.document_loaders import PyPDFLoader
from PyPDF2 import PdfReader
from langchain.text_splitter import TokenTextSplitter, RecursiveCharacterTextSplitter
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
    returns dict with page_content and source
    '''
    # print(uploaded_file)
    pdf_reader = PdfReader(uploaded_file)

    text =""

    for page in pdf_reader.pages:
        text += page.extract_text()

    data = {
        'page_content': text,
        'source': uploaded_file
    }
    return data

def load_data_multiple_docs(uploaded_files):
    '''
    loads uploaded file and
    returns dict with page_content and source
    '''
    # print(uploaded_file)

    data = {}

    for index, uploaded_file in enumerate(uploaded_files):
        pdf_reader = PdfReader(uploaded_file)

        text =""

        for page in pdf_reader.pages:
            text += page.extract_text()

        data[index] = {
            'page_content': text,
            'source': uploaded_file
        }
    
    return data


def split_text_multiple_docs(data, chunk_size, chunk_overlap):
    '''
    splits texts accroding to chunk size and overlap
    returns list of splitted texts
    '''
    text_splitter = TokenTextSplitter(model_name='gpt-3.5-turbo', 
                                      chunk_size=chunk_size, 
                                      chunk_overlap=chunk_overlap)

    document = []
    
    for i in data:
        text_chunk = text_splitter.split_text(data[i]['page_content'])
        metadata = {
            'source': data[i]['source']
        }

        for t in text_chunk:
            doc = Document(page_content=t, metadata=metadata)
            document.append(doc)
        # doc = [Document(page_content=t, metadata=metadata) for t in text_chunk]

        # document.append(doc)

    return document


def split_text(text, source, chunk_size, chunk_overlap):
    '''
    splits texts accroding to chunk size and overlap
    returns list of splitted texts
    '''
    text_splitter = TokenTextSplitter(model_name='gpt-3.5-turbo', 
                                      chunk_size=chunk_size, 
                                      chunk_overlap=chunk_overlap)

    text_chunk = text_splitter.split_text(text)
    metadata = {
        'source': source
    }

    document = [Document(page_content=t, metadata=metadata) for t in text_chunk]

    return document

# def load_and_split_data(uploaded_file, chunk_size, chunk_overlap):
#     '''
#     loads uploaded file and splits doc accroding to chunk size and overlap
#     returns list of splitted docs
#     '''
#     loader = PyPDFLoader(uploaded_file)
#     pages = loader.load()
#     # print(f'pages: {len(pages)}')
#     text_splitter = TokenTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
#     r_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(separators=["\n\n", "\n", "(?<=\. )", " ", ""], chunk_size=chunk_size, chunk_overlap=chunk_overlap)
#     docs = r_splitter.split_documents(pages)
#     # print(f'docs: {len(docs)}')
#     return docs

def initialize_llm(model, temperature):
    '''
    initializes llm model with specified temperature
    returns llm model
    '''
    llm = ChatOpenAI(model=model, 
                     temperature=temperature)

    return llm

def generate_questions(llm, chain_type, documents):
    '''
    generates questions based on given documents
    returns questions
    '''
    qa_chain = load_summarize_chain(llm=llm, 
                                    chain_type=chain_type, 
                                    question_prompt=PROMPT_QUESTIONS, 
                                    refine_prompt=REFINE_PROMPT_QUESTIONS, 
                                    verbose=True)

    questions = qa_chain.run(documents)

    return questions

def create_retrieval_qa_chain(documents, llm):
    '''
    stores documents to vector db and
    returns answers to generated questions
    '''
    persist_directory = 'db'

    embeddings = OpenAIEmbeddings()

    vector_db = Chroma.from_documents(documents=documents, 
                                      embedding=embeddings, 
                                      persist_directory=persist_directory)
    
    retriever = vector_db.as_retriever()

    retrieval_qa_chain = RetrievalQA.from_chain_type(llm=llm, 
                                                     chain_type="stuff", 
                                                     retriever=retriever,
                                                     return_source_documents=True,
                                                     verbose=True)

    return retrieval_qa_chain