from langchain.document_loaders import PyPDFLoader, UnstructuredURLLoader, YoutubeLoader
from PyPDF2 import PdfReader
from langchain.text_splitter import TokenTextSplitter, RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.chat_models import ChatOpenAI
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chains import RetrievalQA
from langchain.chains.summarize import load_summarize_chain
from langchain.docstore.document import Document
from prompts import PROMPT_QUESTIONS, REFINE_PROMPT_QUESTIONS, PROMPT_SUMMARY, REFINE_PROMPT_SUMMARY
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler


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
    loads uploaded files and
    returns dict with page_content and source of all files
    '''
    # print(uploaded_file)
    '''
    Data Structure:
    data = {
        0: {
            'page_content': str,
            'source': str,
            'page': int
        }
    }
    '''
    data = {}

    # Looping through uploaded_files
    for index, uploaded_file in enumerate(uploaded_files):
        # Init 
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

def load_url(url):
    """
    loads url and extracts data from url
    """
    urls = [url]
    loader = UnstructuredURLLoader(urls=urls)
    data = loader.load()

    return data

def split_url_data(data, chunk_size, chunk_overlap):
    text_splitter = TokenTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    r_splitter = RecursiveCharacterTextSplitter(separators=["\n\n", "\n", "(?<=\. )", " ", ""], chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    docs = text_splitter.split_documents(data)
    return docs  

def initialize_llm(model: str, temperature: float, stream: bool = False):
    '''
    initializes llm model with specified temperature
    returns llm model
    '''
    if stream:
        llm = ChatOpenAI(model=model, 
                        temperature=temperature,
                        streaming=stream,
                        callbacks=[StreamingStdOutCallbackHandler()])
    else:
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
                                    # verbose=True
                                    )

    questions = qa_chain.run(documents)
    # for chunk in qa_chain.stream(documents):
    #     print(chunk.get('output_text'), flush=True)
    return questions


def generate_summary(llm, chain_type, documents):
    '''
    generates summary based on given documents
    returns summary
    '''
    qa_chain = load_summarize_chain(llm=llm, 
                                    chain_type=chain_type, 
                                    question_prompt=PROMPT_SUMMARY, 
                                    refine_prompt=REFINE_PROMPT_SUMMARY, 
                                    verbose=True
                                    )

    summary = qa_chain.stream(documents)
    # for chunk in qa_chain.stream(documents):
    #     print(chunk.get('output_text'), flush=True)
    return summary



def create_persistant_vectordb(persist_directory, documents, embeddings):
    '''
    creates persistant vector db
    '''
    # persist_directory = 'db'

    # embeddings = OpenAIEmbeddings()

    vector_db = Chroma.from_documents(documents=documents, 
                                      embedding=embeddings, 
                                      persist_directory=persist_directory)
    
    vector_db.persist()
    vector_db = None

def load_persistant_vectordb(persist_directory, embeddings):
    '''
    loads persistant vector db and returns it
    '''
    vector_db = Chroma(persist_directory=persist_directory, embedding_function=embeddings)

    return vector_db

def create_retrieval_qa_chain(documents, llm):
    '''
    stores documents to vector db and
    returns answers to generated questions
    '''
    persist_directory = 'db'

    embeddings = OpenAIEmbeddings()

    # vector_db = Chroma.from_documents(documents=documents, 
    #                                   embedding=embeddings, 
    #                                   persist_directory=persist_directory)
    
    # vector_db.persist()
    # vector_db = None

    create_persistant_vectordb(persist_directory, documents, embeddings)

    # vector_db = Chroma(persist_directory=persist_directory, embedding_function=embeddings)

    vector_db = load_persistant_vectordb(persist_directory, embeddings)

    retriever = vector_db.as_retriever(search_kwargs={"k": 3})

    retrieval_qa_chain = RetrievalQA.from_chain_type(llm=llm, 
                                                     chain_type="stuff", 
                                                     retriever=retriever,
                                                     return_source_documents=True,
                                                    #  verbose=True
                                                     )

    return retrieval_qa_chain