import streamlit as st
from lc_functions import load_data, split_text, initialize_llm, generate_questions
from tempfile import NamedTemporaryFile
import os

st.title('AI Study Buddy')

if 'questions' not in st.session_state:
    st.session_state['questions'] = 'empty'
    st.session_state['seperated_question_list'] = 'empty'
    st.session_state['questions_to_answers'] = 'empty'
    st.session_state['submitted'] = 'empty'

os.environ['OPENAI_API_KEY'] = st.text_input(label='OpenAI API Key', placeholder='Ex: sk-4ewt5jsdhfh4...', key='openai_api_key')

uploaded_file = st.file_uploader(label='Upload Study Material', type=['pdf'])

if uploaded_file:
    # Load data from pdf
    # st.write(uploaded_file.name)
    bytes_data = uploaded_file.read()
    with NamedTemporaryFile(delete=False) as tmp:
        tmp.write(bytes_data)
        text_from_pdf = load_data(tmp.name)
    os.remove(tmp.name)
    # text_from_pdf = load_data(uploaded_file.name)
    # st.write(text_from_pdf)
    # Split text for question gen
    documents_for_question_gen = split_text(text=text_from_pdf, chunk_size=700, chunk_overlap=50) 
    
    # Split text for question asnwering
    documents_for_question_ans = split_text(text=text_from_pdf, chunk_size=400, chunk_overlap=50)
    # st.write(documents_for_question_gen)
    st.write('Number of documents for question generation: ', len(documents_for_question_gen))
    # st.write(documents_for_question_ans)
    st.write('Number of documents for question answering: ', len(documents_for_question_ans))

    # init llm for question reneration
    llm_question_gen = initialize_llm(model='gpt-3.5-turbo', temperature=0.4)

    # init llm for question answering
    llm_question_ans = initialize_llm(model='gpt-3.5-turbo', temperature=0.1)

    if st.session_state['questions'] == 'empty':
        with st.spinner('Generating questions...'):
            st.session_state['questions'] = generate_questions(llm=llm_question_gen, chain_type='refine', documents=documents_for_question_gen)

    if st.session_state['questions'] != 'empty':
        st.info(st.session_state['questions'])