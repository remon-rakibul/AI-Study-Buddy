import streamlit as st
from lc_functions import load_data, split_text, initialize_llm, generate_questions, create_retrieval_qa_chain
# from tempfile import NamedTemporaryFile
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
    # st.write(uploaded_file.name)
    bytes_data = uploaded_file.read()

    _, file_extension = os.path.splitext(uploaded_file.name)
    # st.write(_, file_extension)

    # Create file directory
    if not os.path.exists('files'):
        os.makedirs('files')

    # Write uploaded file to disk
    with open(f'files/{uploaded_file.name}', 'wb') as f:
        f.write(bytes_data)

    # Load uploaded file
    data = load_data(f'files/{uploaded_file.name}')
    text = data['page_content']
    source = data['source']

    # with NamedTemporaryFile(delete=False) as tmp:
    #     tmp.write(bytes_data)
    #     text_from_pdf = load_data(tmp.name)
    
    # Removing uploaded file from disk
    os.remove(f'files/{uploaded_file.name}')
    
    # st.write(text_from_pdf)

    # Split doc for question gen
    documents_for_question_gen = split_text(text=text, source=source, chunk_size=700, chunk_overlap=50)
    
    # Load and split text for question asnwering
    documents_for_question_ans = split_text(text=text, source=source, chunk_size=400, chunk_overlap=50)

        # # Load and split doc for question gen
        # documents_for_question_gen = load_and_split_data(uploaded_file=tmp.name, chunk_size=700, chunk_overlap=50) 
    
        # # Load and split text for question asnwering
        # documents_for_question_ans = load_and_split_data(uploaded_file=tmp.name, chunk_size=400, chunk_overlap=50)

    # test
    # test = load_and_split_data(uploaded_file=tmp.name, chunk_size=4000, chunk_overlap=100)
    # Removing uploaded file from file system
    # os.remove(tmp.name)
    # Test
    # st.write('Number of documents for test: ', len(test))
    st.write(documents_for_question_gen)
    st.write('Number of documents for question generation: ', len(documents_for_question_gen))
    st.write(documents_for_question_ans)
    st.write('Number of documents for question answering: ', len(documents_for_question_ans))

    # init llm for question reneration
    llm_question_gen = initialize_llm(model='gpt-3.5-turbo', temperature=0.4)

    # init llm for question answering
    llm_question_ans = initialize_llm(model='gpt-3.5-turbo', temperature=0.1)

    if st.session_state['questions'] == 'empty':
        with st.spinner('Generating questions. Please wait.'):
            st.session_state['questions'] = generate_questions(llm=llm_question_gen, chain_type='refine', documents=documents_for_question_gen)

    if st.session_state['questions'] != 'empty':
        st.info(st.session_state['questions'])

        st.session_state['questions_list'] = st.session_state['questions'].split('\n')

        with st.form(key='my_form'):
            st.session_state['questions_to_answer'] = st.multiselect(label='Select Questions to Answer', options=st.session_state['questions_list'])

            submitted = st.form_submit_button('Generate Answer')

            if submitted:
                st.session_state['submitted'] = True

        if st.session_state['submitted']:
            with st.spinner('Generating Answers. Please wait.'):
                generate_answer_chain = create_retrieval_qa_chain(documents=documents_for_question_ans, llm=llm_question_ans)

                for question in st.session_state['questions_to_answer']:
                    ans = generate_answer_chain.run(question)

                    st.write(f'Question: {question}')
                    st.info(f'Answer: {ans}')
