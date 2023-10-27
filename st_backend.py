import streamlit as st
from utils.document import process_file_multi_docs
from lc_functions import (load_data, 
                          split_text, 
                          initialize_llm, 
                          generate_questions, 
                          create_retrieval_qa_chain, 
                          load_data_multiple_docs, 
                          split_text_multiple_docs)
# from tempfile import NamedTemporaryFile
import glob
import os

# Show app title
st.title('AI Study Buddy')

# Create empty session states
if 'questions' not in st.session_state:
    st.session_state['questions'] = 'empty'
    st.session_state['seperated_question_list'] = 'empty'
    st.session_state['questions_to_answers'] = 'empty'
    st.session_state['submitted'] = 'empty'

# Initialize openai api key
os.environ['OPENAI_API_KEY'] = st.text_input(label='OpenAI API Key', placeholder='Ex: sk-4ewt5jsdhfh4...', key='openai_api_key')

# Get uploaded files 
uploaded_files = st.file_uploader(label='Upload Study Material', type=['pdf'], accept_multiple_files=True)

# Check if files are uploaded
if uploaded_files:

    # Process files
    files = process_file_multi_docs(uploaded_files)

    # Load uploaded file
    data = load_data_multiple_docs(files)

    # Split doc for question gen
    documents_for_question_gen = split_text_multiple_docs(data=data, chunk_size=700, chunk_overlap=50)
    
    # Load and split text for question asnwering
    documents_for_question_ans = split_text_multiple_docs(data=data, chunk_size=400, chunk_overlap=50)

    # st.write(documents_for_question_gen)
    # st.write('Number of documents for question generation: ', len(documents_for_question_gen))
    # st.write(documents_for_question_ans)
    # st.write('Number of documents for question answering: ', len(documents_for_question_ans))

    # init llm for question reneration
    llm_question_gen = initialize_llm(model='gpt-3.5-turbo', temperature=0.4)

    # init llm for question answering
    llm_question_ans = initialize_llm(model='gpt-3.5-turbo', temperature=0.1)

    # Check If questions is empty
    if st.session_state['questions'] == 'empty':
        # Generate questions
        with st.spinner('Generating questions. Please wait.'):
            st.session_state['questions'] = generate_questions(llm=llm_question_gen, chain_type='refine', documents=documents_for_question_gen)

    # Add generated question to session state
    if st.session_state['questions'] != 'empty':
        
        # Show questions
        st.info(st.session_state['questions'])

        # Convert questions to a list of questions
        st.session_state['questions_list'] = st.session_state['questions'].split('\n')

        # Multiselect option for selecting questions
        with st.form(key='my_form'):

            # Add selected question to a session state
            st.session_state['questions_to_answer'] = st.multiselect(label='Select Questions to Answer', options=st.session_state['questions_list'])

            # Create submit button
            submitted = st.form_submit_button('Generate Answer')

            # Check if questions are submitted for generating answer
            if submitted:

                # Change session state value
                st.session_state['submitted'] = True

        # Check if questions are submitted
        if st.session_state['submitted']:

            # Show spinner while generating answers
            with st.spinner('Generating Answers. Please wait.'):

                # Generate answers for submitted questions
                generate_answer_chain = create_retrieval_qa_chain(documents=documents_for_question_ans, llm=llm_question_ans)

                # Loop through all the questions
                for question in st.session_state['questions_to_answer']:
                    
                    # Generate answer for each question
                    # ans = generate_answer_chain.run(question)
                    response = generate_answer_chain(question)

                    # Show question
                    st.write(f'Question: {question}')

                    # Show answer
                    st.info(f"Answer: {response['result']}")

                    # Show sources of answer
                    st.caption("Sources:")

                    # Initialize a set to keep unique sources                    
                    sources = set()

                    # Loop through sources
                    for source in response['source_documents']:

                        # Add source to initialized set
                        sources.add(source.metadata['source'])

                    # Loop through unique sources
                    for source in sources:

                        # Show source
                        st.caption(source)

                    # Show a divider at the end of each answer
                    st.divider()