import streamlit as st
from fpdf import FPDF
import base64
from io import BytesIO
from utils.document import process_file_multi_docs, delete_uploaded_files_and_db
from lc_functions import (load_data, 
                          split_text, 
                          initialize_llm, 
                          generate_questions, 
                          generate_summary,
                          create_retrieval_qa_chain, 
                          load_data_multiple_docs, 
                          split_text_multiple_docs)
# from tempfile import NamedTemporaryFile
import atexit
import os

# Show app title
st.title('AI Study Buddy')

# Create empty session states
if 'questions' not in st.session_state:
    st.session_state['questions'] = 'empty'
    st.session_state['seperated_question_list'] = 'empty'
    st.session_state['questions_to_answers'] = 'empty'
    st.session_state['submitted'] = 'empty'
    st.session_state['summary'] = 'empty'

# Initialize openai api key
os.environ['OPENAI_API_KEY'] = st.text_input(label='OpenAI API Key', placeholder='Ex: sk-4ewt5jsdhfh4...', key='openai_api_key')

# Get uploaded files 
uploaded_files = st.file_uploader(label='Upload Study Material', 
                                  type=['pdf'], 
                                  accept_multiple_files=True)

# Show Delete Study Materials button
delete_uploaded_files_btn = st.button("Delete Study Materials")
# generate_questions_btn = st.button("Generate Questions")
# Create submit button
summary_gen = st.button('Generate Summary')

# Check if btn is pressed
if delete_uploaded_files_btn:
    # Delete uploaded files
    delete_uploaded_files_and_db()

# Check if files are uploaded
if uploaded_files:
    # if generate_questions_btn:
    # Process files
    files = process_file_multi_docs(uploaded_files)

    # Load uploaded file
    data = load_data_multiple_docs(files)

    # Split doc for question gen
    documents_for_question_gen = split_text_multiple_docs(data=data, chunk_size=700, chunk_overlap=50)

    # Split doc for summary gen
    documents_for_summary_gen = split_text_multiple_docs(data=data, chunk_size=800, chunk_overlap=80)
    
    # Split docs for question asnwering
    documents_for_question_ans = split_text_multiple_docs(data=data, chunk_size=400, chunk_overlap=50)

    # st.write(documents_for_question_gen)
    # st.write('Number of documents for question generation: ', len(documents_for_question_gen))
    # st.write(documents_for_question_ans)
    # st.write('Number of documents for question answering: ', len(documents_for_question_ans))

    # init llm for question reneration
    llm_question_gen = initialize_llm(model='gpt-3.5-turbo', temperature=0.4)

    # init llm for question reneration
    llm_summary_gen = initialize_llm(model='gpt-3.5-turbo', temperature=0.6, stream=True)

    # init llm for question answering
    llm_question_ans = initialize_llm(model='gpt-3.5-turbo', temperature=0.1)

    # Check If questions is empty
    if st.session_state['questions'] == 'empty':
        # Generate questions
        with st.spinner('Generating questions. This may take a while. Please wait.'):
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

                    # Show first source document
                    st.warning(response['source_documents'][0].page_content)

                    # Show a divider at the end of each answer
                    st.divider()
    
    
    if summary_gen:
        # Process files
        # files = process_file_multi_docs(uploaded_files)
        # st.write(files)
        # Load uploaded file
        # data = load_data_multiple_docs(files)

        # Split doc for summary gen
        documents_for_summary_gen = split_text_multiple_docs(data=data, chunk_size=800, chunk_overlap=80)

        # init llm for question reneration
        llm_summary_gen = initialize_llm(model='gpt-3.5-turbo', temperature=0.6, stream=True)

        # if st.session_state['summary'] == 'empty':
            # st.session_state['summary'] = generate_summary(llm=llm_summary_gen, chain_type='refine', documents=documents_for_summary_gen)
        res = generate_summary(llm=llm_summary_gen, chain_type='refine', documents=documents_for_summary_gen)
            # st.write(type(res))
        st.session_state['summary'] = ''
        for i in res:
            st.session_state['summary'] += i.get('output_text')
            st.info(st.session_state['summary'])

    export_as_pdf = st.button("Download Summary")

    # Function to generate PDF from text
    def text_to_pdf(text):
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Arial", size=12)
        pdf.multi_cell(0, 10, text)
        return pdf

    if export_as_pdf:
        pdf = text_to_pdf(st.session_state['summary'])
        pdf_output = BytesIO()
        pdf_content = pdf.output(dest='S').encode('latin1')  # Get PDF content as byte string
        pdf_output.write(pdf_content)  # Write the content to BytesIO object
        pdf_output.seek(0)
        b64 = base64.b64encode(pdf_output.read()).decode()

        href = f'<a href="data:application/pdf;base64,{b64}" download="summary.pdf">Download PDF</a>'
        st.markdown(href, unsafe_allow_html=True)

            # summary_gen = False

atexit.register(delete_uploaded_files_and_db)