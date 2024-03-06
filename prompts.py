from langchain.prompts import PromptTemplate

prompt_template_questions = """
You are an expert in creating practice questions based on study material.
Your goal is to prepare a student for their an exam. You do this by asking questions about the text below:

------------
{text}
------------

Create questions that will prepare the student for their exam. Make sure not to lose any important information.

QUESTIONS:
"""
PROMPT_QUESTIONS = PromptTemplate(template=prompt_template_questions, input_variables=["text"])

refine_template_questions = ("""
You are an expert in creating practice questions based on study material.
Your goal is to help a student prepare for an exam.
We have received some practice questions to a certain extent: {existing_answer}.
We have the option to refine the existing questions or add new ones.
(only if necessary) with some more context below.
------------
{text}
------------

Given the new context, refine the original questions in English.
If the context is not helpful, please provide the original questions.
QUESTIONS:
"""
)
REFINE_PROMPT_QUESTIONS = PromptTemplate(
    input_variables=["existing_answer", "text"],
    template=refine_template_questions,
)


prompt_template_summary = """
You are an expert in creating a summary based on study material.
Your goal is to prepare a summary from that can help a person get idea of the important concepts from the given context.
Include bullet points and numbering to make the summary more easier to understand.
You do this by creating a summary about the document below:

------------
{text}
------------

Create summary that will give a person learn about the concepts faster. Make sure not to lose any important information.

SUMMARY:
"""
PROMPT_SUMMARY = PromptTemplate(template=prompt_template_summary, input_variables=["text"])


refine_template_summary = ("""
You are an expert in creating summary based on study material.
Your goal is to help a person prepare for an exam.
We have received some summary to a certain extent: {existing_answer}.
We have the option to refine the existing summary or make it even better.
(only if necessary) with some more context below.
------------
{text}
------------

Given the new context, refine the original summary in English.
If the context is not helpful, please provide the original summary.
SUMMARY:
"""
)
REFINE_PROMPT_SUMMARY = PromptTemplate(
    input_variables=["existing_answer", "text"],
    template=refine_template_summary,
)