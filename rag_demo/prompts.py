from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from pydantic import BaseModel, Field


rag_system_prompt = """You are an assistant for question-answering tasks. 
Use the following pieces of retrieved context to answer the question. 
If you don't know the answer, just say that you don't know. 
Use three sentences maximum and keep the answer concise.

Context:
{context}
"""


class GradeDocument(BaseModel):
    '''Grading Chunks of chunks in respect to the query'''
    binary_score: str = Field(
        description='Document are relevant to the question "yes" or "no"'
    )

# System Grading Prompt
system = """You are a grader assessing relevance of a retrieved document to a user question. \n 
    If the document contains keyword(s) or semantic meaning related to the question, grade it as relevant. \n
    Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question."""

gradingPrompt = ChatPromptTemplate.from_messages(
    [
        ('system', system),
        ('human', 'Retrieved Document: \n\n{document} \n\n User Question: {question}')    ]
)

system = """You a question re-writer that converts an input question to a better version that is optimized \n 
     for web search. Look at the input and try to reason about the underlying semantic intent / meaning. Make sure it is STRICTLY less than 200 words. AND ONLY GIVE THE IMPROVE AND OPTIMIZED QUESTION"""

re_write_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system),
        (
            "human",
            "Here is the initial question: \n\n {question} \n Formulate an improved question.",
        ),
    ]
)


multiquery_prompt = PromptTemplate(
            input_variables=['query', 'n'],
            template=(
                """You are an AI language model assistant. Your task is to generate {n}
                different versions of the given user question to retrieve relevant documents from a vector 
                database. By generating multiple perspectives on the user question, your goal is to help
                the user overcome some of the limitations of the distance-based similarity search.
                Provide these alternative questions separated by newlines. 
                Original question: {query} \n\n
                Rewritten questions:                 """
            )
        )


##############################################STEP BACK PROMPT##################################################
from langchain_core.prompts import ChatPromptTemplate, FewShotChatMessagePromptTemplate

examples = [
    {
        "input":"Could the members of The Police perform lawful arrests?",
        "output": "what can the members of The Police do?",
    },
    {
        "input": "Jan Sindel’s was born in what country?",
        "output": "what is Jan Sindel’s personal history?",
    },
    {
        "input": "Can students use Calculus Early Transcendentals for derivatives?",
        "output": "what mathematical topics does Calculus Early Transcendentals address?",
    },
    {
        "input": "Can General Chemistry by Petrucci explain atomic structure?",
        "output": "what chemistry concepts does General Chemistry by Petrucci cover?",
    },
    {
        "input": "Can artificial intelligence write poetry as well as humans?",
        "output": "what are the creative writing capabilities of artificial intelligence?",
    },
]

examplePrompt = ChatPromptTemplate.from_messages(
    [
        ('human', '{input}'),
        ('ai',"{output}"),
    ]
)

fewShotPrompt = FewShotChatMessagePromptTemplate(
    example_prompt= examplePrompt,
    examples=examples
)

stepBackPrompt = ChatPromptTemplate.from_messages(
    [
        ("system", 
         """You are an expert at world knowledge. Your task is to step back and paraphrase a question to a more generic step-back question, which is easier to answer. Here are a few examples:""",
        ),
        fewShotPrompt,
        ('user', "{question}")
    ]
)
############################################################################################################################################################

rag_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", rag_system_prompt),
        ("human", "{question}"),
    ]
)