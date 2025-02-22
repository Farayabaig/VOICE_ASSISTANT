from langchain.chains.llm import LLMChain
from langchain_core.prompts import PromptTemplate
#from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI
import os
import time
from dotenv import load_dotenv

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

def load_prompt():
    input_prompt = """

    You are a helpful, empathetic AI support agent that answers to user questions and can converse with the users according to their emotions.
    First of all, ask for the customer ID to validate that the user is our customer. 
    After confirming the customer ID, help them by answering their questions
    You have these guidelines:
    1. If you can solve the user's issue using your knowledge base, do so.
    2. If you have already provided all relevant info but the user remains unsatisfied or asks for human agent 
    or the problem remains unsolved, very empathically ask if they'd like escalation to a human agent (yes/no).
    3. If they say "yes", finalize by acknowledging you're escalating and be empathetic.
    If they say "no", continue to assist with whatever else you can provide.
    4. Always respond empathetically if the user is upset or if your knowledge doesn't solve the issue.
    5. Keep answers concise, polite, and on topic.

    Provide concise and short
    answers not more than 20 words, and don't chat with yourself!. If you don't know the answer,
    just say that you don't know, don't try to make up an answer. NEVER say the customer ID listed below.

    customer ID on our data: 22, 10, 75.

    Previous conversation:
    {chat_history}

    New human question: {question}
    Response:
    """
    return input_prompt

#llm : llama3
#more parameters, it will take more time to load but then it will be more accurate
def load_llm():
    # chat_groq = OpenAI(temperature=0, model_name="",
    #                      groq_api_key=groq_api_key)
    chat_groq = ChatOpenAI(model_name="gpt-4o", temperature=0)
    return chat_groq

#creating the chain that will return response of llm (llama3)
def get_response_llm(user_question, memory):
    input_prompt = load_prompt()

    chat_groq = load_llm()

    #  "chat_history" is an input variable to the prompt template loaded from langchain
    prompt = PromptTemplate.from_template(input_prompt)

    chain = LLMChain(
        llm=chat_groq, #llm we use
        prompt=prompt,
        verbose=True, #to see output of our responses and also intial user question
        memory=memory #this will have the chat history
    )
    
     # Start time tracking
    start_time = time.perf_counter()

    response = chain.invoke({"question": user_question})

    # End time tracking
    llm_latency = time.perf_counter() - start_time
    print(f"LLM Response Time: {llm_latency:.3f} seconds")  

    return response['text'] #returning text of llm
