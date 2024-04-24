import langchain
import os
import getpass
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
import bs4
from langchain import hub
from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import PromptTemplate
from langchain_community.document_loaders import AsyncChromiumLoader
from langchain_community.document_transformers import BeautifulSoupTransformer
from langchain_community.document_loaders import WebBaseLoader
import requests
from bs4 import BeautifulSoup
import json
from dotenv import load_dotenv
import os
load_dotenv()

api_key = os.getenv("GOOGLE_API_KEY")

llm = ChatGoogleGenerativeAI(model = "gemini-pro", google_api_key = api_key)
gemini_embeddings = GoogleGenerativeAIEmbeddings(model = "models/embedding-001", google_api_key = api_key)
text_splitter = RecursiveCharacterTextSplitter(chunk_size = 500, chunk_overlap = 200)





def scrap_jd(link):
    response = requests.get(link)
    if response:
        html = response.text
        soup = BeautifulSoup(html, "html.parser")
        result = soup.find("div", class_ = "description__text description__text--rich")
        jd = result.text
        return json.dumps({"jd":jd})
    



def jdParser(jd):
    docs = [Document(page_content=jd)]
    splits = text_splitter.split_documents(docs)
    vectorstore = Chroma.from_documents(documents = splits , embedding =  gemini_embeddings)

    retriever  = vectorstore.as_retriever()

    def format_docs(docs):
        return "/n/n".join(doc.page_content for doc in docs)
    

    template = """Use the following pieces of context to answer the question at the end.
    If you don't know the answer, just say that None, don't try to make up an answer.
    Reply only what are going to ask with adding any prilimanery and post sentence.

    {context}

    Question: {question}

    Helpful Answer:"""

    custom_rag_prompt = PromptTemplate.from_template(template)

    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | custom_rag_prompt
        | llm
        | StrOutputParser()
    )

    
    address = rag_chain.invoke("give me the job location which are present in job description")
    experience = rag_chain.invoke("total expereince requered in the job description")
    qualification = rag_chain.invoke("give me the python list of education required in job description")
    skills  = rag_chain.invoke("give me python list of name of technical skills required in job description")
    skills_embedding = gemini_embeddings.embed_query(skills)


    dic = {
        'address':address,
        'experience':experience,
        'qualification':qualification,
        'skills':skills,
        'skill_embedding':skills_embedding,
    }

    return dic


 