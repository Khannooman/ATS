import langchain
import os
import getpass
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
import bs4
from langchain import hub
from langchain_community.document_loaders import PyPDFLoader
from langchain_chroma import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import PromptTemplate
from mongo import resume
from dotenv import load_dotenv
import os
load_dotenv()

# if "GOOGLE_API_KEY" not in os.environ:
#     os.environ["GOOGLE_API_KEY"] = getpass.getpass("pass your gemini key")

api_key = os.getenv("GOOGLE_API_KEY")
llm = ChatGoogleGenerativeAI(model = "gemini-pro", google_api_key = api_key)
gemini_embeddings = GoogleGenerativeAIEmbeddings(model = "models/embedding-001", google_api_key = api_key)
text_splitter = RecursiveCharacterTextSplitter(chunk_size = 500, chunk_overlap = 200)


def parse_resume(path):
    loader = PyPDFLoader(path)
    parser = loader.load_and_split()
    
    splits = text_splitter.split_documents(parser)
    vectorstore = Chroma.from_documents(documents = splits , embedding =  gemini_embeddings)

    retriever  = vectorstore.as_retriever()

    def format_docs(docs):
        return "/n/n".join(doc.page_content for doc in parser)

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

    name = rag_chain.invoke("give me the  of name of resume holder")
    address = rag_chain.invoke("give me the location of resume holder")
    email = rag_chain.invoke("give me the email of resume holder")
    phone_number = rag_chain.invoke("give me the phone number of resume holder")
    experience = rag_chain.invoke("give total working experience candidate have in months")
    organisation = rag_chain.invoke("give me the     python list of organisations resume holder work")
    Institute = rag_chain.invoke("give me the python list of Institute resume holder Study")
    skills  = rag_chain.invoke("give me python list of technical skills resume holder have")
    skills_embedding = gemini_embeddings.embed_query(skills)

    dic = {
        'name' : name,
        'address':address,
        'email':email,
        'phone_number':phone_number,
        'experience':experience,
        'organisation':organisation,
        'Institute':Institute,
        'skills':skills,
        'skill_embedding' : skills_embedding    
    }

    return dic

path = '/home/nooman/Downloads/Nooman Khan.pdf'

par_resume = parse_resume(path)
if par_resume:
    email = par_resume["email"]
    result = resume.find({"email":email})
    if result:
        resume.delete_one({"email":email})
        resume.insert_one(par_resume)