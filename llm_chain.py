from prompt_templates import memory_prompt_template, pdf_chat_prompt
from langchain.chains import LLMChain
from langchain.chains.retrieval_qa.base import RetrievalQA
from langchain_community.embeddings import HuggingFaceInstructEmbeddings
from langchain.memory import ConversationBufferWindowMemory
from langchain.prompts import PromptTemplate
from langchain_community.llms import CTransformers
from langchain_community.vectorstores import Chroma
from operator import itemgetter
# from utils import load_config
# import chromadb
import yaml
import chromadb


with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)


def create_llm(model_path = config["model_path"]["small"],model_type=config["model_type"],model_config=config["model_config"]):
    llm = CTransformers(model = model_path,model_type = model_type,config= model_config)
    return llm

def create_embedding(embeddings_path = config["embeddings_path"]):
    return HuggingFaceInstructEmbeddings(model_name= embeddings_path)

def load_vectordb(embeddings):
    persistent_client = chromadb.PersistentClient(config["chromadb"]["chromadb_path"])

    langchain_chroma = Chroma(
        client=persistent_client,
        collection_name=config["chromadb"]["collection_name"],
        embedding_function=embeddings,
    )
    return langchain_chroma


def load_pdf_chat_chain(chat_history):
    return PdfChatChain(chat_history)


def create_embeddings(embeddings_path = config["embeddings_path"]):
    return HuggingFaceInstructEmbeddings(model_name=embeddings_path)

def create_llm_chain(llm,chat_prompt,memory):
    return LLMChain(llm=llm,prompt=chat_prompt,memory=memory)

def create_chat_memory(chat_history):
    return ConversationBufferWindowMemory(memory_key="history", chat_memory=chat_history, k=3)


def create_prompt_from_template(template):
    return PromptTemplate.from_template(template)


def load_normal_chain(chat_history):
    return chatChain(chat_history)

def load_retrieval_chain(llm, memory, vector_db):
    return RetrievalQA.from_llm(llm=llm, memory=memory, retriever = vector_db.as_retriever())

class PdfChatChain:
    def __init__(self,chat_history):
        self.memory = create_chat_memory(chat_history)
        self.vector_db = load_vectordb(create_embeddings())
        llm = create_llm()
        self.llm_chain = load_retrieval_chain(llm, self.memory,self.vector_db)

    def run(self, user_input):
        print("Pdf chat chain is running...")
        return self.llm_chain.invoke(input= user_input,history=self.memory.chat_memory.messages,stop="Human:")
 
class chatChain:
    def __init__(self,chat_history):
        self.memory = create_chat_memory(chat_history)
        self.llm = create_llm()
        chat_prompt = create_prompt_from_template(memory_prompt_template)
        self.llm_chain = create_llm_chain(self.llm,chat_prompt,self.memory)


    def run(self,user_input):
        return self.llm_chain.run(human_input=user_input,history=self.memory.chat_memory.messages,stop="Human:")
