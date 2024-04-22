#------------------llm chain import-----------------------
from prompt_templates import memory_prompt_template, pdf_chat_prompt
from langchain.chains.retrieval_qa.base import RetrievalQA
from langchain_community.embeddings import HuggingFaceInstructEmbeddings
from langchain.memory import ConversationBufferWindowMemory
from langchain_community.vectorstores import Chroma
from langchain.llms import LlamaCpp
from langchain import PromptTemplate,LLMChain
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema.document import Document
import pypdfium2
import yaml
import chromadb
import json
from langchain.schema.messages import HumanMessage, AIMessage
from datetime import datetime

with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)


#-----------------------------pdf-handler----------------------------------

def get_pdf_texts(pdfs_bytes_list):
    return [extract_text_from_pdf(pdf_bytes.getvalue()) for pdf_bytes in pdfs_bytes_list]

def extract_text_from_pdf(pdf_bytes):
    pdf_file = pypdfium2.PdfDocument(pdf_bytes)
    return "\n".join(pdf_file.get_page(page_number).get_textpage().get_text_range() for page_number in range(len(pdf_file)))
    
def get_text_chunks(text):
    splitter = RecursiveCharacterTextSplitter(chunk_size=config["pdf_text_splitter"]["chunk_size"], 
                                              chunk_overlap=config["pdf_text_splitter"]["overlap"],
                                                separators=config["pdf_text_splitter"]["separators"])
    return splitter.split_text(text)

def get_document_chunks(text_list):
    documents = []
    for text in text_list:
        for chunk in get_text_chunks(text):
            documents.append(Document(page_content = chunk))
    return documents

def add_documents_to_db(pdfs_bytes_list):
    texts = get_pdf_texts(pdfs_bytes_list)
    documents = get_document_chunks(texts)
    vector_db = load_vectordb(create_embeddings())
    vector_db.add_documents(documents)
    print("Documents added to db.")

#----------------------------utils------------------
def load_config():
    with open("config.yaml", "r") as f:
        return yaml.safe_load(f)
    
def save_chat_history_json(chat_history, file_path):
    with open(file_path, "w") as f:
        json_data = [message.dict() for message in chat_history]
        json.dump(json_data, f)

def load_chat_history_json(file_path):
    with open(file_path, "r") as f:
        json_data = json.load(f)
        messages = [HumanMessage(**message) if message["type"] == "human" else AIMessage(**message) for message in json_data]
        return messages

def get_timestamp():
    return datetime.now().strftime("%Y-%m-%d-%H:%M:%S")




#-------llm_chain--------------------------#

def create_llm():
    callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])
    n_gpu_layers = 40
    n_batch = 512
    llm = LlamaCpp(
        model_path = "/home/shavak-hcdc/chatwithpdf/chatwithpdf_v.02/models/llama-2-7b-chat.Q8_0.gguf",
        temprature = 0.01,
        n_gpu_layers = n_gpu_layers,
        n_batch = n_batch,
        top_p = 1,
        callback_manager = callback_manager,
        n_ctx=8192
    )
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
    return ConversationBufferWindowMemory(memory_key="history", chat_memory=chat_history)

def create_prompt_from_template(template):
    return PromptTemplate.from_template(template)


def load_retrieval_chain(llm, memory, vector_db):
    return RetrievalQA.from_llm(llm=llm, memory=memory, retriever = vector_db.as_retriever())

class PdfChatChain:
    def __init__(self,chat_history):
        self.memory = create_chat_memory(chat_history)
        self.vector_db = load_vectordb(create_embeddings())
        self.llm=create_llm()
        self.llm_chain = load_retrieval_chain(self.llm, self.memory,self.vector_db)

    def run(self, user_input):
        print("Pdf chat chain is running...")
        return self.llm_chain.invoke(input=user_input,history=self.memory.chat_memory.messages,stop="Human:")

