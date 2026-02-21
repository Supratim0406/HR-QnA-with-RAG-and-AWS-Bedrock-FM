import os
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import BedrockEmbeddings
from langchain.vectorstores import FAISS
from langchain.indexes import VectorstoreIndexCreator
from langchain_aws import BedrockLLM
from langchain_aws.chat_models import ChatBedrockConverse



def hr_index():
    
    # Download the PDF from URL and converts each page into LangChain Dcoument
    data_loader = PyPDFLoader("https://www.upl-ltd.com/images/people/downloads/Leave-Policy-India.pdf")

    # Split the Text into Chunks
    data_split = RecursiveCharacterTextSplitter(separators=["\n\n", "\n", " ", ""], chunk_size= 100, chunk_overlap = 10)

    # Create Embedding Model
    data_embeddings = BedrockEmbeddings(
        credentials_profile_name='default',
        model_id='amazon.titan-embed-text-v2:0'    
    )

    # Create Vector Index Builder
    data_index = VectorstoreIndexCreator(
        text_splitter=data_split,
        embedding=data_embeddings,
        vectorstore_cls=FAISS
    )

    # Build the Vector Database
    db_index=data_index.from_loaders([data_loader])

    return db_index

# Write a function to connect to Bedrock foundation model
def hr_llm():
    llm=ChatBedrockConverse(
        credentials_profile_name='default',
        region_name="us-east-1",
        model_id="anthropic.claude-3-haiku-20240307-v1:0",
        temperature=0.1,
        max_tokens=1000,
    )
    return llm

# Write a function which searches the user prompt, searches the best match from  vector db and sends both to LLM

def hr_rag_response(index, question):
    rag_llm = hr_llm()
    hr_rag_query = index.query(question=question, llm = rag_llm)
    return hr_rag_query