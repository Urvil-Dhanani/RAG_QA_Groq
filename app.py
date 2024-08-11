# loading .env
import os
from dotenv import load_dotenv
load_dotenv()

# importing required libraries 
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
import streamlit as st


# setting up the API keys
groq_API = os.getenv("GROQ_API_KEY")
hf_API = os.getenv("HUGGINGFACE_API_KEY")

# setting up the embedding Object
my_embedding = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# setting up streamlit 
st.title(body="RAG Q&A with Memory")
st.write("Upload your PDF document and ask questions from the uploaded content")

# Getting the API key and other inputs from user
user_groq_api = st.sidebar.text_input(label="Please provide your GROQ API key", type="password")

user_model = st.sidebar.selectbox(label="Please select the Model", 
                                  options=["Gemma2-9b-It","Gemma-7b-It","Llama3-8b-8192"],
                                  )

user_temperature = st.sidebar.slider(label="Temperature",
                                     min_value=0.0,
                                     max_value=1.0,
                                     value=0.75)

user_max_tokens = st.sidebar.slider(label="Max Tokens",
                                    min_value=50,
                                    max_value=300,
                                    value=150)

user_session_id = st.sidebar.text_input(label="Enter session name", value="default_session")

# if api and model are provided 
if user_groq_api and user_model:
    llm = ChatGroq(model=user_model, groq_api_key=user_groq_api)

    # session_stste is a place provided by streamlit where we store our variables for runtime 
    # lets create our chat store which will have session_id:ChatHistory

    if "store" not in st.session_state:
        st.session_state.store={}

    # file uploading 
    user_pdf_files = st.file_uploader(label="Please upload PDF file",
                                     type="pdf",
                                     accept_multiple_files=True)
    
    if user_pdf_files:
        documents = []
        for pdf_file in user_pdf_files:
            temp_file = f"./temp.pdf"
            with open(temp_file, "wb") as file:
                file.write(pdf_file.getvalue())
                file_name=pdf_file.name

            loader = PyPDFLoader(temp_file)
            docs = loader.load()
            documents.extend(docs)

        # st.write("File has been uploaded & docs created")

        # creating retriever (history_aware)
        chunker = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=400)
        chunks = chunker.split_documents(documents=documents)
        vectorstore = FAISS.from_documents(documents=chunks, embedding=my_embedding)
        retriever = vectorstore.as_retriever()
        # st.write("Your vector store is ready")

        # prompt for retriever
        retriever_system_prompt=(
            "Given a chat history and the latest user question"
            "which might reference context in the chat history, "
            "formulate a standalone question which can be understood "
            "without the chat history. Do NOT answer the question, "
            "just reformulate it if needed and otherwise return it as is."
        )
        retriever_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", retriever_system_prompt),
                MessagesPlaceholder("chat_history"),
                ("human", "{input}"),
            ]
        )
                
        
        history_aware_retriever = create_history_aware_retriever(llm=llm,
                                                                 retriever=retriever,
                                                                 prompt=retriever_prompt)
        

        # prompt for llm 
        llm_system_prompt = (
            "You are an assistant for question-answering tasks. "
            "Use the following pieces of retrieved context to answer "
            "the question. If you don't know the answer, say that you "
            "don't know. Use three sentences maximum and keep the "
            "answer concise."
            "\n\n"
            "{context}"
        )

        llm_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", llm_system_prompt),
                MessagesPlaceholder("chat_history"),
                ("human", "{input}")
            ]
        )

        # now we combine our llm with prompt
        combine_docs = create_stuff_documents_chain(llm=llm, prompt=llm_prompt)

        # Combine history_aware_vectorestore with llm & prompt
        rag_chain = create_retrieval_chain(retriever=history_aware_retriever,
                                           combine_docs_chain=combine_docs)

        # lets setup history 
        def fatch_session_history(session_id:str)->BaseChatMessageHistory:
            if session_id not in st.session_state.store:
                st.session_state.store[session_id]=ChatMessageHistory()
            return st.session_state.store[session_id]
        
        # lets make a coversational chain 
        conversational_rag = RunnableWithMessageHistory(runnable=rag_chain,
                                                        get_session_history=fatch_session_history,
                                                        input_messages_key="input",
                                                        history_messages_key="chat_history",
                                                        output_messages_key="answer"
                                                        )
        
        user_input = st.text_input("Your Question:")

        if user_input:
            session_history = fatch_session_history(user_session_id)
            response = conversational_rag.invoke(
                {"input":user_input},
                config={"configurable":{"session_id":user_session_id}}
            )
            st.write(response["answer"])
            # st.write(st.session_state.store)
            # st.write("Chat History:", session_history.messages)
        
                    

else:
    st.write("Please enter API & model name")


            

        
    


    


