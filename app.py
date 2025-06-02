import streamlit as st
from langchain_community.document_loaders import PyPDFLoader, TextLoader, UnstructuredMarkdownLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
import tempfile
import os
import traceback

# Load environment variables
load_dotenv()

# Initialize the LLM
model = ChatGoogleGenerativeAI(model='gemini-2.0-flash-lite')

st.title("FAQ Chatbot")

# Allowed file types
allowed_formats = ['pdf', 'md', 'txt']

# Session state to store vector store & chat history
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

uploaded_file = st.file_uploader("Upload a document", type=allowed_formats)

if uploaded_file:
    st.success(f"Uploaded file: {uploaded_file.name}")

    with tempfile.NamedTemporaryFile(delete=False, suffix=f".{uploaded_file.name.split('.')[-1]}") as tmp_file:
        tmp_file.write(uploaded_file.getbuffer())
        tmp_file_path = tmp_file.name

    # Detect file type from extension for simplicity
    ext = uploaded_file.name.split('.')[-1].lower()
    if ext == "pdf":
        loader = PyPDFLoader(tmp_file_path)
    elif ext == "md":
        loader = UnstructuredMarkdownLoader(tmp_file_path)
    elif ext == "txt":
        loader = TextLoader(tmp_file_path)
    else:
        st.error("Unsupported file type.")
        st.stop()

    docs = loader.load()
    st.info(f"Loaded {len(docs)} document(s).")

    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_documents(docs)
    st.info(f"Split into {len(chunks)} chunks.")

    embedding_function = HuggingFaceEmbeddings(model_name='sentence-transformers/multi-qa-MiniLM-L6-cos-v1')

    try:
        # Use a temp directory for vector db persistence during session
        chroma_dir = tempfile.mkdtemp()
        vector_store = Chroma(
            embedding_function=embedding_function,
            persist_directory=chroma_dir,
            collection_name='sample'
        )
        vector_store.add_documents(chunks)
        st.success("Documents stored in vector database.")
        st.session_state.vector_store = vector_store
    except Exception as e:
        st.error(f"Error storing documents: {e}")
        st.text(traceback.format_exc())
        st.stop()

if st.session_state.vector_store is not None:
    # Chat input
    user_input = st.text_input("Ask a question:")
    if user_input:
        # Add user message to chat history
        st.session_state.chat_history.append({"role": "user", "content": user_input})

        # Retrieve similar docs from vector store
        response = st.session_state.vector_store.similarity_search_with_score(query=user_input, k=4)
        context_text = "\n".join([doc.page_content for doc, _ in response])

        # Create prompt for LLM
        prompt = PromptTemplate(
            template=(
                "You are a helpful assistant. Answer the question based on the context below.\n\n"
                "Context: {context}\n\nQuestion: {question}"
            ),
            input_variables=["context", "question"]
        )
        final_prompt = prompt.format(context=context_text, question=user_input)

        # Call model
        result = model.invoke(final_prompt)
        answer = result.content.strip()

        # Add bot response to chat history
        st.session_state.chat_history.append({"role": "assistant", "content": answer})

    # Display chat history
    for message in st.session_state.chat_history:
        if message["role"] == "user":
            st.markdown(f"**You:** {message['content']}")
        else:
            st.markdown(f"**Bot:** {message['content']}")

else:
    st.info("Upload a document to start chatting.")



# import streamlit as st
# from langchain_community.document_loaders import PyPDFLoader, TextLoader, UnstructuredMarkdownLoader
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain.vectorstores import Chroma
# from langchain.embeddings import HuggingFaceEmbeddings
# from langchain.prompts import PromptTemplate
# from langchain_google_genai import ChatGoogleGenerativeAI
# from dotenv import load_dotenv
# import tempfile
# import os
# import traceback

# # Load environment variables
# load_dotenv()

# # Initialize the LLM
# model = ChatGoogleGenerativeAI(model='gemini-2.0-flash-lite')

# # Streamlit UI
# st.title("FAQ Chatbot")

# # Supported file types
# allowed_formats = ['pdf', 'md', 'txt']
# uploaded_file = st.file_uploader("Upload a document", type=allowed_formats)

# # Temporary directory for Chroma DB
# chroma_dir = tempfile.mkdtemp()

# if uploaded_file:
#     st.success(f"Uploaded file: {uploaded_file.name}")
#     file_type = uploaded_file.type

#     with tempfile.NamedTemporaryFile(delete=False, suffix=f".{uploaded_file.name.split('.')[-1]}") as tmp_file:
#         tmp_file.write(uploaded_file.getbuffer())
#         tmp_file_path = tmp_file.name

#     if file_type == "application/pdf":
#         loader = PyPDFLoader(tmp_file_path)
#     elif file_type == "text/markdown":
#         loader = UnstructuredMarkdownLoader(tmp_file_path)
#     elif file_type == "text/plain":
#         loader = TextLoader(tmp_file_path)
#     else:
#         st.error("Unsupported file type.")
#         st.stop()

#     docs = loader.load()
#     st.info(f"Loaded {len(docs)} document(s).")

#     splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
#     chunks = splitter.split_documents(docs)
#     st.info(f"Split into {len(chunks)} chunks.")

#     embedding_function = HuggingFaceEmbeddings(model_name='sentence-transformers/multi-qa-MiniLM-L6-cos-v1')

#     try:
#         vector_store = Chroma(
#             embedding_function=embedding_function,
#             persist_directory=chroma_dir,
#             collection_name='sample'
#         )
#         vector_store.add_documents(chunks)
#         st.success("Documents stored in vector database.")
#         st.write("Sample stored document:", vector_store.get(include=['documents'])['documents'][:1])

#     except Exception as e:
#         st.error(f"Error storing documents: {e}")
#         st.text(traceback.format_exc())
#         st.stop()

#     user_input = st.text_input("Ask a question:")
#     if st.button("Submit") and user_input:
#         st.write("Query:", user_input)

#         response = vector_store.similarity_search_with_score(query=user_input, k=4)
#         context_text = "\n".join([doc.page_content for doc, _ in response])
#         st.write("Matched context:", context_text)

#         prompt = PromptTemplate(
#             template=(
#                 "You are a helpful assistant. Answer the question based on the context below.\n\n"
#                 "Context: {context}\n\nQuestion: {question}"
#             ),
#             input_variables=["context", "question"]
#         )

#         final_prompt = prompt.format(context=context_text, question=user_input)
#         result = model.invoke(final_prompt)

#         st.write("Answer:")
#         st.write(result.content)




# import streamlit as st
# from langchain_community.document_loaders import PyPDFLoader, TextLoader, UnstructuredMarkdownLoader
# import tempfile
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain.vectorstores import Chroma
# from langchain.embeddings import HuggingFaceEmbeddings  # Wrapper for SentenceTransformer
# from langchain.prompts import PromptTemplate
# from langchain_google_genai import ChatGoogleGenerativeAI
# from dotenv import load_dotenv

# load_dotenv()

# model = ChatGoogleGenerativeAI(model='gemini-2.0-flash-lite')
# # Streamlit UI
# st.title("FAQ-CHATBOT")
# allowed_formats = ['pdf', 'md', 'txt']
# uploaded_file = st.file_uploader("Upload a document", type=allowed_formats)

# if uploaded_file:
#     st.write(f"Uploaded file: {uploaded_file.name}")
#     file_type = uploaded_file.type

#     with tempfile.NamedTemporaryFile(delete=False, suffix=f".{uploaded_file.name.split('.')[-1]}") as tmp_file:
#         tmp_file.write(uploaded_file.getbuffer())
#         tmp_file_path = tmp_file.name

#     # Load documents
#     if file_type == "application/pdf":
#         st.write("This is a PDF file.")
#         loader = PyPDFLoader(tmp_file_path)
#     elif file_type == "text/markdown":
#         st.write("This is a Markdown file.")
#         loader = UnstructuredMarkdownLoader(tmp_file_path)
#     elif file_type == "text/plain":
#         st.write("This is a Text file.")
#         loader = TextLoader(tmp_file_path)
#     else:
#         st.error("Unsupported file type.")
#         st.stop()

#     docs = loader.load()
#     st.write(f"Upload converted to {len(docs)} docs.")

#     # Split text into chunks
#     splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
#     chunk = splitter.split_documents(docs)
#     st.write(f"Document split into {len(chunk)} chunks.")

#     # Embedding function using Hugging Face SentenceTransformer
#     embedding_function = HuggingFaceEmbeddings(model_name='sentence-transformers/multi-qa-MiniLM-L6-cos-v1')

#     # Create vector store
#     vector_store = Chroma(
#         embedding_function=embedding_function,
#         persist_directory='my_chroma_db',
#         collection_name='sample'
#     )

#     vector_store.add_documents(chunk)
#     data = vector_store.get(include=['embeddings', 'metadatas', 'documents'])
#     st.write("Stored data sample:", data['documents'][:1])


#     user_input = st.text_input("Ask a question:")  

#     if st.button("Submit"):
#         st.write("Query:", user_input)

#         user_query_vector = embedding_function.embed_query(user_input)
#         st.write("User query vector:", user_query_vector)

#         response = vector_store.similarity_search_with_score(query=user_input,k=3)
#         context_text = "\n".join([doc.page_content for doc, _ in response])

#         st.write("response:", context_text)

#     prompt = PromptTemplate(
#         template = "you are a helpful assistant. Answer the question based on the context provided.\n\nContext: {context}\n\nQuestion: {question}",
#         input_variables=['context', 'question']
#     )

#     final_prompt = prompt.format(
#         context=context_text,
#         question=user_input
#     )

#     result = model.invoke(final_prompt)
#     st.write(result.content)



