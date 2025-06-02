# from sentence_transformers import SentenceTransformer

# model = SentenceTransformer('multi-qa-MiniLM-L6-cos-v1')  # Fast & small
# embeddings = model.encode(["Hello world", "This is a test"])
# print(embeddings)
# print(embeddings.shape)


from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv  
load_dotenv()


model = ChatGoogleGenerativeAI(model='gemini-2.0-flash-lite')

response = model.invoke("what is the capital of France?")  # Example query
print(response.content)