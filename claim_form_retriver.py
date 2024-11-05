import warnings
from dotenv import load_dotenv
import nest_asyncio
from llama_index.core import (
    StorageContext,
    load_index_from_storage,
    Settings,
)
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI

warnings.filterwarnings('ignore')
_ = load_dotenv()

nest_asyncio.apply()

claim_form_store_location = "./claim_forms_store"

# Index the documents.
embed_model = OpenAIEmbedding(model="text-embedding-3-large")

llm = OpenAI("gpt-4o")

Settings.llm = llm

Settings.embed_model = embed_model

ctx = StorageContext.from_defaults(persist_dir=claim_form_store_location)

index = load_index_from_storage(ctx)

query_engine = index.as_query_engine()

def get_response_form_store_chat_engine(query):
    chat_engine = index.as_chat_engine()
    response = chat_engine.chat(query)
    return str(response)

def get_response_form_store_query_engine(query):
    response = query_engine.query(query)
    return str(response)


# query_1 = "Who filed the insurance claim for the accident that happened on Sunset Blvd?"
# query_2 = "How did Ms. Patel's accident happen?"

# response_1 = get_response_form_store_query_engine(query_1)
# response_2 = get_response_form_store_query_engine(query_2)

# print(response_1)
# print(response_2)

# response_3 = get_response_form_store_chat_engine(query_1)
# print(response_3)
# response_4 = get_response_form_store_chat_engine(query_2)
# print(response_4)