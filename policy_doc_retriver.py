import warnings
import nest_asyncio
from dotenv import load_dotenv
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core import VectorStoreIndex
from llama_index.core import Settings
from llama_index.vector_stores.duckdb import DuckDBVectorStore

warnings.filterwarnings('ignore')
_ = load_dotenv()

nest_asyncio.apply()

llm = OpenAI(model="gpt-4o-mini")

duck_db_disk_location = "persist/pg.duckdb"

embed_model = OpenAIEmbedding(model="text-embedding-3-small")

Settings.llm = llm

Settings.embed_model = embed_model

def query_policy_content(query):
    vector_store = DuckDBVectorStore.from_local(duck_db_disk_location)
    recursive_index_instruction = VectorStoreIndex.from_vector_store(vector_store, embed_model=embed_model)
    query_engine_instruction = recursive_index_instruction.as_query_engine(
        similarity_top_k=5
    )
    response = query_engine_instruction.query(query)
    return response

# test
# query_1 = "I have paid £450 for my glasses. How much of it can I claim back?"
# response_1 = query_policy_content(query_1)
# print("With instructions: ")
# print(response_1)