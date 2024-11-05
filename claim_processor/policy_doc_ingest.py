import warnings
import nest_asyncio
from dotenv import load_dotenv
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core import VectorStoreIndex
from llama_index.core import Settings, StorageContext
from llama_parse import LlamaParse
from llama_index.core.node_parser import MarkdownElementNodeParser
from llama_index.vector_stores.duckdb import DuckDBVectorStore

warnings.filterwarnings('ignore')
_ = load_dotenv()

nest_asyncio.apply()

embed_model = OpenAIEmbedding(model="text-embedding-3-small")

llm = OpenAI(model="gpt-4o-mini")

Settings.llm = llm
Settings.embed_model = embed_model

policy_doc = "./docs/pb116349-business-health-select-handbook-1024-pdfa.pdf"

vector_store = DuckDBVectorStore("pg.duckdb", persist_dir="./persist/")

storage_context = StorageContext.from_defaults(vector_store=vector_store)

documents_with_instruction = LlamaParse(
    result_type="markdown",
    parsing_instruction="""
                            This document is an insurance policy. When a benefits/coverage/exclusion is describe in
                            the document amend to it add a text in the following benefits string format (where coverage 
                            could be an exclusion).
                            For {nameofrisk} and in this condition {whenDoesThecoverageApply} the coverage is 
                            {coverageDescription}. 
                            If the document contain a benefits TABLE that describe coverage amounts, do not output it 
                            as a table, but instead as a list of benefits string.""",).load_data(policy_doc)

def query_policy_content(query):

    node_parser_instruction = MarkdownElementNodeParser(llm=llm, num_workers=8)
    node_parser = MarkdownElementNodeParser( llm=llm, num_workers=8)

    nodes_instruction = node_parser.get_nodes_from_documents(documents_with_instruction)
    (
        base_nodes_instruction,
        objects_instruction,
    ) = node_parser_instruction.get_nodes_and_objects(nodes_instruction)

    recursive_index_instruction = VectorStoreIndex(nodes=base_nodes_instruction + objects_instruction,
                                                   storage_context=storage_context,
                                                   embed_model=embed_model)
    query_engine_instruction = recursive_index_instruction.as_query_engine(similarity_top_k=5)
    response = query_engine_instruction.query(query)
    return response

# test
# query_1 = "I have paid £450 for my glasses. How much of it can I claim back?"
# response_1 = query_policy_content(query_1)
# print("With instructions: ")
# print(response_1)