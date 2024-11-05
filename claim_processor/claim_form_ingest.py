import warnings
from dotenv import load_dotenv
import nest_asyncio
from llama_parse import LlamaParse
import os
import re
from pathlib import Path
import typing as t
from llama_index.core.schema import TextNode, ImageNode
from llama_index.core import (
    VectorStoreIndex,
    Settings,
)
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI

warnings.filterwarnings('ignore')
_ = load_dotenv()

nest_asyncio.apply()

parser = LlamaParse(
    result_type="markdown",
    parsing_instruction="This is an auto insurance claim document.",
    use_vendor_multimodal_model=True,
    vendor_multimodal_model_name="openai-gpt4o",
    show_progress=True,
)

CLAIMS_DIR = "./claims"

claim_form_store_location = "./claim_forms_store"

def get_claims_files(claims_dir=CLAIMS_DIR) -> list[str]:
    files = []
    for f in os.listdir(claims_dir):
        fname = os.path.join(claims_dir, f)
        if os.path.isfile(fname):
            files.append(fname)
    return files

files = get_claims_files()  # get all files from the claims/ directory

md_json_objs = parser.get_json_result(
    files
)  # extract markdown data for insurance claim document

parser.get_images(
    md_json_objs, download_path="data_images"
)  # extract images from PDFs and save them to ./data_images/

# extract list of pages for insurance claim doc
md_json_list = []
for obj in md_json_objs:
    md_json_list.extend(obj["pages"])

# Create helper functions to create a list of TextNodes from the Markdown tables to feed into the VectorStoreIndex.

def get_page_number(file_name):
    """Gets page number of images using regex on file names"""
    match = re.search(r"-page-(\d+)\.jpg$", str(file_name))
    if match:
        return int(match.group(1))
    return 0

def _get_sorted_image_files(image_dir):
    """Get image files sorted by page."""
    raw_files = [f for f in list(Path(image_dir).iterdir()) if f.is_file()]
    sorted_files = sorted(raw_files, key=get_page_number)
    return sorted_files

def get_text_nodes(json_dicts, image_dir) -> t.List[TextNode]:
    """Creates nodes from json + images"""

    nodes = []

    docs = [doc["md"] for doc in json_dicts]  # extract text
    image_files = _get_sorted_image_files(image_dir)  # extract images

    for idx, doc in enumerate(docs):
        # adds both a text node and the corresponding image node (jpg of the page) for each page
        node = TextNode(
            text=doc,
            metadata={"image_path": str(image_files[idx]), "page_num": idx + 1},
        )
        image_node = ImageNode(
            image_path=str(image_files[idx]),
            metadata={"page_num": idx + 1, "text_node_id": node.id_},
        )
        nodes.extend([node, image_node])

    return nodes

text_nodes = get_text_nodes(md_json_list, "data_images")

# Index the documents.
embed_model = OpenAIEmbedding(model="text-embedding-3-large")

llm = OpenAI("gpt-4o")

Settings.llm = llm
Settings.embed_model = embed_model

index = VectorStoreIndex(text_nodes, embed_model=embed_model)

index.storage_context.persist(persist_dir=claim_form_store_location)

query_engine = index.as_query_engine()

response = query_engine.query(
    "Who filed the insurance claim for the accident that happened on Sunset Blvd?"
)
print((str(response)))

response = query_engine.query("How did Ms. Patel's accident happen?")
print((str(response)))

chat_engine = index.as_chat_engine()
response = chat_engine.chat(
    "Given the accident that happened on Lombard Street, name a party that is liable for the damages and explain why."
)
print((str(response)))