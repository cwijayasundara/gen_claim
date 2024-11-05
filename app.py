import warnings
import streamlit as st
import os
from dotenv import load_dotenv
from pathlib import Path
from policy_doc_retriver import query_policy_content
from claim_form_retriver import get_response_form_store_chat_engine
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from prompts.claim_prompts import claim_processing_prompt, cash_back_prompt, final_response_prompt
from invoice_data_extractor.invoice_data_extractor import extract_invoice_data

warnings.filterwarnings('ignore')
_ = load_dotenv()

llm = ChatOpenAI(model="gpt-4o-mini")

claim_processing_prompt_str = ChatPromptTemplate.from_template(claim_processing_prompt)

claim_chain = claim_processing_prompt_str | llm

cashback_prompt_str = ChatPromptTemplate.from_template(cash_back_prompt)

cash_back_chain = cashback_prompt_str | llm

final_response_prompt_str = ChatPromptTemplate.from_template(final_response_prompt)

final_response_chain = final_response_prompt_str | llm

st.title("ClaimGenius : Your AI Assistant for Insurance Claims")

def sanitize_filename(filename):
    """
    Sanitize the filename to prevent path traversal attacks and remove unwanted characters.
    """
    filename = os.path.basename(filename)
    filename = "".join(c for c in filename if c.isalnum() or c in (" ", ".", "_", "-")).rstrip()
    return filename

st.markdown(
    """
    <style>
        section[data-testid="stSidebar"] {
            width: 400px !important; # Set the width to your desired value
        }
    </style>
    """,
    unsafe_allow_html=True,
)

with st.sidebar:
    st.image("images/img_1.png", width=400)
    add_radio = st.radio(
        "What can I do for you today?",
        ("Claim Policy Assistant",
         "Claim Form Assistant",
         "Make a Claim!",
         "ClaimGenius - Design")
    )

if add_radio == "Claim Policy Assistant":
    st.header("Claim Policy Assistant")
    
    st.write("Example questions you can ask:")
    st.write("What is the cashback amount for dental fees per year? ")
    st.write("What is the cashback amount for optical fees per?")

    request = st.text_area(f"How can I help you with the policy document knowledge base today?", height=100)
    submit = st.button("submit", type="primary")

    if request and submit:
        chat_result = query_policy_content(request)
        st.write(f"chat_result: :blue[{chat_result}]")

elif add_radio == "Claim Form Assistant":
    st.header("Claim Form Assistant")

    st.write("Example questions you can ask:")
    st.write("Who filed the insurance claim for the accident that happened on Sunset Blvd?")
    st.write("How did Ms. Patel's accident happen?")

    request = st.text_area(f"How can I help you with the claim form knowledge base today?", height=100)
    submit = st.button("submit", type="primary")

    if request and submit:
        chat_result = get_response_form_store_chat_engine(request)
        st.write(f"chat_result: :blue[{chat_result}]")

elif add_radio == "Make a Claim!":
    st.header("Make a Claim!")
    st.write("Please upload your invoice to get started")
    uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")

    if uploaded_file is not None:
        upload_dir = Path.cwd() / "uploaded_invoices"
        upload_dir.mkdir(parents=True, exist_ok=True)
        original_filename = Path(uploaded_file.name).name  # Extract the filename
        sanitized_filename = sanitize_filename(original_filename)
        save_path = upload_dir / sanitized_filename
        # convert the same path to string
        save_path = str(save_path)
        with open(save_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        st.success(f"File saved successfully to: {save_path}")

        extracted_invoice_data = extract_invoice_data(save_path)
        st.write(extracted_invoice_data)

        # Extract the policy section based on the treatment type
        treatment_type = extracted_invoice_data.treatment_type
        claim_section_prompt = f"""What is the cashback amount for {treatment_type} fees per year?"""
        policy_section = query_policy_content(claim_section_prompt)
        st.write(f"policy_section: :blue[{policy_section}]")

        claim_details_extracted = extracted_invoice_data.invoice_total + " " + extracted_invoice_data.treatment_type
        # print(claim_details_extracted)
        st.write(f"claim_details_extracted: :blue[{claim_details_extracted}]")

        # Process the claim
        claim_response = claim_chain.invoke({"POLICY_SECTION": policy_section,
                                 "CLAIM_DETAILS": claim_details_extracted})
        # write the response to the streamlit app
        st.write(f":blue[{claim_response.content}]")

        # Extract the cashback amount and treatment type from the claim response
        cash_back_amount = cash_back_chain.invoke({"text": claim_response.content})

        st.write(cash_back_amount.content)

        # Generate the final response
        final_response = final_response_chain.invoke({"invoice_data": extracted_invoice_data,
                                                      "claim_decision": claim_response.content})
        st.write(f":red[{final_response.content}]")
    
elif add_radio == "ClaimGenius - Design":

    st.header("ClaimGenius - Business Flow")
    st.image("images/business_flow.jpg")

    st.header("ClaimGenius - Design")
    st.image("images/claims-Page-2.jpg")

    st.write("ClaimGenius is an AI assistant that helps you with your insurance claims.")
    st.write("It can help you with your policy document knowledge base, help you with your claims, and more.")
    st.write("Feel free to chat with your Policy Assistant or make a claim to get started!")