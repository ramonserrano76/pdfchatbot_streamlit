import openai
import streamlit as st
from streamlit_extras.add_vertical_space import add_vertical_space
from PyPDF2 import PdfFileReader, PdfFileWriter, PdfReader

# from pypdf import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import OpenAI
from langchain.chains.question_answering import load_qa_chain
from langchain.callbacks import get_openai_callback
import pickle
import os
from dotenv import load_dotenv
import base64


def add_bg_from_local(image_file):
    with open(image_file, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read())
    st.markdown(
        f"""
    <style>
    .stApp {{
        background-image: url(data:image/{"jpeg"};base64,{encoded_string.decode()});
        background-size: cover;
    }}
    </style>
    """,
        unsafe_allow_html=True
    )


# Load environment variables from a specific file
env_path = os.path.join(os.path.dirname(__file__), '.env', '.env')
load_dotenv(dotenv_path=env_path)
# Obtén la clave de API de OpenAI desde las variables de entorno
openai_api_key = os.getenv("OPENAI_API_KEY")
# openai_api_key = st.sidebar.text_input('OPENAI_API_KEY', type='password')

# Verifica si la clave de API se ha obtenido correctamente
if openai_api_key is None:
    raise ValueError(
        "La variable de entorno OPENAI_API_KEY no está configurada. Asegúrate de haber definido la clave de API en el archivo .env.")

def main():
    st.header("📄Chat with your pdf file🤗")

    with st.sidebar:
        st.title('🦜️🔗 CHATBOT BASADO EN OPENAI-LLM-LANGCHAIN-STREAMLIT - PREGUNTALE A TUS PDF  🤗')
        st.markdown('''
        ## About APP:

        The app's primary resource is utilised to create::

        - [streamlit](https://streamlit.io/)
        - [Langchain](https://docs.langchain.com/docs/)
        - [OpenAI](https://openai.com/)

        ## About me:

        - [Linkedin](https://www.linkedin.com/in/ramonserrano76/)
        
        ''')

        add_vertical_space(4)
        st.write('💡All about pdf based chatbot, created by RS🤗')

    # upload a your pdf file
    pdf = st.file_uploader("Upload your PDF", type='pdf')
    
    if pdf is not None:
        st.write(pdf.name)       
        pdf_reader = PdfReader(pdf)

        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )

        chunks = text_splitter.split_text(text=text)

        store_name = pdf.name[:-4]

        if os.path.exists(f"{store_name}.pkl"):
            with open(f"{store_name}.pkl", "rb") as f:
                vectorstore = pickle.load(f)
        else:
            client = None  # You need to provide your OpenAI API client here
            embeddings = OpenAIEmbeddings(client=client)
            vectorstore = FAISS.from_texts(chunks, embedding=embeddings)
            with open(f"{store_name}.pkl", "wb") as f:
                pickle.dump(vectorstore, f)

        query = st.text_input("Ask questions about related your upload pdf file")

        if query:
            docs = vectorstore.similarity_search(query=query, k=3)
            client = None  # You need to provide your OpenAI API client here
            # Text models such as text-davinci-003, text-davinci-002 and earlier (ada, babbage, curie, davinci, etc.)
            # Temperature low 0 to 0.2 , mid 0.3 to 0.6, high 0.7 to 1 
            
            llm = OpenAI(client=client, temperature=0.7, model="text-davinci-004")
            chain = load_qa_chain(llm=llm, chain_type="stuff")
            with get_openai_callback() as cb:
                response = chain.run(input_documents=docs, question=query)
            st.write(response)
            
    # with st.form('my_form'):
    #     text = st.text_area('Enter text:', 'What are the three key pieces of advice for learning how to code?')
    #     submitted = st.form_submit_button('Submit')
    #     if not openai_api_key.startswith('sk-'):
    #         st.warning('Please enter your OpenAI API key!', icon='⚠')
    #     if submitted and openai_api_key.startswith('sk-'):
    #         main()

if __name__ == "__main__":
    main()
