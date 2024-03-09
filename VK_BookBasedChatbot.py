import openai
import streamlit as st
from streamlit_extras.add_vertical_space import add_vertical_space
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.callbacks import get_openai_callback
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
# from langchain_openai import OpenAI
from langchain.chains.question_answering import load_qa_chain
import pickle
import os
from dotenv import load_dotenv
import base64
from langchain.llms import OpenAI
from langchain.chains.question_answering import load_qa_chain
from langchain.callbacks import get_openai_callback


def add_bg_from_local(image_file):
    with open(image_file, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read())
    st.markdown(
        f"""
    <style>
    .stApp {{
        background-image: url(data:image/jpeg;base64,{encoded_string.decode()});
        background-size: cover;
    }}
    </style>
    """,
        unsafe_allow_html=True,
    )


# Load environment variables from a specific file
env_path = os.path.join(os.path.dirname(__file__), '.env', '.env')
load_dotenv(dotenv_path=env_path)

# Obtén la clave de API de OpenAI desde las variables de entorno
openai.api_key = os.getenv("OPENAI_API_KEY")

# Verifica si la clave de API se ha obtenido correctamente
if openai.api_key is None:
    raise ValueError(
        "La variable de entorno OPENAI_API_KEY no está configurada. Asegúrate de haber definido la clave de API en el archivo .env.")


def main():  # sourcery skip: extract-method, use-join, use-named-expression
    st.header("📄Chat with your pdf file🤗")

    with st.sidebar:
        st.title(
            '🦜️🔗 CHATBOT BASADO EN OPENAI-LLM-LANGCHAIN-STREAMLIT - PREGUNTALE A TUS PDF  🤗')
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
            client = None  # OpenAI()

            embeddings = OpenAIEmbeddings(client=client)
            vectorstore = FAISS.from_texts(chunks, embedding=embeddings)
            with open(f"{store_name}.pkl", "wb") as f:
                pickle.dump(vectorstore, f)

        query = st.text_input(
            "Ask questions about related your upload pdf file")

        if query:
            docs = vectorstore.similarity_search(query=query, k=3)
            client = None  # You need to provide your OpenAI API client here

            llm = OpenAI(client=client, temperature=0.7,
                         model="gpt-3.5-turbo-instruct")
            chain = load_qa_chain(llm=llm, chain_type="stuff")
            with get_openai_callback() as cb:
                response = chain.run(input_documents=docs, question=query)
            st.write(response)


if __name__ == "__main__":
    main()
