import os
from llama_index.llms import OpenAI
from llama_index import VectorStoreIndex, SimpleDirectoryReader
from llama_index import ServiceContext
import os, streamlit as st


os.environ['OPENAI_API_KEY']="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"
st.title("Llama Index test")
query = st.text_input("Ask any question about the data", "")

documents = SimpleDirectoryReader("data").load_data()
llm = OpenAI(model="gpt-3.5-turbo", temperature=0, max_tokens=256)
service_context = ServiceContext.from_defaults(llm=llm, chunk_size=800, chunk_overlap=20)
index = VectorStoreIndex.from_documents(documents, service_context=service_context)

query_engine = index.as_query_engine(streaming=True)
uploaded_file = st.file_uploader("Choose a file to upload", type=['txt', 'pdf', 'docx', 'csv', 'json'])

if uploaded_file is not None:
    try:
        with open(os.path.join("data", uploaded_file.name), "wb") as f:
            f.write(uploaded_file.getbuffer())
            st.success(f"File '{uploaded_file.name}' uploaded successfully!")

    except Exception as e:
        st.error(f"Failed to save file: {e}")


if st.button("Submit"):
    if not query.strip():
        st.error(f"Please provide the search query.")
    else:
        try:
            response = query_engine.query(query)
            # print(response)
            st.success(response)
        except Exception as e:
            st.error(f"An error occurred: {e}")
