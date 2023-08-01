import glob

from langchain.vectorstores import Chroma
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.llms import AzureOpenAI
from langchain.chains import ConversationalRetrievalChain
import os
from dotenv import load_dotenv

load_dotenv('.env')


class ChromaService:

    def __init__(self):
        self.llm = AzureOpenAI(
            model_name=os.environ.get("MODEL_NAME"),
            temperature=os.environ.get("TEMPERATURE"),
            engine=os.environ.get("ENGINE")
        )
        persist_directory = os.environ.get("PERSIST_DIRECTORY")
        if self.does_vectorstore_exist(persist_directory):
            embed = OpenAIEmbeddings(
                model=os.environ.get("EMBEDDINGS_MODEL"), chunk_size=1
            )
            self.vectorstore = Chroma(persist_directory=persist_directory, embedding_function=embed)
            self.qa = ConversationalRetrievalChain.from_llm(
                llm=self.llm,
                retriever=self.vectorstore.as_retriever(),
                return_source_documents=True)
        else:
            self.vectorstore = None
            self.qa = None

    def does_vectorstore_exist(self, persist_directory: str) -> bool:
        """
        Checks if vectorstore exists
        """
        if os.path.exists(os.path.join(persist_directory, 'index')):
            if os.path.exists(os.path.join(persist_directory, 'chroma-collections.parquet')) and os.path.exists(
                    os.path.join(persist_directory, 'chroma-embeddings.parquet')):
                list_index_files = glob.glob(os.path.join(persist_directory, 'index/*.bin'))
                list_index_files += glob.glob(os.path.join(persist_directory, 'index/*.pkl'))
                # At least 3 documents are needed in a working vectorstore
                if len(list_index_files) > 3:
                    return True
        return False

    def insert_into_index(self, texts, namespace=None):
        try:
            if self.vectorstore is not None:
                self.vectorstore.add_documents(texts, namespace=namespace)
            else:
                persist_directory = os.environ.get("PERSIST_DIRECTORY")
                embed = OpenAIEmbeddings(
                    model=os.environ.get("EMBEDDINGS_MODEL"), chunk_size=1
                )
                self.vectorstore = Chroma.from_documents(texts, embed, persist_directory=persist_directory)
            self.vectorstore.persist()
            self.qa = ConversationalRetrievalChain.from_llm(
                llm=self.llm,
                retriever=self.vectorstore.as_retriever(),
                return_source_documents=True)

            return True
        except Exception as ex:
            raise Exception("Unable to add index")

    def get_chat_history(self, inputs) -> str:
        res = []
        if len(inputs) > 3:
            inputs = inputs[-3:]
        for conv in inputs:
            pair = (conv[0], conv[1])
            res.append(pair)
        return res

    def query_index(self, query, chat_history=[]):
        try:
            if self.vectorstore is None:
                raise Exception("Empty vector store")
            else:
                qa = self.qa
            chat_history = self.get_chat_history(chat_history)
            result = qa({"question": query, "chat_history": chat_history})
            return result["answer"]
        except Exception as e:
            print(e)
            raise e

    def reset_vector_store_object(self):
        self.vectorstore = None
        self.qa = None