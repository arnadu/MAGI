#from llama_index.core import Document
#from llama_index.core.node_parser import SentenceSplitter
#from llama_index.core.node_parser import SimpleFileNodeParser
#from llama_index.core.schema import MetadataMode

import constants as c

from langchain_text_splitters import MarkdownHeaderTextSplitter
from langchain_text_splitters import RecursiveCharacterTextSplitter
#from langchain_openai import OpenAIEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings
#from sentence_transformers import SentenceTransformer

import os

import pymupdf4llm

from llm import call_llm_once

from pymongo.operations import SearchIndexModel

from database import get_db_connection

#EMBEDDING_MODEL = "text-embedding-3-small"   #"text-embedding-3-small"  #text-embedding-ada-002
#CHUNK_SIZE = 1024
#CHUNK_OVERLAP = 20
#QUERY_NEIGHBORS = 10
#DOC_COLLECTION = "Documents"   
#CHUNKS_COLLECTION = "Chunks"

EMBEDDING_MODEL = 'all-MiniLM-L6-v2'
CHUNK_SIZE = 1024
CHUNK_OVERLAP = 20
QUERY_NEIGHBORS = 10
DOCUMENTS_COLLECTION = "Documents-all-MiniLM-L6-v2"   
CHUNKS_COLLECTION = "Chunks-all-MiniLM-L6-v2"

class Library():
    """The list of available documents"""
    def __init__(self): 
        #self.project_name = "test"
        self.library_prompt = "You are a useful librarian, and you assist the user in reviewing documents. Answer the question based on document extracts provided below. Do not hallucinate features and return that there is insufficient information when the documents do not contain the answer."
        self.langchain_embeddings_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2") #SentenceTransformer(EMBEDDING_MODEL)

    def compute_vector(self, text):
        """compute the vector embedding of a text"""
        #response = client.embeddings.create(input=text, model=EMBEDDING_MODEL)
        #return response.data[0].embedding
        embeddings = self.langchain_embeddings_model.embed_documents([text])
        return embeddings[0]


    def get_chunks(self, query, project_name, category, document_name, app_name=None):
        """perform a similarity search to find the most relevant chunks to answer a query"""
        db = get_db_connection()
        chunks_collection = db[CHUNKS_COLLECTION]

        vector = self.compute_vector(query)

        filter = []
        if app_name is not None:
            filter.append({"$or": [{"project_name": project_name}, {"project_name": f"{app_name}.*"}]})
        else:
            filter.append({"project_name": project_name})
        if category:
            filter.append({"category": category})
        if document_name:
            filter.append({"document_name": document_name})

        #https://www.mongodb.com/docs/atlas/atlas-vector-search/vector-search-stage/
        #a vector index needs to be defined: #https://cloud.mongodb.com/v2/65a5c3fddd129d6fe671dd42#/clusters/atlasSearch/Cluster0
        pipeline = [{
            '$vectorSearch': {
                'index': 'vector_index', 
                'path': 'vector', 
                'filter': {'$and': filter}, 
                'queryVector': vector, 
                'numCandidates': QUERY_NEIGHBORS,
                'limit': QUERY_NEIGHBORS
            }
        }, {
            '$project': {
                '_id': 0, 
                'category': 1,
                'document_name': 1, 
                'content': 1, 
                'score': {'$meta': 'vectorSearchScore'}
            }
        }]

        result = chunks_collection.aggregate(pipeline)
        return result

    def query(self, llm, query: str, project_name: str, category: str = "", document_name: str = "", app_name=None):
        try:
            chunks = self.get_chunks(query, project_name, category, document_name, app_name=app_name)
            messages = []
            
            messages.append({"role": "system", "content": self.library_prompt})
            
            messages.append({"role": "user", "content": f"Question: ```{query}```"})

            if category or document_name:
                filter_desc = "Limit similarity search to"
                if category:
                    filter_desc += f" category='{category}'"
                if document_name:
                    filter_desc += f" document_name='{document_name}'"
                messages.append({"role": "user", "content": filter_desc})

            messages.append({"role": "user", "content": "Documents Extracts: "})
            for chunk in chunks:
                messages.append({"role": "user", "content": f"Excerpt from document:{chunk['document_name']} (category:{chunk['category']})\n```{chunk['content']}```"})

            answer = call_llm_once(llm, messages)

            msg = answer.choices[0].message.content
            res = c.AssistantAnswer(msg_source=c.MSG_SOURCE_QUERY, tool_input=query, next_action=c.LLM_NEXT_ACTION_PROCESS_INFORMATION, msg_to_user=msg)
            yield res

        except Exception as e:
            res = c.AssistantAnswer(msg_source=c.MSG_SOURCE_QUERY, tool_input=query, next_action=c.LLM_NEXT_ACTION_PROCESS_INFORMATION, msg_to_user="KO: "+str(e))  
            yield res
        
    def split_document(self, project_name, category, document_name, document):
        """Split a document into chunks and calculate the chunks' vector embeddings"""

        headers_to_split_on = [
            ("#", "Header 1"),
            ("##", "Header 2"),
            ("###", "Header 3"),
        ]

        markdown_splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=headers_to_split_on, 
            strip_headers=False
        )

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE, 
            chunk_overlap=CHUNK_OVERLAP
        )

        # Split

        #split by header first and then by characters
        #md_header_splits = markdown_splitter.split_text(document)
        #nodes = text_splitter.split_documents(md_header_splits)

        nodes = text_splitter.split_text(document)

        #node_parser = SentenceSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
        #nodes = node_parser.get_nodes_from_documents([Document(text=document)], show_progress=False)
        
        chunks = []
        for node in nodes:
            #vector= self.compute_vector(node.get_content(metadata_mode=MetadataMode.EMBED))
            vector= self.compute_vector(node)

            chunks.append({"project_name": project_name, 
                           "category": category, 
                           "document_name": document_name, 
#                           "content": node.get_content(metadata_mode=MetadataMode.EMBED), 
#                           "content": node.page_content, 
                           "content": node, 
                           "vector": vector
            })    
        
        return chunks        


    def upload_file(self, project_name, document_name, category, filepath, abstract):
        """Load a document from a file and save it to the database"""
        md_text = pymupdf4llm.to_markdown(filepath)  # get markdown for all pages
        content = md_text
        self.save_document(project_name, document_name, category, content, abstract)

    def save_document(self, project_name, document_name, category, content, abstract):
        '''Save a document to the database; split it into chunks and save the chunks to the database'''

        db = get_db_connection()
        documents_collection = db[DOCUMENTS_COLLECTION]
        chunks_collection = db[CHUNKS_COLLECTION]

        #delete from the database
        query = {"project_name": project_name, "document_name": document_name}
        documents_collection.delete_many(query)
        chunks_collection.delete_many(query)

        #save the document
        data = {
            "project_name": project_name,
            "document_name": document_name,
            "category": category,
            "content": content,
            "abstract": abstract,
        }    
        result = documents_collection.insert_one(data)
        #print(result)

        #split the document into chunks, calculate their embeddings and save the chunks
        chunks = self.split_document(project_name=project_name, category=category, document_name=document_name, document=content)
        result = chunks_collection.insert_many(chunks)
        #print(result)

    def del_document(self, project_name, document_name):
        db = get_db_connection()
        collection = db[DOCUMENTS_COLLECTION]
        result = collection.delete_one({"project_name": project_name, "document_name": document_name})
        return result
    
    def get_document(self, project_name, document_name):
        db = get_db_connection()
        collection = db[DOCUMENTS_COLLECTION]
        result = collection.find_one({"project_name": project_name, "document_name": document_name})
        return result

    def get_documents(self, project_name, app_name=None):
        db = get_db_connection()
        collection = db[DOCUMENTS_COLLECTION]
        if app_name is not None:
            query = {"$or": [
                {"project_name": project_name}, 
                {"project_name": f"{app_name}.*"}  #get the application's default documents
            ]} 
        else:
            query = {"project_name": project_name}
        result = collection.find(query,{"_id":0, "content":0}).sort("category").sort("document_name")
        docs = []
        for doc in result:
            docs.append(doc)
        return docs

    def project_exists(self, project_name):
        """Check if a project exists in the database"""
        db = get_db_connection()
        doc_collection = db[DOCUMENTS_COLLECTION]
        return doc_collection.count_documents({"project_name": project_name}) > 0

    def clone_project(self, source_project_name, target_project_name):
        """Clone all documents and chunks from one project to another"""
        db = get_db_connection()
        doc_collection = db[DOCUMENTS_COLLECTION]
        chunks_collection = db[CHUNKS_COLLECTION]

        #note that if the target project already exists, the documents and chunks will *not* be overwritten

        # Clone documents
        docs = doc_collection.find({"project_name": source_project_name})
        for doc in docs:
            # Create a new document without _id so MongoDB generates a new one
            new_doc = {k:v for k,v in doc.items() if k != '_id'}
            new_doc["project_name"] = target_project_name
            doc_collection.insert_one(new_doc)

        # Clone chunks
        chunks = chunks_collection.find({"project_name": source_project_name})
        for chunk in chunks:
            # Create a new chunk without _id so MongoDB generates a new one
            new_chunk = {k:v for k,v in chunk.items() if k != '_id'}
            new_chunk["project_name"] = target_project_name
            chunks_collection.insert_one(new_chunk)




########
if __name__ == "__main__":

    db_name = os.environ.get("MONGO_DB", default="PrivacyAI")
    print(f"Using database: {db_name}")

    l = Library(openai_api_key=os.environ.get("OPENAI_API_KEY"))
    
    def create_vector_index(size=384, similarity="cosine"):
        """utility to create a vector index on A new chunks collection; call only once"""
        db = get_db_connection()
        chunks_collection = db[CHUNKS_COLLECTION]
        search_index_model = SearchIndexModel(
            definition={
                "fields": [{
                    "type": "vector",
                    "numDimensions": size, #384 for all-MiniLM-L6-v2
                    "path": "vector",
                    "similarity":  similarity
                },
                {
                    "path": "project_name",
                    "type": "filter"
                },
                {
                    "path": "category",
                    "type": "filter"
                },
                {
                    "path": "document_name",
                    "type": "filter"
                }]},
            name="vector_index",
            type="vectorSearch",
        )
        chunks_collection.create_search_index(model=search_index_model)

    def test_save_file():
        l.upload_file(project_name="test", document_name="Quebec.pdf", category="Policy", filepath="./library/Quebec.pdf", abstract="Quebec privacy law")

    def test_query(question):
        for ans in l.query(question, project_name="test", category="Policy", document_name="Quebec.pdf"):
            print(ans)

    #test_save_file()
    create_vector_index(384, "cosine") #use this function only once after creating a new chunks collection
    test_query("What are the governance requirements of the Quebec privacy law ?")


    #TODO: re-arrange split doc. separat function for embedding + should embed the entire batch in one go
    #TODO: some documents should be available to all projects by default
    #TODO: user interface to control method for splitting and embedding a doc?


