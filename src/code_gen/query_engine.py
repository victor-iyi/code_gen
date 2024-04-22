from llama_index.llms.ollama import Ollama
from llama_index.core import VectorStoreIndex
from llama_index.core.embeddings import resolve_embed_model
from llama_index.core.tools import QueryEngineTool, ToolMetadata
from llama_index.core.schema import Document


def query_engine(llm: Ollama, documents: Document) -> QueryEngineTool:
    # Embedding model.
    embed_model = resolve_embed_model('local:BAAI/bge-m3')

    # Vector DB.
    vector_index = VectorStoreIndex.from_documents(
        documents,
        embed_model=embed_model,
    )
    query_engine = vector_index.as_query_engine(llm=llm)

    # result = query_engine.query('what are some of the routes in the api?')
    # print(result)

    tool = QueryEngineTool(
        query_engine=query_engine,
        metadata=ToolMetadata(
            name='api_documentation',
            description='this gives documentation about code for an API./ Use this for reading  docs for the API',
        )
    )

    return tool
