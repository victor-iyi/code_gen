from llama_index.llms.ollama import Ollama
from llama_parse import LlamaParse
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.core.embeddings import resolve_embed_model
from llama_index.core.agent import ReActAgent
from llama_index.core.tools import QueryEngineTool, ToolMetadata
from code_gen.prompts import context
from dotenv import load_dotenv


load_dotenv()


def main() -> None:
    # RAG model.
    llm = Ollama(model='llama2')
    # result = llm.complete('Hello, world')
    # print(result)

    parser = LlamaParse(
        # api_key=os.environ['LLAMA_CLOUD_API_KEY'],
        result_type='markdown')
    file_extractor = {'.pdf': parser}

    documents = SimpleDirectoryReader(
        'res/data',
        file_extractor=file_extractor,
    ).load_data()

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

    tools = [
        QueryEngineTool(
            query_engine=query_engine,
            metadata=ToolMetadata(
                name='api_documentation',
                description='this gives documentation about code for an API./ Use this for reading  docs for the API',
            )
        ),
    ]

    # For code generation.
    code_llm = Ollama(model='codellama')
    agent = ReActAgent.from_tools(
        tools=tools,
        llm=code_llm,
        verbose=True,
        context=context,
    )

    while (prompt := input('Enter a prompt (q to quit): ')) != 'q':
        result = agent.query(prompt)
        print(result)


if __name__ == '__main__':
    main()
