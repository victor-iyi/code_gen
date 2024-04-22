from llama_index.llms.ollama import Ollama
from llama_parse import LlamaParse
from llama_index.core import SimpleDirectoryReader
from llama_index.core.agent import ReActAgent

from code_gen.prompts import context
from code_gen.code_reader import code_reader
from code_gen.query_engine import query_engine

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

    tools = [
        query_engine(llm, documents),
        code_reader,
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
