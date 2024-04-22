import os
from llama_index.core.tools import FunctionTool


def code_reader_func(filename: str, base: str = 'res/data') -> dict[str, str]:
    """Read contents of a file and return a JSON response.

    Arguments:
        filename (str): Name of the file (within `base` directory).
        base (str): Base directory where the file resides.
            Defaults to "res/data"

    Return:
        dict[str, str] - A Response object.

    """
    path = os.path.join(base, filename)
    try:
        with open(path, 'r') as f:
            content = f.read()
            return {'file_content': content}
    except Exception as e:
        return {'error': f'{e}'}


code_reader = FunctionTool.from_defaults(
    fn=code_reader_func,
    name='code_reader',
    description='''this tool can read the contents of code files and return
    their results. Use this when you need to read the contents of a file''',
)
