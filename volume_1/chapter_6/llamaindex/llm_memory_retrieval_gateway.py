"""Chapter 6 retrieval-ready LlamaIndex gateway built on chapter 5 structured memory."""

import os
import sys

CHAPTER_4_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "chapter_4"))
CHAPTER_5_LLAMAINDEX = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "chapter_5", "llamaindex"))

sys.path.append(CHAPTER_4_ROOT)
sys.path.append(CHAPTER_5_LLAMAINDEX)

from llm_memory_structured_gateway import LlamaIndexLLMManager as Chapter5StructuredManager
from utils import interactive_cli


class LlamaIndexLLMManager(Chapter5StructuredManager):
    """Chapter 6 manager with retrieval-oriented naming compatibility."""

    def __init__(self, memory_enabled: bool = True):
        super().__init__(memory_enabled=memory_enabled)
        self.framework = "LlamaIndex+Memory+Retrieval"


def main():
    if len(sys.argv) > 1 and sys.argv[1] == "web":
        from web import run_web_server

        run_web_server(lambda: LlamaIndexLLMManager(memory_enabled=True))
    else:
        interactive_cli(LlamaIndexLLMManager(memory_enabled=True))


if __name__ == "__main__":
    main()
