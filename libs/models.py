# layoutparser 관련 모델 설정

import layoutparser as lp
from contextlib import contextmanager
import sys, os

@contextmanager
def suppress_output():
    with open(os.devnull, "w") as devnull:
        old_stdout = sys.stdout
        old_stderr = sys.stderr
        sys.stdout = devnull
        sys.stderr = devnull
        try:
            yield
        finally:
            sys.stdout = old_stdout
            sys.stderr = old_stderr

with suppress_output():
    layout_model = lp.Detectron2LayoutModel('lp://PubLayNet/faster_rcnn_R_50_FPN_3x/config', 
                                    extra_config=["MODEL.ROI_HEADS.SCORE_THRESH_TEST", 0.8],
                                    label_map={0: "Text", 1: "Title", 2: "List", 3:"Table", 4:"Figure"},
                                    )

layout_ocr_agent = lp.TesseractAgent()



# llm 모델 설정

# from langchain_ollama import OllamaLLM, ChatOllama
from langchain_ollama import ChatOllama
from langchain_anthropic import ChatAnthropic

from dotenv import load_dotenv
load_dotenv("conf/.env")

# exaone3.5:32b
# llama3.2-vision:90b
# llava:34b
# llama3.3:latest
# gemma3:27b

text_summary_llm = ChatOllama(
    model="exaone3.5:32b",
    temperature=0.3,
    base_url="http://localhost:11434",
    streaming=False,
#    num_thread=4,
#    num_predict=1500,
    timeout=60,
)

vision_summary_llm = ChatOllama(
    model="gemma3:27b",
    temperature=0.3,
    base_url="http://localhost:11434",
    streaming=False,
#    num_thread=4,
#    num_predict=1000,
    timeout=60,
)

methods_extraction_llm = ChatOllama(
    model="exaone3.5:32b",
    temperature=0.3,
    base_url="http://localhost:11434",
    streaming=False,
#    num_thread=4,
#    num_predict=2000,
    timeout=60,
)

claude_llm = ChatAnthropic(
    model="claude-3-5-haiku-20241022",
    temperature=0.3,
)