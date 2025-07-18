# import requests
# import json
import os
import time

# from PIL import Image

from typing import TypedDict

import base64
from io import BytesIO
from PIL import Image


# from langchain_core.output_parsers import StrOutputParser
# from langchain_core.messages import HumanMessage, SystemMessage


from datetime import datetime



class GraphState(TypedDict):
    file: str  # path
    output_dir: str # output_dir
    split_pdfimages: list # pdfimages
    title: str # title
    texts: dict # texts
    texts_summary: list # texts summary
    figures: dict # figures
    figures_summary: list # figures summary
    tables: dict # tables
    tables_summary: list # tables summary



class MultiModal:
    def __init__(self, model, system_prompt=None, user_prompt=None):
        self.model = model
        self.system_prompt = system_prompt
        self.user_prompt = user_prompt
        self.init_prompt()

    def init_prompt(self):
        if self.system_prompt is None:
            self.system_prompt = "You are a helpful assistant on parsing images."
        if self.user_prompt is None:
            self.user_prompt = "Explain the given images in-depth."

    # 이미지를 base64로 인코딩하는 함수 (파일)
    def encode_image_from_file(self, file_path):       
        file_ext = os.path.splitext(file_path)[1].lower()
        if file_ext in [".jpg", ".jpeg"]:
            mime_type = "image/jpeg"
            format = "JPEG"
        elif file_ext == ".png":
            mime_type = "image/png"
            format = "PNG"

        file_path = Image.open(file_path)
        buffered = BytesIO()
        file_path.save(buffered, format=format)
        img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
        buffered.close()
        return f"data:{mime_type};base64,{img_str}"

    # 이미지 경로에 따라 적절한 함수를 호출하는 함수
    def encode_image(self, image_path):
        if not (image_path.startswith("http://") or image_path.startswith("https://")):
            return self.encode_image_from_file(image_path)

    def display_image(self, encoded_image):
        display(Image(url=encoded_image))

    def create_messages(
        self, image_url, system_prompt=None, user_prompt=None, display_image=True
    ):
        encoded_image = self.encode_image(image_url)
        if display_image:
            self.display_image(encoded_image)

        system_prompt = (
            system_prompt if system_prompt is not None else self.system_prompt
        )

        user_prompt = user_prompt if user_prompt is not None else self.user_prompt

        # 인코딩된 이미지를 사용하여 다른 처리를 수행할 수 있습니다.
        messages = [
            {
                "role": "system",
                "content": system_prompt,
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": f"{encoded_image}",
                    },
                    {
                        "type": "text",
                        "text": user_prompt,
                    },
                ],
            },
        ]

        return messages

    def invoke(
        self, image_url, system_prompt=None, user_prompt=None, display_image=True
    ):
        messages = self.create_messages(
            image_url, system_prompt, user_prompt, display_image
        )
        # print(f"시작")
        # start_time = datetime.now()
        response = self.model.invoke(messages)
        # print(response.content)
        # end_time = datetime.now()
        # print(f"종료")
        # print(f"소요 시간: {end_time - start_time}")
        return response.content

    def batch(
        self,
        image_urls: list,
        # system_prompts: list = [],
        # user_prompts: list = [],
        system_prompts: list,
        user_prompts: list,
        display_image=False,
    ):
        messages = []
        for image_url, system_prompt, user_prompt in zip(
            image_urls, system_prompts, user_prompts
        ):
            message = self.create_messages(
                image_url, system_prompt, user_prompt, display_image
            )
            messages.append(message)
        response = self.model.batch(messages)

        # return [r for r in response]
        return [r.content for r in response]

    def stream(
        self, image_url, system_prompt=None, user_prompt=None, display_image=True
    ):
        messages = self.create_messages(
            image_url, system_prompt, user_prompt, display_image
        )
        response = self.model.stream(messages)
        return response
    






class TextAgent:
    def __init__(self, model, system_prompt=None, user_prompt=None):
        self.model = model
        self.system_prompt = system_prompt
        self.user_prompt = user_prompt
        self.init_prompt()

    def init_prompt(self):
        if self.system_prompt is None:
            self.system_prompt = "You are a helpful assistant on parsing text."
        if self.user_prompt is None:
            self.user_prompt = "Explain the given text in-depth."


    def create_messages(
        self, system_prompt=None, user_prompt=None
    ):

        system_prompt = (
            system_prompt if system_prompt is not None else self.system_prompt
        )

        user_prompt = user_prompt if user_prompt is not None else self.user_prompt

        # 인코딩된 이미지를 사용하여 다른 처리를 수행할 수 있습니다.
        messages = [
            {
                "role": "system",
                "content": system_prompt,
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": user_prompt,
                    },
                ],
            },
        ]

        return messages

    def invoke(
        self, system_prompt=None, user_prompt=None,
    ):
        messages = self.create_messages(
            system_prompt, user_prompt
        )
        # print(f"시작")
        # start_time = datetime.now()
        response = self.model.invoke(messages)
        # print(response.content)
        # end_time = datetime.now()
        # print(f"종료")
        # print(f"소요 시간: {end_time - start_time}")
        return response.content

    def batch(
        self,
        system_prompts: list, user_prompts: list,
    ):
        messages = []
        for system_prompt, user_prompt in zip(
            system_prompts, user_prompts
        ):
            message = self.create_messages(
                system_prompt, user_prompt
            )
            messages.append(message)
        response = self.model.batch(messages)

        # return [r for r in response]
        return [r.content for r in response]

    def stream(
        self, system_prompt=None, user_prompt=None
    ):
        messages = self.create_messages(
            system_prompt, user_prompt
        )
        response = self.model.stream(messages)
        return response
    



from pathlib import Path
import pickle
import hashlib

class LocalStateCache:
    def __init__(self, cache_dir="./cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
    
    def _generate_cache_key(self, keyword, file_path):
        # 파일 경로와 키워드를 조합하여 유니크한 캐시 키 생성
        combined = f"{file_path}_{keyword}"
        return hashlib.md5(combined.encode()).hexdigest()
    
    def save_state(self, state, keyword, file_path):
        cache_key = self._generate_cache_key(keyword, file_path)
        cache_file = self.cache_dir / f"{cache_key}.pkl"
        
        with open(cache_file, "wb") as f:
            pickle.dump(state, f)
        print(f"캐시가 저장되었습니다: {cache_file}")
    
    def load_state(self, keyword, file_path):
        cache_key = self._generate_cache_key(keyword, file_path)
        cache_file = self.cache_dir / f"{cache_key}.pkl"
        
        if cache_file.exists():
            with open(cache_file, "rb") as f:
                print(f"캐시를 불러왔습니다: {cache_file}")
                return pickle.load(f)
        return None
    
    def clear_cache(self):
        # 캐시 전체 삭제
        for cache_file in self.cache_dir.glob("*.pkl"):
            cache_file.unlink()
        print("캐시가 모두 삭제되었습니다.")