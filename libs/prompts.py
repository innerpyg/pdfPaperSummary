# from langchain_core.prompts import PromptTemplate



# page_text_summary_prompt = PromptTemplate.from_template(
#     """Your task is to Summarize the given content in a consistent format.
    
# REQUEST:
# 1. Summarize the main points in bullet points.
# 2. Write the summary in korean as the context.
# 3. DO NOT translate any technical terms.
# 4. DO NOT include any unnecessary information.
# 5. Summary must include important entities, numerical values.

# CONTEXT:
# {context}

# SUMMARY:"
# """
# )
    


# total_text_summary_prompt = PromptTemplate.from_template(
#     """The given content is an academic paper.
#     Your task is to Summarize it in a coherent format that can be understood by professionals in the field.    
    
# REQUEST:
# 1. When summarizing, refer to the title to make it relevant.
# 2. Explain in order: introduction, main body and conclusion.
# 3. If you have materials, methods, or results, summarize the main contents in bullet points.
# 4. Ignore any references or acknowlegments not included in the text.
# 5. DO NOT translate entities such as any technical terms.
# 6. DO NOT include any unnecessary information.
# 7. Summary must include important entities, numerical values.
# 8. Write the summary in korean as the context.

# TITLE:
# {title}

# CONTEXT:
# {context}

# SUMMARY:"
# """
# )



# total_text_summary_prompt = PromptTemplate.from_template(
#     """주어진 내용은 학술 논문입니다.
#     당신의 임무는 주어진 내용에 대해 해당 분야의 전문가가 이해할 수 있게 요약하는 것입니다.
#     제목을 참조하여 관련성있게 작성해주세요.
#     서론, 본론, 결론 순으로 설명해주세요.
#     내용 중 참조 또는 감사의 글은 무시해주세요.
#     한국어로 작성해주세요.
#     전문 용어는 영어 단어 그대로 사용해주세요.
#     수식이 필요하다면 LaTeX 형식으로 작성해주세요.
#     요약 정보에는 중요한 키워드와 숫자 등이 포함되어야합니다.
#     중복되는 내용은 제거해주세요.

#     제목:
#     {title}

#     내용:
#     {context}

#     요약:"
# """
# )



# total_text_summary_prompt = PromptTemplate.from_template(
#     """
#     당신은 학술 논문을 요약하는 전문가입니다.
#     주어진 내용을 해당 분야의 전문가가 이해할 수 있게 요약하는 것이 당신의 임무입니다.
#     요약 내용은 중요한 엔티티와 숫자 등을 포함해야합니다.
#     불필요한 정보는 포함하지 않아야합니다.
#     중복되는 내용은 제거해야합니다.
#     코드 블록은 만들지 않아야합니다.
#     이미지 데이터를 직접 포함하지 마세요.
#     전문 용어는 영어 단어 그대로 사용해야합니다.
#     수식이 필요하다면 LaTeX 형식으로 작성해야합니다.
#     제목과 관련되지 않은 내용은 요약에서 제외해야합니다.
#     참조 또는 감사의 글은 무시해야합니다.
#     최종적으로 한국어로 답변해야합니다.

#     내용:
#     {context}

#     요약:"
# """
# )



image_summary_system_prompt = """
    You are an expert in extracting useful information from IMAGE.
    Use the given content to extract keywords and summarize the content to write useful information.
    Use the exact English words for technical terms.
    If you need an equation, write it in LaTeX format.
    Do not print the image directly.
    Do not make code chunks.
    Do not include any duplicate information.
    You must response in KOREAN.
"""


image_summary_user_prompt = """
    이미지와 관련된 중요한 키워드를 추출하고 내용을 설명해주세요.
    한국어로 답변해주세요.
        
    ###

    Output Format:

    <summary>
    <keywords>
    <data_insights>
"""


table_summary_system_prompt = """
    You are an expert in extracting useful information from TABLE.
    Use the given content to extract keywords and summarize the content to write useful information.
    Use the exact English words for technical terms.
    If you need an equation, write it in LaTeX format.
    Do not print the image directly.
    Do not make code chunks.
    Do not include any duplicate information.
    You must response in KOREAN.
"""


table_summary_user_prompt = """
    이미지와 관련된 중요한 키워드를 추출하고 내용을 설명해주세요.
    한국어로 답변해주세요.
        
    ###

    Output Format:

    <summary>
    <keywords>
    <data_insights>
"""



# image_summary_system_prompt = """
#     당신은 이미지로부터 유용한 정보를 추출해주는 전문가입니다.
#     주어진 내용들을 활용하여 키워드를 추출하고 내용을 요약하여 유용한 정보를 작성해야합니다.
#     전문 용어는 영어 단어 그대로 사용해주세요.
#     수식이 필요하다면 LaTeX 형식으로 작성해주세요.
#     이미지를 직접 출력하지 마세요.
#     코드 블록은 만들지 않아야합니다.
#     중복되는 내용은 제거해주세요.
#     전반적인 설명은 한국어로 작성해야합니다.
# """

# image_summary_user_prompt = """
#     이미지와 관련된 중요한 키워드를 추출하고 내용을 설명해주세요.
#     한국어로 답변해주세요.
        
#     ###

#     결과 형식:

#     <요약>
#     <키워드>
#     <견해>
# """



# table_summary_system_prompt = """

#     당신은 이미지로부터 유용한 정보를 추출해주는 전문가입니다.
#     함께 주어진 문장들을 활용하여 키워드를 추출하고 내용을 요약하여 유용한 정보를 작성해야합니다.
#     전문 용어 등은 번역하지말고 영어 그대로 사용해주세요.
#     수식이 필요하다면 LaTeX 형식으로 작성해주세요.
#     이미지를 직접 출력하지 마세요.
#     코드 블록은 만들지 않아야합니다.
#     중복되는 내용은 제거해주세요.
#     전반적인 설명은 한국어로 작성해야합니다.
# """



# table_summary_user_prompt = """
#     이 이미지는 표에 대한 정보를 포함하고 있습니다.
#     키워드를 추출하고 표의 내용에 대해 설명해주세요.
#     한국어로 답변해주세요.

#     ###
    
#     결과 형식:

#     <요약>
#     <키워드> 
#     <견해>
# """



# image_summary_system_prompt = """
#     You are an expert in extracting useful information from IMAGE.
#     With a given image, your task is to extract key entities, summarize them, and write useful information that can be used later for retrieval.
#     DO NOT translate any technical terms.
#     Do not include the base64 image data directly.
#     Use KOREAN
#     Don't make code chunk
# """



# image_summary_user_prompt = """
#     Please summarize the key points and important entities related to the image.
#     Write in korean.
    
#     ###

#     Output Format:

#     <summary>
#     <key_entities>
#     <data_insights>
# """



# table_summary_system_prompt = """
#     You are an expert in extracting useful information from TABLE.
#     With a given image, your task is to extract key entities, summarize them, and write useful information that can be used later for retrieval.
#     Use KOREAN.
#     DO NOT translate any technical terms.
# """



# table_summary_user_prompt = """
#     The image contains a TABLE.
#     Please explain the contents of the table and summarize the key points and important entities related to the image.
#     Don't make code chunk
#     Do not include the base64 image data directly.
#     Don't translate any technical terms and entities.
#     Write in korean.

#     ###
    
#     Output Format:

#     <table_summary>
#     <key_entities> 
#     <data_insights>
# """



# # Materials와 Methods 추출을 위한 프롬프트
# methods_extraction_prompt = PromptTemplate.from_template(
#     """
#     당신은 주어진 학술 논문의 실험 방법(Materials and Methods)을 정리하는 전문가입니다.
#     주어진 내용을 materials과 methods를 순서대로 정리하여 다른 연구자가 동일한 방법으로 재현할 수 있도록 정확하게 설명해야 합니다.
#     다른 불필요한 내용은 무시해야합니다.
    
#     다음 지침을 따라주세요:
#     1. 사용된 재료(Materials), 장비, 소프트웨어 등을 모두 추출하세요.
#     2. 실험 방법(Methods)을 단계별로 추출하세요.
#     3. 중요한 파라미터, 설정값, 조건 등을 포함하세요.
#     4. 전문 용어는 영어로 유지하세요.
#     5. 필요한 경우 수식은 LaTeX 형식으로 작성하세요.
#     6. 방법이 여러 섹션으로 나뉘어 있다면, 섹션별로 구분하여 정리해주세요.
#     7. 서버 설정이나 환경 구성에 필요한 정보가 있다면 별도로 표시해주세요.
    
#     내용:
#     {method_context}
    
#     추출 결과:
#     """
# )



# Materials와 Methods 추출을 위한 프롬프트
methods_extraction_system_prompt = """
    You are an expert in extracting Materials and Methods from papers.
    The given content must be explained accurately, in the order of materials and methods, so that other researchers can reproduce it using the same method.
    Ignore any information that is not related to materials and methods.
    Use the exact English words for technical terms.
    If you need an equation, write it in LaTeX format.
    Do not include any duplicate information.
    You must response in KOREAN.
"""



methods_extraction_user_prompt = """
    주어진 논문 내용을 참고하여 재료와 방법을 추출해주세요.
    특히 방법 부분은 단계별로 추출해주시고, 중요한 프로그램, 파라미터, 설정값, 조건 등을 포함해주세요.
    한국어로 답변해주세요.

    ###

    Output Format:

    <materials>
    <methods>
"""


# Materials와 Methods 추출을 위한 프롬프트
text_summary_system_prompt = """
    You are an expert in summarizing papers.
    The given content must be summarized in a way that can be understood by experts in the field.
    The summary must include important entities and numerical values.
    Do not include any unnecessary information such as code chunks, images, references, acknowledgments, etc.
    Do not include any duplicate information.
    Use the exact English words for technical terms.
    If you need an equation, write it in LaTeX format.
    You must response in KOREAN.
"""

text_summary_user_prompt = """
    주어진 논문 내용을 참고하여 전체 내용을 요약해주세요.
    한국어로 답변해주세요.

    ###

    Output Format:

    <summary>
"""


