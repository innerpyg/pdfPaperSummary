import os
from libs.classes import GraphState
import pdf2image
import numpy as np
import math
import cv2
from PIL import Image
import markdown
import shutil
import zipfile

# custom libs 임포트
from libs.models import (
    layout_model,
    layout_ocr_agent,
    vision_summary_llm,
    methods_extraction_llm,
    text_summary_llm,
    suppress_output,
    claude_llm
)

from libs.prompts import (
    table_summary_system_prompt,
    image_summary_system_prompt,
    table_summary_user_prompt,
    image_summary_user_prompt,
    methods_extraction_system_prompt,
    methods_extraction_user_prompt,
    text_summary_system_prompt,
    text_summary_user_prompt
)

from libs.classes import MultiModal, TextAgent

from langchain_core.documents import Document
from langchain_core.runnables import chain


from concurrent.futures import ProcessPoolExecutor, as_completed
import functools
import multiprocessing

multiprocessing.set_start_method('spawn', force=True)

from contextlib import contextmanager
import sys
import warnings

import torch
# cuda_available = torch.cuda.is_available()
# if cuda_available:
#     print(f"Using CUDA Device: {torch.cuda.get_device_name(torch.cuda.current_device())}")
# deviceToUse=torch.cuda.current_device()

# pdf page 별로 저장
def pdf_to_image(state: GraphState, keyword: str, today: str):
    """
    pdf 파일 이미지로 변환
    """

    file = state["file"]

    output_dir = "Summarize/" + today + "/" + keyword + "/" + os.path.splitext(os.path.basename(file))[0].replace(" ", "")
    page_dir_name = os.path.join(output_dir, "pages")
    if os.path.exists(output_dir):
        # print(f"\n{output_dir} 폴더가 이미 존재합니다. 삭제 후 진행해주세요.\n")
        print(f"\n{output_dir} 폴더가 이미 존재합니다.\n덮어쓰기 진행할까요? (y/n) : ", end="")
        answer = input()
        if answer == "y" or answer == "Y":
            shutil.rmtree(output_dir)
            # print(f"\n{output_dir} 폴더를 삭제하고 진행합니다.\n")
            os.makedirs(page_dir_name, exist_ok=True)
        else:
            print(f"\n프로그램을 종료합니다.\n")
            os._exit(os.EX_OK)
    else:
        os.makedirs(page_dir_name, exist_ok=True)

    images = pdf2image.convert_from_path(file)

    split_pdfimages = []
    for page_num, image in enumerate(images, 1):
        output_file = f"{page_dir_name}/page{page_num}.png"
        image.save(output_file, dpi=(300, 300))
        split_pdfimages.append(output_file)

    print(f"\npdf를 페이지 별 png로 저장 완료\n")

    return GraphState(output_dir=output_dir,split_pdfimages=split_pdfimages)


def process_single_page(page_data):
    """
    각 페이지 별 레이아웃 및 메타데이터 추출 함수 호출
    """

    try:
        page_num, image_path, output_dir = page_data
        image = Image.open(image_path)
        image_array = np.array(image)
        
        # BGR에서 RGB로 변환
        if len(image_array.shape) == 3:  # 컬러 이미지인 경우에만 변환
            image_array = cv2.cvtColor(image_array, cv2.COLOR_BGR2RGB)

        with suppress_output():
            layout = layout_model.detect(image_array)
        
        page_width = image_array.shape[1]
        page_height = image_array.shape[0]
        
        texts = []
        figures = []
        tables = []
        
        # 블록 타입별 색상 정의 (RGB 형식)
        colors = {
            "Text": (255, 0, 0),    # 빨간색
            "Title": (255, 0, 0),   # 빨간색
            "Figure": (0, 255, 0),  # 초록색
            "Table": (0, 0, 255)    # 파란색
        }

        # 모든 블록에 대해 bbox 그리기
        visualization_image = image_array.copy()
        for block in layout:
            x1, y1, x2, y2 = map(int, block.coordinates)
            color = colors.get(block.type, (128, 128, 128))  # 기본값은 회색
            cv2.rectangle(visualization_image, (x1, y1), (x2, y2), color, 2)
            
            # 블록 타입을 표시하는 텍스트 추가
            cv2.putText(visualization_image, block.type, (x1, y1-10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # 시각화 이미지 저장
        vis_dir = os.path.join(output_dir, "visualizations")
        os.makedirs(vis_dir, exist_ok=True)
        vis_path = os.path.join(vis_dir, f"page{page_num}_blocks.png")
        cv2.imwrite(vis_path, visualization_image)

        # 텍스트와 타이틀 정보 추출
        for block in layout:
            if block.type in ["Text", "Title"]:
                if not is_margin_text(block, page_width, page_height):
                    text_info = process_block_with_ocr(block, image_array, layout_ocr_agent, page_num)
                    text_info['block_type'] = block.type
                    texts.append(text_info)

        # figure와 table 정보 추출
        png_dir_name = os.path.join(output_dir, "pngs")
        os.makedirs(png_dir_name, exist_ok=True)
        
        for block in layout:
            if block.type == "Figure":
                # ... existing figure processing code ...
                if not is_nested_figure(block, figures):
                    # 이미지 추출
                    # 소수점만큼의 여백을 주어 이미지 추출
                    figure_x1, figure_y1, figure_x2, figure_y2 = (
                        int(block.coordinates[0]),  # 좌상단 x좌표 (버림)
                        int(block.coordinates[1]),  # 좌상단 y좌표 (버림)
                        math.ceil(block.coordinates[2]),  # 우하단 x좌표 (올림)
                        math.ceil(block.coordinates[3])   # 우하단 y좌표 (올림)
                    )

                    cv2.rectangle(image_array, (figure_x1, figure_y1), (figure_x2, figure_y2), (0, 255, 0), 2)

                    x1, y1, x2, y2 = map(int, block.coordinates)
                    cropped_image = image_array[figure_y1:figure_y2, figure_x1:figure_x2]

                    # 이미지 저장
                    figure_num = len(figures) + 1
                    file_name = f"{png_dir_name}/page{page_num}_figure{figure_num}.png"
                    cv2.imwrite(file_name, cropped_image)

                    # 캡션 추출
                    caption = find_caption_text(block, texts, is_figure=True)
                    figures.append({
                        "coordinates": block.coordinates,
                        "score": block.score,
                        "block": block,
                        "file": file_name,
                        "text": caption['text'] if caption else None,
                        "page_num": page_num,
                        "figure_num": figure_num
                    })
                    if caption:
                        texts.remove(caption)  # 캡션으로 사용된 텍스트는 제거

            elif block.type == "Table":
                # ... existing table processing code ...
                if not is_nested_table(block, tables):
                    # 이미지 추출
                    table_x1, table_y1, table_x2, table_y2 = map(int, block.coordinates)
                    cropped_image = image_array[table_y1:table_y2, table_x1:table_x2]

                    # 이미지 저장
                    table_num = len(tables) + 1
                    file_name = f"{png_dir_name}/page{page_num}_table{table_num}.png"
                    cv2.imwrite(file_name, cropped_image)

                    # 캡션 추출
                    caption = find_caption_text(block, texts, is_figure=False)
                    tables.append({
                        "coordinates": block.coordinates,
                        "score": block.score,
                        "block": block,
                        "file": file_name,
                        "text": caption['text'] if caption else None,
                        "page_num": page_num,
                        "table_num": table_num
                    })
                    if caption:
                        texts.remove(caption)  # 캡션으로 사용된 텍스트는 제거  

        # 단 구조 확인 및 텍스트 정렬
        is_two_column = check_column_layout(texts, page_width, page_num)
        # texts = remove_crossing_blocks(texts, page_width, page_height, is_two_column)

        # if is_two_column:
        #     sorted_texts = sort_text_blocks_with_columns(texts, page_width, is_two_column)
        # else:
        #     sorted_texts = sorted(texts, key=lambda x: x['coordinates'][1])

        sorted_texts = sort_text_blocks_with_columns(texts, page_width, is_two_column, page_num)

    finally:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            if 'layout' in locals():
                del layout

    return page_num, sorted_texts, figures, tables



def extract_metadata(state: GraphState):
    """
    각 페이지 별 메타데이터 추출
    """

    print(f"\n레이아웃 및 메타데이터 추출 시작\n")
    
    split_pdfimages = state["split_pdfimages"]
    output_dir = state["output_dir"]
    
    # 병렬 처리를 위한 데이터 준비
    page_data = [(page_num, image_path, output_dir) 
                 for page_num, image_path in enumerate(split_pdfimages, 1)]
    
    total_texts = []
    total_figures = []
    total_tables = []
    title = None
    
    # GPU 메모리를 고려하여 동시 실행할 프로세스 수 조정
    # 200GB / 2GB = 100개까지 가능하지만, 여유를 두고 설정
    max_workers = min(len(page_data), 10)  # 최대 30페이지 프로세스 동시 실행
    
    # ProcessPoolExecutor를 사용한 병렬 처리
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(process_single_page, data) for data in page_data]
        
        # 결과를 페이지 번호 순서대로 정렬하기 위한 딕셔너리
        results = {}
        
        for future in as_completed(futures):
            page_num, texts, figures, tables = future.result()
            results[page_num] = (texts, figures, tables)
            # print(f"페이지 {page_num} 처리 완료")
        
        # 페이지 번호 순서대로 결과 정리
        for page_num in sorted(results.keys()):
            texts, figures, tables = results[page_num]

            # 첫 페이지에서 제목 추출
            if page_num == 1 and texts:
                image = Image.open(split_pdfimages[0])
                image_array = np.array(image)
                title = max(texts, key=lambda x: (
                    -x['coordinates'][1],
                    abs(x['coordinates'][0] - image_array.shape[1]/2)
                ))
                title["text"] = title["text"].replace("\n", " ")
            
            if texts:
                total_texts.append(texts)
            if figures:
                total_figures.append(figures)
            if tables:
                total_tables.append(tables)

    print(f"\n추출 완료\n")
    
    print(f"\n불필요한 문단 제거\n")
    # 중단점이 될 섹션 키워드 확장
    stop_sections = [
        # 기본 참고문헌 관련
        'references', 'bibliography', 'works cited', 'literature cited',
        # 감사의 글 관련
        'acknowledgements', 'acknowledgments', 'funding', 
        # 부록 관련
        'appendix', 'appendices', 'supplementary', 'supporting information',
        'supplemental material', 'additional information',
        # 이해관계 관련
        'conflict of interest', 'competing interests', 'disclosure statement',
        # 저자 정보 관련
        'author contributions', 'author information', 'authors\' contributions',
        # 기타 행정적 섹션
        'ethics declarations', 'compliance with ethical standards', 'declarations',
        # 저널 관련 정보
        'journal information', 'copyright notice', 'publisher\'s note'
    ]

    method_sections = [
        "material", "method", "methodology", "experimental",
        "procedure", "implementation", "setup", "protocol",
        "data availability", "code availability",
    ]
    method_stop_sections = [
        "result", "discussion", "conclusions",
    ]

    #  모든 텍스트를 하나의 문자열로 결합
    combined_texts = []
    method_texts = []
    found_stop_section = False
    found_method_section = False
    
    for page_texts in total_texts:
        
        if found_stop_section:
            break
            
        for text_dict in page_texts:
            if "text" in text_dict and text_dict["text"]:
                # Title 블록에 대해서만 stop section 확인
                if text_dict.get('block_type') == 'Title':
                    current_text = text_dict["text"].lower()
                # stop section을 발견하면 플래그를 설정하고 내부 루프 종료
                    if any(section in current_text for section in stop_sections):
                        found_stop_section = True
                        break

                    if any(section in current_text for section in method_sections):
                        found_method_section = True

                    if found_method_section:
                        if any(section in current_text for section in method_stop_sections):
                            found_method_section = False

                if found_method_section:
                    method_texts.append(text_dict["text"])

                combined_texts.append(text_dict["text"])

        if found_stop_section:
            break

    combined_text = " ".join(combined_texts)
    method_text = " ".join(method_texts)

    # return GraphState(figures=total_figures, tables=total_tables, texts=total_texts, title=title)
    return GraphState(figures=total_figures, tables=total_tables, texts=combined_text, method_texts=method_text, title=title)



# # 메타데이터 추출
# def extract_metadata(state: GraphState):

#     print(f"\n레이아웃 및 메타데이터 추출 시작\n")

#     split_pdfimages = state["split_pdfimages"]
#     output_dir = state["output_dir"]

#     png_dir_name = os.path.join(output_dir, "pngs")
#     if not os.path.exists(png_dir_name):
#         os.makedirs(png_dir_name, exist_ok=True)

#     total_texts = []
#     total_figures = []
#     total_tables = []

#     for page_num, image in enumerate(split_pdfimages, 1):
#         image = Image.open(image)
#         image_array = np.array(image)
#         layout = layout_model.detect(image_array)

#         page_width = image_array.shape[1]
#         page_height = image_array.shape[0]

#         texts = []
#         figures = []
#         tables = []

#         # 우선 text와 title 정보 추출
#         for block in layout:
#             if block.type in ["Text", "Title"]:
#                 if not is_margin_text(block, page_width, page_height):
#                     text_info = process_block_with_ocr(block, image_array, layout_ocr_agent, page_num)
#                     text_info['block_type'] = block.type  # Text인지 Title인지 구분       
#                     texts.append(text_info)


#     #             x1, y1, x2, y2 = map(int, block.coordinates)
#     #             # cv2.rectangle(image_array, (x1, y1), (x2, y2), (255, 0, 0), 2)

#         # figure와 table 정보 추출하는데, 캡션으로 활용될 text block 가져오기
#         for block in layout:            
#             if block.type == "Figure":
#                 if is_nested_figure(block, figures):
#                     continue

#                 # 이미지 추출
#                 # 소수점만큼의 여백을 주어 이미지 추출
#                 figure_x1, figure_y1, figure_x2, figure_y2 = (
#                     int(block.coordinates[0]),  # 좌상단 x좌표 (버림)
#                     int(block.coordinates[1]),  # 좌상단 y좌표 (버림)
#                     math.ceil(block.coordinates[2]),  # 우하단 x좌표 (올림)
#                     math.ceil(block.coordinates[3])   # 우하단 y좌표 (올림)
#                 )

#                 # cv2.rectangle(image_array, (figure_x1, figure_y1), (figure_x2, figure_y2), (0, 255, 0), 2)

#                 x1, y1, x2, y2 = map(int, block.coordinates)
#                 cropped_image = image_array[figure_y1:figure_y2, figure_x1:figure_x2]

#             # 이미지 저장
#                 figure_num = len(figures) + 1  # 현재 figure의 번호
#                 file_name = f"{png_dir_name}/page{page_num}_figure{figure_num}.png"
#                 cv2.imwrite(file_name, cropped_image)

#                 caption = find_caption_text(block, texts, is_figure=True)
#                 figures.append({
#                     "coordinates": block.coordinates,
#                     "score": block.score,
#                     "block": block,
#                     "file": file_name,
#                     "text": caption['text'] if caption else None,
#                     "page_num": page_num,
#                     "figure_num": figure_num
#                 })
#                 if caption:
#                     texts.remove(caption)  # 캡션으로 사용된 텍스트는 제거
            
#             elif block.type == "Table":
#                 if is_nested_table(block, tables):
#                     continue

#                 # 이미지 추출
#                 table_x1, table_y1, table_x2, table_y2 = map(int, block.coordinates)
#                 # cv2.rectangle(image_array, (table_x1, table_y1), (table_x2, table_y2), (0, 0, 255), 2)
#                 cropped_image = image_array[table_y1:table_y2, table_x1:table_x2]

#                 # 이미지 저장
#                 table_num = len(tables) + 1  # 현재 table의 번호
#                 file_name = f"{png_dir_name}/page{page_num}_table{table_num}.png"
#                 cv2.imwrite(file_name, cropped_image)

#                 caption = find_caption_text(block, texts, is_figure=False)
#                 tables.append({
#                     "coordinates": block.coordinates,
#                     "score": block.score,
#                     "block": block,
#                     "file": file_name,
#                     "text": caption['text'] if caption else None,
#                     "page_num": page_num,
#                     "table_num": table_num
#                 })
#                 if caption:
#                     texts.remove(caption)  # 캡션으로 사용된 텍스트는 제거

#         # output_image_file = f"{png_dir_name}/page{page_num}_with_boxes.png"
#         # cv2.imwrite(output_image_file, image_array)
        

#         # 첫 페이지에서 위치 기반 제목만 찾아서 별도로 저장 후 문단 title 수정
#         if page_num == 1 and texts:
#             title = max(texts, key=lambda x: (
#                 -x['coordinates'][1],
#                 abs(x['coordinates'][0] - image_array.shape[1]/2)
#             ))
#             title["text"] = title["text"].replace("\n", " ")

            
#         # 단 구조 확인    
#         is_two_column = check_column_layout(texts, page_width, page_num)
#         # 횡단 블록 제거
#         texts = remove_crossing_blocks(texts, page_width, is_two_column)
#         # 구조에 따라 텍스트 정렬
#         if is_two_column:
#             sorted_texts = sort_text_blocks_with_columns(texts, page_width, is_two_column)
#         else:
#             sorted_texts = sorted(texts, key=lambda x: x['coordinates'][1])
#         # 문단 구조화
#         previous_block_type = None
#         for text in sorted_texts:
#             if text['block_type'] == "Title" and previous_block_type != "Title":
#                 text['text'] = "\n" + text['text']  # 뉴라인 추가
#             previous_block_type = text['block_type']

#         if figures:
#             total_figures.append(figures)
#         if tables:
#             total_tables.append(tables)
#         if sorted_texts:
#             total_texts.append(sorted_texts)

#     print(f"\n추출 완료\n")

#     return GraphState(figures=total_figures, tables=total_tables, texts=total_texts, title=title)



# 여백 텍스트 확인
def is_margin_text(block, page_width, page_height, margin_threshold=0.1):
    """
    텍스트 블록이 여백의 세로 글씨인지 확인
    """
    margin_width = page_width * margin_threshold
    x1, y1, x2, y2 = block.coordinates
    
    # 왼쪽 또는 오른쪽 여백에 있는지 확인
    is_left_margin = x1 < margin_width
    is_right_margin = x2 > (page_width - margin_width)
    
    # 세로로 긴 텍스트인지 확인
    height = y2 - y1
    width = x2 - x1
    is_vertical = height > width * 1.5
    
    return (is_left_margin or is_right_margin) and is_vertical



# ocr 처리  
def process_block_with_ocr(block, image_array, ocr_agent, page_num):
    """
    블록을 OCR 처리하고 결과를 포함한 dictionary 반환
    """
    # 이미지 추출 및 OCR 처리
    block_image = block.crop_image(image_array)
    with suppress_output():
        ocr_text = ocr_agent.detect(block_image).split('\n\n', 1)[0].strip() + '\n'
    
    return {
        "coordinates": block.coordinates,
        "score": block.score,
        "block": block,
        "text": ocr_text,
        "page_num": page_num
    }



# 유사 이미지 확인
def is_nested_figure(block, figures):
    """
    주어진 figure가 이미 존재하는 figure 리스트에 90% 이상 포함되는지 확인
    """
    for existing_figure in figures:
        if is_within_percentage(block.coordinates, existing_figure['coordinates'], threshold=0.9):
            return True
    return False



# 유사 표 확인
def is_nested_table(block, tables):
    """
    주어진 table이 이미 존재하는 table 리스트에 90% 이상 포함되는지 확인
    """
    for existing_table in tables:
        if is_within_percentage(block.coordinates, existing_table['coordinates'], threshold=0.9):
            return True
    return False



def is_within_percentage(new_coords, existing_coords, threshold=0.9):
    """
    새로운 좌표가 기존 좌표 범위 내에 있는지 비율로 확인
    """
    new_x1, new_y1, new_x2, new_y2 = new_coords
    existing_x1, existing_y1, existing_x2, existing_y2 = existing_coords
    
    # 새로운 블록의 넓이
    new_area = (new_x2 - new_x1) * (new_y2 - new_y1)
    # 기존 블록의 넓이
    existing_area = (existing_x2 - existing_x1) * (existing_y2 - existing_y1)
    
    # 포함 비율 계산
    if new_area > 0 and existing_area > 0:
        overlap_x1 = max(new_x1, existing_x1)
        overlap_y1 = max(new_y1, existing_y1)
        overlap_x2 = min(new_x2, existing_x2)
        overlap_y2 = min(new_y2, existing_y2)
        
        overlap_area = max(0, overlap_x2 - overlap_x1) * max(0, overlap_y2 - overlap_y1)
        
        # 포함 비율 계산
        return (overlap_area / new_area) >= threshold
    return False



# 캡션 텍스트 찾기  
def find_caption_text(block, texts, is_figure=True):
    """
    Figure나 Table의 캡션 텍스트 찾기
    
    Args:
        block: Figure 또는 Table 블록
        texts: 모든 텍스트 블록 리스트
        is_figure: True면 Figure(아래 찾기), False면 Table(위 찾기)
    """
    figure_x1, figure_y1, figure_x2, figure_y2 = block.coordinates
    figure_width = figure_x2 - figure_x1
    
    closest_text = None
    min_distance = float('inf')
    
    for text in texts:
        text_x1, text_y1, text_x2, text_y2 = text['coordinates']
        
        # x축 중첩 확인 (캡션은 보통 Figure/Table과 가로 위치가 겹침)
        x_overlap = (min(figure_x2, text_x2) - max(figure_x1, text_x1)) > (figure_width * 0.5)
        
        if x_overlap:
            if is_figure:
                # Figure의 경우 아래에서 가장 가까운 텍스트
                distance = text_y1 - figure_y2
                if 0 < distance < min_distance:  # 양수여야 아래에 있는 텍스트
                    min_distance = distance
                    closest_text = text
            else:
                # Table의 경우 위에서 가장 가까운 텍스트
                distance = figure_y1 - text_y2
                if 0 < distance < min_distance:  # 양수여야 위에 있는 텍스트
                    min_distance = distance
                    closest_text = text
    
    return closest_text



# 단 구조 확인
def check_column_layout(texts, page_width, page_num):
    """
    텍스트 블록들이 중앙을 가로지르는지 확인하여 1단/2단 구조 판단
    threshold: 0.3 (30% 이상의 블록이 중앙을 가로지르면 1단으로 판단)
    """
    center = page_width / 2
    width_threshold = page_width * 0.4  # 블록이 중앙을 가로지르는지 판단하는 기준
    layout_threshold = 0.3  # 1단/2단 판단 기준
    
    crossing_center = 0
    total_blocks = len(texts)
    
    layout_threshold = 0.6 if page_num == 1 else 0.3

    for text in texts:
        block_width = text['coordinates'][2] - text['coordinates'][0]
        block_center = (text['coordinates'][0] + text['coordinates'][2]) / 2
        
        # 블록이 중앙을 가로지르는지 확인
        if block_width > width_threshold and abs(block_center - center) < page_width * 0.2:
            crossing_center += 1
    
    crossing_ratio = crossing_center / total_blocks if total_blocks > 0 else 0
    
    # print(f"\n=== 단 구조 분석 ===")
    # print(f"전체 블록 수: {total_blocks}")
    # print(f"중앙 가로지르는 블록 수: {crossing_center}")
    # print(f"가로지르는 비율: {crossing_ratio:.2f}")
    
    # 중앙을 가로지르는 블록이 30% 이상이면 1단, 미만이면 2단
    is_two_column = crossing_ratio < layout_threshold
    # print(f"판단 결과: {'2단' if is_two_column else '1단'} 구조")
    
    return is_two_column



# 횡단 블록 제거
def remove_crossing_blocks(texts, page_width, page_height, is_two_column):
    """
    2단 구조일 경우 페이지를 횡단하는 text block 제거
    """
    if not is_two_column:  # 1단인 경우 그대로 반환
        return texts
        
    center = page_width / 2
    width_threshold = page_width * 0.4
    filtered_texts = []
    
    for text in texts:
        block_width = text['coordinates'][2] - text['coordinates'][0]
        block_center = (text['coordinates'][0] + text['coordinates'][2]) / 2
        block_y = text['coordinates'][1]

        is_top_text = block_y > page_height * 0.1
        is_bottom_text = block_y < page_height * 0.9

        
        # 중앙을 가로지르지 않는 블록만 유지
        if is_top_text or is_bottom_text or not (block_width > width_threshold and abs(block_center - center) < page_width * 0.2):
            filtered_texts.append(text)

    
    # print(f"횡단 블록 제거: {len(texts)} -> {len(filtered_texts)}")
    return filtered_texts



# 단 구조에 따른 텍스트 정렬
def sort_text_blocks_with_columns(texts, page_width, is_two_column, page_num):
    """
    텍스트 블록 정렬
    1단: y축 기준 정렬
    2단: 좌우 단으로 나누어 각각 y축 기준 정렬
    """
    sorted_texts = sorted(texts, key=lambda x: x['coordinates'][1])
    if not is_two_column:
        return sorted_texts

    # 2단인 경우
    center = page_width / 2
    result = []

    # 첫 페이지 일 때
    if page_num == 1:
        i = 0
        while i < len(sorted_texts):
            current = sorted_texts[i]
            current_y1, current_y2 = current['coordinates'][1], current['coordinates'][3]
            current_center = (current['coordinates'][0] + current['coordinates'][2]) / 2
            
            # 다음 블록과 y축 영역이 겹치는지 확인
            if i + 1 < len(sorted_texts):
                next_block = sorted_texts[i + 1]
                next_y1, next_y2 = next_block['coordinates'][1], next_block['coordinates'][3]
                next_center = (next_block['coordinates'][0] + next_block['coordinates'][2]) / 2
                
                # y축 영역이 겹치는 정도 계산
                overlap = min(current_y2, next_y2) - max(current_y1, next_y1)
                total_height = max(current_y2, next_y2) - min(current_y1, next_y1)
                overlap_ratio = overlap / total_height if total_height > 0 else 0
                
                # y축 영역이 충분히 겹치고(예: 30% 이상), 중심점이 서로 다른 열에 있는 경우
                if overlap_ratio > 0.3 and (
                    (current_center < center and next_center > center) or
                    (current_center > center and next_center < center)
                ):
                    # 2단으로 판단하고 좌우 순서대로 처리
                    if current_center < center:
                        result.append(current)
                        result.append(next_block)
                    else:
                        result.append(next_block)
                        result.append(current)
                    i += 2
                    continue

            # 겹치지 않거나 마지막 블록인 경우 현재 블록만 추가
            result.append(current)
            i += 1

    # 두 번째 페이지 이상일 때   
    else:
        left_column = []
        right_column = []

        # 좌우 단으로 분류
        for text in texts:
            # 블록의 중심점 x좌표 계산
            block_center = (text['coordinates'][0] + text['coordinates'][2]) / 2
            
            if block_center < center:
                left_column.append(text)
            else:
                right_column.append(text)

        # 각 단별로 y축 기준 정렬
        left_column = sorted(left_column, key=lambda x: x['coordinates'][1])
        right_column = sorted(right_column, key=lambda x: x['coordinates'][1])

        # 좌측 단 다음 우측 단 순서대로 합치기
        result = left_column + right_column
        
    
    return result





# # 단 구조에 따른 텍스트 정렬
# def sort_text_blocks_with_columns(texts, page_width, is_two_column):
#     """
#     텍스트 블록 정렬
#     1단: y축 기준 정렬
#     2단: 좌우 단으로 나누어 각각 y축 기준 정렬
#     """
#     if not is_two_column:
#         # 1단인 경우 단순히 y축 기준 정렬
#         return sorted(texts, key=lambda x: x['coordinates'][1])
    
#     # 2단인 경우
#     center = page_width / 2
#     left_column = []
#     right_column = []
    
#     # 좌우 단으로 분류
#     for text in texts:
#         # 블록의 중심점 x좌표 계산
#         block_center = (text['coordinates'][0] + text['coordinates'][2]) / 2
        
#         if block_center < center:
#             left_column.append(text)
#         else:
#             right_column.append(text)
    
#     # 각 단별로 y축 기준 정렬
#     left_column = sorted(left_column, key=lambda x: x['coordinates'][1])
#     right_column = sorted(right_column, key=lambda x: x['coordinates'][1])
    
#     # print(f"좌측 단 블록 수: {len(left_column)}")
#     # print(f"우측 단 블록 수: {len(right_column)}")
    
#     # 좌측 단 다음 우측 단 순서로 합치기
#     return left_column + right_column











# def create_total_text_summary(state: GraphState, stuff_text_summary_chain):
#     # state에서 텍스트 데이터를 가져옵니다.
#     texts = state["texts"]

    # # 중단점이 될 섹션 키워드 확장
    # stop_sections = [
    #     # 기본 참고문헌 관련
    #     'references', 'bibliography', 'works cited', 'literature cited',
        
    #     # 감사의 글 관련
    #     'acknowledgements', 'acknowledgments', 'funding', 
        
    #     # 부록 관련
    #     'appendix', 'appendices', 'supplementary', 'supporting information',
    #     'supplemental material', 'additional information',
        
    #     # 이해관계 관련
    #     'conflict of interest', 'competing interests', 'disclosure statement',
        
    #     # 저자 정보 관련
    #     'author contributions', 'author information', 'authors\' contributions',
        
    #     # 기타 행정적 섹션
    #     'ethics declarations', 'compliance with ethical standards', 'declarations',
        
    #     # 저널 관련 정보
    #     'journal information', 'copyright notice', 'publisher\'s note'
    # ]


    # #  모든 텍스트를 하나의 문자열로 결합
    # combined_texts = []
    # found_stop_section = False
    
    # for page_texts in texts:
        
    #     if found_stop_section:
    #         break
            
    #     for text_dict in page_texts:
    #         if "text" in text_dict and text_dict["text"]:
    #             # Title 블록에 대해서만 stop section 확인
    #             if text_dict.get('block_type') == 'Title':
    #                 current_text = text_dict["text"].lower()
    #             # stop section을 발견하면 플래그를 설정하고 내부 루프 종료
    #                 if any(section in current_text.lower() for section in stop_sections):
    #                     found_stop_section = True
    #                     break
    #             combined_texts.append(text_dict["text"])

    #     if found_stop_section:
    #         break

    # combined_text = " ".join(combined_texts)
    # print(combined_text)

    # combined_text = " ".join(
    # text_dict["text"] for page_texts in texts for text_dict in page_texts if "text" in text_dict and text_dict["text"]
    # )
    
    #  # Document 객체 생성
    # inputs = {"context": [Document(page_content=texts)], "title": state["title"].get("text", "")}
    
    # # text_summary_chain을 사용하여 일괄 처리로 요약을 생성합니다.
    # summaries = stuff_text_summary_chain.invoke(inputs) + '\n'

    # print(f"\n전체 문서 요약 완료\n")
    
    # # 요약된 텍스트를 포함한 새로운 GraphState 객체를 반환합니다.
    # return GraphState(stuff_text_summary=summaries)




def create_total_text_summary(state: GraphState):
    """
    전체 문서 요약 생성
    """

    texts = state["texts"]

    llm = text_summary_llm
    text_llm = TextAgent(llm)

    system_prompt = text_summary_system_prompt
    user_prompt_template = f"{texts}\n{text_summary_user_prompt}"

    answer = text_llm.invoke(
        system_prompt, user_prompt_template
    )   

    print(f"\n전체 문서 요약 완료\n")
    
    return GraphState(stuff_text_summary=answer)
    
    
    
    




def extract_method(state: GraphState):
    """
    method 추출
    """

    # state에서 텍스트 데이터를 가져옵니다.
    if len(state["method_texts"]) == 0:
        method_content = state["texts"]
    else:
        method_content = state["method_texts"]

    # 메서드 추출을 위한 모델 생성
    llm = methods_extraction_llm
    # llm = claude_llm
    method_llm = TextAgent(llm)

    # 시스템 프롬프트와 사용자 프롬프트 생성
    system_prompt = methods_extraction_system_prompt
    user_prompt_template = f"{method_content}\n{methods_extraction_user_prompt}"

    answer = method_llm.invoke(
        system_prompt, user_prompt_template
    )

    print(f"\nMethods 추출 완료\n")
    
    # 요약된 텍스트를 포함한 새로운 GraphState 객체를 반환합니다.
    return GraphState(method_extraction_result=answer)



def create_image_summary_data_batches(state: GraphState):
    """
    이미지 요약을 위한 데이터 배치를 생성하는 함수
    """
    
    data_batches = []

    # 현 버전에서는 순서대로 넣어져있음

    figures = state["figures"]

    for figure_list in figures:
        for fig in figure_list:
        # 데이터 배치에 이미지, 관련 텍스트, 페이지 번호, figure_num를 추가
            data_batches.append(
                {
                    "image": fig["file"], # 이미지 파일 경로
                    "text": fig["text"], # 관련 텍스트
                    "page": fig["page_num"], # 페이지 번호
                    "figure_num": fig["figure_num"] # 이미지 번호
                }
            )

    print(f"\n그림 배치 구성 완료\n")

    # 생성된 데이터 배치를 GraphState 객체에 담아 반환
    return GraphState(image_summary_data_batches=data_batches)



def create_table_summary_data_batches(state: GraphState):
    """
    테이블 요약을 위한 데이터 배치를 생성하는 함수
    """
    
    data_batches = []

    # 페이지 번호를 오름차순으로 정렬
    tables = state["tables"]

    for table_list in tables:
        for tbl in table_list:
            data_batches.append(
            {
                "image": tbl["file"], # 이미지 파일 경로로
                "text": tbl["text"], # 관련 텍스트
                "page": tbl["page_num"], # 페이지 번호
                "table_num": tbl["table_num"] # 이미지 번호
            }
        )
            
    print(f"\n표 배치 구성 완료\n")

    # 생성된 데이터 배치를 GraphState 객체에 담아 반환
    return GraphState(table_summary_data_batches=data_batches)



def create_image_summary(state: GraphState):
    """
    이미지 요약 생성
    """

    print(f"\n그림 요약 시작\n")
    # 이미지 요약 추출
    # extract_image_summary 함수를 호출하여 이미지 요약 생성
    image_summaries = extract_image_summary.invoke(
        state["image_summary_data_batches"],
    )

    # 이미지 요약 결과를 저장할 딕셔너리 초기화
    image_summary_output = dict()

    # 각 데이터 배치와 이미지 요약을 순회하며 처리
    for data_batch, image_summary in zip(
        state["image_summary_data_batches"], image_summaries
    ):
        # 데이터 배치의 unique "4페이지의 10번 그림 = 410" 키로 사용하여 테이블 요약 저장
        unique_key = data_batch['page']*100 + data_batch['figure_num']
        image_summary_output[unique_key] = image_summary

    print(f"\n그림 요약 완료\n")

    # 이미지 요약 결과를 포함한 새로운 GraphState 객체 반환
    return GraphState(image_summary=image_summary_output)



def create_table_summary(state: GraphState):
    """
    테이블 요약 생성
    """

    print(f"\n표 요약 시작\n")
    # 테이블 요약 추출
    table_summaries = extract_table_summary.invoke(
        state["table_summary_data_batches"],
    )

    # print(table_summaries)

    # 테이블 요약 결과를 저장할 딕셔너리 초기화
    table_summary_output = dict()

    # 각 데이터 배치와 테이블 요약을 순회하며 처리
    for data_batch, table_summary in zip(
        state["table_summary_data_batches"], table_summaries
    ):
        # 데이터 배치의 unique "4페이지의 10번 표 = 410" 키로 사용하여 테이블 요약 저장
        unique_key = data_batch['page']*100 + data_batch['table_num']
        table_summary_output[unique_key] = table_summary

    print(f"\n표 요약 완료\n")

    # 테이블 요약 결과를 포함한 새로운 GraphState 객체 반환
    return GraphState(table_summary=table_summary_output)



@chain
def extract_image_summary(data_batches):
    """
    LLM을 이용한 이미지 요약 추출
    """

    # 객체 생성
    llm = vision_summary_llm
    multimodal_llm = MultiModal(llm)

    system_prompt = image_summary_system_prompt

    image_paths = []
    system_prompts = []
    user_prompts = []
    answers = []

    # for data_batch in data_batches:
    #     context = data_batch["text"]
    #     if context is None or (context is not None and len(context)) < 10:
    #         context = """이미지와 관련된 다른 내용이 없습니다. 이미지만 이용하여 설명해주세요.
    #         """
    #     else:
    #         context = f"""이미지와 관련된 내용: {context}
    #         """
    #     image_path = data_batch["image"]

    #     user_prompt_template = f"{context}\n{image_summary_user_prompt}"

    #     image_paths.append(image_path)
    #     system_prompts.append(system_prompt)
    #     user_prompts.append(user_prompt_template)

    # print(f"\n총 {len(data_batches)}개 figure 처리 시작\n")
    
    # answers = multimodal_llm.batch(
    #     image_paths, system_prompts, user_prompts, display_image=False
    # )


    for i, data_batch in enumerate(data_batches, start=1):
        context = data_batch["text"]
        if context is None or (context is not None and len(context)) < 10:
            context = """There is no context related to the image.
            Please describe the image in detail.
            """
        else:
            context = f"""There is the context related to the image: {context}
            """
        image_path = data_batch["image"]

        user_prompt_template = f"{context}\n{image_summary_user_prompt}"

        answer = multimodal_llm.invoke(
            image_path, system_prompt, user_prompt_template, display_image=False
        )

        answers.append(answer)

        # if i % 5 == 0:
        print(f"\n{i}개 figure 처리 완료\n")


    return answers



@chain
def extract_table_summary(data_batches):
    """
    LLM을 이용한 테이블 요약 추출
    """

    # 객체 생성
    llm = vision_summary_llm
    multimodal_llm = MultiModal(llm)

    system_prompt = table_summary_system_prompt
    
    image_paths = []
    system_prompts = []
    user_prompts = []
    answers = []

    # for data_batch in data_batches:
    #     context = data_batch["text"]
    #     if context is None or (context is not None and len(context)) < 10:
    #         context = """이미지와 관련된 다른 내용이 없습니다. 이미지만 이용하여 설명해주세요.
    #         """
    #     else:
    #         context = f"""이미지와 관련된 내용: {context}
    #         """

    #     image_path = data_batch["image"]

    #     user_prompt_template = f"{context}\n{table_summary_user_prompt}"

    #     image_paths.append(image_path)
    #     system_prompts.append(system_prompt)
    #     user_prompts.append(user_prompt_template)

    # print(f"\n총 {len(data_batches)}개 table 처리 시작\n")

    # # 이미지 파일로 부터 질의
    # answer = multimodal_llm.batch(
    #     image_paths, system_prompts, user_prompts, display_image=False
    # )

    for i, data_batch in enumerate(data_batches, start=1):
        context = data_batch["text"]
        if context is None or (context is not None and len(context)) < 10:
            context = """There is no context related to the image.
            Please describe the image in detail.
            """
        else:
            context = f"""There is the context related to the image: {context}
            """

        image_path = data_batch["image"]

        user_prompt_template = f"{context}\n{table_summary_user_prompt}"

        answer = multimodal_llm.invoke(
            image_path, system_prompt, user_prompt_template, display_image=False
        )

        answers.append(answer)  

        # if i % 5 == 0:
        print(f"\n{i}개 table 처리 완료\n")

    return answers



# @chain
# def extract_image_summary(data_batches):
#     # 객체 생성
#     llm = vision_summary_llm
#     chain = prompt_func | llm | StrOutputParser()

#     system_prompt = """You are an expert in extracting useful information from IMAGE.
#     With a given image, your task is to extract key entities, summarize them, and write useful information that can be used later for retrieval.
#     Write the summary in korean as the context.
#     DO NOT translate any technical terms.
#     """

#     image_paths = []
#     system_prompts = []
#     user_prompts = []
#     answers = []

#     for data_batch in data_batches:
#         # print(data_batch)
#         context = data_batch["text"]
#         if context is None or (context is not None and len(context) < 10):
#             context = """There is no context related to the image.
#             Please describe the image in detail.
#             """
#         else:
#             context = f"""There is the context related to the image: {context}
#             Please summarize the key points and important entities related to the image.
#             """
#         image_path = data_batch["image"]
#         user_prompt_template = f"""{context}
#         Don't make code chunk
#         Make sure to include a summary in Korean and do not include the base64 image data directly.
#         Don't translate any technical terms and entities.
        
#         ###

#         Output Format:

        
#         <title>
#         <summary>
#         <entities>
#         <data_insights>
        
#         """

#         pil_image = Image.open(image_path)
        
#         image_b64 = convert_to_base64(pil_image)
        
#         response = chain.invoke(
#             {"system": system_prompt, "user": user_prompt_template, "image": image_b64}
#         )

#         print(response)

#         answers.append(response)

#     return answers



# @chain
# def extract_table_summary(data_batches):
#     # 객체 생성
#     llm = vision_summary_llm
#     chain = prompt_func | llm | StrOutputParser()

#     system_prompt = """You are an expert in extracting useful information from TABLE.
#     With a given image, your task is to extract key entities, summarize them, and write useful information that can be used later for retrieval.
#     Write the summary in korean as the context.
#     DO NOT translate any technical terms.
#     """
    
#     image_paths = []
#     system_prompts = []
#     user_prompts = []
#     answers = []

#     for data_batch in data_batches:
#         # print(data_batch)
#         context = data_batch["text"]
#         if context is None or (context is not None and len(context) < 10):
#             context = """There is no context related to the image.
#             The image contains a TABLE.
#             Please explain the contents of the table.
#             """
#         else:
#             context = f"""There is the context related to the image: {context}
#             The image contains a TABLE.
#             Please explain the contents of the table and summarize the key points and important entities related to the image.
#             """
#         image_path = data_batch["image"]
#         user_prompt_template = f"""Here is the context related to the image of table: {context}
#         Don't make code chunk
#         Please summarize the key points and important entities related to the table. 
#         Make sure to include a summary in Korean and do not include the base64 image data directly.
        
#         ###

#         Output Format:

#         <title>
#         <table_summary>
#         <key_entities> 
#         <data_insights>

#         """

#         pil_image = Image.open(image_path)
#         image_b64 = convert_to_base64(pil_image)
#         response = chain.invoke(
#             {"system": system_prompt, "user": user_prompt_template, "image": image_b64}
#         )

#         answers.append(response)

#     return response



@chain
def table_markdown_extractor(data_batches):
    """
    LLM을 이용한 테이블 마크다운 추출
    """

    # 객체 생성
    llm = vision_summary_llm

    system_prompt = "You are an expert in converting image of the TABLE into markdown format. Be sure to include all the information in the table. DO NOT narrate, just answer in markdown format."

    image_paths = []
    system_prompts = []
    user_prompts = []

    for data_batch in data_batches:
        image_path = data_batch["image"]
        user_prompt_template = f"""DO NOT wrap your answer in ```markdown``` or any XML tags.
        
        ###

        Output Format:

        <table_markdown>

        """

        image_paths.append(image_path)
        system_prompts.append(system_prompt)
        user_prompts.append(user_prompt_template)

    # 멀티모달 객체 생성
    multimodal_llm = MultiModal(llm)

    # 이미지 파일로 부터 질의
    answer = multimodal_llm.batch(
        image_paths, system_prompts, user_prompts, display_image=False
    )

    return answer



def create_table_markdown(state: GraphState):
    """
    table_markdown_extractor를 사용하여 테이블 마크다운 생성
    state["table_summary_data_batches"]에 저장된 테이블 데이터를 사용
    """

    table_markdowns = table_markdown_extractor.invoke(
        state["table_summary_data_batches"],
    )

    # 결과를 저장할 딕셔너리 초기화
    table_markdown_output = dict()

    # 각 데이터 배치와 생성된 테이블 마크다운을 매칭하여 저장
    for data_batch, table_summary in zip(
        state["table_summary_data_batches"], table_markdowns
    ):
        # 데이터 배치의 id를 키로 사용하여 테이블 마크다운 저장
        unique_key = data_batch['page']*100 + data_batch['table_num']
        table_markdown_output[unique_key] = table_summary

    print(f"\n표 마크다운 작성 완료\n")

    # 새로운 GraphState 객체 반환, table_markdown 키에 결과 저장
    return GraphState(table_markdown=table_markdown_output)



def make_result(state: GraphState):
    """
    결과 파일(html, md) 생성
    """

    pdf_file = state["file"]
    file_prefix = os.path.splitext(os.path.basename(pdf_file))[0]
    output_dir = state["output_dir"]
    
    # 필요한 폴더 구조 생성
    os.makedirs(output_dir, exist_ok=True)
    
    output_file = os.path.join(output_dir, f"{file_prefix}.md")
    output_html_file = os.path.join(output_dir, f"{file_prefix}.html")
    ## 전체 텍스트 요약하는 걸 기본으로 작성
    with open(output_file, "w", encoding="utf-8") as f:
        # 제목 삽입
        # if "title" in state:
        f.write("# 제목\n") 
        f.write(state["title"].get("text") + "\n\n\n\n")
        # 본문 요약 삽입
        f.write("# 본문\n")
        f.write(state["stuff_text_summary"] + "\n\n\n\n")
        ## 방법 삽입
        if "method_extraction_result" in state:
            f.write("# Methods\n")
            f.write(state["method_extraction_result"] + "\n\n\n\n")
        ## 이미지 및 요약 순서대로 삽입
        if "image_summary" in state:
            f.write("# 이미지\n")
            for idx, key in enumerate(state["image_summary"].keys(), start=1):
                page_num = key // 100
                figure_num = key % 100
                img=f"pngs/page{page_num}_figure{figure_num}.png"
                markdown_img_link = f"![Figure {idx}.]({img})"
                f.write(markdown_img_link + '\n')
                img_summary=state["image_summary"][key]
                f.write(img_summary + "\n\n\n\n")
        ## 표 및 요약 순서대로 삽입
        if "table_markdown" in state:
            f.write("# 표\n")
            for idx, key in enumerate(state["table_markdown"].keys(), start=1):
                page_num = key // 100
                table_num = key % 100
                img=f"pngs/page{page_num}_table{table_num}.png"
                markdown_img_link = f"![Table {idx}.]({img})"
                # f.write(markdown_img_link)
                f.write(state["table_markdown"][key])
                img_summary=state["table_summary"][key]
                f.write(img_summary + "\n\n\n\n")

    markdown.markdownFromFile(input=output_file, output=output_html_file)

    print(f"\n마크다운 파일 생성 완료\n")

    pngs_folder = os.path.join(output_dir, "pngs")
    if os.path.isdir(pngs_folder) and not os.listdir(pngs_folder):
        shutil.rmtree(pngs_folder)

    with zipfile.ZipFile(os.path.join(output_dir, f"{file_prefix}.zip"), "w") as zip_file:
        # pdf 원본 파일 압축
        zip_file.write(pdf_file, os.path.basename(pdf_file))
        # 최상위 폴더의 md, html 파일 압축
        for filename in os.listdir(output_dir):
            if filename.endswith(".md") or filename.endswith(".html"):
                file_path = os.path.join(output_dir, filename)
                zip_file.write(file_path, filename)
        # pngs 폴더 내 모든 파일 압축 (하위 경로 유지)
        if os.path.exists(pngs_folder):
            for root, dirs, files in os.walk(pngs_folder):
                for file in files:
                    file_path = os.path.join(root, file)
                    arcname = os.path.relpath(file_path, output_dir)
                    zip_file.write(file_path, arcname)

        
        # zip_file.write(output_file, f"{file_prefix}.md")
        # zip_file.write(output_file, f"{file_prefix}.html")
        # for root, dirs, files in os.walk(pngs_folder):
        #     for file in files:
        #         file_path = os.path.join(root, file)
        #         arcname = os.path.relpath(file_path, output_folder)  # output_folder를 기준으로 경로 설정
        #         zip_file.write(file_path, arcname)  # 파일 추가

    print(f"\n{file_prefix}.zip 압축 파일 생성 완료\n")

