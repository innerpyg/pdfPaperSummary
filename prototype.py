
import pdf2image
import layoutparser as lp
import os
import numpy as np
import cv2
import math


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
    
    print(f"\n=== 단 구조 분석 ===")
    print(f"전체 블록 수: {total_blocks}")
    print(f"중앙 가로지르는 블록 수: {crossing_center}")
    print(f"가로지르는 비율: {crossing_ratio:.2f}")
    
    # 중앙을 가로지르는 블록이 30% 이상이면 1단, 미만이면 2단
    is_two_column = crossing_ratio < layout_threshold
    print(f"판단 결과: {'2단' if is_two_column else '1단'} 구조")
    
    return is_two_column

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

# 단 구조에 따른 텍스트 정렬
def sort_text_blocks_with_columns(texts, page_width, is_two_column):
    """
    텍스트 블록 정렬
    1단: y축 기준 정렬
    2단: 좌우 단으로 나누어 각각 y축 기준 정렬
    """
    if not is_two_column:
        # 1단인 경우 단순히 y축 기준 정렬
        return sorted(texts, key=lambda x: x['coordinates'][1])
    
    # 2단인 경우
    center = page_width / 2
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
    
    print(f"좌측 단 블록 수: {len(left_column)}")
    print(f"우측 단 블록 수: {len(right_column)}")
    
    # 좌측 단 다음 우측 단 순서로 합치기
    return left_column + right_column

# ocr 처리  
def process_block_with_ocr(block, image_array, ocr_agent):
    """
    블록을 OCR 처리하고 결과를 포함한 dictionary 반환
    """
    # 이미지 추출 및 OCR 처리
    block_image = block.crop_image(image_array)
    ocr_text = ocr_agent.detect(block_image).split('\n\n', 1)[0].strip()
    
    return {
        "coordinates": block.coordinates,
        "score": block.score,
        "block": block,
        "text": ocr_text
    }

# def process_block_without_ocr(block):
#     """
#     OCR 처리 없이 블록의 기본 정보만 반환
#     """
#     return {
#         "coordinates": block.coordinates,
#         "score": block.score,
#         "block": block
#     }

# 횡단 블록 제거
def remove_crossing_blocks(texts, page_width, is_two_column):
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
        
        # 중앙을 가로지르지 않는 블록만 유지
        if not (block_width > width_threshold and abs(block_center - center) < page_width * 0.2):
            filtered_texts.append(text)
    
    print(f"횡단 블록 제거: {len(texts)} -> {len(filtered_texts)}")
    return filtered_texts

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



import torch
from layoutparser import Detectron2LayoutModel
# from layoutparser import WeightsUnpickler

model = Detectron2LayoutModel('lp://PubLayNet/faster_rcnn_R_50_FPN_3x/config', 
                                 extra_config=["MODEL.ROI_HEADS.SCORE_THRESH_TEST", 0.8],
                                 label_map={0: "Text", 1: "Title", 2: "List", 3:"Table", 4:"Figure"},
                                 )



# # 모델선언
# model = lp.Detectron2LayoutModel('lp://PubLayNet/faster_rcnn_R_50_FPN_3x/config', 
#                                  extra_config=["MODEL.ROI_HEADS.SCORE_THRESH_TEST", 0.8],
#                                  label_map={0: "Text", 1: "Title", 2: "List", 3:"Table", 4:"Figure"},
#                                  )

# model = lp.Detectron2LayoutModel('lp://PubLayNet/faster_rcnn_R_50_FPN_3x/config', 
#                                  extra_config=["MODEL.ROI_HEADS.SCORE_THRESH_TEST", 0.8],
#                                  label_map={0: "Text", 1: "Title", 2: "List", 3:"Table", 4:"Figure"},
#                                  weights_only=False)  # weights_only 인자 추가

# ocr_agent = lp.TesseractAgent()


# # input pdf 파일 경로
# images = pdf2image.convert_from_path("miRDeep2_split.pdf")


page_layouts = {}
png_dir_name = "pngs"

# if not os.path.exists(png_dir_name):
#     os.makedirs(png_dir_name)


# for page_num, image in enumerate(images, 1):
#     image_array = np.array(image)
#     layout = model.detect(image_array)

#     texts = []
#     figures = []
#     tables = []

#     print(f"\n{page_num} 페이지 작업 중...\n")

#     page_width = image_array.shape[1]
#     page_height = image_array.shape[0]

#     # 우선 text와 title 정보 추출
#     for block in layout:
#         if block.type in ["Text", "Title"]:
#             if not is_margin_text(block, page_width, page_height):
#                 text_info = process_block_with_ocr(block, image_array, ocr_agent)
#                 text_info['block_type'] = block.type  # Text인지 Title인지 구분
#                 texts.append(text_info)

#     # figure와 table 정보 추출하는데, 캡션으로 활용될 text block 가져오기
#     for block in layout:            
#         if block.type == "Figure":
#             # 이미지 추출
#             # 소수점만큼의 여백을 주어 이미지 추출
#             figure_x1, figure_y1, figure_x2, figure_y2 = (
#                 int(block.coordinates[0]),  # 좌상단 x좌표 (버림)
#                 int(block.coordinates[1]),  # 좌상단 y좌표 (버림)
#                 math.ceil(block.coordinates[2]),  # 우하단 x좌표 (올림)
#                 math.ceil(block.coordinates[3])   # 우하단 y좌표 (올림)
#             )
#             cropped_image = image_array[figure_y1:figure_y2, figure_x1:figure_x2]

#             # 이미지 저장
#             figure_num = len(figures) + 1  # 현재 figure의 번호
#             file_name = f"{png_dir_name}/page{page_num}_{figure_num}.png"
#             cv2.imwrite(file_name, cropped_image)

#             caption = find_caption_text(block, texts, is_figure=True)
#             figures.append({
#                 "coordinates": block.coordinates,
#                 "score": block.score,
#                 "block": block,
#                 "file": file_name,
#                 "text": caption['text'] if caption else None
#             })
#             if caption:
#                 texts.remove(caption)  # 캡션으로 사용된 텍스트는 제거
            
#         elif block.type == "Table":
#             # 이미지 추출
#             table_x1, table_y1, table_x2, table_y2 = block.coordinates
#             cropped_image = image_array[table_y1:table_y2, table_x1:table_x2]

#             # 이미지 저장
#             table_num = len(tables) + 1  # 현재 table의 번호
#             file_name = f"{png_dir_name}/page{page_num}_{table_num}.png"
#             cv2.imwrite(file_name, cropped_image)

#             caption = find_caption_text(block, texts, is_figure=False)
#             tables.append({
#                 "coordinates": block.coordinates,
#                 "score": block.score,
#                 "block": block,
#                 "file": file_name,
#                 "text": caption['text'] if caption else None
#             })
#             if caption:
#                 texts.remove(caption)  # 캡션으로 사용된 텍스트는 제거


#     # 첫 페이지에서 위치 기반 제목만 찾아서 별도로 저장 후 문단 title 수정
#     if page_num == 1 and texts:
#         page_layouts['main_title'] = max(texts, key=lambda x: (
#             -x['coordinates'][1],
#             abs(x['coordinates'][0] - image_array.shape[1]/2)
#         ))
#         page_layouts['block_type'] = 'Main_Title'

#         texts = [text for text in texts if text != page_layouts['main_title']]
    
#     # 단 구조 확인    
#     is_two_column = check_column_layout(texts, page_width, page_num)

#     # 횡단 블록 제거
#     texts = remove_crossing_blocks(texts, page_width, is_two_column)

#     # 구조에 따라 텍스트 정렬
#     if is_two_column:
#         sorted_texts = sort_text_blocks_with_columns(texts, page_width, is_two_column)
#     else:
#         sorted_texts = sorted(texts, key=lambda x: x['coordinates'][1])


#     # 페이지별로 분류된 정보 저장
#     page_layouts[page_num] = {
#         "is_two_column": is_two_column,
#         # "titles": titles,
#         "texts": sorted_texts,
#         "figures": figures,
#         "tables": tables
#     }




###### 결과 체킹 용 ================
###### 결과 체킹 용 ================



# 예: 1페이지 정보 확인
# print(f"\n=== 1페이지 정보 ===")
# print(f"Title 수: {len(page_layouts[1]['titles'])}")
# print(f"Text 수: {len(page_layouts[1]['texts'])}")
# print(f"Figure 수: {len(page_layouts[1]['figures'])}")

# Title 상세 정보
# print("\n=== Main Title 상세 정보 ===")
# print(f"\nmain title")
# print(f"{page_layouts['main_title']}")

# Text 상세 정보
# print("\n=== Text 상세 정보 ===")
# for i, text in enumerate(page_layouts[1]['texts'], 1):
#     print(f"\nText {i}:")
#     print(f"위치: {text['coordinates']}")
#     print(f"신뢰도: {text['score']:.2f}")

# # Figure 상세 정보
# print("\n=== Figure 상세 정보 ===")
# for i, figure in enumerate(page_layouts[1]['figures'], 1):
#     print(f"\nFigure {i}:")
#     print(f"위치: {figure['coordinates']}")
#     print(f"신뢰도: {figure['score']:.2f}")

# # Table 상세 정보
# print("\n=== Table 상세 정보 ===")
# for i, table in enumerate(page_layouts[1]['tables'], 1):
#     print(f"\nTable {i}:")
#     print(f"위치: {table['coordinates']}")
#     print(f"신뢰도: {table['score']:.2f}")


# viz_image = np.array(images[5])  # 첫 페이지 기준

# colors = {
#     "Main": (0, 0, 255),   # 파랑
#     "Title": (0, 0, 255),   # 파랑
#     "Text": (255, 0, 0),    # 빨강
#     "Figure": (0, 255, 0)   # 초록
# }

# main Title 박스 그리기
# box = page_layouts['main_title']['coordinates']
# cv2.rectangle(viz_image,
#                 (int(box[0]), int(box[1])),
#                 (int(box[2]), int(box[3])),
#                 colors["Main"],
#                 2)


# Text 박스 그리기
# for text in page_layouts[6]['texts']:
#     box = text['coordinates']
#     cv2.rectangle(viz_image,
#                  (int(box[0]), int(box[1])),
#                  (math.ceil(box[2]), math.ceil(box[3])),
#                  colors["Text"],
#                  2)

# Figure 박스 그리기
# for figure in page_layouts[6]['figures']:
#     box = figure['coordinates']
#     cv2.rectangle(viz_image,
#                  (int(box[0]), int(box[1])),
#                  (math.ceil(box[2]), math.ceil(box[3])),
#                  colors["Figure"],
#                  2)
    
# Table 박스 그리기
# for figure in page_layouts[6]['tables']:
#     box = table['coordinates']
#     cv2.rectangle(viz_image,
#                  (int(box[0]), int(box[1])),
#                  (math.ceil(box[2]), math.ceil(box[3])),
#                  colors["Figure"],
#                  2)


# cv2.imwrite("layout_visualization.jpg", cv2.cvtColor(viz_image, cv2.COLOR_RGB2BGR))



###### 결과 체킹 용 ================
###### 결과 체킹 용 ================