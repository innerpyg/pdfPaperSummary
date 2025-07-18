from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from datetime import datetime

# 클래스 임포트
from libs.classes import GraphState, LocalStateCache
# 함수 임포트
from libs.utils import (
    pdf_to_image,
    extract_metadata,
    create_total_text_summary,
    create_image_summary_data_batches,
    create_table_summary_data_batches,
    create_image_summary,
    create_table_summary,
    create_table_markdown,
    make_result,
    extract_method,
)


def main(args):
    start_time = datetime.now()
    # 파일 인수 가져오기
    file = args.input_pdf
    keyword = args.keyword
    today = args.today

    # 캐시 초기화
    cache = LocalStateCache()
    cached_state = cache.load_state(keyword, file)

    # 캐시가 있으면 캐시 로드
    if cached_state:
        state = cached_state
    # 캐시가 없으면 새로 생성
    else:
        # 파일명 저장
        state = GraphState(file=file)
        # pdf를 페이지 별 png로 저장
        state_out = pdf_to_image(state, keyword, today)
        state.update(state_out)

        # 메타데이터 추출
        state_out = extract_metadata(state)
        state.update(state_out)

        # 이미지 데이터 배치 생성
        state_out = create_image_summary_data_batches(state)
        state.update(state_out)

        # 표 데이터 배치 생성
        state_out = create_table_summary_data_batches(state)
        state.update(state_out)

        # 이미지 요약
        state_out = create_image_summary(state)
        state.update(state_out)

        # 표 요약
        state_out = create_table_summary(state)
        state.update(state_out)

        # 표 마크다운화
        state_out = create_table_markdown(state)
        state.update(state_out)

        # 전체 텍스트 요약
        state_out = create_total_text_summary(state)
        state.update(state_out)

        # Methods 추출
        state_out = extract_method(state)
        state.update(state_out)

        # 캐시 저장
        cache.save_state(state, keyword, file)

    # 요약 결과 마크다운 작성
    make_result(state)

    # 작업 종료 시간 기록
    end_time = datetime.now()
    elapsed_time = end_time - start_time
    print(f"\n작업 시작 시간: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"작업 종료 시간: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"총 소요 시간: {elapsed_time}\n")


if __name__ == "__main__":    
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument("--input_pdf", type=str, default="", help="입력 PDF 파일 경로")
    parser.add_argument("--keyword", type=str, default="", help="키워드")
    parser.add_argument("--today", type=str, default="", help="오늘 날짜")
    args = parser.parse_args()
    main(args)