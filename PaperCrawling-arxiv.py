from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
import os
import logging
from typing import List, Optional, Dict, Any
from tqdm import tqdm
from pathlib import Path

import arxiv
import pymupdf

# 상수 정의
PAPERS_DIR = "Papers"

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def create_safe_filename(title: str) -> str:
    """
    파일명으로 사용할 수 없는 특수문자를 제거합니다.
    
    Args:
        title (str): 원본 제목
    
    Returns:
        str: 안전한 파일명
    """
    return title.replace(" ", "") + ".pdf"

def is_pdf_downloaded(pdf_name: str, papers_dir: str) -> bool:
    """
    PDF 파일이 이미 다운로드되었는지 확인합니다.
    
    Args:
        pdf_name (str): PDF 파일명
        papers_dir (str): 논문 저장 디렉토리
    
    Returns:
        bool: 이미 다운로드된 경우 True
    """
    for root, _, files in os.walk(papers_dir):
        if pdf_name in files:
            return True
    return False

def download_pdf(paper: arxiv.Result, outdir: str) -> Optional[str]:
    """
    arXiv 논문 PDF를 다운로드합니다.
    
    Args:
        paper (arxiv.Result): arXiv 논문 객체
        outdir (str): 저장 디렉토리
    
    Returns:
        Optional[str]: 다운로드된 PDF 파일 경로 또는 실패 시 None
    """
    try:
        title = paper.title
        pdf_name = create_safe_filename(title)
        
        # 이미 다운로드된 논문인지 확인
        if is_pdf_downloaded(pdf_name, PAPERS_DIR):
            logger.info(f"PDF {pdf_name}는 이미 다운로드되어 있습니다.")
            return None
        
        # 디렉토리 생성
        os.makedirs(outdir, exist_ok=True)
        
        # PDF 다운로드
        pdf_path = os.path.join(outdir, pdf_name)
        paper.download_pdf(filename=pdf_path)
        logger.info(f"PDF {pdf_name} 다운로드 완료")
        return pdf_path
        
    except arxiv.ArxivError as e:
        logger.error(f"arXiv API 오류: {str(e)}")
    except IOError as e:
        logger.error(f"파일 시스템 오류: {str(e)}")
    except Exception as e:
        logger.error(f"예상치 못한 오류: {str(e)}")
    
    return None

def validate_args(args: Any) -> None:
    """
    명령행 인자의 유효성을 검사합니다.
    
    Args:
        args: 명령행 인자 객체
    
    Raises:
        ValueError: 인자가 유효하지 않을 경우
    """
    if not args.key:
        raise ValueError("검색 키워드(-k/--key)는 필수입니다.")
    
    if not args.num or args.num <= 0:
        raise ValueError("논문 수(-n/--num)는 양수여야 합니다.")
    
    if not args.today:
        raise ValueError("날짜(-d/--today)는 필수입니다.")

def main(args: Any) -> None:
    """
    메인 함수: arXiv에서 논문을 검색하고 PDF를 다운로드합니다.
    
    Args:
        args: 명령행 인자 객체
    """
    try:
        validate_args(args)
        
        keyword = args.key
        n_paper = args.num
        today = args.today

        # PDF 저장 디렉토리 생성
        outdir = os.path.join(PAPERS_DIR, today, keyword)
        os.makedirs(outdir, exist_ok=True)

        # arXiv 검색
        client = arxiv.Client()
        search = arxiv.Search(
            query=keyword, 
            max_results=n_paper,  
            sort_by=arxiv.SortCriterion.SubmittedDate
        )
        results = list(client.results(search))
        
        logger.info(f"검색 결과: {len(results)}개의 논문을 찾았습니다.")
        
        # 진행 상황 표시와 함께 PDF 다운로드
        for paper in tqdm(results, desc="PDF 다운로드 중"):
            download_pdf(paper, outdir)
            
    except Exception as e:
        logger.error(f"프로그램 실행 중 오류 발생: {str(e)}")
        raise

if __name__ == "__main__":
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument("-k", "--key", type=str, help="검색 키워드")
    parser.add_argument("-n", "--num", type=int, help="다운로드할 논문 수")
    parser.add_argument("-d", "--today", type=str, help="오늘 날짜 (YYYY-MM-DD)")
    args = parser.parse_args()
    main(args)
