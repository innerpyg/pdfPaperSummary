from Bio import Entrez
import requests
from datetime import datetime
import time
import os
from bs4 import BeautifulSoup
import re
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
import logging
import yaml
from typing import List, Optional, Dict, Any
from tqdm import tqdm
from pathlib import Path
import PyPDF2
import io
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import shutil

# 상수 정의
PMC_BASE_URL = "https://www.ncbi.nlm.nih.gov/pmc/articles/PMC{}/"
DEFAULT_USER_AGENT = 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/56.0.2924.87 Safari/537.36'
CONFIG_FILE = "config.yaml"
RELEVANCE_THRESHOLD = 0  # 연관성 점수 임계값
PROCESSED_PMCS_FILE = "processed_pmcs.txt"  # 처리된 PMC ID를 저장할 파일

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def download_pdf_with_retry(pmc_id: str, outdir: str, headers: Dict[str, str], max_retries: int = 3) -> Optional[str]:
    """
    Selenium을 사용하여 PDF 다운로드를 시도하고, 유효성 검사 후 필요시 재시도합니다.
    Args:
        pmc_id (str): PMC ID
        outdir (str): 저장 디렉토리
        headers (Dict[str, str]): HTTP 요청 헤더
        max_retries (int): 최대 재시도 횟수
    Returns:
        Optional[str]: 다운로드된 PDF 파일 경로 또는 실패 시 None
    """
    try:
        # MEDLINE 형식으로 메타데이터 가져오기
        handle = Entrez.efetch(
            db="pmc",
            id=pmc_id,
            rettype="medline",
            retmode="text"
        )
        records = handle.read()
        title = extract_title_from_medline(records)
        if not title:
            logger.warning(f"PMC{pmc_id}의 제목을 찾을 수 없습니다.")
            return None
        pdf_name = create_safe_filename(title)
        if is_pdf_downloaded(pdf_name, "Papers"):
            logger.info(f"PDF {pdf_name}는 이미 다운로드되어 있습니다.")
            return None

        pmc_url = PMC_BASE_URL.format(pmc_id)
        
        # Selenium 옵션 설정
        chrome_options = Options()
        chrome_options.add_argument('--headless')
        chrome_options.add_argument('--no-sandbox')
        chrome_options.add_argument('--disable-dev-shm-usage')
        
        for attempt in range(max_retries):
            try:
                driver = webdriver.Chrome(options=chrome_options)
                driver.get(pmc_url)
                
                # 페이지 소스에서 PDF URL 찾기
                page_source = driver.page_source
                soup = BeautifulSoup(page_source, 'html.parser')
                
                # PDF URL 찾기 시도
                pdf_url = None
                
                # 1. citation_pdf_url 메타 태그 확인
                pdf_meta = soup.find('meta', attrs={'name': 'citation_pdf_url'})
                if pdf_meta:
                    pdf_url = pdf_meta.get('content')
                
                # 2. PDF 링크 찾기
                if not pdf_url:
                    pdf_links = soup.find_all('a', href=lambda x: x and x.endswith('.pdf'))
                    if pdf_links:
                        pdf_url = pdf_links[0]['href']
                
                # 3. PDF 버튼의 href 속성 확인
                if not pdf_url:
                    pdf_buttons = soup.find_all('a', class_=lambda x: x and ('pdf' in x.lower() or 'download' in x.lower()))
                    for button in pdf_buttons:
                        if button.get('href') and button['href'].endswith('.pdf'):
                            pdf_url = button['href']
                            break
                
                if pdf_url:
                    # 상대 URL을 절대 URL로 변환
                    if not pdf_url.startswith('http'):
                        pdf_url = f"https://www.ncbi.nlm.nih.gov{pdf_url}"
                    
                    logger.info(f"PDF URL 찾음: {pdf_url}")
                    
                    # PDF 다운로드
                    response = requests.get(pdf_url, headers=headers, stream=True)
                    if response.status_code == 200:
                        pdf_path = os.path.join(outdir, pdf_name)
                        with open(pdf_path, 'wb') as f:
                            for chunk in response.iter_content(chunk_size=8192):
                                if chunk:
                                    f.write(chunk)
                        
                        # PDF 유효성 검사
                        if os.path.exists(pdf_path) and os.path.getsize(pdf_path) > 0:
                            logger.info(f"PDF {pdf_name} 다운로드 완료 (시도 {attempt+1})")
                            driver.quit()
                            return pdf_path
                    else:
                        logger.warning(f"PDF 다운로드 실패: HTTP {response.status_code}")
                else:
                    logger.warning("PDF URL을 찾을 수 없습니다.")
                
                driver.quit()
                time.sleep(2)
            except Exception as e:
                logger.error(f"Selenium 오류: {str(e)} (시도 {attempt+1})")
                time.sleep(2)
        logger.error(f"최대 재시도 횟수({max_retries})를 초과했습니다.")
    except Exception as e:
        logger.error(f"예상치 못한 오류: {str(e)}")
    return None

def load_config() -> Dict[str, Any]:
    """
    설정 파일을 로드합니다.
    
    Returns:
        Dict[str, Any]: 설정 정보가 담긴 딕셔너리
    
    Raises:
        FileNotFoundError: 설정 파일이 존재하지 않을 경우
        yaml.YAMLError: YAML 파싱 오류 발생 시
    """
    try:
        with open(CONFIG_FILE, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)
    except FileNotFoundError:
        logger.error(f"설정 파일 {CONFIG_FILE}를 찾을 수 없습니다.")
        raise
    except yaml.YAMLError as e:
        logger.error(f"YAML 파싱 오류: {e}")
        raise

def get_email_address(initial: str) -> str:
    """
    이니셜에 해당하는 이메일 주소를 설정 파일에서 가져옵니다.
    
    Args:
        initial (str): 사용자 이니셜
    
    Returns:
        str: 이메일 주소
    
    Raises:
        ValueError: 이니셜이 설정 파일에 없을 경우
    """
    config = load_config()
    email = config.get('email_addresses', {}).get(initial)
    if not email:
        raise ValueError(f"이니셜 {initial}에 해당하는 이메일 주소를 찾을 수 없습니다.")
    return email

def create_safe_filename(title: str) -> str:
    """
    파일명으로 사용할 수 없는 특수문자를 제거합니다.
    
    Args:
        title (str): 원본 제목
    
    Returns:
        str: 안전한 파일명
    """
    safe_title = "".join(c for c in title if c.isalnum() or c in (' ', '-', '_')).rstrip()
    return f"{safe_title}.pdf".replace(" ", "")

def extract_title_from_medline(medline_text: str) -> str:
    """
    MEDLINE 형식의 텍스트에서 제목을 추출합니다.
    
    Args:
        medline_text (str): MEDLINE 형식의 텍스트
    
    Returns:
        str: 추출된 제목
    """
    for line in medline_text.split('\n'):
        if line.startswith('TI  -'):
            return line.replace('TI  -', '').strip()
    return ""

def extract_abstract_from_medline(medline_text: str) -> str:
    """
    MEDLINE 형식의 텍스트에서 초록을 추출합니다.
    
    Args:
        medline_text (str): MEDLINE 형식의 텍스트
    
    Returns:
        str: 추출된 초록
    """
    abstract = ""
    in_abstract = False
    
    for line in medline_text.split('\n'):
        # 초록 시작 확인
        if line.startswith('AB  -'):
            in_abstract = True
            abstract += line.replace('AB  -', '').strip() + " "
        # 초록의 연속된 줄 확인 (들여쓰기만 있는 줄)
        elif in_abstract and line.startswith('      '):
            abstract += line.strip() + " "
        # 다른 태그가 나오면 초록 종료
        elif in_abstract and line.strip() and not line.startswith('      '):
            in_abstract = False
            
    return abstract.strip()

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

def is_valid_pdf(pdf_content, min_size_kb=10):
    """
    PDF 파일의 유효성을 검사합니다.
    
    Args:
        pdf_content (bytes): PDF 파일 내용
        min_size_kb (int): 최소 파일 크기 (KB)
    
    Returns:
        bool: 유효한 PDF인 경우 True
    """
    # 파일 크기 검사 (KB 단위)
    if len(pdf_content) < min_size_kb * 1024:
        return False
    
    # "Preparing to download..." 텍스트 검사
    if b"Preparing to download" in pdf_content:
        return False
    
    # PyPDF2를 사용한 PDF 구조 검사
    try:
        pdf_file = io.BytesIO(pdf_content)
        pdf_reader = PyPDF2.PdfReader(pdf_file)
        
        # 페이지 수 확인 (최소 1페이지 이상)
        if len(pdf_reader.pages) < 1:
            return False
            
        # 첫 페이지에서 텍스트 추출 시도
        first_page = pdf_reader.pages[0]
        text = first_page.extract_text()
        
        # 텍스트가 비어있거나 너무 짧은 경우
        if not text or len(text.strip()) < 10:
            return False
            
        return True
    except Exception:
        return False

def save_failed_pmc(pmc_id: str, keyword: str, pdf_url: str = "", today: str = "") -> None:
    """
    PDF 다운로드에 실패한 PMC ID와 키워드를 파일에 저장합니다.
    
    Args:
        pmc_id (str): PMC ID
        keyword (str): 검색 키워드
        pdf_url (str): PDF 다운로드 URL (선택 사항)
        today (str): 오늘 날짜 (선택 사항)
    """
    try:
        with open("failed_pmcs.txt", 'a', encoding='utf-8') as f:
            f.write(f"{pmc_id}\t{keyword}\t{pdf_url}\t{today}\n")
        logger.info(f"PMC ID {pmc_id}, 키워드 {keyword}, URL {pdf_url}, 날짜 {today}를 실패 목록에 저장했습니다.")
    except Exception as e:
        logger.error(f"실패 목록 저장 중 오류 발생: {str(e)}")

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
    
    if not args.initial:
        raise ValueError("이니셜(-i/--initial)은 필수입니다.")

def calculate_relevance_score(text: str, keyword: str) -> float:
    """
    텍스트와 키워드 간의 연관성 점수를 계산합니다.
    
    Args:
        text (str): 검사할 텍스트
        keyword (str): 검색 키워드
    
    Returns:
        float: 연관성 점수 (0~1 사이 값)
    """
    # 텍스트가 비어있거나 공백만 있는 경우 0 반환
    if not text or not text.strip():
        return 0.0
        
    # 키워드를 소문자로 변환
    text = text.lower()
    keyword = keyword.lower()
    
    # 키워드가 여러 단어로 구성된 경우 처리
    keywords = keyword.split()
    
    # 각 키워드의 출현 빈도 계산
    total_score = 0
    for kw in keywords:
        count = text.count(kw)
        total_score += count
    
    # 텍스트 길이로 정규화
    words = text.split()
    if not words:  # 추가 안전장치
        return 0.0
    return total_score / len(words)

def calculate_total_score(pmc_id: str, keyword: str) -> float:
    """
    논문의 총 연관성 점수를 계산합니다.
    
    Args:
        pmc_id (str): PMC ID
        keyword (str): 검색 키워드
    
    Returns:
        float: 총 연관성 점수
    """
    handle = Entrez.efetch(db="pmc", id=pmc_id, rettype="medline", retmode="text")
    records = handle.read()
    
    title = extract_title_from_medline(records)
    abstract = extract_abstract_from_medline(records)
    
    # 제목에 더 높은 가중치 부여
    title_score = calculate_relevance_score(title, keyword) * 2
    abstract_score = calculate_relevance_score(abstract, keyword)
    
    return title_score, abstract_score

def filter_by_relevance(pmc_id: str, keyword: str) -> bool:
    """
    논문의 연관성을 검사합니다.
    
    Args:
        pmc_id (str): PMC ID
        keyword (str): 검색 키워드
    
    Returns:
        bool: 연관성이 임계값을 넘으면 True
    """
    # MEDLINE 메타데이터 가져오기
    handle = Entrez.efetch(db="pmc", id=pmc_id, rettype="medline", retmode="text")
    records = handle.read()
    
    # 제목과 초록 추출
    title = extract_title_from_medline(records)
    abstract = extract_abstract_from_medline(records)
    
    # 키워드 관련성 점수 계산
    title_score = calculate_relevance_score(title, keyword)
    abstract_score = calculate_relevance_score(abstract, keyword)
    
    # 임계값 기반 필터링
    logger.info(f"title_score: {title_score:.4f}, abstract_score: {abstract_score:.4f}")
    return (title_score + abstract_score) > RELEVANCE_THRESHOLD

def get_best_match(pmc_ids: List[str], keyword: str) -> Optional[str]:
    """
    가장 연관성 높은 논문 ID를 반환합니다.
    
    Args:
        pmc_ids (List[str]): PMC ID 목록
        keyword (str): 검색 키워드
    
    Returns:
        Optional[str]: 가장 연관성 높은 PMC ID 또는 None
    """
    best_score = 0
    best_pmc_id = None
    
    # 이미 작업한 논문 목록 가져오기
    processed_pmcs = get_processed_pmcs()
    
    for pmc_id in pmc_ids:
        # 유사성 확인
        if filter_by_relevance(pmc_id, keyword):
            # 이미 작업한 논문인 경우 건너뛰기
            if pmc_id in processed_pmcs:
                logger.info(f"PMC{pmc_id}는 이미 작업한 논문이므로 건너뜁니다.")
                continue
                
            # 연관성 점수 계산
            title_score, abstract_score = calculate_total_score(pmc_id, keyword)
            if title_score > 0.1 or abstract_score > 0.01:
                total_score = title_score + abstract_score
                if total_score > best_score:
                    best_score = total_score
                    best_pmc_id = pmc_id
            else:
                logger.info(f"PMC{pmc_id}는 연관성 점수가 낮아 건너뜁니다.")
    
    return best_pmc_id

def save_processed_pmc(pmc_id: str) -> None:
    """
    처리된 PMC ID를 파일에 저장합니다.
    
    Args:
        pmc_id (str): 저장할 PMC ID
    """
    try:
        with open(PROCESSED_PMCS_FILE, 'a', encoding='utf-8') as f:
            f.write(f"{pmc_id}\n")
        logger.info(f"PMC ID {pmc_id}를 처리된 목록에 저장했습니다.")
    except Exception as e:
        logger.error(f"PMC ID 저장 중 오류 발생: {str(e)}")

def get_processed_pmcs() -> List[str]:
    """
    이미 작업한 논문의 PMC ID 목록을 반환합니다.
    processed_pmcs.txt와 failed_pmcs.txt 파일에서 PMC ID를 가져옵니다.
    
    Returns:
        List[str]: 이미 작업한 논문의 PMC ID 목록
    """
    processed_pmcs = []
    
    # 처리된 PMC ID 파일이 존재하는 경우 읽기
    if os.path.exists(PROCESSED_PMCS_FILE):
        try:
            with open(PROCESSED_PMCS_FILE, 'r', encoding='utf-8') as f:
                processed_pmcs = [line.strip() for line in f.readlines() if line.strip()]
        except Exception as e:
            logger.error(f"처리된 PMC ID 파일 읽기 중 오류 발생: {str(e)}")
    
    # 실패한 PMC ID 파일이 존재하는 경우 읽기
    if os.path.exists("failed_pmcs.txt"):
        try:
            with open("failed_pmcs.txt", 'r', encoding='utf-8') as f:
                for line in f.readlines():
                    if line.strip():
                        # 탭으로 구분된 첫 번째 열(PMC ID) 추출
                        pmc_id = line.strip().split('\t')[0]
                        if pmc_id and pmc_id not in processed_pmcs:
                            processed_pmcs.append(pmc_id)
        except Exception as e:
            logger.error(f"실패한 PMC ID 파일 읽기 중 오류 발생: {str(e)}")
    
    return processed_pmcs

def main(args: Any) -> None:
    """
    메인 함수: PMC에서 논문을 검색하고 PDF를 다운로드합니다.
    
    Args:
        args: 명령행 인자 객체
    """
    try:
        validate_args(args)
        
        keyword = args.key
        n_paper = args.num
        today = args.today
        initial = args.initial

        Entrez.email = get_email_address(initial)

        # PDF 저장 디렉토리 생성
        outdir = os.path.join("Papers", today, keyword)
        os.makedirs(outdir, exist_ok=True)

        # User-Agent 설정
        headers = requests.utils.default_headers()
        headers['User-Agent'] = DEFAULT_USER_AGENT

        # 검색 쿼리 생성
        search_query = f'"{keyword}"[Title] OR "{keyword}"[Abstract]'
        
        # PMC 검색
        handle = Entrez.esearch(
            db="pmc",
            term=search_query,
            retmax=10,  # 최신 10개 논문만 검색
            sort="date"
        )
        records = Entrez.read(handle)
        pmc_ids = records["IdList"]

        logger.info(f"검색 결과: {len(pmc_ids)}개의 논문을 찾았습니다.")
        
        # 가장 연관성 높은 논문 ID 찾기
        best_pmc_id = get_best_match(pmc_ids, keyword)
        
        if best_pmc_id:
            logger.info(f"가장 연관성 높은 논문 ID: {best_pmc_id}")
                
            # PMC 페이지에서 PDF 링크 찾기
            pmc_url = PMC_BASE_URL.format(best_pmc_id)
            response = requests.get(pmc_url, headers=headers)
            soup = BeautifulSoup(response.content, 'lxml')
            
            # citation_pdf_url 메타 태그 찾기
            pdf_meta = soup.find('meta', attrs={'name': 'citation_pdf_url'})
            pdf_url = pdf_meta.get('content') if pdf_meta else ""
                
            # 가장 연관성 높은 논문 다운로드
            pdf_path = download_pdf_with_retry(best_pmc_id, outdir, headers)

            # 다운로드 성공 시 PMC ID 저장
            if pdf_path:
                save_processed_pmc(best_pmc_id)
            else:
                # 다운로드 실패 시 실패 목록에 저장
                save_failed_pmc(best_pmc_id, keyword, pdf_url, today)
        else:
            logger.warning("연관성 높은 논문을 찾을 수 없거나 모든 논문이 이미 처리되었습니다.")
            
    except Exception as e:
        logger.error(f"프로그램 실행 중 오류 발생: {str(e)}")
        raise

if __name__ == "__main__":
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument("-k", "--key", type=str, help="검색 키워드")
    parser.add_argument("-n", "--num", type=int, help="다운로드할 논문 수")
    parser.add_argument("-d", "--today", type=str, help="오늘 날짜 (YYYY-MM-DD)")
    parser.add_argument("-i", "--initial", type=str, help="사용자 이니셜")
    args = parser.parse_args()
    main(args) 