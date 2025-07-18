from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
import smtplib
from email.mime.text import MIMEText
from email.mime.base import MIMEBase
from email.encoders import encode_base64
from email.mime.multipart import MIMEMultipart
from email.utils import formatdate
from email.header import Header
from email import encoders
from datetime import datetime
import os
import sys
import socket
from dotenv import load_dotenv


def get_email_address(initial):
	try:
		with open("email_address.txt", "r", encoding="utf-8") as f:
			for line in f:
				parts = line.strip().split('\t')
				if parts[0] == initial:
					return parts[1]
		raise ValueError(f"Initial {initial} not found in email_address.txt")
	except FileNotFoundError:
		raise FileNotFoundError("email_address.txt not found")
	except IndexError:
		raise IndexError("Invalid format in email_address.txt")
	except Exception as e:
		raise Exception(f"Error: {e}")


def main(args):
	zip_path = args.zip_path
	zip_name = os.path.basename(zip_path).split('.')[0]
	keyword = args.keyword
	today = args.today
	initial = args.initial

	email_address = get_email_address(initial)
	
	load_dotenv("conf/.env")

	# 메일 서버 설정
	smtp_server = os.getenv("SMTP_SERVER")
	smtp_port = os.getenv("SMTP_PORT")
	sender = os.getenv("SENDER_EMAIL")
	sender_password = os.getenv("SENDER_PASSWORD")
	
	# 메일 쓰기
	msg = MIMEMultipart()
	msg['From'] = sender
	msg['To'] = email_address
	msg['Date'] = formatdate(localtime=True)
	msg['Subject'] = f"[{today}] {keyword} {zip_name} 논문 요약 완료"

	body = f"""
	{zip_name} 논문 요약 완료
	"""
	msg.attach(MIMEText(body, 'plain', 'utf-8'))

	if os.path.exists(zip_path):
		attachment = MIMEBase('application', 'zip')
		with open(zip_path, 'rb') as f:
			attachment.set_payload(f.read())
		encoders.encode_base64(attachment)
		attachment.add_header(
			'Content-Disposition',
            'attachment',
            filename=os.path.basename(zip_path)
        )
		msg.attach(attachment)

	try:
		with smtplib.SMTP(smtp_server, 25, timeout=30) as server:
			server.login(sender, sender_password)
			server.sendmail(sender, email_address, msg.as_string())
		print(f"메일 전송 완료: {email_address}")
	except smtplib.SMTPException as e:
		print(f"SMTP 오류 발생: {e}")
		print(f"서버: {smtp_server}")
		print(f"포트: {smtp_port}")
		print(f"보내는 사람: {sender}")
	except socket.timeout:
		print("서버 연결 시간 초과")
	except Exception as e:
		print(f"이메일 전송 실패: {e}")


if __name__ == "__main__":    
	parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
	parser.add_argument("--zip_path", type=str, required=True, help="처리할 파일 경로")
	parser.add_argument("--keyword", type=str, required=True, help="검색 키워드")
	parser.add_argument("--today", type=str, required=True, help="오늘 날짜")
	parser.add_argument("--initial", type=str, required=True, help="메일 주소 이름 이니셜")
	args = parser.parse_args()
	main(args)
