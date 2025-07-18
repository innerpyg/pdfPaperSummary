from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
import os

import arxiv

import pymupdf

def main(args):
	keyword=args.key
	n_paper=args.num
	today=args.today

	client = arxiv.Client()
	search = arxiv.Search(query=keyword, max_results=n_paper,  sort_by=arxiv.SortCriterion.SubmittedDate)
	results = client.results(search)
	outdir = "Papers/" + today + "/" + keyword

	for i in client.results(search):
		cut_page = 0
		title = i.title.replace(" ", "")
		pdf_name = title + ".pdf"

		# 이미 다운로드 된 논문인지 확인
		check = False
		for root, dirs, files in os.walk("Papers"):
			if pdf_name in files:
				check = True
				break
		if check:
			continue

		os.makedirs(outdir, exist_ok=True)

		i.download_pdf(filename=outdir + "/" + pdf_name)


if __name__ == "__main__":
	parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
	parser.add_argument("-k", "--key", type=str)
	parser.add_argument("-n", "--num", type=int)
	parser.add_argument("-d", "--today", type=str)
	args = parser.parse_args()
	main(args)
