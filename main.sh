#!/bin/bash
# 논문 요약 프로그램
usage () {
	echo "Usage : $0 [ -k keyword ] [ -n the number of paper ] [ -p platform ] [ -i initial(optional) ] "
	exit 1
}

# 인수 확인
if [ $# -ne 6 ] && [ $# -ne 8 ]
then
	usage
fi

# 인수 확인
KEYWORD_set=false
N_PAPER_set=false

## argument input
while getopts "k:n:i:p:" Option
do
	case ${Option} in
		k ) KEYWORD=$OPTARG ;;
		n ) N_PAPER=$OPTARG ;;
		p ) PLATFORM=$OPTARG ;;
		i ) INITIAL=$OPTARG && INITIAL_set=true ;;
		\? ) usage ;;
	esac
done



if [[ -z ${KEYWORD} || -z ${N_PAPER} ]]
then
	echo "Error: Both -k and -n options are required." >&2
	usage
fi

if ! [[ ${N_PAPER} =~ ^[0-9]+$ ]]
then
	echo "-n argument must be numeric value"
	usage
fi

# path 설정
TODAY=$(date '+%Y%m%d')

## keyword에 따른 paper crawling
if [ ${PLATFORM} == "arxiv" ]
then
	${python} PaperCrawling-arxiv.py -k "${KEYWORD}" -n ${N_PAPER} -d ${TODAY}
elif [ ${PLATFORM} == "pmc" ]
then
	${python} PaperCrawling-pmc.py -k "${KEYWORD}" -n ${N_PAPER} -d ${TODAY} -i ${INITIAL}
fi

## crawling 한 paper summarize
if [ -d "Papers/${TODAY}/${KEYWORD}" ]
then
	ls -tr Papers/${TODAY}/"${KEYWORD}"/*pdf | head -${N_PAPER} | while read PDF
	do
		movie=$(echo ${PDF} | awk -F "/" '{print $NF}' | sed 's/.pdf//g')
		zip_file=Summarize/${TODAY}/${KEYWORD}/${movie}/${movie}.zip
		zip_path=Summarize/${TODAY}/${KEYWORD}/${movie}

		if [ ! -f ${zip_file} ]
		then
			mkdir -p Summarize/${TODAY}/${KEYWORD} /dlst/prom/workspace.pyg/Summarize/${TODAY}/${KEYWORD}/
			python PaperSummarize.py --input_pdf ${PDF} --keyword ${KEYWORD} --today ${TODAY}

		# if ${INITIAL_set}
		# then
		# 	${python} notimail-jobfinish.py --zip_path ${zip_file} --keyword ${KEYWORD} --today ${TODAY} --initial ${INITIAL}
		# fi
		fi
	done
else
	echo -e "\n요약 할 신규 논문이 없습니다.\n"
fi
