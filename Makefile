install:
	pip install -r requirements.txt
	git clone https://github.com/HPI-Information-Systems/GutenTAG.git

run:
	python -m timeeval_gui

clear:
	sudo rm -r GutenTAG
