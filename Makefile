
all: build serve

build:
	jupyter-book build . --config docs/_config.yml --toc docs/_toc.yml

serve:
	open _build/html/index.html
