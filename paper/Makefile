# Build the draft (needs TeX Live: latexmk + pdflatex + bibtex). Tables first: make tables.
.PHONY: all tables clean tbd
all: main.pdf
main.pdf: main.tex refs.bib sections/*.tex tables/*.tex
	latexmk -pdf -interaction=nonstopmode main.tex
tables:
	cd .. && python scripts/make_paper_tables.py
tbd:
	@grep -o '\\TBD' sections/*.tex tables/*.tex | cut -d: -f1 | sort | uniq -c
clean:
	latexmk -C
