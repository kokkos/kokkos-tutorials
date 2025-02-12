
default: release-45

workaround-on:
	sed -i.bak 's|%\\input|\\input|g' KokkosTutorial_PreTitle.tex
	rm KokkosTutorial_PreTitle.tex.bak

workaround-off:
	sed -i.bak 's|\\input|%\\input|g' KokkosTutorial_PreTitle.tex
	sed -i.bak 's|%%\\input|%\\input|g' KokkosTutorial_PreTitle.tex
	rm KokkosTutorial_PreTitle.tex.bak

release-36:
	pdflatex release-36.tex

release-37:
	pdflatex release-37.tex

release-40:
	pdflatex release-40.tex

release-41:
	pdflatex release-41.tex

release-42:
	pdflatex release-42.tex

release-43:
	pdflatex release-43.tex

release-44:
	pdflatex release-44.tex

release-45:
	pdflatex release-45.tex

clean:
	rm -r *.aux *.log *.nav *.out *.snm *.toc *.vrb
