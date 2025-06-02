
default: kug-2025

workaround-on:
	sed -i.bak 's|%\\input|\\input|g' KokkosTutorial_PreTitle.tex
	rm KokkosTutorial_PreTitle.tex.bak

workaround-off:
	sed -i.bak 's|\\input|%\\input|g' KokkosTutorial_PreTitle.tex
	sed -i.bak 's|%%\\input|%\\input|g' KokkosTutorial_PreTitle.tex
	rm KokkosTutorial_PreTitle.tex.bak

kug-2025:
	pdflatex kug-2025.tex

clean:
	rm -r *.aux *.log *.nav *.out *.snm *.toc *.vrb
