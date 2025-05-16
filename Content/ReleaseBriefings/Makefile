LATEXMK_OPTS = -pdf -interaction=nonstopmode -shell-escape

default: release-45

workaround-on:
	sed -i.bak 's|%\\input|\\input|g' KokkosTutorial_PreTitle.tex
	rm KokkosTutorial_PreTitle.tex.bak

workaround-off:
	sed -i.bak 's|\\input|%\\input|g' KokkosTutorial_PreTitle.tex
	sed -i.bak 's|%%\\input|%\\input|g' KokkosTutorial_PreTitle.tex
	rm KokkosTutorial_PreTitle.tex.bak

release-36:
	latexmk $(LATEXMK_OPTS) release-36

release-37:
	latexmk $(LATEXMK_OPTS) release-37

release-40:
	latexmk $(LATEXMK_OPTS) release-40

release-41:
	latexmk $(LATEXMK_OPTS) release-41

release-42:
	latexmk $(LATEXMK_OPTS) release-42

release-43:
	latexmk $(LATEXMK_OPTS) release-43

release-44:
	latexmk $(LATEXMK_OPTS) release-44

release-45:
	latexmk $(LATEXMK_OPTS) release-45

clean:
	latexmk -c
