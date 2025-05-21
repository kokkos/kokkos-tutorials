
default: release-46

workaround-on:
	sed -i.bak 's|%\\input|\\input|g' KokkosTutorial_PreTitle.tex
	rm KokkosTutorial_PreTitle.tex.bak

workaround-off:
	sed -i.bak 's|\\input|%\\input|g' KokkosTutorial_PreTitle.tex
	sed -i.bak 's|%%\\input|%\\input|g' KokkosTutorial_PreTitle.tex
	rm KokkosTutorial_PreTitle.tex.bak

release-36:
	latexmk release-36

release-37:
	latexmk release-37

release-40:
	latexmk release-40

release-41:
	latexmk release-41

release-42:
	latexmk release-42

release-43:
	latexmk release-43

release-44:
	latexmk release-44

release-45:
	latexmk release-45

release-46:
	latexmk release-46

clean:
	latexmk -c
