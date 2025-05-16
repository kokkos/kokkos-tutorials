LATEXMK_OPTS = -pdf -interaction=nonstopmode -shell-escape

default: all

workaround-on:
	sed -i.bak 's|%\\input|\\input|g' KokkosTutorial_PreTitle.tex
	rm KokkosTutorial_PreTitle.tex.bak

workaround-off:
	sed -i.bak 's|\\input|%\\input|g' KokkosTutorial_PreTitle.tex
	sed -i.bak 's|%%\\input|%\\input|g' KokkosTutorial_PreTitle.tex
	rm KokkosTutorial_PreTitle.tex.bak

overview:
	latexmk $(LATEXMK_OPTS) Kokkos_ShortOverview
short:
	latexmk $(LATEXMK_OPTS) KokkosTutorial_Short
medium:
	latexmk $(LATEXMK_OPTS) KokkosTutorial_Medium
module1:
	latexmk $(LATEXMK_OPTS) KokkosTutorial_01_Introduction
module2:
	latexmk $(LATEXMK_OPTS) KokkosTutorial_02_ViewsAndSpaces
module3:
	latexmk $(LATEXMK_OPTS) KokkosTutorial_03_MDRangeMoreViews
module4:
	latexmk $(LATEXMK_OPTS) KokkosTutorial_04_HierarchicalParallelism
module5:
	latexmk $(LATEXMK_OPTS) KokkosTutorial_05_SIMDStreamsTasking
module6:
	latexmk $(LATEXMK_OPTS) KokkosTutorial_06_InternodeMPIAndPGAS
module7:
	latexmk $(LATEXMK_OPTS) KokkosTutorial_07_Tools
module8:
	latexmk $(LATEXMK_OPTS) KokkosTutorial_08_KokkosKernels

all: module1 module2 module3 module4 module5 module6 module7 module8

clean:
	latexmk -c
