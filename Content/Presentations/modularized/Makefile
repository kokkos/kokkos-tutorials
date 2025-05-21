
default: all

workaround-on:
	sed -i.bak 's|%\\input|\\input|g' KokkosTutorial_PreTitle.tex
	rm KokkosTutorial_PreTitle.tex.bak

workaround-off:
	sed -i.bak 's|\\input|%\\input|g' KokkosTutorial_PreTitle.tex
	sed -i.bak 's|%%\\input|%\\input|g' KokkosTutorial_PreTitle.tex
	rm KokkosTutorial_PreTitle.tex.bak

overview:
	latexmk Kokkos_ShortOverview
short:
	latexmk KokkosTutorial_Short
medium:
	latexmk KokkosTutorial_Medium
module1:
	latexmk KokkosTutorial_01_Introduction
module2:
	latexmk KokkosTutorial_02_ViewsAndSpaces
module3:
	latexmk KokkosTutorial_03_MDRangeMoreViews
module4:
	latexmk KokkosTutorial_04_HierarchicalParallelism
module5:
	latexmk KokkosTutorial_05_SIMDStreamsTasking
module6:
	latexmk KokkosTutorial_06_InternodeMPIAndPGAS
module7:
	latexmk KokkosTutorial_07_Tools
module8:
	latexmk KokkosTutorial_08_KokkosKernels

all: module1 module2 module3 module4 module5 module6 module7 module8

clean:
	latexmk -c
