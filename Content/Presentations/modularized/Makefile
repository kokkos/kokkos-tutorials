
default: all

workaround-on:
	sed -i.bak 's|%\\input|\\input|g' KokkosTutorial_PreTitle.tex
	rm KokkosTutorial_PreTitle.tex.bak

workaround-off:
	sed -i.bak 's|\\input|%\\input|g' KokkosTutorial_PreTitle.tex
	sed -i.bak 's|%%\\input|%\\input|g' KokkosTutorial_PreTitle.tex
	rm KokkosTutorial_PreTitle.tex.bak

overview:
	pdflatex Kokkos_ShortOverview
short:
	pdflatex KokkosTutorial_Short
medium:
	pdflatex KokkosTutorial_Medium
module1:
	pdflatex KokkosTutorial_01_Introduction
module2:
	pdflatex KokkosTutorial_02_ViewsAndSpaces
module3:
	pdflatex KokkosTutorial_03_MDRangeMoreViews
module4:
	pdflatex KokkosTutorial_04_HierarchicalParallelism
module5:
	pdflatex KokkosTutorial_05_SIMDStreamsTasking
module6:
	pdflatex KokkosTutorial_06_InternodeMPIAndPGAS.tex
module7:
	pdflatex KokkosTutorial_07_Tools.tex
module8:
	pdflatex KokkosTutorial_08_KokkosKernels.tex

all: module1 module2 module3 module4 module5 module6 module7 module8

clean:
	rm -r *.aux *.log *.nav *.out *.snm *.toc *.vrb
