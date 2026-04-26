# Sandbox

This folder contains scripts and files that were used to create graphics or other elements for this lecture series.


## tikz

We will usually transform these files into png files

```bash
pdflatex myfile.tex
pdftoppm -png -singlefile -rx 300 -ry 300 myfile.pdf -o myfile
``` 

