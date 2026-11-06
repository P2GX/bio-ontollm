# Generating the lecture files

These lectures were created using [quarto](https://quarto.org/). A GitHub [build](../.github/workflows/publish.yml) is used to publish the lectures on a GitHub io [site](https://p2gx.github.io/bio-ontollm/). The page explains how to generate the HTML pages locally and how to create the corresponding PDF files.




## Setting up Quarto

- Download the quarto executable from https://quarto.org/
- Follow installation instructions for your OS.


Additionally, we need to add a quarto library for some of the lectures

```bash
quarto add leovan/quarto-pseudocode
```

## VS Code

We recommend use [VS Code](https://code.visualstudio.com) for creating or editing lectures in quarto.


## Virtual environment

Some of the slides use Python to create figures or run code snippets. Therefore, we need to install a virtual environment to create the lecture slides. It is probably possible to do this with conda environments, but we prefer the virtual environment package for its simplicity. 

```bash
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install numpy matplotlib pandas jupyter ipykernel networkx scipy
```

## Preview

For running individual lectures and checking the result on your local browser, use the preview functionality.
The following command will preview the entire series of lectures.

```bash
quarto preview index.qmd
```

This will open up a browser window, e.g., `http://localhost:7783/`. Copy the address, go back to VS Code, using the command pallette enter Simple Browser Show, and paste in the address. Drag this window to the right and you will have side by side windows.

If there is a hiccup, then this command to find out the process id of quarto and kill it may come in handy

```bash
lsof -i :7783
kill -9 <pid>
```
where `pid` is the process id revealed by the lsof command.


## Reset

Occasionally, we have found it necessary to reset the entire environment.

rm -rf .quarto/
rm -rf _site/  


## Out PDF files

In order to create a PDF file from a given lecture, first use the `render` command

```bash
quarto render lectures/ontology1.qmd 
```

This will create a self contained HTML file in the `_site` directory.

```bash
npx decktape reveal --size 1280x720 --pause 3000 _site/lectures/ontology1.html ontology1.pdf
```

`npx` is included in the `node` package. On mac, it can be installed with

```bash
brew install node
```

Other install options can be found at [node.js](https://nodejs.org/en)


## 2 slides per page

To output a PDF file with 2 slides per page, install the `pdfcpu` package:

```brew
brew install pdfcpu
```

Then enter the following command:

```bash
pdfcpu nup "formsize:A4P, border:on" ontology1-2up.pdf 2 ontology1.pdf
```

The file `ontology1-2up.pdf` contains the lecture with 2 slides per PDF page.