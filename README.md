# llmintro
Large Language Model Introduction Lectures

## Quarto

Let's test out the quarto ecosystem for creating slides.

In brief:

- https://quarto.org/
- Follow installation instructions
- I am using VS Code for creating slides

## Running quarto on VS Code

You will need to create a virtual environment and use to install some packages needed by the quarto ecosystem

```bash
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install numpy matplotlib pandas jupyter ikernel
```

I have had some trouble setting up the preview within VS Code, but a workaround was to start quarto in some shell

```bash
quarto preview token.qmd
```

This will open up a browser window, e.g., `http://localhost:7783/`. Copy the address, go back to VS Code, using the command pallette enter Simple Browser Show, and paste in the address. Drag this window to the right and you will have side by side windows.

If there is a hiccup, then this command to find out the process id of quarto and kill it may come in handy

```bash
lsof -i :7783
kill -9 <pid>
```
where `pid` is the process id revealed by the lsof command.
