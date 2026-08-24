# bio-ontollm

Approaches to biomedical knowledge: Ontologies and Large Language Models.

A lecture series built with [Quarto](https://quarto.org/). Each lecture is a
reveal.js slide deck written as a `.qmd` file in `lectures/`. Slides are
rendered and published to GitHub Pages automatically on every push to `master`
(see `.github/workflows/publish.yml`).

See [LECTURES.md](LECTURES.md) for the plan of the series.

## Repository layout

| Path | Contents |
| --- | --- |
| `index.qmd` | Course home page |
| `lectures/*.qmd` | One file per lecture |
| `lectures/_metadata.yml` | Shared reveal.js settings for all lectures |
| `lectures/utils/` | Python helper modules imported by the code chunks |
| `lectures/img/` | Figures |
| `style/styles.css` | Custom slide styling |
| `_extensions/` | Vendored Quarto extensions (checked in — no `quarto add` needed) |
| `sandbox/` | Scratch work, excluded from the render |
| `_site/` | Render output (git-ignored) |

## Setup

### 1. Quarto

```bash
brew install --cask quarto     # macOS; or follow https://quarto.org/docs/get-started/
quarto --version
```

### 2. Python environment

The code chunks need Python **3.12 or newer** (see `pyproject.toml`). Note that
a system or Homebrew default Python is often a different version without the
required packages, so create a dedicated virtual environment in the repo root:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

Or, with [uv](https://docs.astral.sh/uv/):

```bash
uv venv .venv --python 3.12
uv pip install --python .venv/bin/python -r requirements.txt
```

`.venv/` is git-ignored.

### 3. Jupyter kernel

The lectures declare `jupyter: python3` in their YAML header, so Quarto looks
up a kernel *named* `python3`. If you already have a global `python3` kernel it
will win, and the render fails on the imports. Register the virtual
environment's own kernel under that name so it takes precedence whenever the
environment is active:

```bash
source .venv/bin/activate
python -m ipykernel install --sys-prefix --name python3 \
    --display-name "Python 3 (bio-ontollm)"
```

### 4. Verify

```bash
source .venv/bin/activate
quarto check jupyter
```

The reported Python path should point into `.venv`.

## Editing slides

Activate the environment first — that is what selects the right kernel — then
start a preview of the lecture you are working on:

```bash
source .venv/bin/activate
quarto preview lectures/ontology1.qmd --port 7783
```

Quarto watches the file and reloads the browser on every save.

To render without the live server:

```bash
quarto render lectures/ontology1.qmd    # one lecture
quarto render                           # the whole site
```

Output is written to `_site/`, since the repo is a Quarto *website* project.

### Working directory and helper modules

Quarto executes code chunks with the working directory set to `lectures/`.
That is why the chunks can do `sys.path.append('utils')`. New helper modules
belong in `lectures/utils/`.

### Side-by-side preview in VS Code

Setting up the preview inside VS Code directly can be troublesome; a reliable
workaround is to start `quarto preview` in a shell as above, copy the
`http://localhost:7783/...` address, then in VS Code open the command palette,
run *Simple Browser: Show*, and paste the address. Drag that tab to the right
for a side-by-side view.

### Stopping a stuck preview

```bash
lsof -i :7783
kill -9 <pid>
```

where `pid` is the process id revealed by `lsof`.

## Reset

```bash
rm -rf .quarto/
rm -rf _site/
```
