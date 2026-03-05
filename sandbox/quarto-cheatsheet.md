# Quarto Cheatsheet

# Callout notes

```bash
::: {.callout-note icon=false, title="The Linearity Property (Relative Positioning)"}

For any fixed offset $\Delta k$, the encoding at a shifted position can be represented as a linear transformation of the original:

$$PE_{pos + \Delta k} = \mathbf{M}_{\Delta k} \cdot PE_{pos}$$

:::
```


# TODO

- [Todo: explain learning procedure (or refer to previous slides)]{.neon-todo}


# FIgure

![Caption for the image](path/to/image.png){width=80% fig-align="center"}

fig-link (Makes image clickable link)

multiple figures

::: {layout-ncol=2}
![Dog](dog.jpg)

![Cat](cat.jpg)
:::

## Slide Title {background-image="forest.jpg" background-opacity="0.5"}

This content sits on top of the image.

![Click to zoom](big-map.png){.lightbox}  (zoom in with click during presentation)


## Python

fig-width/fig-height should match with the matplotlib code

```{python}
#| echo: false
#| fig-width: 8
#| fig-height: 4.8
#| out-width: "100%"
#| fig-align: center
import sys
sys.path.append('utils') 
from ann_plots import plot_neuron

plot_neuron()
```
