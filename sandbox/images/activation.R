library(ggplot2)
library(patchwork)

# Define functions
sigmoid <- function(x) 1 / (1 + exp(-x))
d_sigmoid <- function(x) sigmoid(x) * (1 - sigmoid(x))
relu <- function(x) pmax(0, x)
d_relu <- function(x) ifelse(x > 0, 1, 0)

x_vals <- seq(-5, 5, length.out = 1000)
df <- data.frame(x = x_vals)

# Common styling
base_theme <- theme_minimal(base_size = 14) + 
  theme(legend.position = "bottom")

# 1. Sigmoid Plot
p1 <- ggplot(df, aes(x = x)) +
  stat_function(fun = sigmoid, aes(color = "Function"), linewidth = 1) +
  stat_function(fun = d_sigmoid, aes(color = "Derivative"), linewidth = 1, linetype = "dashed") +
  scale_color_manual(values = c("Function" = "#2c3e50", "Derivative" = "#e74c3c")) +
  labs(title = "Sigmoid", y = NULL, x = NULL, color = "Type") +
  base_theme

# 2. ReLU Plot
p2 <- ggplot(df, aes(x = x)) +
  stat_function(fun = relu, aes(color = "Function"), linewidth = 1) +
  stat_function(fun = d_relu, aes(color = "Derivative"), linewidth = 1, linetype = "dashed") +
  scale_color_manual(values = c("Function" = "#2c3e50", "Derivative" = "#e74c3c")) +
  labs(title = "ReLU", y = NULL, x = NULL, color = "Type") +
  base_theme

p <- p1 + p2 + plot_layout(guides = "collect") & theme(legend.position = 'bottom')
ggsave("activations.png", p, width = 9, height = 4, dpi = 300)