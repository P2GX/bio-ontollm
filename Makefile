# Clean up temporary Quarto and Jupyter artifacts recursively across all folders
clean:
	find . -name "*.quarto_ipynb*" -delete
	find . -name ".quarto" -type d -exec rm -rf {} +