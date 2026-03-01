import pandas as pd
import numpy as np

# --- 1. Data Setup ---
tokens = ["I", "am", "an", "automaton"]
d_model = 4
n = 100
rows = []

# --- 2. Logic: Building the "String Table" ---
for k in range(len(tokens)):
    # Start the row with Text and Index
    row = [tokens[k], k]
    
    # Calculate Sin/Cos pairs based on d_model
    for j in range(d_model // 2):
        denom = n**(2 * j / d_model)
        sin_val = np.sin(k / denom)
        cos_val = np.cos(k / denom)
        
        # Add formatted strings to the row
        row.append(f"{sin_val:.2f}")
        row.append(f"{cos_val:.2f}")
        
    rows.append(row)

# --- 3. Header Generation ---
headers = ["Sequence", "Index (k)"]
for j in range(d_model // 2):
    headers.extend([f"i={j} (sin)", f"i={j} (cos)"])

# Create DataFrame
df = pd.DataFrame(rows, columns=headers)

# --- 4. Styling for Quarto ---
def style_pe_table(styler):
    # Set headers to bold and light grey background
    styler.set_table_styles([
        {'selector': 'th', 'props': [('background-color', '#f2f2f2'), ('color', 'black'), ('font-weight', 'bold')]}
    ])
    
    # Column 0 (Sequence): White background, black text
    styler.set_properties(subset=["Sequence"], **{'background-color': 'white', 'color': 'black'})
    
    # Column 1 (Index): Light grey background
    styler.set_properties(subset=["Index (k)"], **{'background-color': '#fafafa', 'color': 'black'})
    
    # All other columns (PE values): Reddish text to match your graphic
    pe_cols = [col for col in df.columns if "i=" in col]
    styler.set_properties(subset=pe_cols, **{'color': '#a52a2a', 'background-color': '#fffafa'})
    
    return styler

# Display styled table
df.style.pipe(style_pe_table)