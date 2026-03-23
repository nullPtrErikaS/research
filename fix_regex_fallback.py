# Fix the file by removing the regex fallback code
with open('streamlit_app/streamlit_app.py', 'r', encoding='utf-8') as f:
    lines = f.readlines()

# Find the start of the regex fallback section
start_idx = None
end_idx = None

for i, line in enumerate(lines):
    if '# No token column found' in line:
        start_idx = i
        print(f"Found start at line {i+1}: {line.rstrip()}")
    if start_idx is not None and 'return df_local, tok_col, av_kw' in line and i > start_idx:
        end_idx = i + 1
        print(f"Found end at line {i+1}")
        break

if start_idx is not None and end_idx is not None:
    # Remove lines from start_idx to end_idx-1, keep the return statement
    new_lines = lines[:start_idx]
    
    # Add blank line and return statement
    new_lines.append("        \n")
    new_lines.append("    return df_local, tok_col, av_kw\n")
    
    # Add remaining lines after the return
    new_lines.extend(lines[end_idx:])
    
    with open('streamlit_app/streamlit_app.py', 'w', encoding='utf-8') as f:
        f.writelines(new_lines)
    
    print(f"Successfully removed regex fallback (removed {end_idx - start_idx} lines)")
else:
    print(f"Could not find section. start_idx={start_idx}, end_idx={end_idx}")
