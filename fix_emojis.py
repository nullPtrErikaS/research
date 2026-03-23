#!/usr/bin/env python
# -*- coding: utf-8 -*-

with open('streamlit_app/streamlit_app.py', 'r', encoding='utf-8', errors='replace') as f:
    lines = f.readlines()

# Fix the lines
new_lines = []
for i, line in enumerate(lines):
    # Fix the delete button
    if "st.button" in line and "del_" in line and ("ðŸ" in line or "📋" in line or "🗑" in line):
        line = "                    if st.button('Delete', key=f\"del_{name}\"):\n"
    # Fix the Session Reasoning Trail
    elif "Session Reasoning Trail" in line and "📋" in line:
        line = line.replace("📋 ", "")
    # Fix the Export button
    elif "Export as TXT" in line and "📥" in line:
        line = line.replace("📥 ", "")
    
    new_lines.append(line)

with open('streamlit_app/streamlit_app.py', 'w', encoding='utf-8') as f:
    f.writelines(new_lines)

print("Fixed emojis")
