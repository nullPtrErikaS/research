# Script to move Distance Analysis from tab_docs to tab_stats

with open('streamlit_app/streamlit_app.py', 'r', encoding='utf-8') as f:
    lines = f.readlines()

# Extract Distance Analysis block (lines 3229-3298, 0-indexed: 3228-3297)
# This is the "if len(st.session_state['selected_ids']) >= 2:" block
distance_analysis_start = 3228  # Line 3229 in 1-indexed
distance_analysis_end = 3298    # Up to but not including line 3299

distance_block = lines[distance_analysis_start:distance_analysis_end]

# Remove the Distance Analysis block from tab_docs
new_lines = lines[:distance_analysis_start] + lines[distance_analysis_end:]

# Now find where to insert it in tab_stats
# We need to find the line with "update_selection(cluster_docs" and insert after its closing brace
for i, line in enumerate(new_lines):
    if 'update_selection(cluster_docs' in line:
        # This line is around where we want to insert
        # We need to insert after the next indentation level closes
        # Insert after the line containing "update_selection"
        # Look for the end of the if statement
        insert_point = i + 1
        # Make sure we're inserting within the container block, before the else
        print(f"Found update_selection at line {i+1}")
        print(f"Will insert Distance Analysis after line {insert_point}")
        break

# Add the distance analysis block with proper indentation
# The block starts with 4 spaces of indentation (inside the if statement in tab_docs)
# We need to add it inside the st.container in tab_stats with 12 spaces (within the with block)
new_distance_block = []
for line in distance_block:
    # Remove the leading 4 spaces from distance_analysis block
    if line.startswith('    '):
        line = line[4:]
    # Add 12 spaces to put it inside the tab_stats container
    if line.strip():  # Don't add leading spaces to empty lines
        line = '            ' + line
    new_distance_block.append(line)

# Insert with a markdown divider
insert_lines = [ '            st.markdown("---")\n']
insert_lines += new_distance_block

# Insert at the proper location
new_lines = new_lines[:insert_point] + insert_lines + new_lines[insert_point:]

# Write the modified file
with open('streamlit_app/streamlit_app.py', 'w', encoding='utf-8') as f:
    f.writelines(new_lines)

print(f"✓ Moved Distance Analysis from tab_docs to tab_stats")
print(f"✓ Removed {distance_analysis_end - distance_analysis_start} lines from tab_docs")
print(f"✓ Inserted documentation after update_selection call")
