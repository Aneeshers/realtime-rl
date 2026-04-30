import glob
files = glob.glob('plot_*.py')
files.remove('plot_config.py')

new_colors = ['C_BLACK', 'C_WHITE', 'C_DARK_GRAY', 'C_MID_GRAY', 'C_LIGHT_GRAY']

for fpath in files:
    with open(fpath, 'r') as f:
        content = f.read()
        
    if 'C_BLACK' not in content:
        continue # Doesn't use it
        
    # check if they are actually in the import list
    lines = content.split('\n')
    in_import = False
    import_found = False
    
    for i, line in enumerate(lines):
        if line.startswith('from plot_config import'):
            import_found = True
            if '(' in line:
                in_import = True
                lines.insert(i+1, '    C_BLACK, C_WHITE, C_DARK_GRAY, C_MID_GRAY, C_LIGHT_GRAY,')
                break
            else:
                lines[i] = line + ', C_BLACK, C_WHITE, C_DARK_GRAY, C_MID_GRAY, C_LIGHT_GRAY'
                break
                
    content = '\n'.join(lines)
    with open(fpath, 'w') as f:
        f.write(content)

