import glob
files = glob.glob('plot_*.py')
files.remove('plot_config.py')
for fpath in files:
    with open(fpath, 'r') as f:
        content = f.read()
    if 'C_BLACK' in content and 'C_BLACK,' not in content:
        if 'from plot_config import (' in content:
            content = content.replace('from plot_config import (', 'from plot_config import (\n    C_BLACK, C_WHITE, C_DARK_GRAY, C_MID_GRAY, C_LIGHT_GRAY,')
        else:
            # simple from plot_config import ...
            lines = content.split('\n')
            for i, line in enumerate(lines):
                if line.startswith('from plot_config import'):
                    lines[i] = line + ', C_BLACK, C_WHITE, C_DARK_GRAY, C_MID_GRAY, C_LIGHT_GRAY'
                    break
            content = '\n'.join(lines)
        with open(fpath, 'w') as f:
            f.write(content)
