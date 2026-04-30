import os
import glob
import re

files = glob.glob('plot_*.py')
files.remove('plot_config.py')

import_pattern = re.compile(r'from plot_config import \(')
single_import_pattern = re.compile(r'from plot_config import (.*)')

replacements = {
    '"black"': 'C_BLACK',
    '"white"': 'C_WHITE',
    '"gray"': 'C_MID_GRAY',
    '"#cccccc"': 'C_LIGHT_GRAY',
    '"#e0e0e0"': 'C_LIGHT_GRAY',
    '"#aaaaaa"': 'C_MID_GRAY',
    '"#d0d0d0"': 'C_LIGHT_GRAY',
    '"#888888"': 'C_MID_GRAY',
    '"#555555"': 'C_DARK_GRAY',
    '"#333333"': 'C_DARK_GRAY',
    '"#222222"': 'C_DARK_GRAY',
    "'black'": 'C_BLACK',
    "'white'": 'C_WHITE',
    "'gray'": 'C_MID_GRAY',
}

for fpath in files:
    with open(fpath, 'r') as f:
        content = f.read()

    # Determine if any of the colors need to be imported
    needs_update = False
    for k in replacements.keys():
        if k in content:
            needs_update = True
            break
            
    if not needs_update:
        continue

    print(f"Updating {fpath}")

    # Make the replacements
    for k, v in replacements.items():
        content = content.replace(f'color={k}', f'color={v}')
        content = content.replace(f'ecolor={k}', f'ecolor={v}')
        content = content.replace(f'lcolor={k}', f'lcolor={v}')
        content = content.replace(f'ec={k}', f'ec={v}')
        content = content.replace(f'linecolor={k}', f'linecolor={v}')
        content = content.replace(f'txt_color = {k}', f'txt_color = {v}')

    # Add the imports to the plot_config import statement
    new_imports = "C_BLACK, C_WHITE, C_DARK_GRAY, C_MID_GRAY, C_LIGHT_GRAY"
    
    if "C_BLACK" not in content:
        # Multi-line import
        if "from plot_config import (" in content:
            content = content.replace(
                "from plot_config import (",
                f"from plot_config import (\n    {new_imports},"
            )
        # Single-line import
        else:
            match = single_import_pattern.search(content)
            if match:
                old_imports = match.group(1)
                content = content.replace(
                    match.group(0),
                    f"from plot_config import {old_imports}, {new_imports}"
                )

    with open(fpath, 'w') as f:
        f.write(content)

print("Done updating colors.")
