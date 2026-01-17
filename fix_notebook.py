import json
import re

nb_path = "Tire Defects Detection.ipynb"

try:
    with open(nb_path, 'r', encoding='utf-8') as f:
        nb = json.load(f)

    changed = False
    
    # Regex to match DROP_COLS assignment
    # Looks for DROP_COLS = [...]
    pattern = re.compile(r'DROP_COLS\s*=\s*\[(.*?)\]', re.DOTALL)

    for cell in nb['cells']:
        if cell['cell_type'] == 'code':
            source_str = "".join(cell['source'])
            
            # Check if this cell defines DROP_COLS
            if 'DROP_COLS' in source_str and 'tire_id' in source_str:
                
                # Check if TireProductionID is already there
                if 'TireProductionID' in source_str:
                    print("TireProductionID already present.")
                    continue

                # Naive replace is safer if regex is complex for list structure
                # We know the specific items we expect
                if '"tire_id"' in source_str:
                     new_source_str = source_str.replace('"tire_id"', '"tire_id", "TireProductionID"')
                     
                     # Split back into lines for JSON format (jupyter usually keeps \n in strings)
                     # But simple splitlines(keepends=True) works
                     new_source_lines = []
                     # We need to be careful to preserve newline structure as Jupyter expects
                     # Usually source is a list of strings ending with \n
                     
                     # Let's just do a direct string replace on the list content
                     new_source = []
                     for line in cell['source']:
                         if '"tire_id"' in line and '"TireProductionID"' not in line:
                             new_line = line.replace('"tire_id"', '"tire_id", "TireProductionID"')
                             new_source.append(new_line)
                             changed = True
                             print(f"Updated line: {new_line.strip()}")
                         else:
                             new_source.append(line)
                     
                     if changed:
                        cell['source'] = new_source

    if changed:
        with open(nb_path, 'w', encoding='utf-8') as f:
            json.dump(nb, f, indent=1)
        print("Notebook updated successfully.")
    else:
        print("No changes made. Pattern might be strictly different.")

except Exception as e:
    print(f"Error: {e}")
