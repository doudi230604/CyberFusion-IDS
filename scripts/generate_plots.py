#!/usr/bin/env python3
"""Generate plots by running model scripts and save figures to plots/.

Behavior:
- Uses matplotlib Agg backend and monkeypatches plt.show so each call to show() saves the current figure to plots/<module>_fig<N>.png
- Looks for model scripts in the models/ folder that define a main() or ultra_fast_demo() and executes them (ultra_fast_demo preferred if present)
- Appends a '## Figures générées' section to report.md with links to generated images
"""
import os
import sys
import glob
import importlib.util
import traceback
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
PLOTS_DIR = os.path.join(ROOT, 'plots')
REPORT_MD = os.path.join(ROOT, 'report.md')
MODELS_GLOB = os.path.join(ROOT, 'models', '**', '*.py')

os.makedirs(PLOTS_DIR, exist_ok=True)

saved_files = []

# Monkeypatch plt.show to save each figure
_show_counters = {}
_original_show = plt.show

def _patched_show(*args, **kwargs):
    # Inspect stack to find calling module name (best-effort)
    try:
        import inspect
        frame = inspect.currentframe()
        # climb a few frames to find the module
        module_name = None
        for _ in range(6):
            if not frame:
                break
            module = frame.f_globals.get('__name__')
            if module and module != 'matplotlib.pyplot':
                module_name = module.split('.')[-1]
                break
            frame = frame.f_back
        if not module_name:
            module_name = 'plot'
    except Exception:
        module_name = 'plot'

    count = _show_counters.get(module_name, 0) + 1
    _show_counters[module_name] = count
    filename = f"{module_name}_fig{count}.png"
    filepath = os.path.join(PLOTS_DIR, filename)
    try:
        plt.savefig(filepath, bbox_inches='tight', dpi=150)
        print(f"Saved figure: {filepath}")
        saved_files.append(filepath)
    except Exception as e:
        print(f"Failed to save figure {filepath}: {e}")
    finally:
        plt.close()

# Replace plt.show with our patched version
plt.show = _patched_show


def run_module(path):
    name = os.path.splitext(os.path.basename(path))[0]
    print(f"\n=== Running module: {name} ===")
    try:
        spec = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        # Prefer ultra_fast_demo() if available
        if hasattr(module, 'ultra_fast_demo'):
            print(f"Found ultra_fast_demo in {name}, running it...")
            try:
                module.ultra_fast_demo()
            except TypeError:
                module.ultra_fast_demo()
        elif hasattr(module, 'main'):
            # Avoid running interactive mains that call input()
            with open(path, 'r', encoding='utf-8') as fh:
                src = fh.read()
            if 'input(' in src:
                print(f"main() in {name} looks interactive (contains input()), skipping to avoid blocking.")
            else:
                print(f"Found main() in {name}, running it...")
                try:
                    module.main()
                except TypeError:
                    module.main()
        else:
            print(f"No runnable entrypoint found in {name}, skipping.")
    except Exception as e:
        print(f"Error running {name}: {e}")
        traceback.print_exc()


def find_model_scripts():
    files = glob.glob(MODELS_GLOB, recursive=True)
    # Filter out __init__ files
    files = [f for f in files if not f.endswith('__init__.py')]
    return files


def update_report_with_plots():
    if not saved_files:
        print("No plots to insert into report.md")
        return

    # Build markdown block
    md_lines = []
    md_lines.append('\n\n<!-- GENERATED FIGURES -->')
    md_lines.append('\n## Figures générées')
    md_lines.append('\nLes figures suivantes ont été générées automatiquement par les scripts de modèles et enregistrées dans le dossier `plots/`.')

    # Group by module prefix
    grouped = {}
    for p in saved_files:
        fname = os.path.basename(p)
        key = fname.split('_fig')[0]
        grouped.setdefault(key, []).append(fname)

    for key, files in grouped.items():
        md_lines.append(f"\n### {key}")
        for f in sorted(files):
            md_lines.append(f"\n![{f}](plots/{f})")

    md_block = '\n'.join(md_lines) + '\n'

    # Insert just after the '## 7. Résultats Expérimentaux' header if present, else append to end
    with open(REPORT_MD, 'r', encoding='utf-8') as fh:
        content = fh.read()

    insert_point = None
    # Try to find '## 7. Résultats Expérimentaux' OR the end of the Sommaire block
    for marker in ['## 7. Résultats Expérimentaux', '## Résultats Expérimentaux', '---\n\n## 1. Introduction']:
        idx = content.find(marker)
        if idx != -1:
            insert_point = content.find('\n', idx)  # find newline after header
            break

    if insert_point is None:
        content = content + '\n' + md_block
    else:
        # Insert after the 'Résultats Expérimentaux' header text block (after first section's header line)
        # Safer: append md_block before the 1. Introduction section
        intro_idx = content.find('\n## 1. Introduction')
        if intro_idx != -1:
            content = content[:intro_idx] + md_block + '\n' + content[intro_idx:]
        else:
            content = content + '\n' + md_block

    with open(REPORT_MD, 'w', encoding='utf-8') as fh:
        fh.write(content)

    print(f"Inserted {len(saved_files)} figures into {REPORT_MD}")


if __name__ == '__main__':
    scripts = find_model_scripts()
    # Prefer running lightweight/fast scripts first
    preferred = [
        'randomforesttoniot.py',
        'isolation_forest_unsw_nb15.py',
        'randomforestunsw.py',
        'randomforestcicidi.py',
        'decision_tree02.py'
    ]

    ordered = []
    for p in scripts:
        if os.path.basename(p) in preferred:
            ordered.append(p)
    for p in scripts:
        if p not in ordered:
            ordered.append(p)

    for p in ordered:
        run_module(p)

    if saved_files:
        update_report_with_plots()
    else:
        print("No figures were saved.")

    print("\nDone.")
