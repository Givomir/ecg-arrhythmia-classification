"""
Builds the desktop app into ONE .exe (dist/ECGAnalyzer.exe) with PyInstaller.

Use a clean environment with the same scikit-learn/numpy versions the
models were saved with (otherwise the pickles may not load):
    py -3.11 -m venv .venv
    .venv\\Scripts\\python -m pip install -r desktop\\requirements-desktop.txt
    .venv\\Scripts\\python desktop\\build_exe.py

Needs (generated locally, not in git): artifacts_cascade/, artifacts/,
artifacts_mlp/, rag_index/ with query_embeddings.json (python
build_rag_index.py creates it) and the ONNX embedding model in
desktop/onnx_model/ - export it once in an environment that has PyTorch:
    python rag_embedder.py --export desktop/onnx_model
"""
import os
import sys
import shutil
import subprocess

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
UI_DIR = os.path.join(HERE, 'ui')
ONNX_DIR = os.path.join(HERE, 'onnx_model')
VENDOR_JS = os.path.join(UI_DIR, 'vendor', 'plotly.min.js')

MODEL_DIRS = ['artifacts_cascade', 'artifacts', 'artifacts_mlp']
INDEX_FILES = ['chunks.json', 'config.json', 'embeddings.npy', 'query_embeddings.json']

# Not needed at run time - kept out of the .exe to save space
EXCLUDES = ['matplotlib', 'torch', 'tensorflow', 'keras', 'sentence_transformers',
            'transformers', 'streamlit', 'IPython', 'jupyter', 'notebook', 'tkinter',
            'PIL', 'fastapi', 'uvicorn', 'pytest']


def copy_plotly_js():
    """plotly.js from the installed plotly package -> ui/vendor (offline use)."""
    import plotly
    src = os.path.join(os.path.dirname(plotly.__file__), 'package_data', 'plotly.min.js')
    os.makedirs(os.path.dirname(VENDOR_JS), exist_ok=True)
    shutil.copyfile(src, VENDOR_JS)
    print(f"plotly.js {plotly.__version__} -> {os.path.relpath(VENDOR_JS, ROOT)}")


def check_inputs():
    missing = [d for d in MODEL_DIRS if not os.path.exists(os.path.join(ROOT, d, 'model.pkl'))]
    missing += [os.path.join('rag_index', f) for f in INDEX_FILES
                if not os.path.exists(os.path.join(ROOT, 'rag_index', f))]
    missing += [os.path.join('desktop', 'onnx_model', f)
                for f in ('model.onnx', 'tokenizer.json', 'embedder.json')
                if not os.path.exists(os.path.join(ONNX_DIR, f))]
    if missing:
        sys.exit("Missing build inputs:\n  " + "\n  ".join(missing))


def main():
    copy_plotly_js()
    check_inputs()
    sep = os.pathsep   # ';' on Windows
    data = [(UI_DIR, 'ui'), (os.path.join(ROOT, 'rag_index'), 'rag_index'),
            (ONNX_DIR, 'embedder')]
    data += [(os.path.join(ROOT, d), d) for d in MODEL_DIRS]

    cmd = [sys.executable, '-m', 'PyInstaller', '--noconfirm', '--clean',
           '--onefile', '--windowed', '--name', 'ECGAnalyzer',
           '--icon', os.path.join(HERE, 'icon.ico'),
           '--distpath', os.path.join(ROOT, 'dist'),
           '--workpath', os.path.join(ROOT, 'build'),
           '--specpath', os.path.join(ROOT, 'build'),
           '--paths', ROOT,
           '--hidden-import', 'cascade',
           '--hidden-import', 'rag_embedder',
           '--collect-binaries', 'onnxruntime',
           '--collect-submodules', 'sklearn.ensemble',
           '--collect-submodules', 'sklearn.tree',
           '--collect-submodules', 'sklearn.neural_network']
    for src, dst in data:
        cmd += ['--add-data', f"{src}{sep}{dst}"]
    for mod in EXCLUDES:
        cmd += ['--exclude-module', mod]
    cmd.append(os.path.join(ROOT, 'desktop_app.py'))

    print("Running PyInstaller...")
    subprocess.run(cmd, check=True, cwd=ROOT)
    exe = os.path.join(ROOT, 'dist', 'ECGAnalyzer.exe')
    print(f"\nBuilt: {exe} ({os.path.getsize(exe) / 1e6:.0f} MB)")


if __name__ == '__main__':
    main()
