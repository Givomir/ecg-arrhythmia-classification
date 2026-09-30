"""
Building the RAG index from the clinical guidelines
====================================================
Reads the guideline text files, splits them into meaningful chunks,
creates embeddings locally (sentence-transformers) and saves an index.

Some files have a .pdf extension but are actually plain text - we read those directly.

Usage (once, and again when new guidelines are added):
    python build_rag_index.py --guidelines_dir guidelines --out rag_index

Arguments:
    --guidelines_dir  folder with the guideline files (.pdf/.txt - read as text)
    --out             where to save the index
    --chunk_size      words per chunk (default 250)
    --overlap         overlap between chunks in words (default 50)
"""
import os
import re
import glob
import argparse
import json
import numpy as np


def read_text_file(path):
    """
    Extracts text from a file. If it is a real PDF (starts with %PDF), a
    PDF parser is used. Otherwise it is read as plain text.
    """
    with open(path, 'rb') as f:
        head = f.read(5)

    # Real PDF -> extract the text with a parser
    if head.startswith(b'%PDF'):
        return _extract_pdf_text(path)

    # Otherwise - plain text
    with open(path, 'r', encoding='utf-8', errors='ignore') as f:
        return f.read()


def _extract_pdf_text(path):
    """Extracts text from a PDF. Tries PyMuPDF, then pypdf."""
    # Attempt 1: PyMuPDF (most robust, best extraction quality)
    try:
        import pymupdf
        doc = pymupdf.open(path)
        parts = [page.get_text() for page in doc]
        doc.close()
        text = "\n".join(parts)
        if len(text.strip()) > 200:
            return text
    except Exception as e:
        print(f"    (PyMuPDF failed: {e}; trying pypdf)")

    # Attempt 2: pypdf
    try:
        from pypdf import PdfReader
        reader = PdfReader(path)
        parts = [(page.extract_text() or '') for page in reader.pages]
        return "\n".join(parts)
    except Exception as e:
        raise RuntimeError(f"Cannot extract text from {path}: {e}")


def clean_text(text):
    """Cleans the text - removes repeated whitespace, normalizes lines."""
    text = text.replace('\r', ' ')
    # Remove broken control characters from the PDF->text conversion
    text = re.sub(r'[\x00-\x08\x0b-\x1f\x7f]', '', text)
    text = re.sub(r'\s+', ' ', text)          # collapse whitespace
    text = re.sub(r'-\s+', '', text)          # rejoin words hyphenated across lines
    return text.strip()


def chunk_quality(text):
    """
    Estimates what share of the chunk is meaningful text (ASCII letters, digits,
    common punctuation). Binary leftovers from PDF streams have low
    quality and are discarded.
    """
    if not text:
        return 0.0
    good = sum(1 for ch in text
               if ch.isascii() and (ch.isalnum() or ch in ' .,;:()-%/\n'))
    return good / len(text)


def chunk_text(text, source, chunk_size=250, overlap=50):
    """
    Splits the text into overlapping chunks of ~chunk_size words.
    The overlap keeps the context at the chunk boundaries.
    Low-quality chunks (binary garbage from PDF streams) are discarded.
    """
    words = text.split()
    chunks = []
    start = 0
    dropped = 0
    while start < len(words):
        end = min(start + chunk_size, len(words))
        chunk_words = words[start:end]
        chunk = ' '.join(chunk_words)
        if len(chunk.strip()) > 100:          # skip very short chunks
            if chunk_quality(chunk) >= 0.85:  # quality text only
                chunks.append({'text': chunk, 'source': source})
            else:
                dropped += 1
        if end == len(words):
            break
        start += chunk_size - overlap
    if dropped:
        print(f"    (discarded {dropped} chunks of binary garbage)")
    return chunks


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--guidelines_dir', required=True)
    ap.add_argument('--out', default='rag_index')
    ap.add_argument('--chunk_size', type=int, default=250)
    ap.add_argument('--overlap', type=int, default=50)
    ap.add_argument('--model', default='all-MiniLM-L6-v2',
                    help='sentence-transformers model for the embeddings')
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)

    # Find all files in the folder (both .pdf and .txt)
    files = (glob.glob(os.path.join(args.guidelines_dir, '*.pdf')) +
             glob.glob(os.path.join(args.guidelines_dir, '*.txt')))
    if not files:
        raise FileNotFoundError(f"No files in {args.guidelines_dir}")

    print(f"Found {len(files)} guideline files")

    # --- Reading and chunking ---
    all_chunks = []
    for path in files:
        name = os.path.basename(path)
        # A short human-readable name for the source
        short = name[:60]
        text = clean_text(read_text_file(path))
        chunks = chunk_text(text, short, args.chunk_size, args.overlap)
        all_chunks.extend(chunks)
        print(f"  {short}: {len(text.split())} words -> {len(chunks)} chunks")

    print(f"\nTotal chunks: {len(all_chunks)}")

    # --- Creating the embeddings ---
    print("Loading the embedding model...")
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(args.model)

    print("Creating embeddings (may take a minute)...")
    texts = [c['text'] for c in all_chunks]
    embeddings = model.encode(texts, batch_size=64, show_progress_bar=True,
                              convert_to_numpy=True, normalize_embeddings=True)

    # --- Saving the index ---
    np.save(os.path.join(args.out, 'embeddings.npy'), embeddings.astype('float32'))
    with open(os.path.join(args.out, 'chunks.json'), 'w', encoding='utf-8') as f:
        json.dump(all_chunks, f, ensure_ascii=False)
    with open(os.path.join(args.out, 'config.json'), 'w', encoding='utf-8') as f:
        json.dump({'model': args.model, 'n_chunks': len(all_chunks)}, f)

    print(f"\nIndex saved to {args.out}/")
    print(f"  embeddings.npy: {embeddings.shape}")
    print(f"  chunks.json: {len(all_chunks)} chunks")


if __name__ == '__main__':
    main()