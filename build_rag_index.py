"""
Изграждане на RAG индекс от клиничните guidelines
====================================================
Чете текстовите guideline файлове, разбива ги на смислени части (chunks),
създава embeddings локално (sentence-transformers) и записва индекс.

Файловете са с разширение .pdf, но реално са чист текст - четем ги директно.

Стартиране (веднъж, при добавяне на нови guidelines):
    python build_rag_index.py --guidelines_dir guidelines --out rag_index

Аргументи:
    --guidelines_dir  папка с guideline файловете (.pdf/.txt - четат се като текст)
    --out             къде да се запише индексът
    --chunk_size      брой думи на част (по подразбиране 250)
    --overlap         припокриване между частите в думи (по подразбиране 50)
"""
import os
import re
import glob
import argparse
import json
import numpy as np


def read_text_file(path):
    """
    Извлича текст от файл. Ако е истински PDF (започва с %PDF), ползва
    PDF парсер. Иначе го чете като чист текст.
    """
    with open(path, 'rb') as f:
        head = f.read(5)

    # Истински PDF -> извличаме текста с парсер
    if head.startswith(b'%PDF'):
        return _extract_pdf_text(path)

    # Иначе - чист текст
    with open(path, 'r', encoding='utf-8', errors='ignore') as f:
        return f.read()


def _extract_pdf_text(path):
    """Извлича текст от PDF. Пробва PyMuPDF, после pypdf."""
    # Опит 1: PyMuPDF (най-устойчив, най-качествено извличане)
    try:
        import pymupdf
        doc = pymupdf.open(path)
        parts = [page.get_text() for page in doc]
        doc.close()
        text = "\n".join(parts)
        if len(text.strip()) > 200:
            return text
    except Exception as e:
        print(f"    (PyMuPDF не успя: {e}; пробвам pypdf)")

    # Опит 2: pypdf
    try:
        from pypdf import PdfReader
        reader = PdfReader(path)
        parts = [(page.extract_text() or '') for page in reader.pages]
        return "\n".join(parts)
    except Exception as e:
        raise RuntimeError(f"Не мога да извлека текст от {path}: {e}")


def clean_text(text):
    """Изчиства текста - маха повтарящи се интервали, нормализира редовете."""
    text = text.replace('\r', ' ')
    # Махаме счупени контролни символи от PDF->текст конверсията
    text = re.sub(r'[\x00-\x08\x0b-\x1f\x7f]', '', text)
    text = re.sub(r'\s+', ' ', text)          # свива празни пространства
    text = re.sub(r'-\s+', '', text)          # слепва разкъсани от нов ред думи
    return text.strip()


def chunk_quality(text):
    """
    Оценява каква част от chunk-а е смислен текст (ASCII букви, цифри,
    обичайна пунктуация). Двоичните остатъци от PDF stream-ове имат ниско
    качество и се изхвърлят.
    """
    if not text:
        return 0.0
    good = sum(1 for ch in text
               if ch.isascii() and (ch.isalnum() or ch in ' .,;:()-%/\n'))
    return good / len(text)


def chunk_text(text, source, chunk_size=250, overlap=50):
    """
    Разбива текста на припокриващи се части от ~chunk_size думи.
    Припокриването пази контекста на границите между частите.
    Части с ниско качество (двоичен боклук от PDF stream) се изхвърлят.
    """
    words = text.split()
    chunks = []
    start = 0
    dropped = 0
    while start < len(words):
        end = min(start + chunk_size, len(words))
        chunk_words = words[start:end]
        chunk = ' '.join(chunk_words)
        if len(chunk.strip()) > 100:          # пропускаме много къси части
            if chunk_quality(chunk) >= 0.85:  # само качествен текст
                chunks.append({'text': chunk, 'source': source})
            else:
                dropped += 1
        if end == len(words):
            break
        start += chunk_size - overlap
    if dropped:
        print(f"    (изхвърлени {dropped} части с двоичен боклук)")
    return chunks


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--guidelines_dir', required=True)
    ap.add_argument('--out', default='rag_index')
    ap.add_argument('--chunk_size', type=int, default=250)
    ap.add_argument('--overlap', type=int, default=50)
    ap.add_argument('--model', default='all-MiniLM-L6-v2',
                    help='sentence-transformers модел за embeddings')
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)

    # Намираме всички файлове в папката (и .pdf, и .txt)
    files = (glob.glob(os.path.join(args.guidelines_dir, '*.pdf')) +
             glob.glob(os.path.join(args.guidelines_dir, '*.txt')))
    if not files:
        raise FileNotFoundError(f"Няма файлове в {args.guidelines_dir}")

    print(f"Намерени {len(files)} guideline файла")

    # --- Четене и разбиване на части ---
    all_chunks = []
    for path in files:
        name = os.path.basename(path)
        # Кратко човешко име за източника
        short = name[:60]
        text = clean_text(read_text_file(path))
        chunks = chunk_text(text, short, args.chunk_size, args.overlap)
        all_chunks.extend(chunks)
        print(f"  {short}: {len(text.split())} думи -> {len(chunks)} части")

    print(f"\nОбщо части: {len(all_chunks)}")

    # --- Създаване на embeddings ---
    print("Зареждане на embedding модел...")
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(args.model)

    print("Създаване на embeddings (може да отнеме минута)...")
    texts = [c['text'] for c in all_chunks]
    embeddings = model.encode(texts, batch_size=64, show_progress_bar=True,
                              convert_to_numpy=True, normalize_embeddings=True)

    # --- Запис на индекса ---
    np.save(os.path.join(args.out, 'embeddings.npy'), embeddings.astype('float32'))
    with open(os.path.join(args.out, 'chunks.json'), 'w', encoding='utf-8') as f:
        json.dump(all_chunks, f, ensure_ascii=False)
    with open(os.path.join(args.out, 'config.json'), 'w', encoding='utf-8') as f:
        json.dump({'model': args.model, 'n_chunks': len(all_chunks)}, f)

    print(f"\nИндексът е записан в {args.out}/")
    print(f"  embeddings.npy: {embeddings.shape}")
    print(f"  chunks.json: {len(all_chunks)} части")


if __name__ == '__main__':
    main()