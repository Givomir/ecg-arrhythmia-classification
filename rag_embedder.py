"""
Sentence embeddings without PyTorch (ONNX)
===========================================
The RAG index is built with sentence-transformers (all-MiniLM-L6-v2, PyTorch).
For the desktop app the same model is exported once to ONNX and run with
onnxruntime + tokenizers - a few MB instead of PyTorch's hundreds - so the
app can embed new documents and any query by itself, with no service and no
installation. The pipeline reproduces sentence-transformers exactly:
tokenize (truncate to max_seq_length) -> transformer -> mean pooling over the
attention mask -> L2 normalization.

Export (needs torch + transformers, done by desktop/build_exe.py):
    python rag_embedder.py --export desktop/onnx_model
"""
import os
import json
import argparse
import numpy as np

MODEL_FILE = 'model.onnx'
TOKENIZER_FILE = 'tokenizer.json'
META_FILE = 'embedder.json'


class OnnxEmbedder:
    """encode(texts) -> L2-normalized float32 embeddings, like SentenceTransformer."""

    def __init__(self, model_dir):
        import onnxruntime as ort
        from tokenizers import Tokenizer
        with open(os.path.join(model_dir, META_FILE), encoding='utf-8') as f:
            meta = json.load(f)
        self.model_name = meta['model_name']
        self.tokenizer = Tokenizer.from_file(os.path.join(model_dir, TOKENIZER_FILE))
        self.tokenizer.enable_truncation(max_length=meta['max_seq_length'])
        self.tokenizer.enable_padding(pad_id=meta.get('pad_id', 0),
                                      pad_token=meta.get('pad_token', '[PAD]'))
        opts = ort.SessionOptions()
        opts.log_severity_level = 3
        self.session = ort.InferenceSession(os.path.join(model_dir, MODEL_FILE), opts,
                                            providers=['CPUExecutionProvider'])
        self.inputs = {i.name for i in self.session.get_inputs()}

    def encode(self, texts, batch_size=32):
        out = []
        for start in range(0, len(texts), batch_size):
            enc = self.tokenizer.encode_batch(list(texts[start:start + batch_size]))
            ids = np.array([e.ids for e in enc], dtype=np.int64)
            mask = np.array([e.attention_mask for e in enc], dtype=np.int64)
            feeds = {'input_ids': ids, 'attention_mask': mask}
            if 'token_type_ids' in self.inputs:
                feeds['token_type_ids'] = np.array([e.type_ids for e in enc], dtype=np.int64)
            hidden = self.session.run(None, feeds)[0]               # (batch, tokens, dim)
            m = mask[..., None].astype(np.float32)
            pooled = (hidden * m).sum(1) / np.clip(m.sum(1), 1e-9, None)   # mean pooling
            out.append(pooled / np.clip(np.linalg.norm(pooled, axis=1, keepdims=True), 1e-12, None))
        if not out:
            return np.zeros((0, 0), dtype=np.float32)
        return np.vstack(out).astype(np.float32)


def export_onnx(model_name, out_dir):
    """Exports a sentence-transformers model to ONNX (transformer part only;
    pooling and normalization are done in numpy by OnnxEmbedder)."""
    import torch
    from sentence_transformers import SentenceTransformer

    st = SentenceTransformer(model_name, device='cpu')
    transformer = st[0].auto_model.eval()
    tokenizer = st.tokenizer
    os.makedirs(out_dir, exist_ok=True)

    class Wrapper(torch.nn.Module):
        def __init__(self, m):
            super().__init__()
            self.m = m

        def forward(self, input_ids, attention_mask, token_type_ids):
            return self.m(input_ids=input_ids, attention_mask=attention_mask,
                          token_type_ids=token_type_ids).last_hidden_state

    sample = tokenizer(["export sample"], return_tensors='pt')
    axes = {0: 'batch', 1: 'tokens'}
    torch.onnx.export(
        Wrapper(transformer),
        (sample['input_ids'], sample['attention_mask'], sample['token_type_ids']),
        os.path.join(out_dir, MODEL_FILE),
        input_names=['input_ids', 'attention_mask', 'token_type_ids'],
        output_names=['last_hidden_state'],
        dynamic_axes={'input_ids': axes, 'attention_mask': axes, 'token_type_ids': axes,
                      'last_hidden_state': axes},
        opset_version=14, dynamo=False)
    tokenizer.backend_tokenizer.save(os.path.join(out_dir, TOKENIZER_FILE))
    with open(os.path.join(out_dir, META_FILE), 'w', encoding='utf-8') as f:
        json.dump({'model_name': model_name, 'max_seq_length': st.max_seq_length,
                   'pad_id': tokenizer.pad_token_id, 'pad_token': tokenizer.pad_token,
                   'dim': st.get_sentence_embedding_dimension()}, f, indent=2)
    return out_dir


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--export', required=True, help='output folder for the ONNX model')
    ap.add_argument('--model', default='all-MiniLM-L6-v2')
    args = ap.parse_args()
    print("Exported to", export_onnx(args.model, args.export))
