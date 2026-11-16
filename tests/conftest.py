import sys
from pathlib import Path

import pytest

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE / "fake_laya"))      # the test double, never the real model
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))

import fake_data  # noqa: E402


@pytest.fixture(scope="session")
def data_root(tmp_path_factory):
    return fake_data.make(tmp_path_factory.mktemp("data"))


@pytest.fixture(scope="session")
def tiny_st(tmp_path_factory):
    """A tiny random sentence-transformer, built offline, standing in for BGE-large in tests."""
    import string
    from transformers import BertConfig, BertModel, BertTokenizerFast
    from sentence_transformers import SentenceTransformer, models
    d = tmp_path_factory.mktemp("tiny_st")
    vocab = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"] + list(string.ascii_lowercase + string.digits + ",:_.") + ["##" + c for c in string.ascii_lowercase + string.digits]
    (d / "vocab.txt").write_text("\n".join(vocab))
    BertTokenizerFast(vocab_file=str(d / "vocab.txt")).save_pretrained(str(d / "hf"))
    BertModel(BertConfig(vocab_size=len(vocab), hidden_size=32, num_hidden_layers=1, num_attention_heads=2, intermediate_size=64)).save_pretrained(str(d / "hf"))
    SentenceTransformer(modules=[models.Transformer(str(d / "hf"), max_seq_length=128), models.Pooling(32, "cls"), models.Normalize()]).save(str(d / "st"))
    return d / "st"
