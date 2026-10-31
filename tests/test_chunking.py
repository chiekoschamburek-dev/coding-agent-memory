"""Chunker fidelity, entity extraction, and tokenizer behaviour.

The load-bearing property here is **losslessness**: a chunk set must
concatenate back to the exact source string. If that breaks, the "raw text is
the floor" guarantee from plan.md section 2.3 silently stops holding, and every
SpanCitation later points at shifted offsets.
"""

from __future__ import annotations


from codemem.common.chunking import extract_entities, build_chunks
from codemem.common.store import MessageInput, tokenize


MIXED = """Refactoring the loader.

```python
def parseConfigFile(path: str) -> Config:
    return Config.from_path(path)
```

The failure we hit:
Traceback (most recent call last):
  File "src/config/loader.py", line 88, in parseConfigFile
    raise ConfigParseError("bad toml header")
ConfigParseError: bad toml header

We then ran:
$ pytest tests/test_loader.py -q
4 passed, 1 failed in 0.31s

See src/config/loader.py and the CONFIG_PATH env var for resolution order.
"""


def test_chunks_reconstruct_source_exactly():
    records = build_chunks(
        user_id="u", request_id="r", messages=[MessageInput("user", MIXED)]
    )
    rebuilt = "".join(record.text for record in records)
    assert rebuilt == MIXED


def test_chunks_reconstruct_source_exactly_for_multiple_messages():
    messages = [
        MessageInput("user", MIXED),
        MessageInput("assistant", "Fixed by normalising the header.\n\n4 passed in 0.29s\n"),
        MessageInput("user", "没有任何结构的纯文本" * 300),
    ]
    records = build_chunks(user_id="u", request_id="r", messages=messages)
    for index, message in enumerate(messages):
        rebuilt = "".join(
            record.text for record in records if record.message_index == index
        )
        assert rebuilt == message.content


def test_no_chunk_spans_two_messages():
    messages = [MessageInput("user", "aaa\n"), MessageInput("assistant", "bbb\n")]
    records = build_chunks(user_id="u", request_id="r", messages=messages)
    assert len({record.message_id for record in records}) == 2


def test_chunk_offsets_are_accurate():
    records = build_chunks(
        user_id="u", request_id="r", messages=[MessageInput("user", MIXED)]
    )
    for record in records:
        assert MIXED[record.char_start : record.char_end] == record.text


def test_chunk_ids_are_deterministic_and_scope_sensitive():
    def ids(user_id):
        return [
            r.chunk_id
            for r in build_chunks(
                user_id=user_id, request_id="r", messages=[MessageInput("user", MIXED)]
            )
        ]

    assert ids("u") == ids("u")
    assert ids("u") != ids("v")


def test_structural_kinds_are_recognised():
    records = build_chunks(
        user_id="u", request_id="r", messages=[MessageInput("user", MIXED)]
    )
    kinds = {record.kind for record in records}
    assert "code" in kinds
    assert "stack" in kinds
    assert "test_result" in kinds
    assert "prose" in kinds


def test_plain_string_without_structure_is_handled():
    text = "just an ordinary sentence with no markdown, no code, nothing else."
    records = build_chunks(
        user_id="u", request_id="r", messages=[MessageInput("user", text)]
    )
    assert records
    assert "".join(r.text for r in records) == text


def test_very_long_unstructured_text_is_split_and_lossless():
    text = "word " * 20_000
    records = build_chunks(
        user_id="u", request_id="r", messages=[MessageInput("user", text)]
    )
    assert len(records) > 1
    assert "".join(r.text for r in records) == text
    assert all(len(r.text) <= 2100 for r in records)


def test_empty_message_produces_no_chunks():
    assert build_chunks(user_id="u", request_id="r", messages=[MessageInput("user", "")]) == []


def test_unicode_offsets_are_character_based_not_byte_based():
    text = "前缀 emoji 🚀 中间 src/a.py 后缀\n第二行内容\n"
    records = build_chunks(
        user_id="u", request_id="r", messages=[MessageInput("user", text)]
    )
    assert "".join(r.text for r in records) == text
    for record in records:
        assert text[record.char_start : record.char_end] == record.text


# --------------------------------------------------------------------------
# entity extraction
# --------------------------------------------------------------------------


def _entity_values(records):
    return {(entity.kind, entity.value) for entity in extract_entities(records)}


def test_extracts_paths_symbols_errors_tests_and_config_keys():
    records = build_chunks(
        user_id="u", request_id="r", messages=[MessageInput("user", MIXED)]
    )
    found = _entity_values(records)
    kinds = {kind for kind, _ in found}
    assert "path" in kinds
    assert "error" in kinds
    assert "test_name" in kinds
    assert "config_key" in kinds
    values = {value for _, value in found}
    assert any("loader.py" in value for value in values)
    assert any("configparseerror" in value for value in values)
    assert any("config_path" in value for value in values)


def test_entities_are_lowercased_for_cross_case_matching():
    records = build_chunks(
        user_id="u",
        request_id="r",
        messages=[MessageInput("user", "ParseConfigFile in SRC/Config/Loader.PY")],
    )
    for entity in extract_entities(records):
        assert entity.value == entity.value.lower()


def test_prose_does_not_flood_the_index_with_fake_entities():
    records = build_chunks(
        user_id="u",
        request_id="r",
        messages=[
            MessageInput(
                "user",
                "The quick brown fox jumps over the lazy dog and then goes home "
                "because it is late and the weather is cold outside.",
            )
        ],
    )
    found = _entity_values(records)
    assert not any(kind == "path" for kind, _ in found)
    assert not any(kind == "symbol" for kind, _ in found)


def test_entities_are_deduplicated_per_chunk():
    records = build_chunks(
        user_id="u",
        request_id="r",
        messages=[MessageInput("user", "src/a.py src/a.py src/a.py")],
    )
    entities = extract_entities(records)
    keys = [(e.kind, e.value, e.chunk_id) for e in entities]
    assert len(keys) == len(set(keys))


# --------------------------------------------------------------------------
# tokenizer
# --------------------------------------------------------------------------


def test_identifier_splitting():
    tokens = set(tokenize("parseConfigFile"))
    assert "parseconfigfile" in tokens
    assert "config" in tokens or "parse" in tokens


def test_snake_case_splitting():
    tokens = set(tokenize("refresh_token_window"))
    assert "refresh_token_window" in tokens
    assert "refresh" in tokens
    assert "token" in tokens


def test_paths_index_both_full_and_basename():
    tokens = set(tokenize("src/auth/token.py"))
    assert "src/auth/token.py" in tokens
    assert "token.py" in tokens


def test_cjk_produces_bigrams():
    tokens = set(tokenize("令牌过期"))
    assert "令牌" in tokens
    assert "过期" in tokens


def test_tokenizer_is_stable_for_empty_and_symbols():
    assert tokenize("") == []
    assert tokenize("   \n\t ") == []


def test_short_tokens_are_not_dropped_for_identifiers():
    # `id`, `db`, `os` are real symbols in code and must survive.
    tokens = set(tokenize("db_id os.path"))
    assert "db" in tokens
    assert "id" in tokens
    assert "os" in tokens
