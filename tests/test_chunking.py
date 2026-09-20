"""Chunker and entity extraction: determinism and structural fidelity.

Both are deterministic by design, which is what makes reindexing and score
reproduction safe. These tests pin the behaviours the retriever depends on.
"""

from __future__ import annotations

from codemem.add.chunker import (
    CMD,
    CODE,
    CONFIG,
    DIFF,
    LOG,
    PROSE,
    STACKTRACE,
    TEST,
    chunk_content,
)
from codemem.add.entities import extract_entities

SAMPLE = """The suite fails intermittently.

```bash
$ pytest tests/test_tokenizer.py -k unicode
```

```
Traceback (most recent call last):
  File "src/parser/tokenizer.py", line 142, in read_token
    return self.buffer.pop(0)
IndexError: pop from empty list
```

The root cause is a drained buffer when a BOM is present.

```diff
--- a/src/parser/tokenizer.py
+++ b/src/parser/tokenizer.py
@@ -139,6 +139,8 @@ class Tokenizer:
     def read_token(self):
-        return self.buffer.pop(0)
+        if not self.buffer:
+            return None
+        return self.buffer.pop(0)
```
"""


def _kinds(content: str) -> list[str]:
    return [c.kind for c in chunk_content(content)]


def test_chunking_is_deterministic():
    first = [(c.kind, c.text) for c in chunk_content(SAMPLE)]
    second = [(c.kind, c.text) for c in chunk_content(SAMPLE)]
    assert first == second


def test_structural_kinds_are_detected():
    kinds = _kinds(SAMPLE)
    assert CMD in kinds
    assert STACKTRACE in kinds
    assert DIFF in kinds
    assert PROSE in kinds


def test_code_fence_is_never_split_mid_line():
    content = "```python\n" + "\n".join(f"x{i} = {i}" for i in range(50)) + "\n```"
    chunks = chunk_content(content, target_tokens=10, max_tokens=40)
    assert chunks
    for chunk in chunks:
        # Every line must be a complete statement, never a fragment.
        for line in chunk.text.splitlines():
            assert line == "" or line.startswith(("x", "…")), line


def test_diff_is_kept_whole():
    chunks = [c for c in chunk_content(SAMPLE) if c.kind == DIFF]
    assert len(chunks) == 1
    text = chunks[0].text
    assert "--- a/src/parser/tokenizer.py" in text
    assert "@@ -139,6 +139,8 @@" in text
    assert "+        if not self.buffer:" in text


def test_markdown_bullets_are_not_a_diff():
    # "+"/"-" begin a bullet just as they begin a changed line, so a diff is
    # only a diff when it announces itself with a header.
    content = (
        "What I verified:\n\n"
        "- The handler exists\n"
        "- The method is async\n"
        "- The subclass can call it\n"
        "- The test passes\n"
    )
    assert DIFF not in _kinds(content)


def test_unfenced_diff_with_headers_is_still_diff():
    content = (
        "--- a/src/parser/tokenizer.py\n"
        "+++ b/src/parser/tokenizer.py\n"
        "@@ -139,6 +139,8 @@ class Tokenizer:\n"
        "-        return self.buffer.pop(0)\n"
        "+        if not self.buffer:\n"
        "+            return None\n"
    )
    assert DIFF in _kinds(content)


def test_indented_prose_is_not_config():
    # An issue description carries `key: value` lines, but it opens with a
    # sentence and its values are prose. Labelling it config also gave it a
    # `yaml` lang and unlocked the codeish entity extractors.
    content = (
        "Fix this bug to solve the issue based on manual.yaml:\n"
        "  instance_id: django__django-12915\n"
        "  repo: django/django\n"
        "  base_commit: 4652f1f0aa459a7b980441d629648707c32e36bf\n"
        "  problem_statement: Add get_response_async to the static files handler\n"
        "  so that ASGI applications stop falling back to the sync path\n"
    )
    chunks = chunk_content(content)
    assert [c.kind for c in chunks] == [PROSE]
    assert chunks[0].lang is None


def test_fenced_test_output_is_detected_without_a_language_tag():
    content = (
        "```\n"
        "============================= test session starts ==============================\n"
        "platform linux -- Python 3.11.2, pytest-8.3.5\n"
        "tests/test_tokenizer.py::test_unicode_bom PASSED                       [ 50%]\n"
        "tests/test_tokenizer.py::test_empty_buffer FAILED                      [100%]\n"
        "========================= 1 failed, 1 passed in 0.21s =========================\n"
        "```\n"
    )
    assert TEST in _kinds(content)


def test_fenced_test_source_stays_code():
    content = (
        "```python\n"
        "def test_read_token_guards_empty_buffer():\n"
        "    assert tokenizer.read_token() is None\n"
        "\n"
        "def test_read_token_consumes_buffer():\n"
        "    assert tokenizer.read_token() == 'x'\n"
        "```\n"
    )
    kinds = _kinds(content)
    assert CODE in kinds
    assert TEST not in kinds


def test_log_block_detected():
    content = "\n".join(
        f"2026-01-0{i} 10:0{i}:00 INFO  worker started job={i}" for i in range(1, 6)
    )
    assert LOG in _kinds(content)


def test_config_block_detected():
    content = "[server]\nhost = 0.0.0.0\nport = 8080\nworkers = 4\n"
    assert CONFIG in _kinds(content)


def test_prose_paragraphs_are_packed_then_split():
    content = "\n\n".join(f"Paragraph number {i} with several words in it." for i in range(30))
    chunks = chunk_content(content, target_tokens=40, max_tokens=80)
    assert len(chunks) > 1
    assert all(c.kind == PROSE for c in chunks)


def test_oversize_single_line_is_hard_cut():
    content = "x" * 50_000
    chunks = chunk_content(content, hard_chars=1000, max_tokens=10_000)
    assert len(chunks) > 1
    assert all(len(c.text) <= 1000 for c in chunks)


def test_empty_and_whitespace_only_content():
    assert chunk_content("") == []
    assert chunk_content("   \n\n \t ") == []


# ------------------------------------------------------------- entities ----


def _entities(text: str, kind: str = "prose") -> dict[str, set[str]]:
    out: dict[str, set[str]] = {}
    for entity in extract_entities(text, kind=kind, lang=None):
        out.setdefault(entity.etype, set()).add(entity.value_norm)
    return out


def test_extracts_paths_and_names():
    found = _entities("We edited src/parser/tokenizer.py and tests/test_tokenizer.py.")
    assert "src/parser/tokenizer.py" in found["file_path"]
    assert "tokenizer.py" in found["file_name"]


def test_extracts_exception_types_including_camel_case():
    found = _entities("It raised IndexError and asyncio.TimeoutError.")
    assert "IndexError" in found["exception"]
    assert "asyncio.TimeoutError" in found["exception"]


def test_ignores_version_numbers_as_paths():
    found = _entities("Upgraded to python 3.11.2 and numpy 1.26.4 today.")
    assert "3.11.2" not in found.get("file_path", set())
    assert "1.26.4" not in found.get("file_path", set())


def test_ignores_status_code_tokens_as_issue_ids():
    found = _entities("Requests returned UTF-8 bytes and HTTP-404 errors.")
    assert "UTF-8" not in found.get("issue_id", set())
    assert "HTTP-404" not in found.get("issue_id", set())


def test_extracts_real_issue_ids():
    found = _entities("Fixed by PR #4821 and closes GH-1234.")
    assert any(v.endswith("4821") for v in found.get("issue_id", set()))
    assert "GH-1234" in found.get("issue_id", set())


def test_extracts_symbols_from_code():
    found = _entities(
        "def read_token(self):\n    return self.buffer.pop(0)", kind="code"
    )
    assert "read_token" in found["symbol"]


def test_extraction_is_deterministic():
    text = "Fixed src/a/b.py with Tokenizer.read_token raising IndexError."
    first = extract_entities(text, kind="prose")
    second = extract_entities(text, kind="prose")
    assert first == second


def test_entities_have_no_duplicates():
    text = "src/a.py src/a.py src/a.py"
    entities = extract_entities(text, kind="prose")
    keys = [(e.etype, e.value_norm) for e in entities]
    assert len(keys) == len(set(keys))
