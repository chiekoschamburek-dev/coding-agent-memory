'''Session-scoped experience cards and versioned representations.

Legacy repair cards use problem/locate/implement plus deterministic packets.
Opt-in v5 cards describe individual state/action/mechanism/result attempts;
their own field evidence, never a session-final patch/test run, bounds facts.
Both formats share persistence and retrieval interfaces without conversion.
'''

from __future__ import annotations

import re
from dataclasses import dataclass, replace
from typing import Sequence

from .attempt import (
    PAYLOAD_VERSION as ATTEMPT_PAYLOAD_VERSION,
)
from .attempt import (
    SYSTEM_PROMPT as ATTEMPT_SYSTEM_PROMPT,
)
from .attempt import (
    FieldEvidence,
    admission_problems,
    bounded_text,
    extraction_prompt,
    is_v5,
    prepare_attempt,
    read_evidence,
    render_attempt,
    result_from_evidence,
)
from ..common.casebook import (
    CASE_PAYLOAD_VERSION,
    CONDITION_TYPES,
    MAX_FIELD_CHARS,
    Condition,
    VerifyRecipe,
    build_packet,
    render_packet,
)
from ..common.store import ChunkRecord, StoredChunk, tokenize

EXTRACTION_VERSION_DEFAULT = 'card-v4'

# The narrative fields injected to the answer model, plus the understanding
# face added in card-v2.  ``problem`` is the retrieval face (issue vocabulary,
# symptoms, trigger conditions) and the only narrative field that feeds the
# lexical index; ``mechanism`` is the post-hoc understanding face -- rendered
# to the model, never indexed.  conditions/verify/files/symbols/patch stay
# storage/audit-only in v2 and do not enter the answer render.
_CARD_FIELDS = ('problem', 'locate', 'implement')
_VALID_KINDS = frozenset({'debug', 'feature'})
_KIND_ALIASES = {
    'bugfix': 'debug', 'bug': 'debug', 'fix': 'debug', 'defect': 'debug',
    'feature': 'feature', 'enhancement': 'feature', 'dev': 'feature',
    'development': 'feature',
}
_FEATURE_CUES = (
    'support ', 'add ', 'implement', 'introduce', 'extend', 'allow ',
    'new feature', 'feature request', 'enhancement', '支持', '新增', '实现',
    '扩展', '增强',
)
_DEBUG_CUES = (
    'error', 'exception', 'traceback', 'bug', 'fix', 'broken', 'crash',
    'regression', 'fails', 'failing', 'wrong', '报错', '异常', '修复', '失败',
)


def normalize_kind(value: object) -> str | None:
    if not isinstance(value, str):
        return None
    low = value.strip().lower()
    if low in _VALID_KINDS:
        return low
    return _KIND_ALIASES.get(low)


def resolve_kind(value: object, *texts: object) -> str:
    """Reconcile the model's ``kind`` with a deterministic signal.

    gpt-4o-mini routinely labels a feature request as ``debug`` because every
    example repair it has seen is a bug fix.  When the model says ``debug`` but
    the text carries feature cues and no debug cues, the deterministic signal
    wins; an explicit ``feature`` is always kept.
    """
    kind = normalize_kind(value)
    inferred = infer_kind(*texts)
    if kind == "debug" and inferred == "feature":
        return "feature"
    return kind or inferred


def infer_kind(*texts: object) -> str:
    """Feature/debug fallback when the extractor omits ``kind``.

    Debug cues win when both appear: a feature request that mentions an error
    in passing is still usually a defect fix.  The model's explicit ``kind``
    always takes precedence over this.
    """
    text = ' '.join(str(t) for t in texts if t).lower()
    if any(cue in text for cue in _DEBUG_CUES):
        return 'debug'
    if any(cue in text for cue in _FEATURE_CUES):
        return 'feature'
    return 'debug'


@dataclass(frozen=True)
class ExperienceCard:
    '''Versioned card: legacy repair packet or evidence-bound v5 attempt.'''

    experience_id: str
    user_id: str
    session_id: str
    kind: str = 'debug'
    extraction_version: str = EXTRACTION_VERSION_DEFAULT
    created_at: str = ''
    problem: str | None = None
    mechanism: str | None = None
    locate: str | None = None
    implement: str | None = None
    # card-v4 implement spec: the acceptance-tested requirement shape of this
    # class of fix, not the historical change narrative (near-transfer cards
    # whose patch already ships read as a no-op in narrative form).
    implement_shape: str | None = None
    implement_boundaries: str | None = None
    implement_acceptance: str | None = None
    source_chunk_ids: tuple[str, ...] = ()
    # -- case-packet structure ----------------------------------------------
    # ``conditions`` are facts, never an applicability verdict;
    # ``patch``/``verify`` are the artifacts the current repository cannot
    # reproduce, derived from the cited evidence rather than model output.
    conditions: tuple[Condition, ...] = ()
    files: tuple[str, ...] = ()
    symbols: tuple[str, ...] = ()
    patch: str | None = None
    verify: VerifyRecipe | None = None
    pitfalls: tuple[str, ...] = ()

    # v5 semantic content; legacy fields are intentionally NOT synthesized.
    state: str | None = None
    action: str | None = None
    result: str | None = None
    evidence: tuple[FieldEvidence, ...] = ()

    @property
    def is_attempt(self) -> bool:
        return is_v5(self.extraction_version)

    @property
    def stable_id(self) -> str:
        return f'experience:{self.experience_id}'

    @property
    def has_narrative_fields(self) -> bool:
        if self.is_attempt:
            return bool(self.state and self.action and self.result)
        return any(isinstance(getattr(self, name), str) and getattr(self, name).strip()
                   for name in _CARD_FIELDS)

    @property
    def has_packet_fields(self) -> bool:
        return bool(
            self.files
            or self.patch
            or self.conditions
            or self.symbols
            or self.pitfalls
            or (self.verify is not None and not self.verify.is_empty())
        )


def _clean_text(value: object) -> str | None:
    if not isinstance(value, str):
        return None
    value = value.strip()
    return value[:MAX_FIELD_CHARS] or None


def _clip_text(value: object) -> str | None:
    if not isinstance(value, str):
        return None
    return value[:MAX_FIELD_CHARS] or None


def card_to_payload(card: ExperienceCard) -> dict:
    if card.is_attempt:
        return {
            'payload_version': ATTEMPT_PAYLOAD_VERSION,
            'experience_id': card.experience_id, 'user_id': card.user_id,
            'session_id': card.session_id, 'kind': card.kind,
            'extraction_version': card.extraction_version, 'created_at': card.created_at,
            'state': bounded_text(card.state), 'action': bounded_text(card.action),
            'mechanism': bounded_text(card.mechanism), 'result': bounded_text(card.result),
            'source_chunk_ids': list(card.source_chunk_ids),
            'evidence': [ref.to_dict() for ref in card.evidence],
        }
    payload = {
        'experience_id': card.experience_id,
        'user_id': card.user_id,
        'session_id': card.session_id,
        'extraction_version': card.extraction_version,
        'created_at': card.created_at,
        'problem': _clip_text(card.problem),
        'mechanism': _clip_text(card.mechanism),
        'locate': _clip_text(card.locate),
        'implement': _clip_text(card.implement),
        'implement_shape': _clip_text(card.implement_shape),
        'implement_boundaries': _clip_text(card.implement_boundaries),
        'implement_acceptance': _clip_text(card.implement_acceptance),
        'source_chunk_ids': list(card.source_chunk_ids),
    }
    if card.kind:
        payload['kind'] = card.kind
    if card.has_packet_fields:
        payload['payload_version'] = CASE_PAYLOAD_VERSION
        payload['conditions'] = [c.to_dict() for c in card.conditions]
        payload['files'] = list(card.files)
        payload['symbols'] = list(card.symbols)
        payload['patch'] = card.patch or ''
        payload['verify'] = card.verify.to_dict() if card.verify else {}
        payload['pitfalls'] = list(card.pitfalls)
    return payload


def _string_list(raw: object, limit: int) -> tuple[str, ...]:
    if not isinstance(raw, list):
        return ()
    out: list[str] = []
    for value in raw:
        if isinstance(value, str) and value.strip() and value not in out:
            out.append(value.strip())
        if len(out) >= limit:
            break
    return tuple(out)


def card_from_payload(raw: dict) -> ExperienceCard:
    source_ids = raw.get('source_chunk_ids') or []
    if not isinstance(source_ids, list):
        source_ids = []
    source_ids = tuple(value for value in source_ids if isinstance(value, str) and value)
    if is_v5(raw.get('extraction_version')) or raw.get('payload_version') == ATTEMPT_PAYLOAD_VERSION:
        evidence = read_evidence(raw.get('evidence'))
        result = result_from_evidence(evidence)
        # Persisted v5 cards must retain the same evidence/result contract;
        # never render an edited success summary over failed observations.
        if (result != bounded_text(raw.get('result'))
                or not all(any(ref.field == name for ref in evidence)
                           for name in ('state', 'action', 'result'))
                or any(ref.chunk_id not in source_ids for ref in evidence)):
            result = None
        return ExperienceCard(
            experience_id=str(raw.get('experience_id', '')),
            user_id=str(raw.get('user_id', '')), session_id=str(raw.get('session_id', '')),
            kind=normalize_kind(raw.get('kind')) or 'debug', extraction_version='card-v5',
            created_at=str(raw.get('created_at', '')),
            state=bounded_text(raw.get('state')), action=bounded_text(raw.get('action')),
            mechanism=bounded_text(raw.get('mechanism')) if any(ref.field == 'mechanism' for ref in evidence) else None,
            result=result, source_chunk_ids=source_ids, evidence=evidence,
        )
    fields = {name: _clean_text(raw.get(name)) for name in _CARD_FIELDS}
    mechanism = _clean_text(raw.get('mechanism'))
    spec = {name: _clean_text(raw.get(name))
            for name in ('implement_shape', 'implement_boundaries',
                         'implement_acceptance')}
    conditions_raw = raw.get('conditions')
    conditions = tuple(
        condition
        for condition in (
            Condition.from_dict(value) for value in conditions_raw or []
        )
        # same closed enum as the write path (:415): a rogue type from an
        # old/hand-edited payload would render into the injected 条件 line
        if condition is not None and condition.type in CONDITION_TYPES
    )
    return ExperienceCard(
        experience_id=str(raw.get('experience_id', '')),
        user_id=str(raw.get('user_id', '')),
        session_id=str(raw.get('session_id', '')),
        kind=normalize_kind(raw.get('kind')) or 'debug',
        extraction_version=str(raw.get('extraction_version', EXTRACTION_VERSION_DEFAULT)),
        created_at=str(raw.get('created_at', '')),
        problem=fields['problem'],
        mechanism=mechanism,
        locate=fields['locate'],
        implement=fields['implement'],
        implement_shape=spec['implement_shape'],
        implement_boundaries=spec['implement_boundaries'],
        implement_acceptance=spec['implement_acceptance'],
        source_chunk_ids=source_ids,
        conditions=conditions,
        files=_string_list(raw.get('files'), 8),
        symbols=_string_list(raw.get('symbols'), 12),
        patch=raw.get('patch') if isinstance(raw.get('patch'), str) else None,
        verify=VerifyRecipe.from_dict(raw.get('verify')),
        pitfalls=_string_list(raw.get('pitfalls'), 3),
    )


_SYSTEM_PROMPT = '''你是工程历史修复案例抽取器。只整理输入原文，不生成当前任务方案，不判断是否适用于别的任务。
所有字段内容一律用英文书写；只有逐字引用的报错串、符号名、路径、命令保持原文。
每张 card 输出：kind、problem、mechanism、locate、implement、source_chunk_ids，
以及可选的 files、conditions、pitfalls。
kind 只能是 "debug"（缺陷/报错/回归修复）或 "feature"（新能力/接口/功能增强）。
每个事实只放一处，不要跨字段复制：
problem 是症状面（检索面）：只站在“还没动工、只看过 issue”的视角，用 issue 自己的词汇写
症状和触发条件（什么场景/输入/操作下出现什么现象）。
不写根因、不写修复思路；版本、报错串、配置不写在 problem，放进 conditions
（type=version|module|symbol|error|config），报错串逐字保留，不得泛化。
mechanism 是理解面（做完之后才可能知道的）：根因是什么、违反/建立了什么不变量或约定、
以及当时如何确认了这个诊断（一句话）。可以点名关键符号或函数，但解释机制，不罗列步骤。
locate 是定位要点：一到两条模式级定位要点，每条落到一个具体符号或路径锚点
（在哪个模块/函数/分支、为什么在那里）；不要罗列路径或符号清单——精确路径只进 files。
implement 改为三段子断言（implement_shape/boundaries/acceptance），回答"这类修复被验收过的
需求形状"而不是"当时怎么改的"（历史补丁若已随版本发布，叙事读起来是空操作）：
implement_shape（形态）: 变更形态与决定性锚点——守卫/改动放在哪个分派点之前、签名是否变化。
  只能依据下方"证据预消化"里的 diff 分支结构。
implement_boundaries（边界）: 必须区分的输入空间分片——哪些 case 走哪条分支；
  依据 diff 分支结构与测试的参数化用例；负断言（不要碰什么）须能在证据中找到对应。
implement_acceptance（验收）: 测试断言的行为语义——期望返回什么、禁止什么、抛什么；
  必须逐字引用决定性断言行（像 conditions 的报错串一样保留原文）。
  历史 session 没有测试证据（没跑过测试/没有断言原文）时诚实留空——宁缺毋滥。
每条断言必须能 grounding 到引用的 chunk；不要写 diff 全文，失败尝试只进 pitfalls。
files 只填 diff/命令里出现的精确路径；conditions 是事实条件，格式 [{"type":"version|module|symbol|error|config","value":"...","basis":"原文引用"}]；
pitfalls 只填真实失败尝试（必须能在原文找到对应内容，含具体符号/路径/命令），最多 3 条。没有依据时留空。
conditions 不是“是否可迁移”的结论；不要输出 success、failure、verified 等状态。

不为以下内容出卡——它们不是工程问题，没有可迁移的修法：
- 解释器/包管理器/虚拟环境自身的报错（路径含 site-packages、dist-packages、anaconda、miniconda、venv、.pth，或调用帧为 <string>）；
- 依赖安装、镜像源、网络失败；IDE/编辑器提示；纯环境配置差异。

出卡门槛：problem 必须非空，且 implement_shape 必须非空（acceptance/boundaries 可空）。
只有现象、没有实际改动的内容，不要输出。
宁缺毋滥：一个 session 最多 3 张 card，可以只出 1 张或不出；同质/低价值内容不要强行凑卡。
多个报错若指向同一根因或同一处改动，只出一张 card，其余作为该卡的 pitfalls；
根因不同、涉及不同文件或不同阶段目标的问题，必须分别出卡，不要为了少出卡而合并。

只输出 JSON：{'cards': [{'kind':'debug|feature','problem':'','mechanism':'','locate':'','implement':'','implement_shape':'','implement_boundaries':'','implement_acceptance':'','files':[],'conditions':[],'pitfalls':[],'source_chunk_ids':[]}]}'''

_DIFF_HUNK_RE = re.compile(r"^@@.*$|^[-+][^-+].*$", re.M)
_TEST_LINE_RE = re.compile(
    r"^.*(?:assert\b|def test_|raises\b|pytest\.raises|assert_equal|assert_allclose"
    r"|assert_raises|FAILED|PASSED).*$", re.I | re.M)


def _implement_evidence_digest(window: Sequence[StoredChunk],
                               max_chars: int = 2600) -> str:
    """Mechanical pre-digest of the extraction window for the implement spec.

    The three-part implement (shape/boundaries/acceptance) is grounded in two
    evidence streams the raw batch hides in prose noise: the diff's branch
    structure (@@ context + decisive +/- lines) and the test assertion lines.
    Extracted deterministically so the model condenses evidence instead of
    hunting for it -- the same cure as the pitfalls yield problem.
    """
    diff_parts: list[str] = []
    test_parts: list[str] = []
    used_diff = used_test = 0
    for chunk in window:
        text = chunk.text
        if "diff --git" in text and used_diff < 1300:
            hunks = [m.group(0) for m in _DIFF_HUNK_RE.finditer(text)][:12]
            if hunks:
                block = " | ".join(h.strip()[:120] for h in hunks)
                diff_parts.append(block)
                used_diff += len(block)
        elif _TEST_LINE_RE.search(text) and used_test < 1200:
            lines = [ln.strip()[:120] for ln in text.splitlines()
                     if _TEST_LINE_RE.match(ln)][:8]
            if lines:
                block = " | ".join(lines)
                test_parts.append(block)
                used_test += len(block)
    head = []
    if diff_parts:
        head.append("diff 分支结构（@@ 上下文与决定性增删行）:\n" + "\n".join(diff_parts))
    if test_parts:
        head.append("测试断言行（逐字）:\n" + "\n".join(test_parts))
    if not head:
        return ""
    return ("\n".join(head))[:max_chars]


def extraction_system_prompt(version: str) -> str:
    return ATTEMPT_SYSTEM_PROMPT if is_v5(version) else _SYSTEM_PROMPT


def build_extraction_prompt(
    window: Sequence[StoredChunk],
    quote_max_chars: int,
    multi_case: bool = False,
    max_cards: int = 3,
    extraction_version: str = EXTRACTION_VERSION_DEFAULT,
) -> str:
    if is_v5(extraction_version):
        return extraction_prompt(window, max_cards)
    parts = [
        '修复案例 schema v2（字段可为空，没有依据就留空；所有字段内容用英文，逐字引用除外；每个事实只放一处）：',
        'kind: "debug"（缺陷/报错/回归）或 "feature"（新能力/接口/增强）',
        'problem（症状面/检索面）: 只用 issue 的词汇写症状与触发条件——什么场景/输入/操作下出现什么现象；不写根因，报错串/版本/配置不在这里，进 conditions',
        'mechanism（理解面）: 做完后对问题的机制理解——根因、违反/建立的不变量，以及当时如何确认了这个诊断（一句话）；可点名关键符号',
        'locate（定位要点）: 一到两条模式级定位要点，每条落到具体符号或路径锚点（哪个模块/函数/分支、为什么）；不要罗列路径清单',
        'implement_shape（形态）: 变更形态与决定性锚点（守卫放在哪个分派点、签名是否变化）——只依据证据预消化里的 diff 分支结构',
        'implement_boundaries（边界）: 输入空间分片——哪些 case 走哪条分支；依据 diff 分支 + 测试参数化用例；负断言须有证据对应',
        'implement_acceptance（验收）: 测试断言的行为语义——逐字引用决定性断言行；无测试证据则留空（宁缺毋滥）',
        'files: diff 或命令中出现的精确路径（如 src/a.py）——路径只放这里（存储/审计用，不进模型上下文）',
        'conditions: 事实条件数组 [{"type":"version|module|symbol|error|config","value":"...","basis":"原文"}]；报错串逐字保留为 type=error（存储/审计用）',
        'pitfalls: 真实失败尝试，最多 3 条（没有就留空；内容必须能在上述 chunk 中找到依据）',
        'source_chunk_ids: 使用到的输入 chunk_id 列表',
        '',
    ]
    if multi_case:
        parts.extend([
            '这个窗口是一个完整 session，可能包含多个不同问题/阶段。',
            f'每个独立问题输出一张 card，最多 {max_cards} 张，可以只出 1 张或不出。',
            '同一根因或同一处改动只出一张，其余报错作为该卡的 pitfalls；',
            '但根因不同、涉及不同文件或不同阶段目标时，必须分别出卡，不要为了少出卡而合并。',
            '每张 card 的 problem 必须非空，且 implement（实际改法）必须非空；',
            '只有现象、没有实际改动的问题不要输出。',
            '环境/依赖/虚拟环境自身的问题（site-packages、anaconda、venv、.pth、<string> 帧）不要出卡。',
            '每张 card 的 source_chunk_ids 必须指向该问题实际使用的 chunk；证据不足的问题不要输出。',
            '',
        ])
    digest = _implement_evidence_digest(window)
    if digest:
        parts.append('证据预消化（机械提取，仅供 implement 三段断言引用，不得虚构其外内容）：')
        parts.append(digest)
        parts.append('')
    parts.append('历史片段（按来源顺序）：')
    for chunk in window:
        parts.append(f'<<<CHUNK {chunk.chunk_id} | session {chunk.session_id} | role {chunk.role}>>>')
        parts.append(chunk.text)
    parts.append(f'\n只输出 JSON。kind 只能是 debug 或 feature；字段没有依据时留空；files/conditions 必须来自上述 chunk；source_chunk_ids 必须来自上述 chunk。输入窗口最多 {quote_max_chars} 字符的来源片段。')
    return '\n'.join(parts)


_NARRATIVE_PATH_RE = re.compile(
    r'[A-Za-z0-9_./\\~-]+\.(?:py|pyi|js|jsx|mjs|cjs|ts|tsx|go|rs|java|kt|rb|'
    r'php|c|h|cc|cpp|hpp|cs|swift|scala|sh|bash|zsh|ps1|sql|json|ya?ml|toml|'
    r'ini|cfg|conf|gradle|mod|lock)\b'
)
_NARRATIVE_SYMBOL_RE = re.compile(
    # each dotted segment is at least two chars: single-letter segments are
    # English abbreviations (e.g. / i.e.), not qualified symbols
    r'[A-Za-z_][A-Za-z0-9_]{1,}(?:\.[A-Za-z_][A-Za-z0-9_]{1,})+'
    r'|[A-Z][A-Za-z0-9]*(?:Error|Exception|Warning|Fault)'
)
# v5 attempts only: any identifier carrying an underscore names a code fact
# specifically (snake_case and _prefixed alike), so it must ground too.
# Legacy packet cards keep the narrower heuristic (compatibility of existing
# semantics).
_ATTEMPT_SYMBOL_RE = re.compile(r'\b(?=\w*_)\w+\b')


def _strip_ungrounded_anchors(text: str | None, source_text: str) -> str | None:
    """Narrative is not a grounding backdoor: a path or qualified symbol the
    model writes into problem/locate must occur in the cited chunks, or it is
    stripped before the text reaches the index.  Plain prose words never match
    these patterns, so paraphrase is unaffected."""
    if not text:
        return text

    def drop(match: re.Match) -> str:
        token = match.group(0)
        low = token.lower()
        if low in source_text or low.replace('\\', '/') in source_text:
            return token
        return ''

    out = _NARRATIVE_PATH_RE.sub(drop, text)
    out = _NARRATIVE_SYMBOL_RE.sub(drop, out)
    out = re.sub(r'[ \t]{2,}', ' ', out)
    out = re.sub(r' +([,.;:!?)])', r'\1', out)
    out = re.sub(r'([(]) +', r'\1', out)
    return out.strip() or None


def _path_grounded(path: str, source_text: str) -> bool:
    normalized = str(path).strip().replace('\\', '/').lower()
    if not normalized:
        return False
    if "/" in normalized:
        # A declared directory path must appear in full; a shared basename
        # (`config.py` under src/) must not ground a different vendor path.
        return normalized in source_text
    return normalized in source_text


_PITFALL_TOKEN_RE = re.compile(r'[A-Za-z_][A-Za-z0-9_.]{3,}')
_PITFALL_GENERIC_TOKENS = frozenset({
    'error', 'errors', 'test', 'tests', 'failed', 'failing', 'failure',
    'assert', 'patch', 'file', 'files', 'function', 'code', 'change',
    'changes', 'tried', 'trying', 'instead', 'still', 'with', 'that',
    'this', 'from', 'when', 'then', 'only', 'also', 'first',
})


def _pitfall_grounded(pitfall: str, source_text: str) -> bool:
    """A pitfall is a claim about a real failed attempt; like files and
    conditions it must leave a trace in the cited evidence.  A pure-prose
    pitfall with no verifiable token is exactly the hallucination-prone
    case, so it is dropped rather than trusted."""
    low = ' '.join(pitfall.lower().split())
    if low in source_text:
        return True
    tokens = {
        match.group(0).lower()
        for match in _PITFALL_TOKEN_RE.finditer(pitfall)
    } - _PITFALL_GENERIC_TOKENS
    return any(token in source_text for token in tokens)


def _unsupported_attempt_fields(attempt: dict) -> list[str]:
    """Narrative fields naming paths/symbols absent from their own evidence."""
    unsupported: list[str] = []
    for name in ('state', 'action', 'mechanism'):
        value = attempt[name]
        source = '\n'.join(ref.quote for ref in attempt['evidence']
                           if ref.field == name).lower()
        normalized_source = source.replace('\\', '/')
        if any(
            match.group(0).lower().replace('\\', '/') not in normalized_source
            for pattern in (_NARRATIVE_PATH_RE, _NARRATIVE_SYMBOL_RE, _ATTEMPT_SYMBOL_RE)
            for match in pattern.finditer(value or '')
        ):
            unsupported.append(name)
    return unsupported


def card_admission_problems(
    proposed: list,
    *,
    chunks_by_id: dict,
    user_id: str,
    session_id: str,
) -> list[str]:
    """Labeled rejection reasons across the proposed cards, for repair feedback."""
    problems: list[str] = []
    for i, raw_card in enumerate(proposed[:4], 1):
        if not isinstance(raw_card, dict):
            problems.append(f"card{i}: 不是 JSON 对象")
            continue
        errs = admission_problems(raw_card, chunks_by_id,
                                  user_id=user_id, session_id=session_id)
        if errs:
            problems.append(f"card{i}: " + "；".join(errs[:3]))
            continue
        attempt = prepare_attempt(raw_card, chunks_by_id,
                                  user_id=user_id, session_id=session_id)
        unsupported = _unsupported_attempt_fields(attempt) if attempt else []
        if any(name != 'mechanism' for name in unsupported):
            problems.append(
                f"card{i}: 叙述中的路径/符号没有出现在该字段引用的原文里，"
                "改写叙述或改引包含该符号的行")
    if not problems:
        problems.append("上一轮的 card 没有通过程序化准入")
    return problems


def _new_card_from_raw(
    raw_card: dict,
    *,
    chunks_by_id: dict[str, StoredChunk],
    user_id: str,
    session_id: str,
    experience_id: str,
    extraction_version: str,
    created_at: str,
    grounding_chunks_by_id: dict[str, StoredChunk] | None = None,
) -> ExperienceCard | None:
    if is_v5(extraction_version):
        attempt = prepare_attempt(raw_card, chunks_by_id, user_id=user_id, session_id=session_id)
        if attempt is None:
            return None
        # Per-field anchors must occur in that field's cited evidence. Never
        # repair an unsupported claim by deleting a path from its sentence.
        unsupported = _unsupported_attempt_fields(attempt)
        if any(name != 'mechanism' for name in unsupported):
            return None
        if 'mechanism' in unsupported:
            attempt['mechanism'] = None
            attempt['evidence'] = tuple(ref for ref in attempt['evidence']
                                        if ref.field != 'mechanism')
        attempt['source_chunk_ids'] = tuple(cid for cid in attempt['source_chunk_ids']
                                            if any(ref.chunk_id == cid for ref in attempt['evidence']))
        return ExperienceCard(
            experience_id=experience_id, user_id=user_id, session_id=session_id,
            kind=resolve_kind(raw_card.get('kind'), attempt['state'], attempt['action']),
            extraction_version=extraction_version, created_at=created_at, **attempt,
        )
    fields = {name: _clean_text(raw_card.get(name)) for name in _CARD_FIELDS}
    spec = {name: _clean_text(raw_card.get(name))
            for name in ('implement_shape', 'implement_boundaries',
                         'implement_acceptance')}
    # format-drift salvage: the model sometimes writes the labeled spec into
    # the narrative implement instead of the separate keys (observed ~1 card
    # per library build); split it back out deterministically
    if fields['implement'] and not any(spec.values()) and \
            re.search(r'implement_shape\s*:', fields['implement'], re.I):
        original = fields['implement']
        parts = re.split(
            r'implement_(shape|boundaries|acceptance)\s*:\s*',
            original, flags=re.I)
        for i in range(1, len(parts) - 1, 2):
            key = 'implement_' + parts[i].lower()
            value = parts[i + 1].strip()
            if key in spec and value and not spec[key]:
                spec[key] = value
        if any(spec.values()):
            fields['implement'] = None
    raw_ids = raw_card.get('source_chunk_ids')
    source_ids = tuple(value for value in raw_ids or [] if isinstance(value, str) and value in chunks_by_id)
    # ``problem`` is what a query is matched against; a card without it has no
    # index face on the problem field and is effectively unfindable.  A card
    # without any implement content (narrative, or the card-v4 shape spec)
    # is a phenomenon report, not a repair case.
    if not fields['problem'] or not source_ids:
        return None
    if not fields['implement'] and not spec['implement_shape']:
        return None
    # Model-declared facts must be grounded in the chunks the card itself
    # cites, not in an unrelated part of the batch/session.  The narrative
    # anchors are the exception: the model synthesizes them across the whole
    # timeline, so they are checked against the full session's kept chunks
    # (``grounding_chunks_by_id``) rather than only the batch it saw.
    cited_text = '\n'.join(
        chunks_by_id[chunk_id].text
        for chunk_id in source_ids
        if chunk_id in chunks_by_id
    ).lower()
    source_text = cited_text
    if grounding_chunks_by_id:
        source_text = '\n'.join(
            chunk.text for chunk in grounding_chunks_by_id.values()
        ).lower()
    fields['problem'] = _strip_ungrounded_anchors(fields['problem'], source_text)
    fields['locate'] = _strip_ungrounded_anchors(fields['locate'], source_text)
    fields['implement'] = _strip_ungrounded_anchors(fields['implement'], source_text)
    for name in ('implement_shape', 'implement_boundaries', 'implement_acceptance'):
        spec[name] = _strip_ungrounded_anchors(spec[name], source_text)
    # The admission gate above ran on the pre-strip text; stripping can empty
    # a face the gate already passed.  A card whose problem or implement did
    # not survive grounding is a phenomenon report -- the same reject as
    # above, applied after the clamp.
    if not fields['problem']:
        return None
    if not fields['implement'] and not spec['implement_shape']:
        return None
    # The mechanism face may name symbols; anything the session evidence does
    # not contain is clamped away, exactly like the other narrative fields.
    mechanism = _strip_ungrounded_anchors(_clean_text(raw_card.get('mechanism')),
                                          source_text)
    conditions_raw = raw_card.get('conditions')
    conditions_list = []
    for value in conditions_raw or []:
        condition = Condition.from_dict(value)
        if condition is None:
            continue
        # Types outside the closed enum can never be matched by a query
        # channel; they would only occupy a ``case_anchors`` row.
        if condition.type not in CONDITION_TYPES:
            continue
        # A factual condition must at least occur in the supplied evidence.
        if condition.value.lower() not in cited_text:
            continue
        basis = condition.basis.strip()
        if basis and basis.lower() not in source_text and basis not in {
            'diff', 'chunk', 'problem'
        }:
            condition = Condition(condition.type, condition.value, '')
        if condition.type == 'symbol' and re.fullmatch(
            r'[A-Z][A-Za-z0-9]*(?:Error|Exception|Warning|Fault)', condition.value
        ):
            condition = Condition('error', condition.value, condition.basis)
        conditions_list.append(condition)
    conditions = tuple(conditions_list)[:8]
    pitfalls = tuple(
        pitfall for pitfall in _string_list(raw_card.get('pitfalls'), 3)
        if _pitfall_grounded(pitfall, cited_text)
    )
    files = tuple(
        path for path in _string_list(raw_card.get('files'), 8)
        if _path_grounded(path, cited_text)
    )
    symbols = tuple(
        symbol for symbol in _string_list(raw_card.get('symbols'), 12)
        if symbol.lower() in cited_text
    )
    # The extraction prompt does not ask for a patch or verification recipe.
    # Both are derived deterministically from the cited evidence; accepting a
    # model-written diff/verify here would let it bypass path grounding.
    patch = None
    verify = None
    kind = resolve_kind(raw_card.get('kind'), fields['problem'], fields['implement'])
    return ExperienceCard(
        experience_id=experience_id,
        user_id=user_id,
        session_id=session_id,
        kind=kind,
        problem=fields['problem'],
        mechanism=mechanism,
        locate=fields['locate'],
        implement=fields['implement'],
        implement_shape=spec['implement_shape'],
        implement_boundaries=spec['implement_boundaries'],
        implement_acceptance=spec['implement_acceptance'],
        source_chunk_ids=source_ids,
        extraction_version=extraction_version,
        created_at=created_at,
        conditions=conditions,
        files=files,
        symbols=symbols,
        patch=patch,
        verify=verify,
        pitfalls=pitfalls,
    )


def enrich_card(card: ExperienceCard, chunks: Sequence[StoredChunk]) -> ExperienceCard:
    """Fill the deterministic packet fields from the extraction window.

    The model writes problem/locate/implement/pitfalls/conditions; paths,
    diff, symbols and the verification recipe are extracted mechanically so
    the packet stays self-contained even when the model omits them.
    """
    if card.is_attempt:
        # A session-wide patch/result could belong to a later successful
        # revision, not this (possibly failed or reverted) attempt.
        return card
    packet = build_packet(card, chunks)
    return replace(
        card,
        conditions=packet.conditions or card.conditions,
        files=packet.files or card.files,
        symbols=packet.symbols or card.symbols,
        patch=packet.patch or card.patch,
        verify=packet.verify if not packet.verify.is_empty() else card.verify,
        pitfalls=packet.pitfalls or card.pitfalls,
    )


def build_cards(
    *,
    raw: object,
    chunks_by_id: dict[str, StoredChunk],
    user_id: str,
    session_id: str,
    id_for,
    extraction_version: str,
    created_at: str,
    max_cards: int,
    grounding_chunks_by_id: dict[str, StoredChunk] | None = None,
) -> list[ExperienceCard]:
    if not isinstance(raw, dict) or not isinstance(raw.get('cards'), list):
        return []
    cards: list[ExperienceCard] = []
    for local_index, raw_card in enumerate(raw['cards']):
        if len(cards) >= max_cards or not isinstance(raw_card, dict):
            continue
        card = _new_card_from_raw(
            raw_card,
            chunks_by_id=chunks_by_id,
            user_id=user_id,
            session_id=session_id,
            experience_id=id_for(local_index),
            extraction_version=extraction_version,
            created_at=created_at,
            grounding_chunks_by_id=grounding_chunks_by_id,
        )
        if card is not None:
            cards.append(card)
    return cards


def _join_field_parts(parts: Sequence[object]) -> str:
    """Join one field's parts, keeping each fact at most once.

    The index face is wide on purpose (narrative + structure), but a value
    that already occurs in the narrative must not be appended again as a
    structured fact: the repeat only inflates tf for that token, it adds no
    matchable surface.  Containment is tested against the whitespace-
    normalised lowercase text accumulated so far, so ``parser.py`` is
    dropped when ``src/parser.py`` (or the sentence naming it) is already in.
    """
    out: list[str] = []
    seen_text = ''
    for part in parts:
        value = ' '.join(str(part).split()) if part else ''
        if not value:
            continue
        low = value.lower()
        if low in seen_text:
            continue
        out.append(value)
        seen_text = f'{seen_text} {low}' if seen_text else low
    return ' '.join(out)


def card_field_texts(card: ExperienceCard) -> dict[str, str]:
    """The index face: state-only in v5, field-aware repair facts in legacy.

    A case packet has narrative *and* machine-extracted structure, and the
    narrative fields are populated too, so the two shapes cannot be an either/or
    choice -- reading only the three strings leaves files, symbols, conditions,
    patch and verify out of ``experience_terms`` entirely.  Measured over the
    stage-27 pooled corpus with the shipped channel (stage 28, 6 tasks with a
    signal positive, pool of 67 cards): narrative-only puts a positive in the
    top 8 for 5/6 tasks at mean rank 2.17, the wider face for 6/6 at mean rank
    0.83.

    Field identity is what lets a query anchor get weighted where it matches: an
    error literal on problem, a path or symbol on locate, a test node on test
    (``Store.experience_field_search``).
    """
    if card.is_attempt:
        anchors = card_entities(card, {})
        return {
            'problem': card.state or '',
            'locate': _join_field_parts([value for kind, value in anchors
                                        if kind in ('path', 'symbol', 'error', 'config_key')]),
            'implement': '', 'test': '',
        }
    conditions = list(card.conditions or ())
    verify = card.verify
    test_parts: list[object] = [
        *((verify.cmd, verify.expect, verify.observed) if verify else ()),
        *(c.value for c in conditions if c.type == "test_name"),
    ]
    groups: dict[str, list[object]] = {
        "problem": [_clip_text(card.problem),
                    *(c.value for c in conditions if c.type in ("error", "version"))],
        "locate": [_clip_text(card.locate), *card.files, *card.symbols,
                   *(c.value for c in conditions if c.type in ("module", "symbol", "frame"))],
        "implement": [_clip_text(card.implement), card.patch, *card.pitfalls,
                      _clip_text(getattr(card, "implement_shape", None)),
                      _clip_text(getattr(card, "implement_boundaries", None)),
                      # acceptance vocabulary (assert verbs, expected values)
                      # is a retrievable face: queries name desired behaviour
                      _clip_text(getattr(card, "implement_acceptance", None))],
        "test": test_parts,
    }
    return {field: _join_field_parts(parts) for field, parts in groups.items()}


def card_field_terms(card: ExperienceCard) -> list[str]:
    """Field-prefixed terms for the experience_terms index."""
    tokens: list[str] = []
    for field_name, text in card_field_texts(card).items():
        if not text:
            continue
        for term in tokenize(text):
            tokens.append(f"{field_name}\x1f{term}")
    return tokens


def card_lexical_text(card: ExperienceCard) -> str:
    if card.is_attempt:
        return card.state or ''
    return '\n'.join(
        value for value in (card.problem, card.locate, card.implement) if value
    )


def card_retrieval_text(card: ExperienceCard) -> str:
    """Un-fielded view of the index face, for probes that score one term bag."""
    if card.is_attempt:
        return card.state or ''
    return '\n'.join(value for value in card_field_texts(card).values() if value)


def card_entities(card: ExperienceCard, chunks_by_id: dict[str, StoredChunk]) -> list[tuple[str, str]]:
    text = card_lexical_text(card)
    if card.is_attempt:
        text += '\n' + '\n'.join(ref.quote for ref in card.evidence if ref.field == 'state')
    for chunk_id in (() if card.is_attempt else card.source_chunk_ids):
        chunk = chunks_by_id.get(chunk_id)
        if chunk is not None:
            text += '\n' + chunk.text
    from ..common.chunking import extract_entities
    entities = extract_entities([
        ChunkRecord(
            chunk_id='card',
            message_id='card',
            message_index=0,
            ord=0,
            char_start=0,
            char_end=len(text),
            kind='code',
            text=text,
            parent_chunk_id=None,
            tokens=tuple(tokenize(text)),
        ),
    ])
    return [(item.kind, item.value) for item in entities]


def scope_footprint(card: ExperienceCard, max_chars: int = 400) -> str:
    """One-line scope prior derived mechanically from the stored patch.

    Under the official tests-hidden protocol the issue text under-specifies
    how far a fix must reach; the historical patch is the only in-memory
    carrier of that scope.  Files, the functions each hunk lands in (from
    the @@ context line) and the magnitude, parsed from the diff -- no model
    output, grounded by construction.
    """
    patch = card.patch or ""
    if not patch.strip():
        return ""
    files = sorted(set(re.findall(r"^diff --git a/(\S+)", patch, re.M)))
    funcs = []
    for ctx in re.findall(r"^@@.*?@@\s*(.*)$", patch, re.M):
        ctx = (ctx or "").strip()
        m = re.search(r"def\s+(\w+)|class\s+(\w+)|(\w+)", ctx)
        if m:
            token = next(g for g in m.groups() if g)
            if token not in funcs:
                funcs.append(token)
        if len(funcs) >= 4:
            break
    added = len(re.findall(r"^\+(?!\+)", patch, re.M))
    removed = len(re.findall(r"^-(?!-)", patch, re.M))
    parts = []
    if files:
        parts.append("文件 " + ", ".join(f.split("/")[-1] for f in files[:3]))
    if funcs:
        parts.append("触及 " + ", ".join(funcs))
    parts.append(f"约 +{added}/-{removed} 行")
    line = "范围足迹（历史同类修复的改动范围先验）：" + "；".join(parts)
    return line[:max_chars]


def render_experience(
    card: ExperienceCard,
    chunks_by_id: dict[str, StoredChunk],
    counter,
    max_tokens: int,
    quote_max_chars: int,
    allow_patch: bool = True,
    evidence: Sequence[str] = (),
    first_action_hint: str = "",
    patch_render: str = "before_after",
    include_chunk_provenance: bool = True,
    persisted: bool = False,
    mode: str = "packet",
) -> str:
    if card.is_attempt:
        # All render modes share the tuple. Legacy hints, patch synthesis and
        # scope footprints must not turn a historical attempt into an order.
        return render_attempt(card, counter, max_tokens)
    # card-v2 abstract mode: only the three narrative fields plus the
    # mechanism face enter the model context -- no packet structure, no chunk
    # excerpts.  The retrieval face stays ``problem``; grounding was applied
    # at extraction time, so nothing here dereferences beyond the card.
    if mode in ("abstract", "abstract_scope"):
        lines = ['[历史经验卡]']
        if first_action_hint:
            lines.append(first_action_hint)
        faces = (
            ('问题（症状/触发）', card.problem),
            ('机制（根因/确认）', card.mechanism),
        )
        if mode == "abstract":
            faces += (('定位要点', card.locate),)
        # card-v4: the implement spec replaces the historical narrative when
        # present -- shape/boundaries/acceptance answer what a strong model
        # cannot derive from the pre-fix repo
        spec_faces = (
            ('修复形态', getattr(card, 'implement_shape', None)),
            ('边界分片', getattr(card, 'implement_boundaries', None)),
            ('验收语义', getattr(card, 'implement_acceptance', None)),
        )
        if any(v for _, v in spec_faces):
            faces += spec_faces
        else:
            faces += (('修改策略', card.implement),)
        for label, value in faces:
            if value:
                lines.append(f'{label}：{value}')
        if mode == "abstract_scope":
            footprint = scope_footprint(card)
            if footprint:
                lines.append(footprint)
        return counter.truncate('\n'.join(lines), max_tokens)
    # Payload v2: a self-contained case packet.  The renderer never depends on
    # the reader being able to dereference a chunk id, because the API only
    # exposes ``content``.
    if card.has_packet_fields:
        packet = build_packet(card, list(chunks_by_id.values()), allow_patch=allow_patch,
                              persisted=persisted)
        body = render_packet(packet, counter, max_tokens, evidence=evidence,
                             allow_patch=allow_patch, first_action_hint=first_action_hint,
                             patch_render=patch_render,
                             include_chunk_provenance=include_chunk_provenance)
        footprint = scope_footprint(card)
        if footprint:
            body = f"{footprint}\n{body}"
        return body
    lines = ['[历史经验卡]', f'来源：session {card.session_id} · {card.stable_id}', '', '【历史经验】']
    labels = {'problem': 'problem', 'locate': 'locate', 'implement': 'implement'}
    for name in _CARD_FIELDS:
        value = getattr(card, name)
        if value:
            lines.append(f'{labels[name]}：{value}')
    lines.extend(['', '【实施证据】'])
    if card.source_chunk_ids:
        # Chunk ids remain in the stored card for audit; the answer render can
        # omit the source-id line while keeping the evidence text.
        if include_chunk_provenance:
            lines.append('来源 chunk：' + ', '.join(card.source_chunk_ids))
        for chunk_id in card.source_chunk_ids:
            chunk = chunks_by_id.get(chunk_id)
            if chunk is not None:
                excerpt = chunk.text[:quote_max_chars]
                label = f'原文 [{chunk_id}]：' if include_chunk_provenance else '原文：'
                lines.append(label + excerpt)
    lines.append('')
    lines.append('【验证证据】')
    return counter.truncate('\n'.join(lines), max_tokens)


def card_anchors(card: ExperienceCard) -> list[tuple[str, str]]:
    """Structural anchors for the case index: files, symbols, conditions."""
    if card.is_attempt:
        kinds = {'path': 'file', 'symbol': 'symbol', 'error': 'error',
                 'config_key': 'config', 'test_name': 'test_name'}
        return list(dict.fromkeys((kinds[kind], value) for kind, value in card_entities(card, {})
                                  if kind in kinds))
    anchors: list[tuple[str, str]] = []
    for path in card.files:
        anchors.append(("file", path.lower()))
    for symbol in card.symbols:
        anchors.append(("symbol", symbol.lower()))
    for condition in card.conditions:
        anchors.append((condition.type, condition.value.lower()))
    return anchors


_SETUP_RE = re.compile(
    r"git clone|git checkout|pip install|npm install|Creating virtualenv|"
    r"Downloading |Collecting |conda create|cd /testbed && git |"
    r"remove.*testbed|Cloning into|"
    r"pytest.*[0-9]\.[0-9]+.*or (higher|later)|"
    r"update pytest|pytest version", re.I)


def is_setup_chunk(text: str) -> bool:
    """Setup-phase boilerplate (clone/checkout/install) carries no transferable
    experience; these chunks are excluded from card extraction windows.

    No bare ``swebench``/``testbed`` word match: on the 300Lite corpus every
    working path carries a ``./swebench_<ts>/testbed/`` prefix, so a bare
    token match excluded ~14% of chunks / ~20% of window chars per session
    (repository reads, find/grep commands) from extraction."""
    head = text[:600]
    return bool(_SETUP_RE.search(head)) and len(text) < 2000
