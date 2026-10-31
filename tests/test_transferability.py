"""Production grouping, grounding and fail-open tests, without network calls."""
import json
from dataclasses import replace
from types import SimpleNamespace

import pytest

from codemem.common.config import Settings
from codemem.common.embeddings import reset_model_limiter_for_tests
from codemem.add.experience import ExperienceCard, render_experience
from codemem.common.guard import ModuleBudget
from codemem.add.minimal import VerifySemantics
from codemem.search.retrieval import Candidate, SearchEngine
from codemem.service import MemoryService
from codemem.common.store import Store
from codemem.search.transferability import HYP_PROMPT, MinimalTransferabilityJudge, TransferCard, _evidence_segments

QUERY = "Synthetic parser loses empty fields in nested records."


class FakeLLM:
    configured = True

    def __init__(self, mutate=None):
        self.calls = []
        self.mutate = mutate

    def complete_json(self, **kwargs):
        self.calls.append(kwargs)
        if kwargs['system'] == HYP_PROMPT:
            request = json.loads(kwargs['user'])
            raw = {'hypotheses': [{'subsystem': 'parser', 'defect_shape': 'empty fields',
                                   'evidence_id': request['issue_evidence'][0]['id']}]}
        else:
            request = json.loads(kwargs['user'])
            raw = {'items': [
                {'id': c['id'], 'verdict': ('irrelevant' if i == 0 else
                 'transferable' if i == 1 else 'topic_related'),
                 'root_cause_unit_id': c['mechanism_evidence'][0]['id'], 'reason': 'controlled mechanism comparison',
                 'matched_hypothesis': 'H1' if i == 1 else None}
                for i, c in enumerate(request['cards'])]}
        if self.mutate:
            raw = self.mutate(len(self.calls), raw)
        return raw, SimpleNamespace(raw_finish_reason='stop')


def cfg(tmp_path, **kwargs):
    options = dict(db_path=tmp_path/'test.db', llm_model='gpt-4o-mini',
                   llm_base_url=None, llm_api_key=None, embed_base_url=None,
                   embed_api_key=None, embed_model='', rerank_base_url=None,
                   transferability_enabled=True, search_max_items=100,
                   applicability_enabled=False, model_concurrency=4)
    options.update(kwargs)
    reset_model_limiter_for_tests()
    return Settings(**options)


def cards(n):
    return [TransferCard(str(i), f'Problem {i}', f'Historical mechanism number {i}.')
            for i in range(n)]


def test_twenty_cards_are_judged_in_one_batch_after_hypotheses(tmp_path):
    llm = FakeLLM()
    judge = MinimalTransferabilityJudge(llm, cfg(tmp_path))
    result = judge.judge(user_id='u', query=QUERY, cards=cards(25))
    assert len(llm.calls) == 2
    assert len(json.loads(llm.calls[1]['user'])['cards']) == 20
    assert json.loads(llm.calls[1]['user'])['candidate_ids'] == [f'C{i:02d}' for i in range(1,21)]
    assert json.loads(llm.calls[1]['user'])['expected_items'] == 20
    assert len(result.verdicts) == 20
    assert all(call['max_retries'] == 0 for call in llm.calls)
    assert result.verdicts['1'].verdict == 'transferable'


def test_cache_is_scope_content_and_query_bound(tmp_path):
    llm = FakeLLM(); judge = MinimalTransferabilityJudge(llm, cfg(tmp_path))
    result = judge.judge(user_id='u', query=QUERY, cards=cards(2))
    assert not result.cached
    assert judge.judge(user_id='u', query=QUERY, cards=cards(2)).cached
    assert len(llm.calls) == 2
    judge.judge(user_id='other', query=QUERY, cards=cards(2))
    assert len(llm.calls) == 4
    changed = cards(2); changed[0] = replace(changed[0], mechanism='Changed mechanism.')
    judge.judge(user_id='u', query=QUERY, cards=changed)
    assert len(llm.calls) == 6


def test_ungrounded_hypothesis_stops_before_batch(tmp_path):
    llm = FakeLLM(lambda n, raw: {'hypotheses': [{'subsystem': 'parser',
          'defect_shape': 'empty fields', 'evidence_id': 'Q999'}]} if n == 1 else raw)
    result = MinimalTransferabilityJudge(llm, cfg(tmp_path)).judge(
        user_id='u', query=QUERY, cards=cards(2))
    assert not result.verdicts and result.reason == 'no_valid_hypotheses'
    assert len(llm.calls) == 1


@pytest.mark.parametrize('change', ['quote', 'hypothesis', 'duplicate', 'bad_label'])
def test_invalid_verdict_is_unjudged_not_a_demotion(tmp_path, change):
    def mutate(n, raw):
        if n == 2:
            item = raw['items'][1]
            if change == 'quote': item['root_cause_unit_id'] = 'C01.M01'  # belongs to the other card
            elif change == 'hypothesis': item['matched_hypothesis'] = 'H99'
            elif change == 'duplicate': raw['items'].append(dict(item))
            elif change == 'bad_label': item['verdict'] = ['transferable']
        return raw
    result = MinimalTransferabilityJudge(FakeLLM(mutate), cfg(tmp_path)).judge(
        user_id='u', query=QUERY, cards=cards(2))
    assert '1' not in result.verdicts
    assert result.reason == 'partial' and '0' in result.verdicts


def test_deadline_and_backend_exception_preserve_no_decisions(tmp_path):
    llm = FakeLLM(); judge = MinimalTransferabilityJudge(llm, cfg(tmp_path))
    result = judge.judge(user_id='u', query=QUERY, cards=cards(2), deadline=ModuleBudget(0))
    assert not result.verdicts and not llm.calls
    def fail(n, raw): raise TimeoutError('fake transport timeout')
    judge = MinimalTransferabilityJudge(FakeLLM(fail), cfg(tmp_path))
    assert not judge.judge(user_id='u', query=QUERY, cards=cards(2)).verdicts


def test_judge_requires_pinned_memory_model(tmp_path):
    judge = MinimalTransferabilityJudge(FakeLLM(), cfg(tmp_path, llm_model='other-model'))
    assert not judge.available


class ForbiddenReranker:
    available = True
    def rerank(self, *args, **kwargs):
        raise AssertionError('qwen must not run in replacement mode')


def engine(tmp_path, *, mutate=None, score=1.0, n=25, **kwargs):
    settings = cfg(tmp_path, **kwargs)
    llm = FakeLLM(mutate)
    judge = MinimalTransferabilityJudge(llm, settings)
    engine = SearchEngine(Store(settings), settings, transferability_judge=judge,
                          reranker=ForbiddenReranker())
    candidates = []
    for i in range(n):
        card = ExperienceCard(experience_id=str(i), user_id='u', session_id=str(i),
               extraction_version='card-v5-minimal', problem=f'Problem {i} involving records.',
               mechanism=f'Historical mechanism number {i}.',
               verify_semantics=VerifySemantics(invariants=('Empty fields remain visible.',)))
        candidates.append(Candidate(str(i), 'experience', f'Unique body number {i}',
                          fused_score=score, payload={'card': card, 'chunks_by_id': {}}))
    engine._recall_cards = lambda *args: {c.item_id: c for c in candidates}
    engine._rank = lambda pool, features: list(pool.values())
    return engine, candidates, llm


def test_grouping_keeps_all_cards_and_unjudged_tail_and_original_scores(tmp_path):
    e, candidates, llm = engine(tmp_path)
    data, diag = e.search(user_id='u', query=QUERY, top_k=100)
    assert len(data) == 25  # no hidden ten/twenty-item return cap
    assert [x['id'] for x in data] == ['1'] + [str(i) for i in range(2,25)] + ['0']
    assert all(c.fused_score == 1.0 for c in candidates)
    assert diag.rerank['reason'] == 'replaced_by_transferability'
    assert len(llm.calls) == 2
    assert all('score' not in x for x in data)
    assert '未经当前代码验证' in data[0]['content']
    assert 'loses empty fields' in data[0]['content']
    original = render_experience(candidates[1].payload['card'], {}, e.counter, 1600, 600)
    assert data[0]['content'].startswith(original)
    assert all('查询时适用性推测' not in x['content'] for x in data[1:])
    assert sum(e.counter.count(x['content']) for x in data) == diag.tokens_used


def test_top_k_is_honoured_after_promotion(tmp_path):
    e, _, _ = engine(tmp_path)
    data, _ = e.search(user_id='u', query=QUERY, top_k=1)
    assert len(data) == 1 and data[0]['id'] == '1'


def test_judge_cannot_turn_zero_recall_evidence_into_admission(tmp_path):
    e, _, _ = engine(tmp_path, score=0)
    data, _ = e.search(user_id='u', query=QUERY, top_k=100)
    assert data == []


def test_failure_returns_original_order_without_annotations_or_qwen(tmp_path):
    e, _, _ = engine(tmp_path, mutate=lambda n, raw: None if n == 2 else raw)
    data, diag = e.search(user_id='u', query=QUERY, top_k=100)
    assert [x['id'] for x in data] == [str(i) for i in range(25)]
    assert all('查询时适用性推测' not in x['content'] for x in data)
    assert not diag.applicability['applied']


def test_service_disables_external_reranker_in_replacement_mode(tmp_path):
    settings = cfg(tmp_path, rerank_base_url='http://do-not-call.invalid')
    service = MemoryService(Store(settings), settings, llm_client=FakeLLM())
    assert not service.search_engine.reranker.available
    assert service.search_engine.transferability_judge.available


def test_evidence_segments_preserve_source_and_all_nonwhitespace_characters():
    source = "`Parser.visit_Call` preserves `evaluate=False`.\nA second sentence; " + "x"*520 + " tail."
    segments = _evidence_segments(source, 'C01.M', 900)
    assert len(segments) >= 4
    assert all(s['text'] in source and len(s['text']) <= 240 for s in segments)
    assert segments[0]['text'] == '`Parser.visit_Call` preserves `evaluate=False`.'
    assert [s['id'] for s in segments] == [f'C01.M{i:02d}' for i in range(1,len(segments)+1)]
    assert ''.join(''.join(s['text'].split()) for s in segments) == ''.join(source.split())
    assert _evidence_segments('  \n ', 'Q', 1800) == []
    assert _evidence_segments('a'*2000, 'Q', 1800)[-1]['text'] in 'a'*1800


def test_program_backfills_original_quotes_and_ignores_model_supplied_copies(tmp_path):
    query = 'The `Parser` loses empty fields.\nSecond report line.'
    candidates = [TransferCard('a', 'problem', '`Parser.parse` merges adjacent separators.\nEmpty fields disappear.')]

    def mutate(n, raw):
        if n == 1:
            raw['hypotheses'][0]['evidence'] = 'model rewrote the evidence'
        else:
            raw['items'][0]['verdict'] = 'transferable'
            raw['items'][0]['matched_hypothesis'] = 'H1'
            raw['items'][0]['root_cause_unit'] = 'model rewrote the mechanism'
        return raw

    llm = FakeLLM(mutate)
    result = MinimalTransferabilityJudge(llm, cfg(tmp_path)).judge(user_id='u',query=query,cards=candidates)
    assert result.reason == 'ok'
    assert result.hypotheses[0]['evidence'] == 'The `Parser` loses empty fields.'
    assert result.verdicts['a'].root_cause_unit == '`Parser.parse` merges adjacent separators.'
    assert result.hypotheses[0]['evidence'] in query
    assert result.verdicts['a'].root_cause_unit in candidates[0].mechanism
    request = json.loads(llm.calls[1]['user'])
    assert 'mechanism' not in request['cards'][0]
    assert request['cards'][0]['mechanism_evidence'][0]['id'] == 'C01.M01'
    assert result.diagnostics['items_accepted'] == 1
    assert [s['outcome'] for s in result.diagnostics['stages']] == ['ok','ok']


@pytest.mark.parametrize('bad_id', ['C01.M01', 'C02.M99', 'Q01', ['C02.M01'], None])
def test_wrong_card_or_malformed_evidence_id_is_rejected(tmp_path,bad_id):
    def mutate(n, raw):
        if n == 2: raw['items'][1]['root_cause_unit_id'] = bad_id
        return raw
    result = MinimalTransferabilityJudge(FakeLLM(mutate),cfg(tmp_path)).judge(user_id='u',query=QUERY,cards=cards(2))
    assert result.reason == 'partial' and '1' not in result.verdicts
    assert result.diagnostics['rejected'] == {'mechanism_evidence_id':1}
    assert result.diagnostics['unjudged_candidates'] == 1


def test_unknown_without_source_is_retained_without_claiming_a_root_cause(tmp_path):
    def mutate(n,raw):
        if n == 2:
            raw['items'][1].update(verdict='unknown',root_cause_unit_id=None,matched_hypothesis=None)
        return raw
    result=MinimalTransferabilityJudge(FakeLLM(mutate),cfg(tmp_path)).judge(user_id='u',query=QUERY,cards=cards(2))
    assert result.reason == 'ok'
    assert result.verdicts['1'].verdict == 'unknown' and result.verdicts['1'].root_cause_unit == ''


def test_missing_and_duplicate_items_are_counted_separately(tmp_path):
    def mutate(n,raw):
        if n == 2:
            raw['items'] = [raw['items'][0],dict(raw['items'][0])]
        return raw
    result=MinimalTransferabilityJudge(FakeLLM(mutate),cfg(tmp_path)).judge(user_id='u',query=QUERY,cards=cards(2))
    assert not result.verdicts
    assert result.diagnostics['missing_candidates'] == 1
    assert result.diagnostics['rejected'] == {'duplicate_candidate_id':1}


def test_diagnostics_distinguish_bad_hypothesis_from_empty_hypotheses(tmp_path):
    bad=FakeLLM(lambda n,raw: {'hypotheses':[{'subsystem':'parser','defect_shape':'empty fields','evidence_id':'Q99'}]} if n==1 else raw)
    result=MinimalTransferabilityJudge(bad,cfg(tmp_path)).judge(user_id='u',query=QUERY,cards=cards(2))
    assert result.diagnostics['rejected'] == {'hypothesis_evidence_id':1}
    empty=FakeLLM(lambda n,raw: {'hypotheses':[]} if n==1 else raw)
    result=MinimalTransferabilityJudge(empty,cfg(tmp_path)).judge(user_id='u',query=QUERY,cards=cards(2))
    assert result.diagnostics['hypotheses_returned'] == 0
    assert not result.diagnostics['rejected']


@pytest.mark.parametrize('failure', ['timeout','length','invalid_json','budget','busy'])
def test_stage_failure_diagnostics_do_not_contain_prompt_or_exception_text(tmp_path,monkeypatch,failure):
    class Broken(FakeLLM):
        def complete_json(self,**kwargs):
            if failure=='timeout': raise TimeoutError('private-input-or-key-must-not-be-logged')
            if failure=='length': return {},SimpleNamespace(raw_finish_reason='length')
            if failure=='invalid_json': return None,SimpleNamespace(raw_finish_reason='stop')
            return super().complete_json(**kwargs)
    if failure=='busy':
        monkeypatch.setattr('codemem.search.transferability.get_model_limiter',lambda settings:SimpleNamespace(acquire=lambda **kw:False))
    result=MinimalTransferabilityJudge(Broken(),cfg(tmp_path)).judge(user_id='u',query=QUERY,cards=cards(2),
                deadline=ModuleBudget(0) if failure=='budget' else None)
    expected={'timeout':'backend_error','length':'output_truncated','invalid_json':'invalid_json',
              'budget':'budget_exhausted','busy':'model_slots_busy'}[failure]
    assert result.diagnostics['stages'][0]['outcome'] == expected
    output=json.dumps(result.diagnostics)
    assert QUERY not in output and 'private-input-or-key' not in output


def test_cached_result_reports_no_new_model_stages(tmp_path):
    judge=MinimalTransferabilityJudge(FakeLLM(),cfg(tmp_path))
    judge.judge(user_id='u',query=QUERY,cards=cards(2))
    result=judge.judge(user_id='u',query=QUERY,cards=cards(2))
    assert result.cached and result.calls == 0
    assert result.diagnostics['cache_hit'] and result.diagnostics['stages'] == []
