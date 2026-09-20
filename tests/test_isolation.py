import math
import unittest

from jev.decision import Engine, validate
from jev.server import execute
from jev.corpus import Corpus


def request(mode='separate', samples='auto'):
    return {'state': 'The object is red.', 'rag': False, 'mode': mode, 'samples': samples,
            'questions': {'color': {'type': 'choice', 'instructions': 'Color?',
                                    'criteria': {'red': None, 'blue': None}},
                          'ambiguous': {'type': 'noul', 'instructions': 'Uncertain?'}}}


class RecordingEngine(Engine):
    def __init__(self):
        super().__init__(None, 'http://unused', 'unused')
        self.calls = []

    def read(self, questions, state, seed):
        self.calls.append((questions, state, seed))
        return [dict(probs=[.5, .5] if q['instructions'] == 'Uncertain?' else [.99, .01],
                     entropy=math.log(2) if q['instructions'] == 'Uncertain?' else .01,
                     label_mass=1., argmax_is_label=True) for q in questions]


class IsolationTests(unittest.TestCase):
    def test_default_preserves_joint_reads(self):
        body = request()
        body.pop('mode')
        engine = RecordingEngine()
        result = engine.decide(validate(body))
        self.assertEqual(len(engine.calls), 4)
        self.assertTrue(all(len(qs) == 2 for qs, _, _ in engine.calls))
        self.assertEqual(result['diagnostics']['mode'], 'joint')

    def test_separate_preserves_answers_and_adapts_each_question(self):
        engine = RecordingEngine()
        result = engine.decide(validate(request()))
        self.assertEqual(list(result['answers']), ['color', 'ambiguous'])
        self.assertEqual(result['answers']['color']['choice'], 'red')
        self.assertEqual(result['answers']['ambiguous']['noul'], .5)
        self.assertEqual(result['diagnostics']['reads'], 5)
        self.assertEqual(result['diagnostics']['questions']['color']['reads'], 1)
        self.assertEqual(result['diagnostics']['questions']['ambiguous']['reads'], 4)
        self.assertTrue(all(len(qs) == 1 for qs, _, _ in engine.calls))
        self.assertTrue(all(qs[0]['id'] == 'answer' for qs, _, _ in engine.calls))

    def test_question_order_and_neighbors_do_not_change_read_inputs(self):
        first = RecordingEngine()
        body = request(samples=2)
        first.decide(validate(body), seed=123)
        second = RecordingEngine()
        body['questions'] = dict(reversed(list(body['questions'].items())))
        second.decide(validate(body), seed=123)
        self.assertEqual(first.calls[:2], second.calls[2:])
        self.assertEqual(first.calls[2:], second.calls[:2])

    def test_invalid_mode_fails_before_backend(self):
        for mode in ['packed', '', None, 1, [], {}]:
            with self.subTest(mode=mode), self.assertRaises(ValueError):
                validate(request(mode))

    def test_rag_support_question_remains_self_contained(self):
        class Capture(RecordingEngine):
            def decide(self, schema, seed=42):
                self.schema = schema
                return {'answers': {q['id']: {'type': 'noul', 'noul': 1.}
                                    for q in schema['questions']}, 'diagnostics': {}}
        engine = Capture()
        body = request(samples=1)
        body['rag'] = {'number': 40}
        body['sources_only'] = True
        corpus = Corpus([{'id': 'n40', 'number': 40, 'title': '40', 'text': 'red'}])
        result = execute(body, engine, corpus)
        support = engine.schema['questions'][-1]
        self.assertIn('Color?', support['instructions'])
        self.assertIn('Uncertain?', support['instructions'])
        self.assertIn('red', support['instructions'])
        self.assertNotIn('_dg_supported', result['answers'])
        self.assertEqual(engine.schema['mode'], 'separate')
