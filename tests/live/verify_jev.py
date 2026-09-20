"""Bounded real-server checks for joint/separate Jev decisions; keeps credentials local."""
import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import time
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[2]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    key = dict(line.split('=', 1) for line in (ROOT / '.env').read_text().splitlines()
               if '=' in line)['DG_BERT_API_KEY']
    questions = {
        'red': {'type': 'noul', 'instructions': 'Are there red balls in the box?'},
        'blue': {'type': 'noul', 'instructions': 'Are there blue balls in the box?'},
        'color': {'type': 'choice', 'instructions': 'What color are the balls?',
                  'criteria': {'red': None, 'blue': None, 'green': None}},
        'count': {'type': 'score', 'instructions': 'How many balls are in the box?',
                  'criteria': ['zero', 'one', 'two', 'three']}}
    body = {'state': 'The box contains exactly three red balls and no other objects.',
            'rag': False, 'samples': 1, 'seed': 42, 'questions': questions}
    records = []

    def post(label, request_body, route='/v1/systemone'):
        start = time.monotonic()
        req = Request('http://127.0.0.1:8011' + route,
                      data=json.dumps(request_body).encode(),
                      headers={'Authorization': 'Bearer ' + key, 'Content-Type': 'application/json'})
        with urlopen(req, timeout=180) as response:
            result = json.load(response)
        if route == '/v1/chat/completions':
            result = json.loads(result['choices'][0]['message']['content'])
        records.append({'case': label, 'seconds': time.monotonic() - start, 'result': result})
        return result

    status = 'failed'
    try:
        for mode in ('joint', 'separate'):
            for phase in ('first', 'warm'):
                result = post(mode + '-' + phase, {**body, 'mode': mode})
                assert result['diagnostics']['mode'] == mode
                assert result['diagnostics']['reads'] == (1 if mode == 'joint' else 4)
                assert result['diagnostics']['calibrated'] is False
                assert set(result['answers']) == set(questions)
                for answer in result['answers'].values():
                    assert answer is not None, 'unexpected abstention on simple state'
                    values = list(answer['probabilities'].values())
                    assert all(math.isfinite(v) and 0 <= v <= 1 for v in values)
                    assert math.isclose(sum(values), 1.)
                records[-1]['correct'] = [result['answers']['red']['noul'] >= .5,
                                          result['answers']['blue']['noul'] < .5,
                                          result['answers']['color']['choice'] == 'red',
                                          max(result['answers']['count']['probabilities'],
                                              key=result['answers']['count']['probabilities'].get) == '3']
        baseline = result
        reversed_result = post('separate-reversed', {**body, 'mode': 'separate',
                               'questions': dict(reversed(list(questions.items())))})
        records[-1]['answers_equal'] = baseline['answers'] == reversed_result['answers']
        single = post('separate-single', {**body, 'mode': 'separate', 'questions': {'red': questions['red']}})
        records[-1]['answer_equal'] = single['answers']['red'] == baseline['answers']['red']
        chat_body = {'messages': [
            {'role': 'system', 'content': json.dumps({'rag': False, 'samples': 1, 'mode': 'separate',
                                                     'questions': {'red': questions['red']}})},
            {'role': 'user', 'content': json.dumps(body['state'])}]}
        chat = post('chat-wrapper', chat_body, '/v1/chat/completions')
        assert chat['diagnostics']['mode'] == 'separate'
        grounded = post('separate-rag', {'state': {'number': 40}, 'rag': {'number': 40}, 'sources_only': True,
                         'samples': 'auto', 'mode': 'separate', 'questions': {
                             'caution': {'type': 'noul', 'instructions': 'この資料は慢心に注意するよう述べていますか。'}}})
        assert grounded['sources'] and '_dg_supported' not in grounded['answers']
        assert grounded['answers']['caution'] is not None
        assert grounded['answers']['caution']['noul'] >= .5
        unsupported = post('separate-rag-unknown', {'state': {'number': 40}, 'rag': {'number': 40}, 'sources_only': True,
                            'mode': 'separate', 'questions': {'weather': {
                                'type': 'noul', 'instructions': '2026年9月19日の東京の気温は20度ですか。'}}})
        assert unsupported['answers']['weather'] is None
        assert unsupported['abstention'] == 'insufficient_source_evidence'
        status = 'complete'
    finally:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps({'status': status, 'time_utc': datetime.now(timezone.utc).isoformat(),
                                          'cases': records}, ensure_ascii=False, indent=2), encoding='utf-8')
        print(json.dumps({'status': status, 'cases': [
            {k: v for k, v in r.items() if k != 'result'} for r in records]}, ensure_ascii=False))


if __name__ == '__main__':
    main()
