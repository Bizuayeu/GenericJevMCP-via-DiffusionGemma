import base64
import contextlib
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
from jev import client


class ClientTests(unittest.TestCase):
    def call(self, args):
        with patch.object(sys, 'argv', ['client.py', '--ssh-config', 'ssh_config', *args]), \
                patch('jev.client.subprocess.run', return_value=subprocess.CompletedProcess([], 0, b'{"answers":{}}', b'')) as run, \
                contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            client.main()
            return json.loads(run.call_args.kwargs['input'])

    def test_request_file_preserves_unicode_and_api_fields(self):
        body = {'state': '赤いボールが3個', 'rag': False, 'mode': 'separate', 'samples': 1,
                'questions': {'red': {'type': 'noul', 'instructions': '赤いですか'}}}
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'request.json'
            path.write_text(json.dumps(body, ensure_ascii=False), encoding='utf-8-sig')
            payload = self.call(['decide', '--request', str(path)])
        self.assertEqual(payload, {'service': 'dg-bert', 'body': body})

    def test_old_question_cli_keeps_rag(self):
        payload = self.call(['decide', '--number', '40', '--question', '慢心に注意？'])
        self.assertEqual(payload['body']['rag'], {'number': 40})
        self.assertEqual(payload['body']['questions']['answer']['instructions'], '慢心に注意？')

    def test_request_rejects_ambiguous_flags_and_invalid_json(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'request.json'
            path.write_text('{}', encoding='utf-8')
            for args in [['--number', '40'], ['--query', 'x'], ['--question', 'q'], []]:
                with self.subTest(args=args), self.assertRaises(SystemExit):
                    self.call(['decide', '--request', str(path), *args])
            for text in ['{"state":NaN}', '{"state":1,"state":2}', '[]']:
                path.write_text(text, encoding='utf-8')
                with self.subTest(text=text), self.assertRaises(SystemExit):
                    self.call(['decide', '--request', str(path)])

    def test_image_path_is_encoded_without_reading_remote_keys(self):
        with tempfile.TemporaryDirectory() as tmp:
            image = Path(tmp) / 'image.png'
            data = b'\x89PNG\r\n\x1a\nfixture'
            image.write_bytes(data)
            payload = self.call(['decide', '--question', 'Image color?', '--image', str(image)])
        self.assertEqual(payload['body']['image'], 'data:image/png;base64,'+base64.b64encode(data).decode())
        self.assertIs(payload['body']['rag'], False)

    def test_plain_question_uses_model_knowledge(self):
        payload = self.call(['decide', '--question', 'Capital of Japan?'])
        self.assertIs(payload['body']['rag'], False)
        self.assertEqual(payload['body']['state'], '')

    def test_explicit_local_config_preserves_ssh_installation(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            (root/'client-config.json').write_text(json.dumps({'ssh_config':'my-config','ssh_host':'my-spark','remote_root':'/opt/jev'}),encoding='utf-8')
            with patch.object(client,'ROOT',root), \
                 patch.object(sys,'argv',['client.py','decide','--question','q']), \
                 patch('jev.client.subprocess.run',return_value=subprocess.CompletedProcess([],0,b'{"answers":{}}',b'')) as run, \
                 contextlib.redirect_stdout(io.StringIO()):
                client.main()
            self.assertEqual(run.call_args.args[0][1:3],['-F','my-config'])
            self.assertIn('my-spark',run.call_args.args[0])
            self.assertEqual(run.call_args.args[0][-1],'cd /opt/jev && python3 -m jev.client receive')

    def test_choice_question_runs_without_request_file(self):
        payload=self.call(['decide','--question','日本の首都はどれですか','--choices','東京','大阪','京都'])
        self.assertEqual(payload['body']['questions']['answer'], {
            'type':'choice','instructions':'日本の首都はどれですか',
            'criteria':{'東京':None,'大阪':None,'京都':None}})
        self.assertIs(payload['body']['rag'],False)

    def test_choice_question_preserves_optional_state(self):
        payload=self.call(['decide','--question','色は？','--state','赤いボール','--choices','赤','青'])
        self.assertEqual(payload['body']['state'],'赤いボール')

    def test_choice_input_validation(self):
        for choices in [['東京'],['東京','東京'],[str(i) for i in range(27)]]:
            with self.subTest(choices=choices),self.assertRaises(SystemExit):
                self.call(['decide','--question','q','--choices',*choices])

    def test_request_json_avoids_temporary_file(self):
        body = {'questions': {'a': {'type': 'noul', 'instructions': '赤い？'}}}
        self.assertEqual(self.call(['decide', '--request-json', json.dumps(body)])['body'], body)

    def test_state_files_are_loaded_with_provenance(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / '資料.md'
            path.write_text('赤いボール', encoding='utf-8-sig')
            payload = self.call(['decide', '--question', '赤い？', '--state-file', str(path)])
            self.assertEqual(payload['body']['state'], {'sources': [
                {'source': str(path.resolve()), 'text': '赤いボール'}]})
            path.write_text('a' * 32769, encoding='utf-8')
            with self.assertRaises(SystemExit):
                self.call(['decide', '--question', 'q', '--state-file', str(path)])

    def test_client_elapsed_includes_transport_and_preserves_engine_time(self):
        result = {'answers': {'a': None}, 'diagnostics': {'elapsed_seconds': 0.25}}
        for transport in ['ssh', 'direct']:
            with self.subTest(transport=transport), \
                 patch.object(client, 'ROOT', Path('/unconfigured-jev-test')), \
                 patch.object(sys, 'argv', ['client.py', *(['--ssh-config','config'] if transport=='ssh' else []),
                                           'decide','--question','q']), \
                 patch('jev.client.time.monotonic', side_effect=[10,12.5]), \
                 patch('jev.client.receive', return_value=dict(result)), \
                 patch('jev.client.subprocess.run', return_value=subprocess.CompletedProcess([],0,json.dumps(result).encode(),b'')), \
                 contextlib.redirect_stdout(io.StringIO()) as output:
                client.main()
                value=json.loads(output.getvalue())
                self.assertEqual(value['elapsed_seconds'],2.5)
                self.assertEqual(value['diagnostics']['elapsed_seconds'],0.25)
                self.assertEqual(value['timing'],{'total_seconds':2.5,'decision_seconds':.25})
                self.assertIn('判定時間: 0.250 秒',value['display'])

    def test_text_rendering_preserves_decisions_and_abstention(self):
        result={'answers': {
            'capital': {'type':'choice','choice':'東京','probabilities':{'東京':.9,'大阪':.1}},
            'truth': {'type':'noul','noul':.2,'probabilities':{'yes':.2,'no':.8}},
            'scale': {'type':'score','score':.75,'legend':{'0':'低','1':'高'},'probabilities':{'0':.25,'1':.75}},
            'missing': None}, 'elapsed_seconds':2.5,'diagnostics':{'elapsed_seconds':.25}}
        text=client.render_decision(result)
        for part in ['東京','90.00%','no','0.750','低','高','保留','2.500','0.250']:
            self.assertIn(part,text)

    def test_inline_request_rejects_invalid_and_conflicting_input(self):
        for body in ['{"questions":{},"questions":{}}','{"state":NaN}','[]']:
            with self.subTest(body=body), self.assertRaises(SystemExit):
                self.call(['decide','--request-json',body])
        body=json.dumps({'state':'existing','questions':{'q':{'type':'noul'}}})
        with self.assertRaises(SystemExit):
            self.call(['decide','--request-json',body,'--state-file','missing.md'])
        with self.assertRaises(SystemExit):
            self.call(['decide','--question','q','--state-file','missing.md'])

    def test_text_cli_uses_result_and_includes_source_path(self):
        result={'answers':{'q':None},'diagnostics':{'reads':0}}
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'source.txt'
            path.write_text('source text',encoding='utf-8')
            with patch.object(sys,'argv',['client.py','--ssh-config','config','decide',
                                         '--question','q','--state-file',str(path),'--format','text']), \
                 patch('jev.client.subprocess.run',return_value=subprocess.CompletedProcess([],0,json.dumps(result).encode(),b'')), \
                 contextlib.redirect_stdout(io.StringIO()) as output:
                client.main()
            self.assertIn(str(path.resolve()),output.getvalue())
            self.assertIn('保留',output.getvalue())
            self.assertIn('所要時間（全体）:',output.getvalue())

    def test_request_from_stdin_preserves_quotes_and_unicode(self):
        body={'questions':{'q':{'type':'noul','instructions':'「赤」ですか'}}}
        with patch.object(sys,'stdin',io.StringIO(json.dumps(body,ensure_ascii=False))):
            self.assertEqual(self.call(['decide','--request-json','-'])['body'],body)

    def test_timing_has_distinct_labels_even_with_complexity_question(self):
        value={'answers':{'complexity':{'type':'score','score':.106,
               'legend':{'0':'low','1':'high'},'probabilities':{'0':.894,'1':.106}}},
               'elapsed_seconds':.882,'diagnostics':{'elapsed_seconds':.106}}
        display=client.render_decision(value)
        self.assertIn('所要時間（全体）: 0.882 秒',display)
        self.assertIn('判定時間: 0.106 秒',display)
        self.assertIn('complexity: 0.106',display)
        value['diagnostics']={'reads':0}
        self.assertIn('判定時間: 未計測',client.render_decision(value))
