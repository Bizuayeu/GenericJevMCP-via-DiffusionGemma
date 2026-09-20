import base64
import json
import unittest
from unittest.mock import patch
from decision import Engine, validate
from media import encode_image, validate_image, MAX_IMAGE_BYTES
from server import execute
from corpus import Corpus

PNG = 'data:image/png;base64,' + base64.b64encode(b'\x89PNG\r\n\x1a\n' + b'fixture').decode()

class KnowledgeImageTests(unittest.TestCase):
    def test_additional_input_optional_and_no_rag_by_default(self):
        class Fake:
            def decide(self, schema, seed):
                self.schema = schema
                return {'answers': {'capital': {'type':'choice', 'choice':'Tokyo'}}, 'diagnostics':{}}
        engine = Fake()
        body = {'questions': {'capital': {'type':'choice', 'instructions':'Capital of Japan?',
                                         'criteria': {'Tokyo':None,'Osaka':None}}}}
        result = execute(body, engine, Corpus([]))
        self.assertEqual(engine.schema['state'], '')
        self.assertEqual(len(engine.schema['questions']), 1)
        self.assertEqual(result['answers']['capital']['choice'], 'Tokyo')

    def test_image_validation_rejects_remote_urls_and_wrong_mime(self):
        base = {'questions': {'q': {'type':'noul'}}}
        for image in ['https://example.com/x.png', '/tmp/x.png', '', None, 1,
                      'data:image/png;base64,?', 'data:image/jpeg;base64,'+base64.b64encode(b'bad').decode()]:
            with self.subTest(image=image), self.assertRaises(ValueError):
                validate({**base, 'image': image})
        self.assertEqual(validate({**base, 'image': PNG})['image'], PNG)

    def test_actual_image_is_forwarded_in_chat_content(self):
        class Encoder:
            def get_vocab_size(self): return 100
        class Response:
            def __enter__(self): return self
            def __exit__(self, *args): pass
        q = validate({'questions':{'q':{'type':'noul'}}})['questions']
        engine = Engine(Encoder(), 'http://unused', 'unused')
        value = {'choices':[{'logprobs':{'content':[{'top_logprobs':[
            {'token':'token_id:1','logprob':-.1}, {'token':'token_id:2','logprob':-3.}]}]}}]}
        with patch('decision.template_for',return_value=([1],[{'pos':0,'ids':[1,2]}])), \
             patch('decision.urlopen',return_value=Response()) as call, \
             patch('decision.json.load',return_value=value):
            engine.read(q, '', 42, image=PNG)
        payload=json.loads(call.call_args.args[0].data)
        content=payload['messages'][1]['content']
        self.assertEqual(content[-1], {'type':'image_url','image_url':{'url':PNG}})
        self.assertIn('knowledge', payload['messages'][0]['content'])

    def test_separate_mode_retains_image_for_each_question(self):
        class Capture(Engine):
            def __init__(self):
                super().__init__(None,'unused','unused')
                self.images=[]
            def read(self, questions, state, seed, image=None):
                self.images.append(image)
                return [dict(probs=[.99,.01],entropy=.01,label_mass=1.,argmax_is_label=True)]
        engine=Capture()
        engine.decide(validate({'image':PNG,'mode':'separate','samples':1,
                               'questions':{'one':{'type':'noul'},'two':{'type':'noul'}}}))
        self.assertEqual(engine.images,[PNG,PNG])

    def test_image_transport_budget_and_mime_mismatch(self):
        with self.assertRaises(ValueError):
            encode_image(b'\x89PNG\r\n\x1a\n' + b'x' * MAX_IMAGE_BYTES)
        with self.assertRaises(ValueError):
            validate_image(PNG.replace('image/png', 'image/jpeg'))
        with self.assertRaises(ValueError):
            validate_image('data:image/png;base64,' + 'A' * (4*((MAX_IMAGE_BYTES+2)//3)+4))

    def test_retrieval_does_not_disable_model_knowledge_by_default(self):
        class Capture:
            def decide(self,schema,seed):
                self.schema=schema
                return {'answers':{'q':{'type':'noul','noul':1.}},'diagnostics':{}}
        engine=Capture()
        execute({'rag':{'number':40},'questions':{'q':{'type':'noul'}}},engine,
                Corpus([{'id':'n40','number':40,'title':'40','text':'reference'}]))
        self.assertEqual(len(engine.schema['questions']),1)
        self.assertNotIn('only',engine.schema['questions'][0]['instructions'])

    def test_sources_only_is_explicit_and_works_without_registered_rag(self):
        class Capture:
            def decide(self,schema,seed):
                self.schema=schema
                return {'answers':{'q':{'type':'noul','noul':1.},'_dg_supported':{'type':'noul','noul':.1}},'diagnostics':{}}
        engine=Capture()
        result=execute({'state':'no relevant information','sources_only':True,
                        'questions':{'q':{'type':'noul'}}},engine,Corpus([]))
        self.assertEqual(engine.schema['questions'][-1]['id'],'_dg_supported')
        self.assertIsNone(result['answers']['q'])
        with self.assertRaises(ValueError):
            execute({'sources_only':'yes','questions':{'q':{'type':'noul'}}},engine,Corpus([]))
