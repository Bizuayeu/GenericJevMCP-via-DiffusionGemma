import threading
import unittest
import json
from urllib.error import HTTPError
from urllib.request import Request,urlopen
from http.server import ThreadingHTTPServer
from jev.server import execute,make_handler
from jev.corpus import Corpus

class Fake:
    upstream='http://127.0.0.1:1'
    def decide(self,schema,seed):
        return {'answers':{q['id']:{'type':'noul','noul':1.} for q in schema['questions']},'diagnostics':{'reads':1}}
class ApiTests(unittest.TestCase):
    def test_no_retrieval_match_never_calls_model(self):
        class NoCall:
            def decide(self,*args): raise AssertionError('must not call')
        result=execute({'state':'zzzyyxxx','rag':{},'questions':{'q':{'type':'noul'}}},NoCall(),Corpus([]))
        self.assertEqual(result['abstention'],'no_retrieval_match')
        self.assertIsNone(result['answers']['q'])
    def test_grounding_rejection_hides_answers(self):
        class Unsupported(Fake):
            def decide(self,schema,seed):
                r=super().decide(schema,seed)
                r['answers']['_dg_supported']['noul']=.1
                return r
        corpus=Corpus([{'id':'n40','number':40,'title':'40','text':'無敵運'}])
        r=execute({'state':'数霊40','rag':{'number':40},'sources_only':True,'questions':{'q':{'type':'noul'}}},Unsupported(),corpus)
        self.assertIsNone(r['answers']['q'])
        self.assertEqual(r['sources'][0]['id'],'n40')
    def test_auth_and_bad_input_over_http(self):
        http=ThreadingHTTPServer(('127.0.0.1',0),make_handler(Fake(),Corpus([]),'secret-client','secret-backend'))
        worker=threading.Thread(target=http.serve_forever,daemon=True)
        worker.start()
        try:
            url='http://127.0.0.1:'+str(http.server_port)+'/v1/systemone'
            for header,body,expected in [
                ({},b'{}',401),
                ({'Authorization':'Bearer secret-client'},b'{"state":NaN}',422),
                ({'Authorization':'Bearer secret-client'},b'{"state":1,"state":2}',422)]:
                with self.assertRaises(HTTPError) as caught:
                    urlopen(Request(url,data=body,headers=header),timeout=2)
                self.assertEqual(caught.exception.code,expected)
        finally:
            http.shutdown()
            http.server_close()
            worker.join()
