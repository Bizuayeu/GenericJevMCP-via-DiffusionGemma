import math
import os
import unittest
from pathlib import Path
from decision import distribution, validate, template_for, aggregate
from corpus import extract_page, split_chunks, Corpus

class DecisionTests(unittest.TestCase):
    def test_ambiguous_labels_have_normalized_entropy(self):
        d = distribution({1: math.log(.0005), 2: math.log(.0005), 3: math.log(.999)}, [1,2])
        self.assertAlmostEqual(d['entropy'], math.log(2))
        self.assertAlmostEqual(d['label_mass'], .001)
        self.assertFalse(d['argmax_is_label'])
    def test_missing_and_nonfinite_labels_fail(self):
        for row in [{1: -.1}, {1: -math.inf, 2: 0}, {1: float('nan'), 2: 0}]:
            with self.assertRaises(ValueError):
                distribution(row,[1,2])
    def test_duplicates_and_unbounded_samples_fail(self):
        base={'state':'x','questions':{'x':{'type':'choice','criteria':{'A':None,'B':None}}}}
        for v in [0,33,True,float('nan')]:
            with self.assertRaises(ValueError):
                validate(dict(base,samples=v))
        with self.assertRaises(ValueError):
            validate({'state':'x','questions':{'x':{'type':'score','criteria':['same','same']}}})
    def test_score_is_zero_based(self):
        q=validate({'state':'x','samples':1,'questions':{'s':{'type':'score','criteria':['low','high']}}})['questions'][0]
        out=aggregate(q,[{'probs':[.25,.75],'entropy':.5,'label_mass':1,'argmax_is_label':True}])
        self.assertEqual(out['score'],.75)
        self.assertEqual(out['legend'],{'0':'low','1':'high'})
    def test_template_uses_actual_tokenizer(self):
        from tokenizers import Tokenizer
        tok=Tokenizer.from_file(os.environ.get('DG_TEST_TOKENIZER', str(Path(__file__).parent/'tokenizer.json')))
        qs=validate({'state':'x','samples':1,'questions':{'根拠':{'type':'noul'},'kind':{'type':'choice','criteria':{'数霊':None,'手相':None}}}})['questions']
        template,slots=template_for(qs,tok,256)
        self.assertEqual(len(slots),2)
        self.assertTrue(all(len(s['ids'])==2 for s in slots))

class CorpusTests(unittest.TestCase):
    def test_extract_preserves_text_and_images_but_removes_navigation(self):
        html='<title>題</title><div id="hpb-main"><h3>節</h3><ul><li>数霊 ４０<li>運勢 無敵運</ul><p>本文</p><img src="a.jpg" alt="図"><div id="pagetop">戻る</div></div><!-- main end --><nav>広告</nav>'
        doc=extract_page(html)
        self.assertIn('数霊 ４０',doc['text'])
        self.assertNotIn('広告',doc['text'])
        self.assertNotIn('戻る',doc['text'])
        self.assertEqual(doc['images'][0]['src'],'a.jpg')
    def test_number_records_do_not_bleed(self):
        chunks=split_chunks({'text':'1〜10\n数霊 １\n太陽\n数霊 2\n月','title':'数霊表1〜10'},'japanese-digitalroot/1-10.html')
        self.assertEqual([c['number'] for c in chunks],[1,2])
        self.assertNotIn('月',chunks[0]['text'])
    def test_unknown_number_and_unmatched_query_abstain(self):
        c=Corpus([{'id':'a','number':40,'text':'無敵運 カリスマ','title':'40'}])
        self.assertEqual(c.search('irrelevant xyz'),[])
        self.assertEqual(c.search('',number=99),[])
        self.assertEqual(c.search('',number=40)[0]['id'],'a')

if __name__=='__main__':
    unittest.main()
