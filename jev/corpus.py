"""Local HTTrack corpus extraction and Japanese character-bigram retrieval."""
import argparse
from collections import Counter
import hashlib
from html.parser import HTMLParser
import json
import math
from pathlib import Path
import re
import unicodedata
from urllib.parse import urljoin

class Page(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.active=False
        self.skip=False
        self.title=False
        self.title_parts=[]
        self.parts=[]
        self.images=[]
    def handle_starttag(self, tag, attrs):
        a=dict(attrs)
        if tag=='title':
            self.title=True
        if a.get('id')=='hpb-main':
            self.active=True
        if a.get('id')=='pagetop':
            self.skip=True
        if not self.active or self.skip:
            return
        if tag in {'p','div','li','br','tr','h1','h2','h3','h4','hr'}:
            self.parts.append('\n')
        if tag in {'h1','h2','h3','h4'}:
            self.parts.append('## ')
        if tag in {'td','th'}:
            self.parts.append(' | ')
        if tag=='img':
            self.images.append({'src':a.get('src',''),'alt':a.get('alt','')})
    def handle_endtag(self, tag):
        if tag=='title':
            self.title=False
        if self.active and not self.skip and tag in {'p','div','li','tr','h1','h2','h3','h4'}:
            self.parts.append('\n')
    def handle_comment(self, data):
        if data.strip()=='main end':
            self.active=False
    def handle_data(self, data):
        if self.title:
            self.title_parts.append(data)
        if self.active and not self.skip:
            self.parts.append(data)

def extract_page(html):
    parser=Page()
    parser.feed(html)
    lines=[' '.join(line.split()) for line in ''.join(parser.parts).splitlines()]
    return {'title':''.join(parser.title_parts).strip(),
            'text':'\n'.join(l for l in lines if l),'images':parser.images}

def split_chunks(doc, relative):
    text=doc['text']
    normalized=unicodedata.normalize('NFKC',text)
    if relative.startswith('japanese-digitalroot/'):
        matches=list(re.finditer(r'(?m)^数霊\s+(\d+)\s*$',normalized))
        if not matches:
            raise ValueError('Number table without number records: '+relative)
        return [{'number':int(m[1]),'title':'数霊 '+m[1],
                 'text':normalized[m.start():matches[i+1].start() if i+1<len(matches) else len(normalized)].strip()}
                for i,m in enumerate(matches)]
    parts=re.split(r'(?m)(?=^## )',text)
    out=[]
    for part in parts:
        if not part.strip():
            continue
        lines=part.strip().splitlines()
        title=lines[0].removeprefix('## ') if lines[0].startswith('## ') else doc['title']
        # Initial retrieval context budget: cap a passage at 1800 characters.
        # Long sections retain their section title and source, split on line boundaries.
        buf=''
        for line in lines:
            while len(line)>1800:
                if buf:
                    out.append({'number':None,'title':title,'text':buf.strip()})
                    buf=''
                out.append({'number':None,'title':title,'text':line[:1800]})
                line=line[1800:]
            if len(buf)+len(line)>1800 and buf:
                out.append({'number':None,'title':title,'text':buf.strip()})
                buf=''
            buf+=line+'\n'
        if buf.strip():
            out.append({'number':None,'title':title,'text':buf.strip()})
    return out

def build(source, output):
    source=Path(source).resolve()
    output=Path(output)
    if output.resolve().is_relative_to(source):
        raise ValueError('Keep derived corpus outside original archive')
    site=source/'www.surei.net'
    docs=[]
    seen=set()
    stats={'pages':0,'duplicates':0,'images':0,'empty_pages':[]}
    for path in sorted(site.rglob('*.html')):
        raw=path.read_bytes()
        encoding=re.search(br'charset=["\x27]?([\w-]+)',raw[:2000],re.I)
        codec=encoding[1].decode() if encoding else 'cp932'
        if codec.lower().replace('-','_')=='shift_jis':
            codec='cp932'
        html=raw.decode(codec,errors='strict')
        doc=extract_page(html)
        stats['pages']+=1
        if not doc['text']:
            if 'id="hpb-main"' not in html:
                raise ValueError('Main marker missing: '+str(path))
            stats['empty_pages'].append(path.relative_to(site).as_posix())
            continue
        key=hashlib.sha256(doc['text'].encode()).hexdigest()
        if key in seen:
            stats['duplicates']+=1
            continue
        seen.add(key)
        relative=path.relative_to(site).as_posix()
        mirrored=re.search(r'Mirrored from (.*?) by HTTrack.*?\], (.*?) -->',html)
        url='http://'+mirrored[1] if mirrored else 'http://www.surei.net/'+relative
        archive_time=mirrored[2] if mirrored else None
        images=[]
        for image in doc['images']:
            target=(path.parent/image['src']).resolve()
            if not target.is_relative_to(source) or not target.is_file():
                raise ValueError('Missing or external local image: '+str(target))
            images.append({**image,'archive_path':target.relative_to(source).as_posix(),
                           'url':urljoin(url,image['src']), 'text_extracted':False})
        stats['images']+=len(images)
        for index,chunk in enumerate(split_chunks(doc,relative)):
            item={**chunk,'id':relative+'#'+str(index),'source_url':url,
                  'archive_path':path.relative_to(source).as_posix(),
                  'archived_at_source_timezone':archive_time,
                  'source_sha256':hashlib.sha256(raw).hexdigest(),'images':images}
            docs.append(item)
    numbers=[d['number'] for d in docs if d['number'] is not None]
    if sorted(numbers)!=list(range(1,92)):
        raise ValueError('Expected exactly one record for each number 1..91')
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(''.join(json.dumps(d,ensure_ascii=False)+'\n' for d in docs),encoding='utf-8')
    stats.update(chunks=len(docs),numbers=len(numbers),
                 corpus_sha256=hashlib.sha256(output.read_bytes()).hexdigest())
    output.with_suffix('.manifest.json').write_text(json.dumps(stats,indent=2)+'\n',encoding='utf-8')
    return stats

def terms(text):
    text=unicodedata.normalize('NFKC',text).lower()
    result=re.findall(r'[a-z0-9]+',text)
    for run in re.findall(r'[\u3040-\u30ff\u3400-\u9fff]+',text):
        result.extend(run[i:i+2] for i in range(len(run)-1))
        if len(run)==1:
            result.append(run)
    return Counter(result)

class Corpus:
    def __init__(self, docs):
        self.docs=docs
        self.counts=[terms(d['title']+' '+d['text']) for d in docs]
        self.df=Counter(t for count in self.counts for t in count)
        self.lengths=[sum(c.values()) for c in self.counts]
        self.mean=sum(self.lengths)/max(len(self.lengths),1)
    @classmethod
    def load(cls,path):
        return cls([json.loads(line) for line in Path(path).read_text(encoding='utf-8').splitlines()])
    def search(self,query,number=None,limit=3):
        if number is not None:
            return [d for d in self.docs if d.get('number')==number][:limit]
        query_terms=terms(query)
        if not query_terms:
            return []
        scored=[]
        for i,count in enumerate(self.counts):
            score=0.
            for term in query_terms:
                tf=count.get(term,0)
                if tf:
                    idf=math.log(1+(len(self.docs)-self.df[term]+.5)/(self.df[term]+.5))
                    score+=idf*tf*2.2/(tf+1.2*(.25+.75*self.lengths[i]/max(self.mean,1)))
            if score>0:
                scored.append((score,i))
        scored.sort(key=lambda x:(-x[0],x[1]))
        return [{**self.docs[i],'retrieval_score':round(s,5)} for s,i in scored[:limit]]

if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('source')
    p.add_argument('output')
    a=p.parse_args()
    print(json.dumps(build(a.source,a.output),ensure_ascii=False,indent=2))
