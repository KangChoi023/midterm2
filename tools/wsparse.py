import re,sys,json

Q_RE=re.compile(r'const Q\s*=\s*\[')

BASE_RE=re.compile(r'const BASE\s*=\s*\[')
EXTRA_RE=re.compile(r'const EXTRA\s*=\s*\[')

def supported(path):
    """이 파서가 읽는 형식(const Q=[…] 또는 Python Day1 의 BASE + EXTRA) 인가"""
    s=open(path,encoding='utf8').read()
    return bool(Q_RE.search(s) or BASE_RE.search(s))

def extract(path):
    s=open(path,encoding='utf8').read()
    i=Q_RE.search(s).end()-1
    return s,i

class P:
    def __init__(self,s,i): self.s=s; self.i=i
    def ws(self):
        while True:
            while self.s[self.i] in ' \t\r\n': self.i+=1
            if self.s.startswith('//',self.i):
                while self.s[self.i]!='\n': self.i+=1
            elif self.s.startswith('/*',self.i):
                self.i=self.s.index('*/',self.i)+2
            else: break
    def val(self):
        self.ws(); c=self.s[self.i]
        if c=='[':
            self.i+=1; out=[]
            while True:
                self.ws()
                if self.s[self.i]==']': self.i+=1; return out
                out.append(self.val()); self.ws()
                if self.s[self.i]==',': self.i+=1
        if c=='{':
            self.i+=1; out={}
            while True:
                self.ws()
                if self.s[self.i]=='}': self.i+=1; return out
                m=re.compile(r"[A-Za-z_$][\w$]*|'[^']*'|\"[^\"]*\"").match(self.s,self.i)
                k=m.group(0).strip('\'"'); self.i=m.end(); self.ws()
                assert self.s[self.i]==':',(self.s[self.i-20:self.i+20])
                self.i+=1
                out[k]=self.val(); self.ws()
                if self.s[self.i]==',': self.i+=1
        if c in "'\"`":
            q=c; self.i+=1; buf=[]
            while self.s[self.i]!=q:
                ch=self.s[self.i]
                if ch=='\\':
                    n=self.s[self.i+1]; self.i+=2
                    buf.append({'n':'\n','t':'\t','\\':'\\',"'":"'",'"':'"','`':'`'}.get(n,n))
                else: buf.append(ch); self.i+=1
            self.i+=1; return ''.join(buf)
        m=re.compile(r"-?\d+(\.\d+)?|true|false|null").match(self.s,self.i)
        t=m.group(0); self.i=m.end()
        return {'true':True,'false':False,'null':None}.get(t, float(t) if '.' in t else int(t) if t[0] in '-0123456789' else t)

import html as _html
def _unesc(x):
    if isinstance(x,str): return _html.unescape(x)
    if isinstance(x,list): return [_unesc(v) for v in x]
    if isinstance(x,dict): return {k:_unesc(v) for k,v in x.items()}
    return x

def _plain(x):
    """보기·짝·순서 칸은 화면에서 글자로 찍히므로 html 태그를 벗긴다 (<code>Ctrl</code> → Ctrl)"""
    import html as _h
    return _h.unescape(re.sub(r'<[^>]+>','',str(x)))

def _safe_svg(svg):
    """워크시트 그림(SVG 조각)을 위키에 그대로 넣기 전에 스크립트 · on* 속성 · javascript: 링크를 지운다"""
    svg=re.sub(r'<script\b[^>]*>.*?</script\s*>','',svg,flags=re.S|re.I)
    svg=re.sub(r'<(foreignObject|iframe|object|embed)\b[^>]*>.*?</\1\s*>','',svg,flags=re.S|re.I)
    svg=re.sub(r'<(foreignObject|iframe|object|embed)\b[^>]*/>','',svg,flags=re.I)
    svg=re.sub(r'\son\w+\s*=\s*("[^"]*"|\'[^\']*\'|[^\s>]+)','',svg,flags=re.I)
    return re.sub(r'(href\s*=\s*["\']?)\s*javascript:','\\1#',svg,flags=re.I)

def _norm(q, seed):
    """NN 형 워크시트 문항({type: single|multi|ox|short|drop|match|order|code|hot}) → 위키 형식({ty: mc|multi|ox|match|order|short|hot}).
    hot(그림에서 정답 칸 클릭)은 그림 · 클릭 영역을 그대로 옮기고 a = 정답 영역 번호. drop·code 는 정답 보기만 되는 short(ro=1)."""
    t=q.get('type')
    if not t: return q
    base={k:q[k] for k in ('p','w','an') if k in q}
    if t=='single': return dict(base,o=[_plain(x) for x in q['o']],a=q['a'])
    if t=='multi': return dict(base,ty='multi',o=[_plain(x) for x in q['o']],a=list(q['a']))
    if t=='ox': return dict(base,ty='ox',a=1 if q['a'] else 0)
    if t=='match': return dict(base,ty='match',left=[_plain(x) for x in q['left']],right=[_plain(x) for x in q['right']],a=list(q['a']))
    if t=='order':                                           # items 는 정답 순서 → 고정된 섞기로 보여 준다
        n=len(q['items']); perm=sorted(range(n),key=lambda k:((k*7+seed*3)%n, k))
        if perm==list(range(n)): perm=perm[::-1]
        return dict(base,ty='order',steps=[_plain(q['items'][k]) for k in perm],a=[perm.index(k) for k in range(n)])
    if t=='short': return dict(base,ty='short',a=[str(x) for x in q['a']])
    if t=='hot':
        # 사각형 {x, y, w, h} 또는 원 {shape: circle, x, y, r}
        regs=[{k:r[k] for k in ('shape','x','y','w','h','r') if k in r} | {'label':_plain(r.get('label',''))} for r in q['regions']]
        ids=[r['id'] for r in q['regions']]
        if q['a'] not in ids: return None
        return dict(base,ty='hot',svg=_safe_svg(q['svg']),vb=q['viewBox'],sw=q.get('sw'),regions=regs,a=ids.index(q['a']))
    if t=='drop':
        it=iter(q['opts'])
        p=re.sub(r'\{\}',lambda m:'[ '+' / '.join(next(it))+' ]',q['p2'])
        return dict(base,ty='short',p=p,a=[' · '.join(o[k] for o,k in zip(q['opts'],q['a']))],ro=1)
    if t=='code':
        parts=q['code'].split('{{}}')
        code=''.join(x+('____' if k<len(parts)-1 else '') for k,x in enumerate(parts))
        full=''.join(x+(q['a'][k][0] if k<len(parts)-1 else '') for k,x in enumerate(parts))
        return dict(base,ty='short',c=code,a=[full],ro=1)
    return None

def load(path):
    s=open(path,encoding='utf8').read()
    if not Q_RE.search(s) and BASE_RE.search(s):          # 교시별 기본 문항 BASE + 교시별 추가 문항 EXTRA
        out=_unesc(P(s,BASE_RE.search(s).end()-1).val())
        m=EXTRA_RE.search(s)
        extra=_unesc(P(s,m.end()-1).val()) if m else []
        for les,more in zip(out,extra): les['qs']=les.get('qs',[])+more
    else:
        s,i=extract(path)
        out=_unesc(P(s,i).val())
    for les in out:
        les['qs']=[x for x in (_norm(q,k) for k,q in enumerate(les.get('qs',[]))) if x]
    return out

CIRC='①②③④⑤⑥⑦⑧'

def q_answer_text(q):
    ty=q.get('ty','mc'); a=q['a']
    if ty=='ox': return 'O (참)' if a else 'X (거짓)'
    if ty=='multi': return ', '.join(f"{CIRC[i]} {q['o'][i]}" for i in a)
    if ty=='match': return ' / '.join(f"{q['left'][k]} → {q['right'][v]}" for k,v in enumerate(a))
    if ty=='order': return ' → '.join(q['steps'][i] for i in a)
    if ty=='hot': return q['regions'][a]['label']
    if ty=='short': return a[0]
    return f"{CIRC[a]} {q['o'][a]}"

def q_body(q, n, md=True):
    ty=q.get('ty','mc'); lines=[]
    tag={'ox':'[O/X] ','multi':'[복수선택] ','match':'[짝짓기] ','order':'[순서] ','hot':'[그림 클릭] '}.get(ty,'')
    lines.append(f"**Q{n}.** {tag}{q['p']}")
    if q.get('c'): lines.append("```c\n"+q['c']+"\n```" if md else "```\n"+q['c']+"\n```")
    if ty in ('mc','multi'):
        for i,o in enumerate(q['o']): lines.append(f"- {CIRC[i]} {o}")
    elif ty=='match':
        lines.append('- 왼쪽: '+' / '.join(q['left']))
        lines.append('- 오른쪽: '+' / '.join(q['right']))
    elif ty=='order':
        lines.append('- 항목: '+' / '.join(q['steps']))
    elif ty=='hot':
        lines.append('- 클릭할 칸: '+' / '.join(r['label'] for r in q['regions']))
    lines.append(f"- **정답**: {q_answer_text(q)}")
    lines.append(f"- 해설: {q['w']}")
    return '\n'.join(lines)

def render(path):
    Q=load(path); out=[]
    for si,sec in enumerate(Q,1):
        out.append(f"## {sec.get('t')}  [{sec.get('anim','')}]")
        for qi,q in enumerate(sec['qs'],1):
            out.append(q_body(q,qi,md=False)); out.append('')
    return '\n'.join(out)

def render_section(path, sec_no):
    Q=load(path); sec=Q[sec_no-1]
    return '\n\n'.join(q_body(q,qi,md=True) for qi,q in enumerate(sec['qs'],1))

if __name__=='__main__':
    print(render(sys.argv[1]))
