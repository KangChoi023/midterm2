import re,sys,json

def extract(path):
    s=open(path,encoding='utf8').read()
    i=s.index('const Q=')
    i=s.index('[',i)
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

def load(path):
    s,i=extract(path)
    return _unesc(P(s,i).val())

CIRC='①②③④⑤⑥⑦⑧'

def q_answer_text(q):
    ty=q.get('ty','mc'); a=q['a']
    if ty=='ox': return 'O (참)' if a else 'X (거짓)'
    if ty=='multi': return ', '.join(f"{CIRC[i]} {q['o'][i]}" for i in a)
    if ty=='match': return ' / '.join(f"{q['left'][k]} → {q['right'][v]}" for k,v in enumerate(a))
    if ty=='order': return ' → '.join(q['steps'][i] for i in a)
    return f"{CIRC[a]} {q['o'][a]}"

def q_body(q, n, md=True):
    ty=q.get('ty','mc'); lines=[]
    tag={'ox':'[O/X] ','multi':'[복수선택] ','match':'[짝짓기] ','order':'[순서] '}.get(ty,'')
    lines.append(f"**Q{n}.** {tag}{q['p']}")
    if q.get('c'): lines.append("```c\n"+q['c']+"\n```" if md else "```\n"+q['c']+"\n```")
    if ty in ('mc','multi'):
        for i,o in enumerate(q['o']): lines.append(f"- {CIRC[i]} {o}")
    elif ty=='match':
        lines.append('- 왼쪽: '+' / '.join(q['left']))
        lines.append('- 오른쪽: '+' / '.join(q['right']))
    elif ty=='order':
        lines.append('- 항목: '+' / '.join(q['steps']))
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
