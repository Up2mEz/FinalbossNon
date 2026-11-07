"""One-off patch script: shuffle options, fill-in suggestions, stats and review queue. Safe to delete after it ran."""
p = 'tools/app.html'
s = open(p, encoding='utf-8').read()

def rep(old, new, cnt=1):
    global s
    assert s.count(old) == cnt, (s.count(old), old[:80])
    s = s.replace(old, new)

rep(r"""['start','exam','gate','results'].forEach(function(x){$(x).classList.toggle('hidden',x!==v)})""",
    r"""['start','exam','gate','results','stats'].forEach(function(x){$(x).classList.toggle('hidden',x!==v)})""")
rep('<section id="results" class="page hidden"></section>',
    '<section id="results" class="page hidden"></section>\n<section id="stats" class="page hidden"></section>')
rep('<div class="hist card pad"><h3>ประวัติการสอบ</h3><div id="hist" class="muted">ยังไม่มี</div></div>',
    '<div class="hist card pad"><div class="histhead"><h3>ประวัติการสอบ</h3><button class="btn sm" id="statsBtn" type="button">สถิติและคิวทบทวน</button></div><div id="hist" class="muted">ยังไม่มี</div></div>')

rep(r"""function save(){store(K('active'),S)}""", r"""function save(){store(K('active'),S)}
function makePerm(ids){var p={};ids.forEach(function(id){var it=ALL[id];if(it.kind!=='mcq')return;var a=['A','B','C','D'];p[id]=it.ns?a:shuffle(a)});return p}
function perm(id){return (S.perm&&S.perm[id])||['A','B','C','D']}
function dl(id,orig){return 'ABCD'.charAt(perm(id).indexOf(orig))}""")
rep(r"""limit:limit,submitted:false};save();enter();
}""", r"""limit:limit,submitted:false,perm:makePerm(ids)};save();enter();
}""")
rep(r"""cur:0,elapsed:0,t0:Date.now(),limit:0,submitted:false};save();enter();
}""", r"""cur:0,elapsed:0,t0:Date.now(),limit:0,submitted:false,perm:makePerm(wrong)};save();enter();
}""")

rep(r"""h+='<div class="opts" role="radiogroup" aria-label="ตัวเลือก">'+['A','B','C','D'].map(function(L){
      var c='opt',tag='';if(S.ans[id]===L)c+=' sel';
      if(locked){if(L===it.ans){c+=' right';tag='<span class="chip ok tag">ถูก</span>'}else if(S.ans[id]===L){c+=' wrong';tag='<span class="chip bad tag">ที่คุณเลือก</span>'}}
      return '<button type="button" class="'+c+'" data-l="'+L+'" role="radio" aria-checked="'+(S.ans[id]===L)+'"'+(locked?' disabled':'')+'><span class="L">'+L+'</span><span class="t">'+md(it.opts[L])+'</span>'+tag+'</button>'}).join('')+'</div>';""",
    r"""h+='<div class="opts" role="radiogroup" aria-label="ตัวเลือก">'+perm(id).map(function(O,i){var L='ABCD'.charAt(i);
      var c='opt',tag='';if(S.ans[id]===O)c+=' sel';
      if(locked){if(O===it.ans){c+=' right';tag='<span class="chip ok tag">ถูก</span>'}else if(S.ans[id]===O){c+=' wrong';tag='<span class="chip bad tag">ที่คุณเลือก</span>'}}
      return '<button type="button" class="'+c+'" data-l="'+O+'" role="radio" aria-checked="'+(S.ans[id]===O)+'"'+(locked?' disabled':'')+'><span class="L">'+L+'</span><span class="t">'+md(it.opts[O])+'</span>'+tag+'</button>'}).join('')+'</div>';""")
rep(r"""{S.ans[id]=m[k]||k;save();render()}""", r"""{S.ans[id]=perm(id)['ABCD'.indexOf(m[k]||k)];save();render()}""")
rep(r"""head=ok?'ถูก':(S.ans[id]?'ยังไม่ใช่ ที่ถูกคือ '+it.ans:'ไม่ได้ตอบ ที่ถูกคือ '+it.ans);""",
    r"""head=ok?'ถูก':(S.ans[id]?'ยังไม่ใช่ ที่ถูกคือ '+dl(id,it.ans):'ไม่ได้ตอบ ที่ถูกคือ '+dl(id,it.ans));""")
rep(r"""out+='<div class="opts">'+['A','B','C','D'].map(function(L){var c='opt',tag='';if(L===it.ans){c+=' right';tag='<span class="chip ok tag">ถูก</span>'}else if(S.ans[id]===L){c+=' wrong';tag='<span class="chip bad tag">ที่คุณเลือก</span>'}return '<div class="'+c+'"><span class="L">'+L+'</span><span class="t">'+md(it.opts[L])+'</span>'+tag+'</div>'}).join('')+'</div>""",
    r"""out+='<div class="opts">'+perm(id).map(function(O,i){var L='ABCD'.charAt(i),c='opt',tag='';if(O===it.ans){c+=' right';tag='<span class="chip ok tag">ถูก</span>'}else if(S.ans[id]===O){c+=' wrong';tag='<span class="chip bad tag">ที่คุณเลือก</span>'}return '<div class="'+c+'"><span class="L">'+L+'</span><span class="t">'+md(it.opts[O])+'</span>'+tag+'</div>'}).join('')+'</div>""")

rep(r"""function fillCard(it,id){var g=S.grade[id];""", r"""function norm(t){return String(t).toLowerCase().replace(/[,\s]/g,'').replace(/[–−]/g,'-')}
function nums(t){var m=String(t).replace(/,/g,'').match(/-?\d+(?:\.\d+)?/g);return m?m.map(Number):[]}
function suggest(it,ans){
  if(!ans||!String(ans).trim())return {v:0,why:'ยังไม่ได้ตอบ'};
  var parts=String(it.key).split(';').map(function(x){return x.trim()}).filter(Boolean),m=0,un=nums(ans);
  parts.forEach(function(p){var pn=nums(p),ok=false,letters=p.replace(/[0-9.,\s%-]/g,'');
    if(pn.length&&letters.length<=2){ok=pn.every(function(x){return un.some(function(y){return Math.abs(y-x)<=Math.max(1e-9,Math.abs(x)*0.005)})})}
    else{ok=norm(ans).indexOf(norm(p))>=0}
    if(ok)m++});
  return {v:m===parts.length?1:(m>0?0.5:0),why:'ตรง '+m+' จาก '+parts.length+' ช่อง'};
}
function fillCard(it,id){var g=S.grade[id],sg=suggest(it,S.ans[id]);""")
rep(r"""'<div class="grade" data-id="'+id+'"><button type="button" class="btn sm" data-g="1" aria-pressed="'+(g===1)+'">ถูก</button>""",
    r"""'<div class="sugg" data-id="'+id+'">ระบบเสนอ: <b>'+(sg.v===1?'ถูก':sg.v===0.5?'ถูกบางส่วน':'ผิด')+'</b> ('+sg.why+') <button type="button" class="btn sm" data-use="'+sg.v+'">ใช้ตามที่เสนอ</button></div><div class="grade" data-id="'+id+'"><button type="button" class="btn sm" data-g="1" aria-pressed="'+(g===1)+'">ถูก</button>""")
rep(r"""$('tutor').addEventListener('click',function(e){var g=e.target.closest('.grade button');if(g){S.grade[S.ids[S.cur]]=parseFloat(g.dataset.g);save();render()}});""",
    r"""$('tutor').addEventListener('click',function(e){var g=e.target.closest('.grade button'),u=e.target.closest('.sugg button');if(g){S.grade[S.ids[S.cur]]=parseFloat(g.dataset.g);save();render()}else if(u){S.grade[S.ids[S.cur]]=parseFloat(u.dataset.use);save();render()}});""")
rep(r"""  var g=e.target.closest('.grade button');if(!g)return;S.grade[g.parentNode.dataset.id]=parseFloat(g.dataset.g);""",
    r"""  var g=e.target.closest('.grade button'),u=e.target.closest('.sugg button');if(!g&&!u)return;var gid=(g||u).parentNode.dataset.id;S.grade[gid]=parseFloat(g?g.dataset.g:u.dataset.use);recordStat(gid);""")

rep(r"""function submit(auto){
  clearInterval(tick);S.submitted=true;S.auto=!!auto;S.elapsed=Math.floor((Date.now()-S.t0)/1000);
  var T=totals();""", r"""function recordStat(id){var it=ALL[id];if(it.kind==='fill'&&S.grade[id]===undefined)return;var st=store(K('qstat'))||{},e=st[id]||{n:0,w:0,l:null,t:0},prev=S.rec&&S.rec[id];
  if(prev!==undefined){e.n--;if(prev<1)e.w--}
  e.n++;var sv=sc(id);if(sv<1)e.w++;e.l=sv>=1?1:0;e.t=Date.now();st[id]=e;(S.rec=S.rec||{})[id]=sv;store(K('qstat'),st)}
function submit(auto){
  clearInterval(tick);S.submitted=true;S.auto=!!auto;S.elapsed=Math.floor((Date.now()-S.t0)/1000);S.ids.forEach(recordStat);
  var T=totals();""")

rep(r"""setSubject((store('mock-last-subject')||PACK.subjects[0].id));""", r"""/* stats and review queue */
function qstat(){return store(K('qstat'))||{}}
function queueIds(){var st=qstat();return Object.keys(st).filter(function(id){return ALL[id]&&st[id].l===0})}
function startQueue(){
  var ids=shuffle(queueIds()).slice(0,40);if(!ids.length)return;
  S={sid:SUBJ.id,set:'mix',lv:'retry',mode:'tutor',scope:'retry',ids:ids,ans:{},flags:{},rev:{},hint:{},grade:{},cur:0,elapsed:0,t0:Date.now(),limit:0,submitted:false,perm:makePerm(ids)};save();enter();
}
function showStats(){
  show('stats');var st=qstat(),ids=Object.keys(st).filter(function(id){return ALL[id]});
  var tot=0,wr=0,byU={},byL={},byT={};
  ids.forEach(function(id){var e=st[id],it=ALL[id];tot+=e.n;wr+=e.w;var u=byU[it.week]=byU[it.week]||{n:0,w:0},l=byL[it.level]=byL[it.level]||{n:0,w:0};u.n+=e.n;u.w+=e.w;l.n+=e.n;l.w+=e.w;var k=it.week+'|'+it.topic,t=byT[k]=byT[k]||{n:0,w:0,week:it.week,topic:it.topic};t.n+=e.n;t.w+=e.w});
  var q=queueIds().length,all=SUBJ.questions.length;
  var weakT=Object.keys(byT).map(function(k){return byT[k]}).filter(function(t){return t.w>0}).sort(function(a,b){return b.w/b.n-a.w/a.n||b.w-a.w}).slice(0,8);
  var h='<h2>สถิติของคุณ '+esc(SUBJ.title)+'</h2>';
  if(!tot){h+='<p class="muted" style="margin:8px 0 18px">ยังไม่มีข้อมูล ทำข้อสอบหนึ่งชุดแล้วกลับมาดู ระบบจะจำว่าข้อไหนผิดและสร้างคิวทบทวนให้</p>'}
  else{
    h+='<div class="stats" style="margin:14px 0 20px"><div class="stat"><b>'+tot+'</b><span>ครั้งที่ตอบ</span></div><div class="stat"><b>'+Math.round((tot-wr)/tot*100)+'%</b><span>ตอบถูก</span></div><div class="stat"><b>'+ids.length+'/'+all+'</b><span>ข้อที่เคยทำ</span></div><div class="stat"><b>'+q+'</b><span>ข้อในคิวทบทวน</span></div></div>';
    h+='<h3 style="font-size:1rem;margin-bottom:8px">ตามระดับ</h3><div class="lvres">'+Object.keys(byL).sort().map(function(l){var o=byL[l];return '<span class="chip n lv"><span class="meter" aria-hidden="true">'+[1,2,3,4].map(function(i){return '<i'+(i<=l?' class="on"':'')+'></i>'}).join('')+'</span>'+esc(lvName(+l))+' '+Math.round((o.n-o.w)/o.n*100)+'% ('+o.n+')</span>'}).join('')+'</div>';
    h+='<h3 style="font-size:1rem;margin:18px 0 8px">ตาม'+esc(UL)+'</h3>'+UNITS.filter(function(w){return byU[w]}).map(function(w){var o=byU[w],p=Math.round((o.n-o.w)/o.n*100);return '<div class="wkrow"'+hs(w)+'><span><span class="chip">'+esc(US)+w+'</span></span><div class="bar"><i style="width:'+Math.max(p,2)+'%"></i></div><span class="mono">'+p+'%</span></div>'}).join('');
    if(weakT.length)h+='<h3 style="font-size:1rem;margin:18px 0 8px">หัวข้อที่พลาดบ่อย</h3>'+weakT.map(function(t){return '<div class="weakrow"'+hs(t.week)+'><span class="chip">'+esc(US)+t.week+'</span><span class="wt">'+esc(t.topic)+'</span><span class="mono">พลาด '+t.w+' จาก '+t.n+'</span></div>'}).join('');
  }
  h+='<div style="display:flex;gap:10px;flex-wrap:wrap;margin-top:22px"><button class="btn primary" id="qBtn" type="button"'+(q?'':' disabled')+'>ทบทวนข้อที่เคยผิด ('+q+' ข้อ)</button><button class="btn" id="sBack" type="button">กลับ</button><button class="btn" id="sClear" type="button">ล้างสถิติ</button></div>';
  $('stats').innerHTML=h;
}
$('stats').addEventListener('click',function(e){
  if(e.target.closest('#qBtn'))startQueue();
  else if(e.target.closest('#sBack')){initStart();show('start')}
  else if(e.target.closest('#sClear')){var b=$('sClear');if(b.dataset.armed){store(K('qstat'),{});showStats()}else{b.dataset.armed='1';b.textContent='กดอีกครั้งเพื่อยืนยันการล้าง'}}
});
$('statsBtn').addEventListener('click',showStats);
setSubject((store('mock-last-subject')||PACK.subjects[0].id));""")
rep(r"""  var sv=store(K('active'));$('resumeBtn').classList.toggle('hidden',!(sv&&!sv.submitted));
  updatePreview();""", r"""  var sv=store(K('active'));$('resumeBtn').classList.toggle('hidden',!(sv&&!sv.submitted));
  var qn=queueIds().length;$('statsBtn').textContent='สถิติและคิวทบทวน'+(qn?' ('+qn+')':'');
  updatePreview();""")
# queueIds is defined after initStart in source order but is a function declaration: hoisted, fine.
open(p, 'w', encoding='utf-8').write(s)
print('app.html patched')

c = open('tools/app.css', encoding='utf-8').read()
c += """
/* v4: stats, suggestions */
.histhead{display:flex;justify-content:space-between;align-items:center;gap:12px;margin-bottom:10px;flex-wrap:wrap}
.histhead h3{margin:0}
.sugg{margin-top:10px;font-size:.9rem;display:flex;flex-wrap:wrap;gap:8px;align-items:center}
.weakrow{display:grid;grid-template-columns:auto minmax(0,1fr) auto;gap:12px;align-items:center;padding:8px 0;border-bottom:1px solid var(--line);font-size:.92rem}
.weakrow .wt{overflow-wrap:anywhere}
"""
open('tools/app.css', 'w', encoding='utf-8').write(c)

b = open('tools/build_html.py', encoding='utf-8').read()
if 'NOSHUF' not in b:
    b = b.replace("css = open(", """import re
NOSHUF = re.compile(r'ข้อ\\s*[A-D](?![A-Za-z])|(?:option|choice|ตัวเลือก)\\s*[A-D](?![A-Za-z])|(?<![A-Za-z0-9_(])[A-D](?:\\s*[,/และกับ]\\s*[A-D])+(?![A-Za-z0-9])|(?<![A-Za-z0-9_])[A-D]\\s*(?:ผิด|ถูก|เป็นคำตอบ|คือคำตอบ|ตัดทิ้ง)|(?:ทั้ง|ทุก|ไม่มี)\\s*(?:ข้อ|ตัวเลือก)|ข้อข้างต้น|\\([A-D]\\s*=\\s*[A-Z]|[A-D]\\s*=\\s*(?:Sequential|Routing|Supervisor|Debate)|\\([A-D]\\)|\\b[A-D]\\)')
for sj in subjects:
    n = 0
    for q in sj['questions']:
        if q['kind'] == 'mcq':
            blob = ' '.join([q['stem']] + list(q['opts'].values()) + [q.get('expl', '')])
            if NOSHUF.search(blob): q['ns'] = 1; n += 1
    print(sj['id'], 'options not shuffled (refer to letters):', n)
css = open(""", 1)
    open('tools/build_html.py', 'w', encoding='utf-8').write(b)
print('css and build patched')
