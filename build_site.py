import re, sys, pathlib, html
import markdown

HERE = pathlib.Path(__file__).resolve().parent
SRC = HERE / "content"
OUT = HERE / "site"
OUT.mkdir(exist_ok=True)
weeks = range(8, 15)
FIGS = SRC / "figs"

CSS = """
/* Layout: sticky week tabs on top, content column with a sticky outline on wide screens */
:root{
  --bg:#f5f7fb; --surface:#ffffff; --fg:#17233b; --muted:#5a6781; --line:#d9e0ee;
  --accent:#1f4fa3; --accent-soft:#e3ecfb; --must:#8a4b00; --must-bg:#fff1d6; --code-bg:#edf1f8; --ext:#7a3fa0; --ext-bg:#f3e9fa;
  --font-body:'Noto Sans Thai','Sarabun',system-ui,sans-serif; --font-mono:'JetBrains Mono',ui-monospace,Consolas,monospace;
}
@media (prefers-color-scheme: dark){:root:not([data-theme="light"]){
  --bg:#0e1524; --surface:#162036; --fg:#e6ebf6; --muted:#9aa8c4; --line:#2a3957;
  --accent:#7fb0ff; --accent-soft:#1d2d4d; --must:#ffcf7a; --must-bg:#3a2c10; --code-bg:#101a2e; --ext:#d2a8f0; --ext-bg:#2a1d38; color-scheme:dark}}
:root[data-theme="dark"]{
  --bg:#0e1524; --surface:#162036; --fg:#e6ebf6; --muted:#9aa8c4; --line:#2a3957;
  --accent:#7fb0ff; --accent-soft:#1d2d4d; --must:#ffcf7a; --must-bg:#3a2c10; --code-bg:#101a2e; --ext:#d2a8f0; --ext-bg:#2a1d38; color-scheme:dark}
html{scroll-behavior:smooth}
body{background:var(--bg);color:var(--fg);font-family:var(--font-body);font-size:16px;line-height:1.75;margin:0}
.top{position:sticky;top:env(safe-area-inset-top,0px);z-index:20;background:var(--surface);border-bottom:1px solid var(--line)}
body::before{content:"";position:fixed;inset:0 0 auto;height:env(safe-area-inset-top,0px);z-index:30;background:var(--surface)}
.top-in{max-width:1100px;margin:0 auto;padding:10px max(16px,env(safe-area-inset-right,0px)) 10px max(16px,env(safe-area-inset-left,0px));display:flex;align-items:center;gap:12px;flex-wrap:wrap}
.brand{font-weight:700;color:var(--accent);margin-right:auto;font-size:15px}
.tabs{display:flex;gap:6px;flex-wrap:wrap}
.tabs button{font:inherit;font-size:14px;border:1px solid var(--line);background:transparent;color:var(--fg);padding:5px 12px;border-radius:6px;cursor:pointer}
.tabs button[aria-selected="true"]{background:var(--accent);border-color:var(--accent);color:var(--bg)}
.tabs button:focus-visible,.toc a:focus-visible{outline:2px solid var(--accent);outline-offset:2px}
.wrap{max-width:1100px;margin:0 auto;padding-block:24px 64px;padding-left:max(16px,env(safe-area-inset-left,0px));padding-right:max(16px,env(safe-area-inset-right,0px));display:grid;grid-template-columns:minmax(0,1fr);gap:24px}
@media(min-width:960px){.wrap{grid-template-columns:230px minmax(0,1fr)}}
.toc{display:none}
@media(min-width:960px){.toc{display:block;position:sticky;top:76px;align-self:start;max-height:calc(100vh - 100px);overflow:auto;font-size:14px;line-height:1.5}}
.toc a{display:block;color:var(--muted);text-decoration:none;padding:5px 8px;border-left:2px solid var(--line)}
.toc a:hover{color:var(--accent);border-left-color:var(--accent)}
.toc h4{margin:0 0 8px;font-size:12px;letter-spacing:.08em;text-transform:uppercase;color:var(--muted)}
article{min-width:0;background:var(--surface);border:1px solid var(--line);border-radius:10px;padding:clamp(16px,4vw,36px)}
article h1{font-size:clamp(24px,5vw,32px);line-height:1.3;margin:0 0 .5em;text-wrap:balance}
article h2{font-size:clamp(20px,3.5vw,24px);margin:2em 0 .6em;padding-top:.3em;border-top:1px solid var(--line);text-wrap:balance;scroll-margin-top:80px}
article h3{font-size:18px;margin:1.6em 0 .4em;color:var(--accent);scroll-margin-top:80px}
article p,article li{max-width:75ch}
article ul,article ol{padding-left:1.4em}
article li{margin:.2em 0}
code{font-family:var(--font-mono);font-size:.88em;background:var(--code-bg);padding:.1em .35em;border-radius:4px}
pre{background:var(--code-bg);padding:14px 16px;border-radius:8px;overflow-x:auto;line-height:1.5}
pre code{background:none;padding:0}
.tw{overflow-x:auto;margin:1em 0}
table{border-collapse:collapse;font-size:14.5px;line-height:1.5;min-width:100%}
th,td{border:1px solid var(--line);padding:7px 10px;text-align:left;vertical-align:top}
th{background:var(--accent-soft)}
td code,th code{white-space:nowrap}
strong.must{background:var(--must-bg);color:var(--must);padding:.05em .4em;border-radius:4px}
blockquote{border-left:4px solid var(--accent);background:var(--accent-soft);margin:1em 0;padding:.5em 1em;border-radius:0 8px 8px 0}
blockquote p{margin:.35em 0}
details.fold{border:1px solid var(--line);border-radius:8px;margin:1.4em 0;padding:0 14px;background:var(--bg)}
details.fold summary{cursor:pointer;padding:10px 0;list-style:none}
details.fold summary::-webkit-details-marker{display:none}
details.fold summary::before{content:"▸ ";color:var(--accent)}
details.fold[open] summary::before{content:"▾ "}
details.fold summary h2{display:inline;border:0;margin:0;padding:0;font-size:clamp(17px,3vw,21px)}
details.fold[open]{padding-bottom:10px}
p.ext,li.ext{background:var(--ext-bg);border-left:3px solid var(--ext);padding:.25em .6em;border-radius:0 6px 6px 0}
tr.ext td{background:var(--ext-bg)}
.extbadge{display:inline-block;font-size:.78em;font-weight:700;color:var(--ext);border:1px solid var(--ext);border-radius:4px;padding:0 .4em;margin:0 .15em;white-space:nowrap}
.assume{border-bottom:1px dotted var(--muted)}
.legend{font-size:14px;color:var(--muted);margin:0 0 14px;padding:8px 12px;border:1px solid var(--line);border-radius:8px;background:var(--surface)}
main{min-width:0;max-width:100%}
article a{overflow-wrap:anywhere}
article{overflow-wrap:break-word}
pre{max-width:100%}
.figin{max-width:100%}
figure.fig{margin:1.4em 0;padding:0;max-width:100%}
.figin{overflow-x:auto;background:var(--bg);border:1px solid var(--line);border-radius:8px;padding:10px}
.figin svg{display:block;width:100%;min-width:600px;max-width:820px;height:auto;margin:0 auto}
figcaption{font-size:14px;line-height:1.55;color:var(--muted);margin-top:8px;max-width:75ch}
svg text{font-family:var(--font-body);fill:var(--fg);font-size:13px}
svg text.s{font-size:11.5px;fill:var(--muted)}
svg text.h{font-size:14px;font-weight:700}
svg text.ac{fill:var(--accent);font-weight:700}
svg text.wn{fill:var(--must);font-weight:700}
svg .box{fill:var(--surface);stroke:var(--fg);stroke-width:1.4}
svg .acc{fill:var(--accent-soft);stroke:var(--accent);stroke-width:1.6}
svg .warn{fill:var(--must-bg);stroke:var(--must);stroke-width:1.6}
svg .mut{fill:var(--code-bg);stroke:var(--line);stroke-width:1.2}
svg .ln{stroke:var(--fg);stroke-width:1.4;fill:none}
svg .lna{stroke:var(--accent);stroke-width:1.8;fill:none}
svg .lnw{stroke:var(--must);stroke-width:1.8;fill:none}
svg .dash{stroke-dasharray:5 4}
svg .ah{fill:var(--fg);stroke:none}
svg .aha{fill:var(--accent);stroke:none}
svg .ahw{fill:var(--must);stroke:none}
section.week{display:none}
section.week.on{display:block}
hr{border:0;border-top:1px solid var(--line);margin:2em 0}
.theme{font:inherit;font-size:13px;border:1px solid var(--line);background:transparent;color:var(--muted);padding:5px 10px;border-radius:6px;cursor:pointer}
@media(prefers-reduced-motion:reduce){html{scroll-behavior:auto}}
"""

JS = """
(function(){
  var secs=[].slice.call(document.querySelectorAll('section.week'));
  var tabs=[].slice.call(document.querySelectorAll('.tabs button'));
  var toc=document.getElementById('toc');
  function show(id){
    var found=false;
    secs.forEach(function(s){var on=s.id===id; s.classList.toggle('on',on); if(on)found=true;});
    if(!found){id=secs[0].id; secs[0].classList.add('on');}
    tabs.forEach(function(t){t.setAttribute('aria-selected',t.dataset.id===id?'true':'false');});
    var s=document.getElementById(id), h=s.querySelectorAll('h2'), out='<h4>หัวข้อในสัปดาห์นี้</h4>';
    h.forEach(function(x,i){x.id=id+'-s'+i; out+='<a href="#'+x.id+'" data-sec="'+x.id+'">'+x.textContent+'</a>';});
    toc.innerHTML=out;
    document.title=s.dataset.title+' | NLP Final Study';
  }
  function route(){var h=location.hash.replace('#','');var base=h.split('-s')[0];show(base);
    if(h.indexOf('-s')>0){go(document.getElementById(h));} else window.scrollTo(0,0);}
  function go(el){if(!el)return;var d=el.closest('details');if(d)d.open=true;el.scrollIntoView({behavior:'auto',block:'start'});}
  tabs.forEach(function(t){t.addEventListener('click',function(){location.hash=t.dataset.id;});});
  toc.addEventListener('click',function(e){var a=e.target.closest('a'); if(!a)return; e.preventDefault(); go(document.getElementById(a.dataset.sec));});
  window.addEventListener('hashchange',route); route();
  var tb=document.getElementById('theme');
  tb.addEventListener('click',function(){var r=document.documentElement,c=r.getAttribute('data-theme');
    var dark=c? c==='dark' : window.matchMedia('(prefers-color-scheme: dark)').matches;
    r.setAttribute('data-theme',dark?'light':'dark');});
})();
"""

def fix_lists(text):
    """python-markdown needs a blank line before a list that follows a paragraph."""
    out, infence, prev = [], False, ""
    li = re.compile(r"^([-*]|\d+\.)\s+")
    for line in text.split("\n"):
        if line.strip().startswith("```"):
            infence = not infence
        if (not infence and li.match(line) and prev.strip() and not li.match(prev)
                and not prev[:1].isspace() and not prev.lstrip().startswith(("|", ">"))):
            out.append("")
        out.append(line)
        prev = line
    return "\n".join(out)


FOLD_KEYS = ("คลังสูตร", "ชุดฝึกคำนวณ", "คู่ที่มักสับสน")


def fold(body):
    """Collapse the long reference sections so the main lesson reads first."""
    out = []
    for p in re.split(r"(?=<h2>)", body):
        m = re.match(r"<h2>(.*?)</h2>", p, flags=re.S)
        if m and any(k in m.group(1) for k in FOLD_KEYS):
            out.append('<details class="fold"><summary><h2>%s</h2></summary>%s</details>' % (m.group(1), p[m.end():]))
        else:
            out.append(p)
    return "".join(out)


EXT_RE = re.compile(r"\((ต่อยอดจากสไลด์|นอกสไลด์)[^)<]*\)")


def mark_ext(body):
    """Highlight everything that goes beyond the slides (labelled by the authors)."""
    def badge(m):
        return '<span class="extbadge">%s</span>' % m.group(0)[1:-1]

    def block(tag):
        pat = re.compile(r"<%s>((?:(?!<%s>|</%s>).)*?)</%s>" % (tag, tag, tag, tag), re.S)

        def f(m):
            inner = m.group(1)
            if EXT_RE.search(inner):
                return '<%s class="ext">%s</%s>' % (tag, inner, tag)
            return m.group(0)
        return pat
    for tag in ("p", "li", "tr"):
        pat = block(tag)
        body = pat.sub(lambda m, t=tag: ('<%s class="ext">%s</%s>' % (t, m.group(1), t)) if EXT_RE.search(m.group(1)) else m.group(0), body)
    body = EXT_RE.sub(badge, body)
    body = body.replace("ตัวเลขสมมติ", '<span class="assume">ตัวเลขสมมติ</span>')
    return body


md = markdown.Markdown(extensions=["tables", "fenced_code", "sane_lists"])
sections, tabs = [], []
for n in weeks:
    p = SRC / f"week{n:02d}.md"
    if not p.exists():
        print("missing", p)
        continue
    text = p.read_text(encoding="utf-8")
    figs = []

    def grab(m):
        name, cap = m.group(1).strip(), m.group(2).strip()
        f = FIGS / f"{name}.svg"
        if not f.exists():
            print("missing fig", name)
            return ""
        svg = re.sub(r"<\?xml.*?\?>", "", f.read_text(encoding="utf-8")).strip()
        svg = svg.replace("<svg", '<svg role="img" aria-label="%s"' % html.escape(cap, quote=True), 1)
        figs.append('<figure class="fig"><div class="figin">%s</div><figcaption>%s</figcaption></figure>' % (svg, html.escape(cap)))
        return "\n\n@@FIG%d@@\n\n" % (len(figs) - 1)

    text = re.sub(r"^\{\{fig:([^|}]+)\|([^}]+)\}\}[ \t]*$", grab, text, flags=re.M)
    md.reset()
    body = md.convert(fix_lists(text))
    for i, fg in enumerate(figs):
        body = body.replace("<p>@@FIG%d@@</p>" % i, fg)
    body = fold(body)
    body = mark_ext(body)
    body = body.replace("<strong>ต้องจำ", '<strong class="must">ต้องจำ')
    body = re.sub(r"<table>", '<div class="tw"><table>', body)
    body = body.replace("</table>", "</table></div>")
    m = re.search(r"<h1>(.*?)</h1>", body)
    title = re.sub(r"<.*?>", "", m.group(1)) if m else f"สัปดาห์ {n}"
    sid = f"week{n}"
    short = f"สัปดาห์ {n}"
    tabs.append(f'<button type="button" role="tab" data-id="{sid}" aria-selected="false">{short}</button>')
    sections.append(f'<section class="week" id="{sid}" data-title="{html.escape(title)}"><article>{body}</article></section>')

# ---- paper library tab ----
PAP = SRC / "papers"
pap_md, idx_rows = [], []
for n in weeks:
    p = PAP / f"week{n:02d}.md"
    if not p.exists():
        continue
    t = p.read_text(encoding="utf-8")
    t = re.sub(r"^#{1,2} .*\n", "", t, flags=re.M)  # drop any H1/H2 the author added
    titles = re.findall(r"^### (.+)$", t, flags=re.M)
    for ti in titles:
        idx_rows.append((n, ti.strip()))
    pap_md.append(f"## สัปดาห์ {n} ({len(titles)} เปเปอร์)\n\n{t.strip()}\n")
if pap_md:
    intro = ("# คลัง Paper ตามสไลด์\n\nรวมเปเปอร์ที่อาจารย์ยกมาในสไลด์สัปดาห์ 8 ถึง 14 แยกตามสัปดาห์ "
             "แต่ละเรื่องมีลิงก์ที่ตรวจแล้ว ใจความสำคัญ และจุดที่เชื่อมกับข้อสอบ ลิงก์เปิดไปยังหน้า abstract ไม่ได้ดาวน์โหลดไฟล์ PDF ให้\n\n")
    md.reset()
    body = md.convert(fix_lists(intro + "\n".join(pap_md)))
    body = re.sub(r"<table>", '<div class="tw"><table>', body).replace("</table>", "</table></div>")
    body = re.sub(r"(?<![\">=/])(https?://[^\s<\"')]+[^\s<\"'),.;])",
                  r'<a href="\1" target="_blank" rel="noopener">\1</a>', body)
    tabs.append('<button type="button" role="tab" data-id="papers" aria-selected="false">คลัง Paper</button>')
    sections.append('<section class="week" id="papers" data-title="คลัง Paper"><article>%s</article></section>' % body)

page_body = f"""
<header class="top"><div class="top-in">
  <span class="brand">NLP Final Study · สัปดาห์ 8-14</span>
  <nav class="tabs" role="tablist" aria-label="เลือกสัปดาห์">{''.join(tabs)}</nav>
  <button class="theme" id="theme" type="button">สลับธีม</button>
</div></header>
<div class="wrap">
  <nav class="toc" id="toc" aria-label="หัวข้อ"></nav>
  <main><p class="legend"><span class="extbadge">ต่อยอดจากสไลด์</span> / <span class="extbadge">นอกสไลด์</span> = ข้อความแถบสีม่วงคือส่วนที่เกินสไลด์ ไม่ได้มาจากสไลด์โดยตรง ส่วนที่ไม่มีแถบคือเนื้อหาตามสไลด์และเดโม &nbsp;·&nbsp; <span class="assume">ตัวเลขสมมติ</span> = ตัวเลขที่ผู้เขียนตั้งขึ้นเพื่อสาธิตวิธีคิด</p>{''.join(sections)}</main>
</div>
<script>{JS}</script>
"""
fonts = '<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Noto+Sans+Thai:wght@400;600;700&family=JetBrains+Mono:wght@400&display=swap">'
head_common = f'<title>NLP Final Study</title><meta name="claude:viewport-fit-sides" content="cover">{fonts}<style>{CSS}</style>'

# artifact fragment (skeleton added at publish time)
(OUT / "nlp-final-study.artifact.html").write_text(head_common + page_body, encoding="utf-8")
# standalone file for local opening
standalone = (
    '<!doctype html><html lang="th"><head><meta charset="utf-8">'
    '<meta name="viewport" content="width=device-width,initial-scale=1">'
    + head_common.replace('<meta name="claude:viewport-fit-sides" content="cover">', "")
    + "</head><body>" + page_body + "</body></html>"
)
(OUT / "index.html").write_text(standalone, encoding="utf-8")
print("built", len(sections), "weeks")
