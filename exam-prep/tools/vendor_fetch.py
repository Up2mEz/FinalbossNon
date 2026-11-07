"""Download the third-party assets the page uses (KaTeX and the IBM Plex fonts) into tools/vendor/ so that
tools/build_html.py can embed them and the page works offline.
Run once: python tools/vendor_fetch.py"""
import os, re, urllib.request

R = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
V = f'{R}/tools/vendor'
os.makedirs(f'{V}/katex_fonts', exist_ok=True); os.makedirs(f'{V}/fonts', exist_ok=True)
UA = 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36'

def get(url):
    req = urllib.request.Request(url, headers={'User-Agent': UA})
    return urllib.request.urlopen(req, timeout=60).read()

K = 'https://cdnjs.cloudflare.com/ajax/libs/KaTeX/0.16.9'
for name, path in [('katex.min.js', 'katex.min.js'), ('auto-render.min.js', 'contrib/auto-render.min.js'), ('katex.min.css', 'katex.min.css')]:
    open(f'{V}/{name}', 'wb').write(get(f'{K}/{path}')); print('got', name)
css = open(f'{V}/katex.min.css', encoding='utf-8').read()
for fn in sorted(set(re.findall(r'fonts/(KaTeX_[A-Za-z0-9_\-]+)\.woff2', css))):
    open(f'{V}/katex_fonts/{fn}.woff2', 'wb').write(get(f'{K}/fonts/{fn}.woff2'))
print('katex fonts', len(os.listdir(f'{V}/katex_fonts')))

G = 'https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500;600&family=IBM+Plex+Sans+Thai:wght@400;500;600;700&display=swap'
gcss = get(G).decode('utf-8')
blocks = re.findall(r'/\*\s*([\w-]+)\s*\*/\s*(@font-face\s*\{.*?\})', gcss, re.S)
keep = []
for subset, blk in blocks:
    if subset not in ('thai', 'latin'): continue
    url = re.search(r'url\((https://[^)]+)\)', blk).group(1)
    fam = re.search(r"font-family:\s*'([^']+)'", blk).group(1); wt = re.search(r'font-weight:\s*(\d+)', blk).group(1)
    fn = f"{fam.replace(' ', '')}_{wt}_{subset}.woff2"
    open(f'{V}/fonts/{fn}', 'wb').write(get(url))
    keep.append(blk.replace(url, 'FONTFILE:' + fn))
open(f'{V}/fonts.css', 'w', encoding='utf-8').write('\n'.join(keep))
print('google font faces kept', len(keep), 'total size KB', sum(os.path.getsize(f'{V}/fonts/{f}') for f in os.listdir(f'{V}/fonts')) // 1024)
