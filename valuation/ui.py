import base64
from html import escape
from pathlib import Path


def portrait(player_id, name, size=180):
    path = Path("assets/players") / f"{int(player_id)}.jpg"
    if path.exists():
        encoded = base64.b64encode(path.read_bytes()).decode("ascii")
        return f'<img class="portrait" src="data:image/jpeg;base64,{encoded}" alt="{escape(str(name), quote=True)}" style="width:{size}px;height:{size}px;object-fit:contain;object-position:center;background:#cbd0d8;border-radius:20px">'
    initials = escape("".join(word[0] for word in str(name).split()[:2]))
    return (
        f'<div class="initials" style="width:{size}px;height:{size}px">{initials}</div>'
    )


def player_card(row):
    name = escape(str(row.player_name))
    position = escape(str(row.position))
    return f"""<div class="player-card">{portrait(row.player_id, row.player_name, 100)}
<div class="card-position">{position} · {int(row.age)} YEARS</div>
<div class="card-name">{name}</div><div class="card-value">€{row.market_value_eur / 1e6:.1f}<span>m</span></div>
<div class="card-footer">RECORDED VALUE · 2025/26</div>
<div class="card-stats"><span><b>{int(row.goals)}</b> goals</span><span><b>{int(row.assists)}</b> assists</span><span><b>{int(row.minutes_played):,}</b> min</span></div></div>"""


STYLES = """<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Space+Grotesk:wght@400;500;600;700&display=swap');
html,body,[class*="css"],.stApp{font-family:'DM Sans',sans-serif}
.stApp{background:#0b1018;color:#edf2fa}
.block-container{max-width:1440px;padding:5rem 3rem 4rem}
h1,h2,h3,.hero-name,.card-name,.card-value{font-family:'Space Grotesk',sans-serif!important}
h1{font-size:3.3rem!important;letter-spacing:-2px;line-height:1.08!important}
h2,h3{letter-spacing:-.7px}
[data-testid="stSidebar"]{background:#101721;border-right:1px solid #26303c}
[data-testid="stMetric"]{background:#141d29;border:1px solid #263244;padding:20px 24px;border-radius:16px}
[data-testid="stMetricLabel"]{color:#92a2b9;font-size:12px;text-transform:uppercase;letter-spacing:1px}
[data-testid="stMetricValue"]{font-family:'Space Grotesk',sans-serif;font-size:1.8rem}
[data-testid="stForm"]{background:#121b27;border:1px solid #263244;border-radius:18px;padding:24px}
.stTabs [data-baseweb="tab-list"]{gap:32px;border-bottom:1px solid #263244;margin:20px 0}
.stTabs [data-baseweb="tab"]{font-size:15px;padding-bottom:18px}
.stTabs [aria-selected="true"]{color:#ff5470!important}
.topbar{display:flex;justify-content:space-between;align-items:center;margin-bottom:36px;border-bottom:1px solid #25303e;padding-bottom:22px}
.wordmark{font-family:'Space Grotesk',sans-serif;font-size:23px;font-weight:700;letter-spacing:3px}.wordmark span{color:#ff405d}
.status{font-size:11px;letter-spacing:1.5px;background:#142b29;color:#82dac4;border:1px solid #255347;padding:8px 12px;border-radius:30px}
.eyebrow{color:#ff6079;font-size:11px;font-weight:700;letter-spacing:2.5px;margin:0 0 16px}
.intro{color:#91a1b7;font-size:16px;max-width:650px;line-height:1.7;margin-bottom:24px}
.hero{background:radial-gradient(ellipse at 95% 0%,#793149 0%,#302333 28%,#192334 65%);border:1px solid #443144;border-radius:24px;padding:34px 40px;display:flex;align-items:center;justify-content:space-between;gap:24px;overflow:hidden;margin-bottom:24px}
.hero-name{font-size:clamp(28px,4vw,54px);font-weight:700;line-height:1.07;letter-spacing:-1.7px;color:white;margin:14px 0}
.hero-sub{color:#b2bed0;font-size:14px}.hero-tag{display:inline-block;padding:6px 11px;background:#ff405d22;color:#ff8799;border:1px solid #ff405d55;border-radius:6px;font-size:10px;letter-spacing:1.8px}
.hero-stat{color:#fff;font-family:'Space Grotesk',sans-serif;font-size:30px;margin-top:18px}.hero-stat span{font-family:'DM Sans',sans-serif;font-size:11px;color:#a5b1c5;margin-left:10px;letter-spacing:1px}
.hero .portrait{box-shadow:0 12px 50px #0005;border:4px solid #ffffff15;flex-shrink:0}
.player-card{background:linear-gradient(150deg,#1b2637,#121a26);border:1px solid #2b3748;border-radius:18px;padding:24px;text-align:center;margin:8px 0 16px;min-height:340px}
.player-card .portrait{border-radius:50%!important;border:3px solid #35435a}
.card-position{font-size:9px;color:#98aac2;letter-spacing:1.4px;margin-top:16px}.card-name{font-size:20px;font-weight:600;margin-top:7px;color:#f3f6fc;min-height:52px;line-height:1.3}
.card-value{font-size:30px;font-weight:600;color:#fff}.card-value span{font-size:19px;color:#a4b1c7}.card-footer{font-size:8px;letter-spacing:1.2px;color:#8395b0;margin-top:4px}
.card-stats{display:flex;justify-content:space-between;border-top:1px solid #2a3547;padding-top:15px;margin-top:18px;font-size:10px;color:#8496ae}.card-stats b{display:block;color:#dbe4f2;font-size:15px;margin-bottom:3px}
.initials{display:inline-flex;align-items:center;justify-content:center;background:#28384e;border-radius:20px;font-size:36px;color:#a6bad5}
.section-note{font-size:12px;color:#8496af;line-height:1.7;margin:8px 0 20px}
button[kind="primary"]{background:#e72e50!important;border-radius:9px!important;border:0!important;min-height:44px}
@media(max-width:700px){.block-container{padding:1.4rem 1rem}.hero{padding:22px;gap:12px}.hero .portrait{width:110px!important;height:140px!important}h1{font-size:2.5rem!important}.status{display:none}.hero-stat span{display:block;margin-left:0}}
</style>"""
