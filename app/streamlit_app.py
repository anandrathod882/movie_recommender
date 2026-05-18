import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import sys, os, random, requests
from urllib.parse import quote

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.content_based import (
    load_movies, build_movie_soup, build_tfidf_matrix,
    build_cosine_sim, get_recommendations as get_cb_recs
)

st.set_page_config(page_title="CineAI", page_icon="🎬",
                   layout="wide", initial_sidebar_state="collapsed")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&display=swap');

@keyframes gradBG{0%{background-position:0% 50%}50%{background-position:100% 50%}100%{background-position:0% 50%}}
@keyframes fadeUp{from{opacity:0;transform:translateY(40px)}to{opacity:1;transform:translateY(0)}}
@keyframes fadeDown{from{opacity:0;transform:translateY(-40px)}to{opacity:1;transform:translateY(0)}}
@keyframes float1{0%,100%{transform:translateY(0px) rotate(-3deg)}50%{transform:translateY(-18px) rotate(-3deg)}}
@keyframes float2{0%,100%{transform:translateY(0px) rotate(3deg)}50%{transform:translateY(-22px) rotate(3deg)}}
@keyframes float3{0%,100%{transform:translateY(0px) rotate(-2deg)}50%{transform:translateY(-14px) rotate(-2deg)}}
@keyframes float4{0%,100%{transform:translateY(-10px) rotate(2deg)}50%{transform:translateY(-26px) rotate(2deg)}}
@keyframes float5{0%,100%{transform:translateY(-5px) rotate(-4deg)}50%{transform:translateY(-20px) rotate(-4deg)}}
@keyframes float6{0%,100%{transform:translateY(0px) rotate(4deg)}50%{transform:translateY(-16px) rotate(4deg)}}
@keyframes scaleIn{from{opacity:0;transform:scale(0.88)}to{opacity:1;transform:scale(1)}}
@keyframes slideR{from{opacity:0;transform:translateX(-20px)}to{opacity:1;transform:translateX(0)}}
@keyframes pulseGlow{0%,100%{box-shadow:0 0 20px rgba(124,58,237,0.4)}50%{box-shadow:0 0 50px rgba(124,58,237,0.9),0 0 80px rgba(99,102,241,0.4)}}
@keyframes shimmer{0%{background-position:-1000px 0}100%{background-position:1000px 0}}
@keyframes badgePulse{0%,100%{box-shadow:0 0 0 0 rgba(167,139,250,0.4)}50%{box-shadow:0 0 0 8px rgba(167,139,250,0)}}
@keyframes borderGlow{0%,100%{border-color:rgba(124,58,237,0.4)}50%{border-color:rgba(167,139,250,0.9)}}
@keyframes float{0%,100%{transform:translateY(0)}50%{transform:translateY(-7px)}}
@keyframes glow{0%,100%{box-shadow:0 0 15px rgba(124,58,237,0.3)}50%{box-shadow:0 0 35px rgba(124,58,237,0.65)}}

/* *{font-family:'Inter',sans-serif !important;box-sizing:border-box;} */
html,body,.stApp{background:#07080f !important;color:#e2e8f0 !important;}
/* Apply Inter font carefully */
p, h1, h2, h3, h4, h5, h6, button, input, select, textarea {
    font-family: 'Inter', sans-serif;
}
* { box-sizing: border-box; }
footer,.stDeployButton{display:none !important;}
#MainMenu{visibility:hidden;}
.block-container{padding:0 !important;max-width:100% !important;}
::-webkit-scrollbar{width:4px;height:4px;}
::-webkit-scrollbar-thumb{background:rgba(124,58,237,0.5);border-radius:10px;}

/* ── Hamburger toggle button ── */
[data-testid="collapsedControl"], [data-testid="stSidebarCollapsedControl"] {
    position:fixed !important;top:14px !important;left:14px !important;
    width:44px !important;height:44px !important;
    background:rgba(124,58,237,0.2) !important;
    border:1.5px solid rgba(124,58,237,0.5) !important;
    border-radius:10px !important;z-index:99999 !important;
    overflow:hidden !important;cursor:pointer !important;
    transition:all 0.3s ease !important;
    font-size:0 !important;
    color:transparent !important;
    line-height:0 !important;
    letter-spacing:0 !important;
    text-indent:0 !important;
    word-spacing:0 !important;
}
[data-testid="collapsedControl"]:hover, [data-testid="stSidebarCollapsedControl"]:hover {
    background:rgba(124,58,237,0.5) !important;
    box-shadow:0 0 24px rgba(124,58,237,0.7) !important;
}
[data-testid="collapsedControl"] *, [data-testid="stSidebarCollapsedControl"] * {
    display:none !important;
    visibility:hidden !important;
    font-size:0 !important;
    color:transparent !important;
    width:0 !important;
    height:0 !important;
    opacity:0 !important;
}
[data-testid="collapsedControl"]::before, [data-testid="stSidebarCollapsedControl"]::before {
    content:"»" !important;
    display:flex !important;
    align-items:center !important;
    justify-content:center !important;
    position:absolute !important;
    top:48% !important;left:50% !important;
    transform:translate(-50%,-50%) !important;
    color:#a78bfa !important;
    font-size:26px !important;
    font-weight:900 !important;
    transition:all 0.3s ease !important;
    background:transparent !important;
    box-shadow:none !important;
    width:auto !important;height:auto !important;
}
[data-testid="collapsedControl"]:hover::before, [data-testid="stSidebarCollapsedControl"]:hover::before {
    color:#c4b5fd !important;
    transform:translate(-38%,-50%) !important;
}

/* ── Close Sidebar toggle button (Left Arrow) ── */
[data-testid="stBaseButton-headerNoPadding"] {
    background:rgba(124,58,237,0.1) !important;
    border:1px solid rgba(124,58,237,0.3) !important;
    border-radius:8px !important;
    transition:all 0.3s ease !important;
    color: transparent !important;
    position: relative !important;
    width: 36px !important;
    height: 36px !important;
    display: flex !important;
    align-items: center !important;
    justify-content: center !important;
    cursor: pointer !important;
    z-index: 99999 !important;
}
[data-testid="stBaseButton-headerNoPadding"]:hover {
    background:rgba(124,58,237,0.3) !important;
    border-color:rgba(124,58,237,0.6) !important;
}
[data-testid="stBaseButton-headerNoPadding"] * {
    display: none !important;
}
[data-testid="stBaseButton-headerNoPadding"]::before {
    content:"«" !important;
    display:flex !important;
    align-items:center !important;
    justify-content:center !important;
    position:absolute !important;
    top:48% !important; left:50% !important;
    transform:translate(-50%,-50%) !important;
    color:#a78bfa !important;
    font-size:26px !important;
    font-weight:900 !important;
    transition:all 0.3s ease !important;
    background:transparent !important;
    box-shadow:none !important;
    border:none !important;
    width:auto !important; height:auto !important;
}
[data-testid="stBaseButton-headerNoPadding"]:hover::before {
    color:#c4b5fd !important;
    transform:translate(-62%,-50%) !important;
}

[data-testid="stSidebar"]{background:linear-gradient(180deg,#0d0221,#0f172a) !important;border-right:1px solid rgba(124,58,237,0.25) !important;}
[data-testid="stSidebar"] *{color:#e2e8f0 !important;}
.sidebar-stat{background:rgba(124,58,237,0.1);border:1px solid rgba(124,58,237,0.2);border-radius:12px;padding:0.7rem 1rem;margin-bottom:8px;display:flex;align-items:center;gap:10px;}
.sidebar-stat-val{font-size:1.1rem;font-weight:800;color:#a78bfa;}
.sidebar-stat-lbl{font-size:0.75rem;color:#64748b;}

.hero-wrap{
    position:relative;min-height:520px;
    background:linear-gradient(135deg,#0d0221 0%,#1a0533 30%,#0f172a 60%,#1e1b4b 100%);
    overflow:hidden;display:flex;align-items:center;justify-content:center;
    border-bottom:1px solid rgba(124,58,237,0.2);padding:3rem 2rem;
}
.hero-bg-glow{position:absolute;top:50%;left:50%;transform:translate(-50%,-50%);width:700px;height:400px;background:radial-gradient(ellipse,rgba(124,58,237,0.25) 0%,transparent 70%);pointer-events:none;}
.hero-bg-glow2{position:absolute;top:20%;left:10%;width:300px;height:300px;background:radial-gradient(ellipse,rgba(99,102,241,0.15) 0%,transparent 70%);pointer-events:none;}
.hero-bg-glow3{position:absolute;top:20%;right:10%;width:300px;height:300px;background:radial-gradient(ellipse,rgba(139,92,246,0.15) 0%,transparent 70%);pointer-events:none;}
.fp{position:absolute;border-radius:12px;overflow:hidden;box-shadow:0 8px 40px rgba(0,0,0,0.6);border:1px solid rgba(255,255,255,0.1);}
.fp img{width:100%;height:100%;object-fit:cover;display:block;}
.fp1{width:110px;height:165px;left:2%;top:15%;animation:float1 5s ease-in-out infinite;}
.fp2{width:90px;height:135px;left:8%;top:55%;animation:float2 6s ease-in-out infinite;}
.fp3{width:95px;height:143px;left:14%;top:20%;animation:float3 4.5s ease-in-out infinite;}
.fp4{width:110px;height:165px;right:2%;top:15%;animation:float4 5.5s ease-in-out infinite;}
.fp5{width:90px;height:135px;right:8%;top:55%;animation:float5 6.5s ease-in-out infinite;}
.fp6{width:95px;height:143px;right:14%;top:20%;animation:float6 4s ease-in-out infinite;}
.hero-fade-left{position:absolute;top:0;left:0;width:35%;height:100%;background:linear-gradient(to right,#0d0221 0%,transparent 100%);pointer-events:none;z-index:2;}
.hero-fade-right{position:absolute;top:0;right:0;width:35%;height:100%;background:linear-gradient(to left,#0d0221 0%,transparent 100%);pointer-events:none;z-index:2;}
.hero-fade-bottom{position:absolute;bottom:0;left:0;right:0;height:120px;background:linear-gradient(to top,#07080f 0%,transparent 100%);pointer-events:none;z-index:2;}
.hero-content{position:relative;z-index:3;text-align:center;max-width:680px;}
.hero-badge{display:inline-flex;align-items:center;gap:8px;background:rgba(124,58,237,0.15);border:1px solid rgba(167,139,250,0.4);color:#c4b5fd;padding:7px 20px;border-radius:100px;font-size:0.72rem;font-weight:700;letter-spacing:2px;text-transform:uppercase;margin-bottom:1.5rem;animation:fadeDown 0.6s ease both,badgePulse 3s ease infinite;}
.hero-badge-dot{width:6px;height:6px;background:#a78bfa;border-radius:50%;animation:badgePulse 2s ease infinite;}
.hero-title{font-size:4rem;font-weight:900;line-height:1.0;letter-spacing:-2.5px;margin-bottom:1rem;animation:fadeDown 0.7s ease 0.1s both;background:linear-gradient(135deg,#ffffff 0%,#c4b5fd 45%,#818cf8 100%);-webkit-background-clip:text;-webkit-text-fill-color:transparent;background-clip:text;}
.hero-sub{font-size:0.95rem;color:#64748b;margin-bottom:2.5rem;animation:fadeUp 0.7s ease 0.2s both;font-weight:400;letter-spacing:0.3px;}
.hero-sub span{color:#94a3b8;}
.hero-stats{display:flex;justify-content:center;gap:1rem;animation:fadeUp 0.8s ease 0.3s both;}
.hstat{background:rgba(255,255,255,0.04);border:1px solid rgba(124,58,237,0.35);border-radius:16px;padding:0.9rem 1.6rem;transition:all 0.35s ease;animation:borderGlow 4s ease infinite,pulseGlow 4s ease infinite;backdrop-filter:blur(10px);min-width:110px;}
.hstat:hover{background:rgba(124,58,237,0.15);transform:translateY(-5px) scale(1.04);}
.hstat-num{font-size:1.8rem;font-weight:900;background:linear-gradient(135deg,#a78bfa,#60a5fa);-webkit-background-clip:text;-webkit-text-fill-color:transparent;background-clip:text;}
.hstat-lbl{font-size:0.68rem;color:#475569;text-transform:uppercase;letter-spacing:1.5px;margin-top:2px;}
.hstat-icon{font-size:1rem;margin-bottom:2px;}

.main-wrap{padding:2rem 2.5rem;max-width:1200px;margin:auto;}
.sec-hdr{display:flex;align-items:center;gap:10px;font-size:1.1rem;font-weight:800;color:#e2e8f0;margin-bottom:1.2rem;}
.sec-hdr::after{content:'';flex:1;height:1px;background:linear-gradient(90deg,rgba(124,58,237,0.4),transparent);}
.sec-accent{width:4px;height:22px;border-radius:4px;background:linear-gradient(180deg,#7c3aed,#3b82f6);flex-shrink:0;}
.glass{background:rgba(255,255,255,0.022);border:1px solid rgba(255,255,255,0.07);border-radius:20px;backdrop-filter:blur(20px);padding:1.8rem;margin-bottom:1.5rem;animation:fadeUp 0.6s ease both;}
.glass:hover{border-color:rgba(124,58,237,0.25);}

.stTextInput>div>div>input{background:rgba(255,255,255,0.05) !important;border:1px solid rgba(124,58,237,0.3) !important;border-radius:12px !important;color:#e2e8f0 !important;font-size:1rem !important;padding:0.7rem 1rem !important;}
.stTextInput>div>div>input:focus{border-color:rgba(124,58,237,0.8) !important;box-shadow:0 0 0 4px rgba(124,58,237,0.14) !important;}
.stTextInput>div>div>input::placeholder{color:#374151 !important;}
.stSelectbox>div>div{background:rgba(255,255,255,0.05) !important;border:1px solid rgba(124,58,237,0.3) !important;border-radius:12px !important;color:#e2e8f0 !important;}
.stButton>button{background:linear-gradient(135deg,#7c3aed,#4f46e5) !important;color:white !important;border:none !important;border-radius:12px !important;font-weight:700 !important;font-size:0.88rem !important;padding:0.5rem 1.2rem !important;transition:all 0.3s ease !important;}
.stButton>button:hover{transform:translateY(-2px) scale(1.03) !important;box-shadow:0 8px 28px rgba(124,58,237,0.5) !important;}
div[data-testid="stHorizontalBlock"] .stButton>button{background:rgba(255,255,255,0.04) !important;color:#94a3b8 !important;border:1px solid rgba(255,255,255,0.1) !important;border-radius:100px !important;padding:0.38rem 1rem !important;font-weight:600 !important;font-size:0.82rem !important;white-space:nowrap !important;}
div[data-testid="stHorizontalBlock"] .stButton>button:hover{background:rgba(124,58,237,0.2) !important;border-color:rgba(124,58,237,0.6) !important;color:#a78bfa !important;transform:translateY(-2px) !important;}

.detail-page{background:linear-gradient(135deg,rgba(30,27,75,0.95),rgba(15,23,42,0.98));border:1px solid rgba(124,58,237,0.35);border-radius:24px;padding:2.5rem;margin-bottom:1.5rem;animation:scaleIn 0.4s ease both;}
.detail-title{font-size:2rem;font-weight:900;color:#e2e8f0;line-height:1.15;margin-bottom:0.7rem;}
.detail-meta{display:flex;gap:8px;flex-wrap:wrap;margin-bottom:1rem;}
.dtag{background:rgba(124,58,237,0.18);border:1px solid rgba(124,58,237,0.35);color:#a78bfa;border-radius:8px;padding:5px 14px;font-size:0.8rem;font-weight:600;}
.imdb-badge{display:inline-flex;align-items:center;gap:5px;background:#f5c518;color:#000;border-radius:6px;padding:4px 12px;font-size:0.8rem;font-weight:800;margin-bottom:1rem;}
.detail-plot{font-size:0.95rem;color:#94a3b8;line-height:1.85;margin-bottom:1rem;padding:1.2rem;background:rgba(255,255,255,0.03);border-radius:14px;border-left:4px solid #7c3aed;}
.detail-crew{font-size:0.85rem;color:#7c3aed;font-weight:600;margin-bottom:4px;}
.wiki-section{background:rgba(124,58,237,0.06);border:1px solid rgba(124,58,237,0.2);border-radius:16px;padding:1.2rem;margin-top:1rem;}
.wiki-section-title{font-size:1rem;font-weight:800;color:#a78bfa;margin-bottom:8px;}
.wiki-section-text{font-size:0.88rem;color:#94a3b8;line-height:1.8;}

.poster-wrap{position:relative;border-radius:16px;overflow:hidden;border:1px solid rgba(255,255,255,0.07);background:#0f172a;transition:all 0.4s ease;animation:scaleIn 0.5s ease both;cursor:pointer;}
.poster-wrap:hover{transform:translateY(-8px) scale(1.02);border-color:rgba(124,58,237,0.5);box-shadow:0 20px 60px rgba(124,58,237,0.3);}
.poster-img{width:100%;aspect-ratio:2/3;object-fit:cover;display:block;transition:transform 0.4s ease;}
.poster-wrap:hover .poster-img{transform:scale(1.06);}
.poster-overlay{position:absolute;bottom:0;left:0;right:0;background:linear-gradient(transparent,rgba(0,0,0,0.96));padding:2rem 0.8rem 0.8rem;}
.poster-title{font-size:0.78rem;font-weight:700;color:#fff;line-height:1.3;overflow:hidden;display:-webkit-box;-webkit-line-clamp:2;-webkit-box-orient:vertical;}
.poster-year{position:absolute;top:8px;right:8px;background:rgba(0,0,0,0.75);color:#fff;font-size:0.68rem;font-weight:700;padding:3px 8px;border-radius:8px;}
.no-poster{aspect-ratio:2/3;display:flex;flex-direction:column;align-items:center;justify-content:center;background:linear-gradient(135deg,#1e1b4b,#0f172a);}
.no-poster-icon{font-size:1.8rem;margin-bottom:6px;animation:float 3s ease infinite;}
.no-poster-text{font-size:0.72rem;font-weight:700;color:#4c1d95;text-align:center;padding:0.5rem;line-height:1.3;}
.card-desc{font-size:0.72rem;color:#64748b;line-height:1.5;margin-top:6px;padding:0 2px 6px;}
.result-item{display:flex;align-items:center;gap:14px;padding:0.85rem 1rem;border-radius:14px;border:1px solid rgba(255,255,255,0.05);background:rgba(255,255,255,0.02);margin-bottom:8px;transition:all 0.3s ease;animation:slideR 0.4s ease both;}
.result-item:hover{background:rgba(124,58,237,0.08);border-color:rgba(124,58,237,0.3);transform:translateX(6px);}
.result-rank{font-size:1.2rem;font-weight:900;background:linear-gradient(135deg,#7c3aed,#3b82f6);-webkit-background-clip:text;-webkit-text-fill-color:transparent;min-width:34px;}
.result-poster{width:44px;height:66px;border-radius:8px;object-fit:cover;flex-shrink:0;border:1px solid rgba(255,255,255,0.1);}
.result-no-poster{width:44px;height:66px;border-radius:8px;background:linear-gradient(135deg,#1e1b4b,#0f172a);display:flex;align-items:center;justify-content:center;font-size:1.2rem;flex-shrink:0;}
.result-title{font-size:0.93rem;font-weight:700;color:#e2e8f0;}
.result-desc{font-size:0.75rem;color:#64748b;margin-top:3px;line-height:1.4;overflow:hidden;display:-webkit-box;-webkit-line-clamp:2;-webkit-box-orient:vertical;}
.result-match{background:linear-gradient(135deg,#7c3aed,#4f46e5);color:white;border-radius:100px;padding:4px 12px;font-size:0.72rem;font-weight:700;flex-shrink:0;margin-left:auto;white-space:nowrap;animation:glow 3s ease infinite;}
.movie-meta{display:flex;gap:8px;flex-wrap:wrap;margin:8px 0 10px;}
.mtag{background:rgba(124,58,237,0.15);border:1px solid rgba(124,58,237,0.3);color:#a78bfa;border-radius:8px;padding:4px 12px;font-size:0.76rem;font-weight:600;}
.movie-plot{font-size:0.88rem;color:#94a3b8;line-height:1.75;border-top:1px solid rgba(255,255,255,0.06);padding-top:10px;margin-top:6px;}
.movie-crew{font-size:0.78rem;color:#6c63ff;margin-top:6px;font-weight:600;}
.rcard{background:rgba(255,255,255,0.03);border:1px solid rgba(255,255,255,0.07);border-radius:16px;overflow:hidden;transition:all 0.35s ease;}
.rcard:hover{border-color:rgba(124,58,237,0.5);transform:translateY(-6px);box-shadow:0 16px 40px rgba(124,58,237,0.25);}
.rcard-img{width:100%;aspect-ratio:16/9;object-fit:cover;display:block;}
.rcard-body{padding:0.8rem;}
.rcard-title{font-size:0.83rem;font-weight:700;color:#e2e8f0;}
.rcard-genre{font-size:0.7rem;color:#7c3aed;margin-top:3px;font-weight:600;}

/* ── Wikipedia error ── */
.wiki-error{background:rgba(239,68,68,0.08);border:1px solid rgba(239,68,68,0.25);border-radius:14px;padding:1rem 1.2rem;color:#fca5a5;font-size:0.88rem;}

/* ── OTT Links ── */
.ott-container { margin-top: 1.5rem; display: flex; flex-direction: column; gap: 1.2rem; }
.ott-item { display: flex; flex-direction: column; gap: 4px; }
.ott-label { display: inline-block; border: 1px solid rgba(255, 255, 255, 0.4); border-radius: 6px; padding: 4px 12px; font-size: 0.8rem; font-weight: 600; color: #e2e8f0; width: max-content; }
.ott-link { display: inline-block; border: 1px solid rgba(255, 255, 255, 0.4); border-radius: 6px; padding: 6px 12px; font-size: 0.85rem; color: #e2e8f0; text-decoration: none; width: max-content; max-width: 100%; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; transition: all 0.3s ease; }
.ott-link:hover { border-color: rgba(124, 58, 237, 0.8); background: rgba(124, 58, 237, 0.1); color: #c4b5fd; }

/* --- Removed invisible button CSS --- */
</style>

<script>
/* Continuously wipe any text node inside the sidebar toggle button and globally */
(function(){
  function nukeTextNodes(node) {
    if (node.nodeType === 3) {
      if (node.nodeValue && node.nodeValue.includes("keyboard_double_arrow_right")) {
        node.nodeValue = "";
      }
    } else if (node.nodeType === 1) {
      node.childNodes.forEach(nukeTextNodes);
      if(node.textContent && node.textContent.trim() === "keyboard_double_arrow_right") {
          node.style.display = "none";
          node.textContent = "";
      }
    }
  }

  function cleanToggle(){
    nukeTextNodes(document.body);
    var btns = document.querySelectorAll('[data-testid="collapsedControl"], [data-testid="stSidebarCollapsedControl"]');
    btns.forEach(function(btn){
      btn.querySelectorAll("span,svg,p").forEach(function(el){
        el.style.cssText="display:none!important;visibility:hidden!important;font-size:0!important;color:transparent!important;width:0!important;height:0!important;";
        el.textContent="";
      });
    });
  }
  /* Run on load and watch for DOM changes */
  var mo = new MutationObserver(cleanToggle);
  document.addEventListener("DOMContentLoaded", function(){
    cleanToggle();
    var target = document.body;
    if(target) mo.observe(target,{childList:true,subtree:true});
  });
  setTimeout(cleanToggle, 500);
  setTimeout(cleanToggle, 1500);
})();
</script>
""", unsafe_allow_html=True)


# ── Load ─────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_models():
    mv = load_movies("data/processed")
    cb = build_movie_soup(mv.copy())
    _, tm = build_tfidf_matrix(cb)
    cs = build_cosine_sim(tm)
    return mv, cb, cs

with st.spinner(""):
    movies, movies_cb, cosine_sim = load_models()

all_titles     = sorted(movies["title"].dropna().tolist())
TOTAL_MOVIES   = len(all_titles)
all_genres_raw = "|".join(movies["genres"].dropna().tolist())
TOTAL_GENRES   = len(set(all_genres_raw.split("|")))
KEY_GENRES     = ["Action","Comedy","Drama","Sci-Fi","Thriller","Romance","Horror","Animation"]
G_EMOJI        = {"Action":"⚔️","Comedy":"😂","Drama":"🎭","Sci-Fi":"🚀",
                  "Thriller":"😱","Romance":"❤️","Horror":"👻","Animation":"✨"}

HERO_POSTERS = [
    "https://image.tmdb.org/t/p/w342/d5NXSklXo0qyIYkgV61RRcV8ald.jpg",
    "https://image.tmdb.org/t/p/w342/9gk7adHYeDvHkCSEqAvQNLV5Uge.jpg",
    "https://image.tmdb.org/t/p/w342/8UlWHLMpgZm9bx6QYh0NFoq67TZ.jpg",
    "https://image.tmdb.org/t/p/w342/velWPhVoxdCbDtkME6J7HQBW79x.jpg",
    "https://image.tmdb.org/t/p/w342/pB8BM7pdSp6B6Ih7QZ4DrQ3PmJK.jpg",
    "https://image.tmdb.org/t/p/w342/2CAL2433ZeIihfX1Hb2139CX0pW.jpg",
]

WIKI_HEADERS = {
    "User-Agent": "CineAI/1.0 (movie-recommender-app; educational project) python-requests/2.x",
    "Accept": "application/json",
}


# ── API helpers ───────────────────────────────────────────────
@st.cache_data(ttl=3600, show_spinner=False)
def get_omdb(title, key):
    if not key: return None
    clean = title.split("(")[0].strip()
    year  = title[-5:-1] if title.endswith(")") else ""
    try:
        p = {"apikey":key,"t":clean,"type":"movie","plot":"full"}
        if year: p["y"] = year
        r = requests.get("http://www.omdbapi.com/", params=p, timeout=6)
        if r.status_code == 200:
            d = r.json()
            if d.get("Response") == "True":
                def c(v): return v if v and v != "N/A" else ""
                return {"poster":c(d.get("Poster","")),
                        "plot":c(d.get("Plot","")),
                        "rating":c(d.get("imdbRating","")),
                        "director":c(d.get("Director","")),
                        "actors":c(d.get("Actors","")),
                        "runtime":c(d.get("Runtime","")),
                        "awards":c(d.get("Awards",""))}
    except: pass
    return None


@st.cache_data(ttl=3600, show_spinner=False)
def get_justwatch_providers(title, country="IN"):
    try:
        from simplejustwatchapi.justwatch import search
        # Search for 1 result
        results = search(title, country, "en", 1)
        if results:
            first = results[0]
            provs = []
            for offer in first.offers:
                # filter unique platforms
                if offer.package.name not in [p['name'] for p in provs]:
                    provs.append({
                        "name": offer.package.name,
                        "url": offer.url
                    })
            return provs
    except Exception:
        pass
    return None


def _clean_title(title: str) -> str:
    """Strip year suffix like (1994) and common suffixes that confuse Wikipedia."""
    import re
    clean = re.sub(r"\s*\(\d{4}\)\s*$", "", title).strip()
    # Remove trailing punctuation artifacts
    clean = re.sub(r"[,;:]\s*$", "", clean).strip()
    return clean


def _wiki_search_candidates(title: str) -> list[str]:
    """
    Return a list of query slugs to try against Wikipedia, from most
    specific to most generic.
    """
    clean = _clean_title(title)
    slug  = clean.replace(" ", "_")
    return [
        slug + "_(film)",
        slug + "_(movie)",
        slug + "_film",
        slug,
    ]


@st.cache_data(ttl=3600, show_spinner=False)
def get_wiki(title: str) -> dict | None:
    """
    Fetch Wikipedia summary for a movie title.
    Tries multiple slug variants and falls back to the OpenSearch API
    if the direct REST lookup misses.
    Returns a dict with keys: title, extract, image, url  — or None.
    """
    # ── 1. Direct REST summary attempts ──────────────────────────────────
    for slug in _wiki_search_candidates(title):
        try:
            url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{quote(slug)}"
            r   = requests.get(url, headers=WIKI_HEADERS, timeout=8)
            if r.status_code == 200:
                d       = r.json()
                extract = d.get("extract", "").strip()
                # Reject disambiguation pages & stubs
                if extract and len(extract) > 80 and "may refer to" not in extract[:120]:
                    return {
                        "title":   d.get("title", title),
                        "extract": extract,
                        "image":   d.get("thumbnail", {}).get("source", ""),
                        "url":     d.get("content_urls", {}).get("desktop", {}).get("page", ""),
                    }
        except Exception:
            continue

    # ── 2. OpenSearch fallback ────────────────────────────────────────────
    clean = _clean_title(title)
    try:
        sr = requests.get(
            "https://en.wikipedia.org/w/api.php",
            params={
                "action":   "opensearch",
                "search":   clean + " film",
                "limit":    3,
                "format":   "json",
                "namespace": 0,
            },
            headers=WIKI_HEADERS,
            timeout=8,
        )
        if sr.status_code == 200:
            results = sr.json()          # [term, [titles], [descs], [urls]]
            if len(results) >= 4:
                for page_title, page_url in zip(results[1], results[3]):
                    slug = page_title.replace(" ", "_")
                    try:
                        r2 = requests.get(
                            f"https://en.wikipedia.org/api/rest_v1/page/summary/{quote(slug)}",
                            headers=WIKI_HEADERS,
                            timeout=8,
                        )
                        if r2.status_code == 200:
                            d       = r2.json()
                            extract = d.get("extract", "").strip()
                            if extract and len(extract) > 80 and "may refer to" not in extract[:120]:
                                return {
                                    "title":   d.get("title", title),
                                    "extract": extract,
                                    "image":   d.get("thumbnail", {}).get("source", ""),
                                    "url":     d.get("content_urls", {}).get("desktop", {}).get("page", ""),
                                }
                    except Exception:
                        continue
    except Exception:
        pass

    # ── 3. Plain title fallback (no "film" suffix) ────────────────────────
    clean = _clean_title(title)
    try:
        slug = quote(clean.replace(" ", "_"))
        r3   = requests.get(
            f"https://en.wikipedia.org/api/rest_v1/page/summary/{slug}",
            headers=WIKI_HEADERS,
            timeout=8,
        )
        if r3.status_code == 200:
            d       = r3.json()
            extract = d.get("extract", "").strip()
            if extract and len(extract) > 80 and "may refer to" not in extract[:120]:
                return {
                    "title":   d.get("title", title),
                    "extract": extract,
                    "image":   d.get("thumbnail", {}).get("source", ""),
                    "url":     d.get("content_urls", {}).get("desktop", {}).get("page", ""),
                }
    except Exception:
        pass

    return None


@st.cache_data(ttl=3600, show_spinner=False)
def get_wiki_img(title: str) -> str:
    """
    Fetch a poster/thumbnail from Wikipedia's pageimages API.
    Tries multiple query variants.
    """
    clean = _clean_title(title)
    queries = [
        clean + " film",
        clean + " movie",
        clean,
    ]
    for q in queries:
        try:
            r = requests.get(
                "https://en.wikipedia.org/w/api.php",
                params={
                    "action":      "query",
                    "titles":      q,
                    "prop":        "pageimages",
                    "format":      "json",
                    "pithumbsize": 500,
                    "pilicense":   "any",
                },
                headers=WIKI_HEADERS,
                timeout=8,
            )
            if r.status_code == 200:
                pages = r.json().get("query", {}).get("pages", {})
                for page in pages.values():
                    img = page.get("thumbnail", {}).get("source", "")
                    if img:
                        return img
        except Exception:
            continue
    return ""


def get_poster(title, key):
    o = get_omdb(title, key)
    if o and o.get("poster"): return o["poster"]
    img = get_wiki_img(title)
    return img if img else ""


def fetch_movie_data_parallel(titles, omdb_key):
    import concurrent.futures
    import threading
    from streamlit.runtime.scriptrunner import add_script_run_ctx, get_script_run_ctx
    ctx = get_script_run_ctx()
    def fetch_with_ctx(t):
        if ctx: add_script_run_ctx(threading.current_thread(), ctx)
        img = get_poster(t, omdb_key)
        od = get_omdb(t, omdb_key)
        wd = get_wiki(t)
        return t, img, od, wd
    with concurrent.futures.ThreadPoolExecutor(max_workers=min(10, len(titles) if titles else 1)) as executor:
        return dict((res[0], res) for res in executor.map(fetch_with_ctx, titles))


def make_poster(img, title, year, delay=0, desc=""):
    yr  = year or ""
    ttl = (title[:36]+"...") if len(title)>36 else title
    dh  = f'<div class="card-desc">{desc[:100]}...</div>' if desc else ""
    if img:
        return (f'<div class="poster-wrap" style="animation-delay:{delay}s">'
                f'<img src="{img}" class="poster-img"/>'
                f'<div class="poster-year">{yr}</div>'
                f'<div class="poster-overlay"><div class="poster-title">{ttl}</div></div>'
                f'</div>{dh}')
    return (f'<div class="poster-wrap" style="animation-delay:{delay}s">'
            f'<div class="no-poster"><div class="no-poster-icon">🎬</div>'
            f'<div class="no-poster-text">{ttl}</div></div>'
            f'<div class="poster-year">{yr}</div>'
            f'<div class="poster-overlay"><div class="poster-title">{ttl}</div></div>'
            f'</div>{dh}')


def inject_wiki_modal_scaffold():
    """
    Inject the modal DOM + global JS once per page load.
    Must be called before any wiki button is rendered.
    Uses postMessage so the iframe can communicate with the parent Streamlit page.
    """
    st.markdown("""
    <!-- Wikipedia Modal -->
    <div id="wiki-modal-overlay" onclick="if(event.target===this)closeWikiModal()">
      <div id="wiki-modal">
        <div id="wiki-modal-topbar"></div>
        <div id="wiki-modal-header">
          <div id="wiki-modal-poster-placeholder">🎬</div>
          <div id="wiki-modal-meta">
            <div id="wiki-modal-badge">📖 Wikipedia</div>
            <div id="wiki-modal-title">Movie Title</div>
            <div id="wiki-modal-tags"></div>
          </div>
          <button id="wiki-modal-close" onclick="closeWikiModal()">✕</button>
        </div>
        <div id="wiki-modal-body">
          <div id="wiki-modal-section-label">📖 About this film</div>
          <div id="wiki-modal-extract">Loading…</div>
        </div>
        <div id="wiki-modal-footer">
          <span id="wiki-modal-powered">WIKIPEDIA · FREE ENCYCLOPEDIA</span>
          <a id="wiki-modal-wiki-link" href="#" target="_blank">🔗 Full Article ↗</a>
        </div>
      </div>
    </div>
    <script>
    function openWikiModal(data){
        var o = document.getElementById('wiki-modal-overlay');
        document.getElementById('wiki-modal-title').textContent = data.title || '';
        var tags = document.getElementById('wiki-modal-tags');
        tags.innerHTML = '';
        (data.tags||[]).forEach(function(t){
            var s=document.createElement('span');
            s.className='wm-tag';s.textContent=t;tags.appendChild(s);
        });
        var ph = document.getElementById('wiki-modal-poster-placeholder');
        var existing = document.getElementById('wiki-modal-poster');
        if(existing) existing.remove();
        if(data.image){
            var img=document.createElement('img');
            img.id='wiki-modal-poster';img.src=data.image;
            img.onerror=function(){this.style.display='none';ph.style.display='flex';};
            ph.parentNode.insertBefore(img,ph);
            ph.style.display='none';
        } else {
            ph.style.display='flex';
        }
        document.getElementById('wiki-modal-extract').innerHTML = data.extract || 'No information available.';
        var lnk = document.getElementById('wiki-modal-wiki-link');
        if(data.url){ lnk.href=data.url; lnk.style.display='inline-flex'; }
        else { lnk.style.display='none'; }
        o.classList.add('open');
        document.body.style.overflow='hidden';
    }
    function closeWikiModal(){
        document.getElementById('wiki-modal-overlay').classList.remove('open');
        document.body.style.overflow='';
    }
    document.addEventListener('keydown',function(e){
        if(e.key==='Escape') closeWikiModal();
    });
    window.addEventListener('message',function(e){
        if(e.data && e.data.type==='open_wiki_modal'){
            openWikiModal(e.data.payload);
        }
    });
    </script>
    """, unsafe_allow_html=True)


def show_wiki_modal(w: dict, title: str, genre: str = "", year: str = ""):
    """
    Opens the A4-style Wikipedia modal popup via a hidden iframe component.
    Sends a postMessage from a Streamlit component iframe to the parent window.
    """
    import json
    tags = [t for t in [genre, year] if t and t != "—"]
    extract = w.get("extract", "")
    extract_html = "".join(
        f"<p style='margin:0 0 0.85rem 0;'>{p.strip()}</p>"
        for p in extract.split("\n") if p.strip()
    ) or f"<p>{extract}</p>"

    payload = {
        "title":   w.get("title", title),
        "extract": extract_html,
        "image":   w.get("image", ""),
        "url":     w.get("url", ""),
        "tags":    tags,
    }
    payload_json = json.dumps(payload).replace("</", "<\\/")

    iframe_html = f"""
    <script>
    (function(){{
        var payload = {payload_json};
        function fire(){{
            if(window.parent && window.parent.openWikiModal){{
                window.parent.openWikiModal(payload);
            }} else {{
                window.parent.postMessage({{type:'open_wiki_modal',payload:payload}},'*');
            }}
        }}
        if(document.readyState==='loading'){{
            document.addEventListener('DOMContentLoaded',fire);
        }} else {{ fire(); }}
    }})();
    </script>
    """
    components.html(iframe_html, height=0, scrolling=False)


# ── Sidebar ───────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='text-align:center;padding:0.8rem 0 0.5rem;'>
      <div style='font-size:2rem;animation:float 3s ease infinite;display:inline-block;'>🎬</div>
      <div style='font-size:1.2rem;font-weight:900;color:#a78bfa;margin-top:6px;'>CineAI</div>
      <div style='font-size:0.72rem;color:#475569;letter-spacing:1px;'>MOVIE DISCOVERY</div>
    </div>""", unsafe_allow_html=True)
    st.divider()
    st.markdown(f"""
    <div class="sidebar-stat">
      <div style='font-size:1.4rem;'>🎬</div>
      <div><div class="sidebar-stat-val">{TOTAL_MOVIES:,}</div>
           <div class="sidebar-stat-lbl">Total Movies</div></div>
    </div>
    <div class="sidebar-stat">
      <div style='font-size:1.4rem;'>🎭</div>
      <div><div class="sidebar-stat-val">{TOTAL_GENRES}</div>
           <div class="sidebar-stat-lbl">Genres</div></div>
    </div>
    <div class="sidebar-stat">
      <div style='font-size:1.4rem;'>🤖</div>
      <div><div class="sidebar-stat-val">AI</div>
           <div class="sidebar-stat-lbl">TF-IDF · Cosine</div></div>
    </div>""", unsafe_allow_html=True)
    st.divider()
    st.markdown("**🔑 OMDb API Key**")
    st.caption("Free at omdbapi.com")
    omdb_key_input = st.text_input("", value=st.session_state.get("omdb_key", ""),
                              placeholder="Paste OMDb key",
                              label_visibility="collapsed",
                              type="password", key="omdb_sidebar_key")
    if omdb_key_input != st.session_state.get("omdb_key", ""):
        st.session_state["omdb_key"] = omdb_key_input
        st.rerun()
    
    if st.session_state.get("omdb_key"): st.success("✅ Real posters ON!")
    else: st.info("📖 Wikipedia images active")

    st.divider()
    top_n = st.slider("🎬 Results", 5, 20, 10)


# ════════════════════════════════════════════
# HERO
# ════════════════════════════════════════════
fp_imgs = HERO_POSTERS
st.markdown(f"""
<div class="hero-wrap">
  <div class="hero-bg-glow"></div>
  <div class="hero-bg-glow2"></div>
  <div class="hero-bg-glow3"></div>
  <div class="hero-fade-left"></div>
  <div class="hero-fade-right"></div>
  <div class="hero-fade-bottom"></div>
  <div class="fp fp1"><img src="{fp_imgs[0]}" onerror="this.style.display='none'"/></div>
  <div class="fp fp2"><img src="{fp_imgs[1]}" onerror="this.style.display='none'"/></div>
  <div class="fp fp3"><img src="{fp_imgs[2]}" onerror="this.style.display='none'"/></div>
  <div class="fp fp4"><img src="{fp_imgs[3]}" onerror="this.style.display='none'"/></div>
  <div class="fp fp5"><img src="{fp_imgs[4]}" onerror="this.style.display='none'"/></div>
  <div class="fp fp6"><img src="{fp_imgs[5]}" onerror="this.style.display='none'"/></div>
  <div class="hero-content">
    <div class="hero-badge">
      <div class="hero-badge-dot"></div>
      AI POWERED MOVIE DISCOVERY
    </div>
    <div class="hero-title">Find Your Next<br/>Favourite Film</div>
    <div class="hero-sub">
      <span>Personalized Smart AI Suggestions</span> ·
      <span>High-Res Posters</span> ·
      <span>Curated Picks</span>
    </div>
    <div class="hero-stats">
      <div class="hstat"><div class="hstat-icon">🎬</div><div class="hstat-num">{TOTAL_MOVIES:,}</div><div class="hstat-lbl">Movies</div></div>
      <div class="hstat"><div class="hstat-icon">🎭</div><div class="hstat-num">{TOTAL_GENRES}</div><div class="hstat-lbl">Genres</div></div>
      <div class="hstat"><div class="hstat-icon">🤖</div><div class="hstat-num">AI</div><div class="hstat-lbl">Powered</div></div>
    </div>
  </div>
</div>
<div class="main-wrap">
""", unsafe_allow_html=True)


# ════════════════════════════════════════════
# SEARCH
# ════════════════════════════════════════════
st.markdown('<div class="glass">', unsafe_allow_html=True)
st.markdown('<div class="sec-hdr"><div class="sec-accent"></div>🔍 Find Movies Fast</div>',
            unsafe_allow_html=True)
sc1, sc2 = st.columns([6,1])
with sc1:
    q = st.text_input("","",placeholder="Search e.g. Dune, Avatar, Batman...",
                      label_visibility="collapsed", key="q")
with sc2: st.button("Search", key="sb")

if q:
    matched = [t for t in all_titles if q.lower() in t.lower()][:15]
    if matched:
        sel = st.selectbox("", matched, label_visibility="collapsed", key="sel")
        yr  = sel[-5:-1] if sel.endswith(")") else "—"
        with st.spinner("Loading..."):
            omdb = get_omdb(sel, st.session_state.get("omdb_key", ""))
            wiki = get_wiki(sel)
            img  = (omdb.get("poster") if omdb and omdb.get("poster")
                    else (wiki.get("image") if wiki and wiki.get("image") else get_wiki_img(sel)))
        row   = movies[movies["title"]==sel]
        genre = row.iloc[0]["genres"].replace("|"," · ") if len(row)>0 else ""

        # ── Movie card: poster left, info right ───────────────────────
        ic1, ic2 = st.columns([1,3])
        with ic1:
            # Poster only — no Full Details button
            st.markdown(make_poster(img, sel, yr), unsafe_allow_html=True)
        with ic2:
            r_tag  = f'<div class="imdb-badge">⭐ {omdb["rating"]}/10 IMDb</div>' if omdb and omdb.get("rating") else ""
            rt_tag = f'<span class="mtag">⏱️ {omdb["runtime"]}</span>' if omdb and omdb.get("runtime") else ""
            desc   = (omdb.get("plot","") if omdb and omdb.get("plot")
                      else (wiki.get("extract","")[:400] if wiki else ""))
            d_html = f'<div class="movie-plot">{desc}</div>' if desc else ""
            d_line = f'<div class="movie-crew">🎬 {omdb["director"]}</div>' if omdb and omdb.get("director") else ""
            a_line = f'<div class="movie-crew" style="color:#64748b;">🎭 {omdb["actors"]}</div>' if omdb and omdb.get("actors") else ""
            st.markdown(f"""
            <div style="font-size:1.5rem;font-weight:800;color:#e2e8f0;">{sel}</div>
            <div class="movie-meta">
              <span class="mtag">🎭 {genre}</span>
              <span class="mtag">📅 {yr}</span>{rt_tag}
            </div>{r_tag}
            <div style="background:rgba(124,58,237,0.06);border:1px solid rgba(124,58,237,0.2);border-radius:14px;padding:1rem;margin-top:0.8rem;">
              {d_html}{d_line}{a_line}
            </div>""", unsafe_allow_html=True)
            st.markdown("<br/>", unsafe_allow_html=True)
            ba, bb = st.columns(2)
            with ba:
                if st.button("🎬 Get Similar", key="gs"):
                    st.session_state["rec_movie"] = sel
                    st.session_state.pop("wiki_show", None)
            with bb:
                wiki_label = "✕ Hide Wikipedia" if st.session_state.get("wiki_for") == sel and st.session_state.get("wiki_show") else "📖 Wikipedia / AI"
                if st.button(wiki_label, key="wi"):
                    # Toggle: if already showing for this movie, hide it
                    if st.session_state.get("wiki_for") == sel and st.session_state.get("wiki_show"):
                        st.session_state.pop("wiki_show", None)
                        st.session_state.pop("wiki_for", None)
                    else:
                        with st.spinner("🔍 Searching Wikipedia..."):
                            w = get_wiki(sel)
                        st.session_state["wiki_show"] = w
                        st.session_state["wiki_for"]  = sel
                    st.rerun()

        # ── OTT Links Section ─────────────────────────────────────────
        with st.spinner("Checking streaming platforms..."):
            providers = get_justwatch_providers(sel, country=st.session_state.get("region", "IN"))
        
        if providers:
            st.markdown('<div class="ott-container">', unsafe_allow_html=True)
            for p in providers[:4]:
                p_name = p["name"]
                p_link = p["url"]
                st.markdown(f'''
                <div class="ott-item">
                    <div class="ott-label">{p_name}</div>
                    <a href="{p_link}" target="_blank" class="ott-link">Watch on {p_name}</a>
                </div>
                ''', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)

        # ── Inline Wikipedia panel (below movie card, full width) ─────
        if st.session_state.get("wiki_for") == sel and "wiki_show" in st.session_state:
            w = st.session_state["wiki_show"]
            if w and w.get("extract"):
                # Build paragraph blocks
                paras = [p.strip() for p in w["extract"].split("\n") if p.strip()]
                extract_html = "".join(
                    f'<p style="margin:0 0 0.9rem 0;color:#cbd5e1;font-size:0.9rem;'
                    f'line-height:1.9;font-family:Georgia,serif;">{p}</p>'
                    for p in paras
                )
                wiki_img_html = ""
                if w.get("image"):
                    wiki_img_html = (
                        f'<img src="{w["image"]}" style="float:right;width:140px;'
                        f'border-radius:10px;margin:0 0 1rem 1.4rem;'
                        f'border:1px solid rgba(124,58,237,0.35);"/>'
                    )
                wiki_url_html = ""
                if w.get("url"):
                    wiki_url_html = (
                        f'<a href="{w["url"]}" target="_blank" '
                        f'style="display:inline-flex;align-items:center;gap:6px;'
                        f'background:linear-gradient(135deg,#7c3aed,#4f46e5);'
                        f'color:white;border-radius:10px;padding:7px 18px;'
                        f'font-size:0.78rem;font-weight:700;text-decoration:none;'
                        f'margin-top:0.5rem;">'
                        f'🔗 Read full article on Wikipedia ↗</a>'
                    )
                st.markdown(f"""
                <div style="
                    margin-top:1.2rem;
                    background:linear-gradient(135deg,rgba(124,58,237,0.07),rgba(99,102,241,0.04));
                    border:1px solid rgba(124,58,237,0.35);
                    border-radius:18px;
                    padding:1.6rem 1.8rem;
                    animation:fadeUp 0.4s ease both;
                    background-image:repeating-linear-gradient(
                        transparent,transparent 30px,
                        rgba(124,58,237,0.05) 30px,rgba(124,58,237,0.05) 31px
                    );
                ">
                  <!-- header row -->
                  <div style="display:flex;align-items:center;gap:10px;margin-bottom:1rem;
                              padding-bottom:0.8rem;border-bottom:1px solid rgba(124,58,237,0.18);">
                    <div style="width:4px;height:28px;border-radius:4px;
                                background:linear-gradient(180deg,#7c3aed,#3b82f6);flex-shrink:0;"></div>
                    <div>
                      <div style="font-size:1rem;font-weight:800;color:#a78bfa;line-height:1.2;">
                        📖 {w.get("title", sel)}
                      </div>
                      <div style="font-size:0.65rem;color:#475569;letter-spacing:1.5px;
                                  text-transform:uppercase;margin-top:2px;">
                        Wikipedia · Free Encyclopedia
                      </div>
                    </div>
                  </div>
                  <!-- body -->
                  <div style="overflow:hidden;">
                    {wiki_img_html}
                    {extract_html}
                    <div style="clear:both;"></div>
                  </div>
                  <!-- footer -->
                  <div style="border-top:1px solid rgba(124,58,237,0.15);
                              padding-top:0.9rem;margin-top:0.5rem;
                              display:flex;align-items:center;justify-content:space-between;
                              flex-wrap:wrap;gap:0.5rem;">
                    <span style="font-size:0.68rem;color:#334155;letter-spacing:1px;">
                      📚 WIKIPEDIA · FREE ENCYCLOPEDIA
                    </span>
                    {wiki_url_html}
                  </div>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="wiki-error" style="margin-top:1rem;">
                  <strong>📖 Wikipedia</strong> — No article found for <em>"{sel}"</em>.<br/>
                  <span style="font-size:0.8rem;color:#94a3b8;">
                    Try searching directly on
                    <a href="https://en.wikipedia.org/wiki/Special:Search?search={quote(_clean_title(sel))}+film"
                       target="_blank" style="color:#a78bfa;">Wikipedia ↗</a>
                  </span>
                </div>
                """, unsafe_allow_html=True)
    else:
        st.warning(f"No movies found for '{q}'")
st.markdown('</div>', unsafe_allow_html=True)

@st.dialog("Movie Details")
def show_movie_details(title):
    with st.spinner("Loading..."):
        img = get_poster(title, st.session_state.get("omdb_key", ""))
        wd  = get_wiki(title)
        wiki_url = wd.get("url") if wd else f"https://en.wikipedia.org/wiki/Special:Search?search={quote(_clean_title(title))}+film"
    
    st.markdown(f"<h3 style='text-align:center;margin-top:0;'>{title}</h3>", unsafe_allow_html=True)
    
    if img:
        st.image(img, use_container_width=True)
    else:
        st.markdown('<div style="aspect-ratio:2/3;background:linear-gradient(135deg,#1e1b4b,#0f172a);display:flex;align-items:center;justify-content:center;font-size:4rem;border-radius:10px;margin-bottom:1rem;">🎬</div>', unsafe_allow_html=True)
        
    c1, c2 = st.columns(2)
    with c1:
        if st.button("Similar", key="dlg_sim", use_container_width=True):
            st.session_state["rec_movie"] = title
            st.session_state.pop("browse", None)
            st.rerun()
    with c2:
        st.link_button("Wikipedia", wiki_url, use_container_width=True)

# ════════════════════════════════════════════
# BROWSE BY GENRE
# ════════════════════════════════════════════
st.markdown('<div class="glass">', unsafe_allow_html=True)
st.markdown('<div class="sec-hdr"><div class="sec-accent"></div>🎭 Browse by Genre</div>',
            unsafe_allow_html=True)
gcols = st.columns(len(KEY_GENRES))
for i, genre in enumerate(KEY_GENRES):
    with gcols[i]:
        if st.button(f"{G_EMOJI[genre]} {genre}", key=f"g_{genre}"):
            st.session_state["browse"] = genre
            st.session_state.pop("rec_movie", None)
if "browse" in st.session_state:
    g   = st.session_state["browse"]
    gm  = movies[movies["genres"].str.contains(g, na=False)]
    gts = sorted(gm["title"].tolist())[:12]
    st.divider()
    st.markdown(f"**{G_EMOJI.get(g,'🎬')} {g}** · {len(gm):,} movies found")
    st.markdown("")
    with st.spinner("Fetching data..."):
        p_data = fetch_movie_data_parallel(gts, st.session_state.get("omdb_key", ""))
    for ri in range(0, 12, 4):
        cols = st.columns(4)
        for j, title in enumerate(gts[ri:ri+4]):
            yr   = title[-5:-1] if title.endswith(")") else ""
            _, img, od, wd = p_data[title]
            desc = (od.get("plot","") if od and od.get("plot")
                    else (wd.get("extract","") if wd else ""))
            with cols[j]:
                st.markdown(make_poster(img, title, yr, j*0.08, desc), unsafe_allow_html=True)
                cb1, cb2 = st.columns(2)
                with cb1:
                    if st.button("📄", key=f"gd_{ri}_{j}", help="Details"):
                        show_movie_details(title)
                with cb2:
                    if st.button("Similar→", key=f"gc_{ri}_{j}"):
                        st.session_state["rec_movie"] = title
                        st.session_state.pop("browse", None)
                        st.rerun()
    if st.button("✖ Close", key="cg"):
        st.session_state.pop("browse", None)
        st.rerun()
st.markdown('</div>', unsafe_allow_html=True)


# ════════════════════════════════════════════
# SIMILAR MOVIES
# ════════════════════════════════════════════
if "rec_movie" in st.session_state:
    rt = st.session_state["rec_movie"]
    st.markdown('<div class="glass">', unsafe_allow_html=True)
    st.markdown(
        f'<div class="sec-hdr"><div class="sec-accent"></div>'
        f'🎬 Similar to <span style="color:#a78bfa;">"{rt}"</span></div>',
        unsafe_allow_html=True)
    with st.spinner("Finding similar movies..."):
        recs = get_cb_recs(rt, movies_cb, cosine_sim, top_n=top_n)
    if isinstance(recs, list) or len(recs) == 0:
        st.warning("No similar movies found.")
    else:
        titles = recs["title"].tolist()
        with st.spinner("Fetching movie details..."):
            p_data = fetch_movie_data_parallel(titles, st.session_state.get("omdb_key", ""))
        for idx, (_, row) in enumerate(recs.iterrows()):
            score = round(row["similarity_score"]*100, 1)
            genre = row["genres"].split("|")[0]
            title = row["title"]
            _, img, od, wd = p_data[title]
            desc  = (od.get("plot","") if od and od.get("plot")
                     else (wd.get("extract","")[:180]+"..." if wd else ""))
            r1, r2 = st.columns([5,1])
            with r1:
                pe = (f'<img src="{img}" class="result-poster"/>'
                      if img else '<div class="result-no-poster">🎬</div>')
                st.markdown(f"""
                <div class="result-item" style="animation-delay:{idx*0.05}s;cursor:pointer;">
                  <span class="result-rank">#{idx+1}</span>{pe}
                  <div style="flex:1;min-width:0;">
                    <div class="result-title">{row['title']}</div>
                    <div class="result-desc">{desc if desc else genre}</div>
                  </div>
                  <span class="result-match">Match {score}%</span>
                </div>""", unsafe_allow_html=True)
            with r2:
                if st.button("📄", key=f"rd_{idx}", help="Details"):
                    show_movie_details(row["title"])
    if st.button("✖ Clear", key="cr"):
        st.session_state.pop("rec_movie", None)
        st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)


# ════════════════════════════════════════════
# RANDOM SUGGESTIONS
# ════════════════════════════════════════════
st.markdown('<div class="glass">', unsafe_allow_html=True)
st.markdown('<div class="sec-hdr"><div class="sec-accent"></div>🎲 Need Inspiration?</div>',
            unsafe_allow_html=True)
if st.button("🎲 Discover Random Movies", key="rand"):
    st.session_state["picks"] = random.sample(all_titles, 6)
if "picks" in st.session_state:
    picks = st.session_state["picks"]
    with st.spinner("Fetching data..."):
        p_data = fetch_movie_data_parallel(picks, st.session_state.get("omdb_key", ""))
    for ri in range(0, 6, 3):
        cols = st.columns(3)
        for j, title in enumerate(picks[ri:ri+3]):
            gv  = movies[movies["title"]==title]["genres"].values
            gs  = gv[0].split("|")[0] if len(gv)>0 else "Unknown"
            yr  = title[-5:-1] if title.endswith(")") else ""
            _, img, od, wd = p_data[title]
            with cols[j]:
                if img:
                    st.markdown(f"""
                    <div class="rcard" style="cursor:pointer;">
                      <img src="{img}" class="rcard-img" style="aspect-ratio:16/9;"/>
                      <div class="rcard-body">
                        <div class="rcard-title">{title}</div>
                        <div class="rcard-genre">🏷️ {gs} · 📅 {yr}</div>
                      </div>
                    </div>""", unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div class="rcard" style="cursor:pointer;">
                      <div style="aspect-ratio:16/9;background:linear-gradient(135deg,#1e1b4b,#0f172a);display:flex;align-items:center;justify-content:center;font-size:2rem;">🎬</div>
                      <div class="rcard-body">
                        <div class="rcard-title">{title}</div>
                        <div class="rcard-genre">🏷️ {gs} · 📅 {yr}</div>
                      </div>
                    </div>""", unsafe_allow_html=True)
                rc1, rc2 = st.columns(2)
                with rc1:
                    if st.button("📄", key=f"rpd_{ri}_{j}", help="Details"):
                        show_movie_details(title)
                with rc2:
                    if st.button("Similar→", key=f"rp_{ri}_{j}"):
                        st.session_state["rec_movie"] = title
                        st.rerun()
st.markdown('</div>', unsafe_allow_html=True)


st.markdown('<div style="text-align:center;padding:2rem 0;color:#1e293b;font-size:0.8rem;">CineAI · MovieLens · OMDb · Wikipedia · TF-IDF</div></div>',
            unsafe_allow_html=True)


import streamlit.components.v1 as components
components.html("""
<script>
const parent = window.parent.document;
parent.addEventListener('click', function(e) {
    let clickable = e.target.closest('.poster-wrap, .result-item, .rcard');
    if(clickable) {
        let container = clickable.closest('[data-testid="stColumn"]');
        let detailBtn = null;
        if(container) {
            detailBtn = Array.from(container.querySelectorAll('button')).find(b => b.innerText.includes('📄'));
        }
        if(!detailBtn) {
            let row = clickable.closest('[data-testid="stHorizontalBlock"]');
            if(row) {
                detailBtn = Array.from(row.querySelectorAll('button')).find(b => b.innerText.includes('📄'));
            }
        }
        if(detailBtn) detailBtn.click();
    }
});
</script>
""", height=0, width=0)