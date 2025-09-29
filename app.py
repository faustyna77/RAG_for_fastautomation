import os
import urllib.parse
from typing import List, Optional, Set, Deque
from collections import deque

import httpx
from bs4 import BeautifulSoup
import google.generativeai as genai
import asyncpg

# --- readability: miękki import + fallback ---
try:
    from readability import Document
    HAVE_READABILITY = True
except Exception:
    Document = None
    HAVE_READABILITY = False

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel


# =======================
# KONFIG (ENV)
# =======================
SITE_BASE      = os.getenv("SITE_BASE")      # np. https://www.fastautomation.pl
NEON_URL       = os.getenv("NEON_URL")       # postgres://user:pass@host/db
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY") # z ai.google.dev

EMBED_MODEL = os.getenv("EMBED_MODEL", "models/text-embedding-004")
CHAT_MODEL  = os.getenv("CHAT_MODEL",  "models/gemini-1.5-flash")

genai.configure(api_key=GOOGLE_API_KEY)


app = FastAPI(title="FastAutomation Chat (Gemini + Neon)")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # zawęź do frontendu w produkcji
    allow_methods=["*"],
    allow_headers=["*"],
)


# =======================
# DB START/STOP
# =======================
@app.on_event("startup")
async def startup():
    app.state.pool = await asyncpg.create_pool(NEON_URL)

@app.on_event("shutdown")
async def shutdown():
    await app.state.pool.close()


# =======================
# POMOCNICZE
# =======================
def embed(text: str) -> List[float]:
    """Embedding przez Google AI Studio (Gemini)."""
    if not GOOGLE_API_KEY:
        raise RuntimeError("Brak GOOGLE_API_KEY – ustaw w Secrets Space.")
    text = text.replace("\n", " ")
    r = genai.embed_content(model=EMBED_MODEL, content=text)
    return r["embedding"]

def chat_llm(system: str, messages: List[dict]) -> str:
    """Prosty czat: system_instruction + historia wklejona do jednego promptu."""
    if not GOOGLE_API_KEY:
        return f"(Demo) Odebrałam: {messages[-1]['content']}"
    model = genai.GenerativeModel(CHAT_MODEL, system_instruction=system)

    buf = []
    for m in messages:
        role = "Użytkownik" if m.get("role") == "user" else "Asystent"
        buf.append(f"{role}: {m.get('content','')}")
    prompt = "\n".join(buf)

    resp = model.generate_content(prompt)
    return (resp.text or "").strip()

def chunk_text(text: str, tokens_est: int = 600) -> List[str]:
    max_chars = tokens_est * 4
    parts, buf, curr = [], [], 0
    for seg in text.split("\n"):
        s = seg.strip()
        if not s:
            s = "\n"
        if curr + len(s) > max_chars:
            parts.append(" ".join(buf).strip())
            buf, curr = [s], len(s)
        else:
            buf.append(s)
            curr += len(s)
    if buf:
        parts.append(" ".join(buf).strip())
    return [p for p in parts if p.strip()]

def same_site(url: str, base: str) -> bool:
    try:
        u = urllib.parse.urlparse(url)
        b = urllib.parse.urlparse(base)
        return u.scheme in ("http", "https") and u.netloc == b.netloc
    except Exception:
        return False

def absolutize(link: str, base: str) -> str:
    return urllib.parse.urljoin(base, link)

def extract_text(html: str) -> str:
    """Wyciągnij czytelny tekst z HTML (readability -> fallback BS4)."""
    if HAVE_READABILITY:
        try:
            doc = Document(html)
            main_html = doc.summary()
            txt = BeautifulSoup(main_html, "lxml").get_text(" ", strip=True)
            if len(txt) > 300:
                return txt
        except Exception:
            pass
    return BeautifulSoup(html, "lxml").get_text(" ", strip=True)

def discover_urls(base_url: str, max_pages: int = 50, use_sitemap: bool = True) -> List[str]:
    urls: List[str] = []
    headers = {"User-Agent": "FastAutomation-RAG-Bot/1.0"}

    # 1) spróbuj sitemap.xml
    if use_sitemap:
        sm = urllib.parse.urljoin(base_url, "/sitemap.xml")
        try:
            with httpx.Client(timeout=10, headers=headers) as cli:
                r = cli.get(sm)
                if r.status_code == 200 and "xml" in r.headers.get("content-type", ""):
                    soup = BeautifulSoup(r.text, "xml")
                    locs = [loc.get_text(strip=True) for loc in soup.find_all("loc")]
                    urls = [u for u in locs if same_site(u, base_url)]
                    if urls:
                        return urls[:max_pages]
        except Exception:
            pass

    # 2) fallback: prosty BFS
    seen: Set[str] = set()
    q: Deque[str] = deque([base_url])
    with httpx.Client(timeout=10, headers=headers, follow_redirects=True) as cli:
        while q and len(urls) < max_pages:
            u = q.popleft()
            if u in seen:
                continue
            seen.add(u)
            try:
                r = cli.get(u)
                ctype = r.headers.get("content-type", "")
                if r.status_code != 200 or "text/html" not in ctype:
                    continue
                urls.append(u)
                soup = BeautifulSoup(r.text, "lxml")
                for a in soup.select("a[href]"):
                    href = absolutize(a["href"], u)
                    if same_site(href, base_url) and href not in seen:
                        q.append(href)
            except Exception:
                continue
    return urls


# =======================
# NEON DB (pgvector)
# =======================
async def rag_insert_pg(pool, content: str, source: str, emb: list[float]):
    async with pool.acquire() as conn:
        await conn.execute(
            "INSERT INTO documents (content, source, embedding) VALUES ($1, $2, $3)",
            content, source, emb
        )

async def rag_search_pg(pool, emb: list[float], k: int = 5):
    async with pool.acquire() as conn:
        rows = await conn.fetch(
            """
            SELECT content, source, embedding <=> $1 AS dist
            FROM documents
            ORDER BY embedding <=> $1
            LIMIT $2
            """,
            emb, k
        )
        return [dict(r) for r in rows]


# =======================
# MODELE (Pydantic)
# =======================
class Msg(BaseModel):
    role: str
    content: str

class IngestIn(BaseModel):
    documents: List[str]
    source: Optional[str] = None

class ChatIn(BaseModel):
    message: str
    history: Optional[List[Msg]] = []

class CrawlIn(BaseModel):
    base_url: Optional[str] = None
    max_pages: int = 50
    use_sitemap: bool = True


# =======================
# ENDPOINTY
# =======================
@app.get("/health/db")
async def health_db():
    try:
        async with app.state.pool.acquire() as conn:
            v = await conn.fetchval("SELECT 1")
        return {"ok": True, "db": v}
    except Exception as e:
        return {"ok": False, "error": str(e)}

@app.get("/health/embedding")
def health_embedding():
    try:
        v = embed("test")
        return {"ok": True, "dim": len(v)}
    except Exception as e:
        return {"ok": False, "error": str(e)}

@app.post("/rag/ingest")
async def rag_ingest(inp: IngestIn):
    inserted = 0
    try:
        for doc in inp.documents:
            for chunk in chunk_text(doc, tokens_est=700):
                emb = embed(chunk)
                await rag_insert_pg(app.state.pool, chunk, inp.source, emb)
                inserted += 1
        return {"ok": True, "ingested": inserted}
    except Exception as e:
        return {"ok": False, "error": str(e)}

@app.post("/rag/ingest_site")
async def rag_ingest_site(inp: CrawlIn):
    base = inp.base_url or SITE_BASE
    if not base:
        return {"ok": False, "error": "Podaj base_url lub ustaw SITE_BASE"}

    urls = discover_urls(base, max_pages=inp.max_pages, use_sitemap=inp.use_sitemap)
    inserted = 0
    headers = {"User-Agent": "FastAutomation-RAG-Bot/1.0"}
    async with httpx.AsyncClient(timeout=20, headers=headers, follow_redirects=True) as cli:
        for u in urls:
            try:
                r = await cli.get(u)
                if r.status_code != 200 or "text/html" not in r.headers.get("content-type", ""):
                    continue
                text = extract_text(r.text)
                for chunk in chunk_text(text, tokens_est=700):
                    emb = embed(chunk)
                    await rag_insert_pg(app.state.pool, chunk, u, emb)
                    inserted += 1
            except Exception:
                continue
    return {"ok": True, "pages": len(urls), "chunks": inserted, "domain": base}

@app.post("/api/chat")
async def api_chat(inp: ChatIn):
    context = ""
    try:
        q_emb = embed(inp.message)
        rows = await rag_search_pg(app.state.pool, q_emb, k=5)
        if rows:
            context = "\n\n".join(f"- {row.get('content','')}" for row in rows)
    except Exception:
        pass

    sys = (
        "Jesteś asystentem firmy FastAutomation. Odpowiadasz po polsku, konkretnie. "
        "Gdy podano KONTEKST, opieraj się przede wszystkim na nim. "
        "Jeśli nie wiesz – powiedz to i zaproponuj kontakt."
    )
    msgs: List[dict] = []
    if context:
        msgs.append({"role": "system", "content": f"KONTEKST:\n{context}"})
    for m in (inp.history or []):
        msgs.append({"role": m.role, "content": m.content})
    msgs.append({"role": "user", "content": inp.message})

    reply = chat_llm(sys, msgs)
    return {"reply": reply}


# =======================
# START (lokalnie)
# =======================
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=int(os.environ.get("PORT", "7860")),
        reload=False,
    )
