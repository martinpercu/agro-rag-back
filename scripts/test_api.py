"""Test end-to-end del API Agroposta."""
import json
import sys
from pathlib import Path

import httpx

BASE = "http://127.0.0.1:8002"


def main() -> int:
    client = httpx.Client(base_url=BASE, timeout=60.0)

    # 1. Health
    r = client.get("/")
    print(f"[GET /] {r.status_code} {r.json()}")
    assert r.status_code == 200

    # 2. Stats
    r = client.get("/stats")
    stats = r.json()
    print(f"[GET /stats] {stats['total']} chunks, secciones={len(stats['by_section'])}")

    # 3. Chat 1 (sin session)
    r = client.post("/chat", json={"question": "Cual es el precio del novillo en feedlot?"})
    print(f"\n[POST /chat #1] HTTP {r.status_code}")
    data = r.json()
    print(f"  intent: {data['intent']}")
    print(f"  session_id: {data['session_id']}")
    print(f"  answer: {data['answer'][:200]}...")
    print(f"  sources: {len(data['sources'])}")
    session_id = data["session_id"]

    # 4. Chat 2 (misma session)
    r = client.post(
        "/chat",
        json={"question": "Y del maiz tardio?", "session_id": session_id},
    )
    data2 = r.json()
    print(f"\n[POST /chat #2] HTTP {r.status_code}")
    print(f"  intent: {data2['intent']}")
    print(f"  answer: {data2['answer'][:200]}...")

    # 5. Verificar que la sesion tiene 4 mensajes
    r = client.get(f"/sessions/{session_id}")
    sess = r.json()
    print(f"\n[GET /sessions/{session_id[:8]}] mensajes: {len(sess['messages'])}")
    for m in sess["messages"]:
        print(f"  {m['role']}: {m['content'][:90]}")
    assert len(sess["messages"]) == 4, f"esperaba 4 mensajes, hay {len(sess['messages'])}"

    # 6. Export PDF
    r = client.post("/export-pdf", json={"session_id": session_id})
    out = Path("/tmp/agroposta_export.pdf")
    out.write_bytes(r.content)
    print(f"\n[POST /export-pdf] HTTP {r.status_code} -> {out} ({len(r.content)} bytes)")
    assert r.status_code == 200
    assert r.headers["content-type"] == "application/pdf"

    # 7. PDF readability
    from pypdf import PdfReader
    reader = PdfReader(str(out))
    print(f"  PDF: {len(reader.pages)} paginas")
    text = reader.pages[0].extract_text()
    assert "Agroposta" in text
    assert "Productor:" in text
    assert "Fuentes citadas" in text
    print("  PDF contiene header, productor, fuentes citadas OK")

    # 8. Test /chat/stream basico (recolectar eventos)
    r = client.post(
        "/chat/stream",
        json={"question": "Cuanto sale la soja?", "session_id": session_id},
    )
    events = []
    text_tokens = []
    for line in r.text.splitlines():
        if line.startswith("event:"):
            events.append(line.split(":", 1)[1].strip())
        elif line.startswith("data:"):
            try:
                payload = json.loads(line.split(":", 1)[1].strip())
                if "text" in payload:
                    text_tokens.append(payload["text"])
            except Exception:
                pass
    print(f"\n[POST /chat/stream] eventos: {events}")
    print(f"  tokens concatenados: {''.join(text_tokens)[:150]}...")

    # 9. Verificar que la sesion ya tiene 6 mensajes (4 + 2 del stream)
    r = client.get(f"/sessions/{session_id}")
    sess = r.json()
    print(f"\n[GET /sessions] despues del stream, mensajes: {len(sess['messages'])}")
    assert len(sess["messages"]) == 6, f"esperaba 6 mensajes, hay {len(sess['messages'])}"
    print("\nTODO OK ✅")
    return 0


if __name__ == "__main__":
    sys.exit(main())
