# Recruiter AI App

Ez az alkalmazás egy több ügynökből álló mesterséges intelligencia rendszer, amely dokumentumokat, webes adatokat és egyéb forrásokat képes feldolgozni, keresni és szűrni. A rendszer támogatja a vektoralapú adatbázist, személyes adatok maszkolását, webes keresést, valamint graf-alapú workflow-t.

## Indítás lépései

1. **uv telepítése (ha még nincs):**
   ```bash
   # macOS és Linux
   curl -LsSf https://astral.sh/uv/install.sh | sh
   
   # Windows
   powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
   
   # pip-pel
   pip install uv
   ```

2. **Függőségek telepítése:**
   ```bash
   uv sync
   ```
   
   Vagy csak a produkciós függőségek telepítése:
   ```bash
   uv sync --no-dev
   ```

3. **Redis Stack indítása Dockerrel:**
   ```bash
   docker run -d --name redis-stack-server -p 6380:6379 -p 8001:8001 redis/redis-stack:latest
   ```
   - Redis szerver elérhető: `localhost:6380`
   - RedisInsight UI: [http://localhost:8001](http://localhost:8001)

4. **Környezeti változók beállítása:**
   Másold a `sample.env` tartalmát `.env` néven, és töltsd ki a szükséges kulcsokat.

5. **Alkalmazás indítása:**
   ```bash
   uv run python src/app.py
   ```
   vagy
   ```bash
   ./run.sh
   ```

## Fő API végpontok röviden

- `/api/graph` – Mesterséges intelligencia válasz generálása több ügynökkel (dokumentum, web, szűrés)
- `/api/vector_db/create` – Dokumentumokból vektoralapú adatbázis létrehozása
- `/api/document/...` – Dokumentumok kezelése
- `/api/personal_data_filter/...` – Személyes adatok maszkolása
- `/api/topic_validation/...` – Témaválidáció
- `/api/web_scraping/...` – Webes adatgyűjtés
- `/api/health-check` – Egészségügyi ellenőrzés

További részletek és példák a `/docs` végponton (FastAPI automatikus dokumentáció) érhetők el.

---

Indítás előtt győződj meg róla, hogy minden szükséges környezeti változót beállítottál, és a függőségek telepítve vannak (`uv sync`).


## LangGraph Studio használata (lokális fejlesztéshez)

A LangGraph Studio segítségével vizuálisan tesztelheted és fejlesztheted a LangGraph-alapú alkalmazásodat.

### 1. Telepítsd a LangGraph CLI-t (ha még nincs):

```bash
uv add --dev "langgraph-cli[inmem]"
```

vagy pip-pel:

```bash
pip install -U "langgraph-cli[inmem]"
```

### 2. Indítsd el a LangGraph fejlesztői szervert:

```bash
uv run langgraph dev
```

Ha Safari böngészőt használsz, vagy problémád van a localhost eléréssel, indítsd a szervert a következőképp:

```bash
uv run langgraph dev --tunnel
```

### 3. Nyisd meg a LangGraph Studio-t böngészőben:

Látogasd meg ezt a címet:

https://smith.langchain.com/studio/?baseUrl=http://127.0.0.1:2024

Ha más porton vagy hoston fut a szerver, módosítsd a `baseUrl` paramétert ennek megfelelően.

További információ: [LangGraph Studio Quickstart](https://langchain-ai.github.io/langgraph/cloud/how-tos/studio/quick_start/#local-development-server)

#### (Opcionális) Debugger csatlakoztatása

Ha lépésenként szeretnéd debuggolni az alkalmazást:

```bash
uv add --dev debugpy
uv run langgraph dev --debug-port 5678
```

Ezután VS Code-ban vagy más IDE-ben csatlakozhatsz a 5678-as portra.

## uv vs Poetry összehasonlítás

A projekt áttért Poetry-ról uv-ra a következő előnyök miatt:

- **Gyorsabb telepítés**: uv Rust-ban íródott, jelentősen gyorsabb dependency resolution és telepítés
- **Egyszerűbb használat**: kevesebb parancs, egyszerűbb workflow
- **Modern Python tooling**: korszerű Python projekt menedzsment
- **Jobb teljesítmény**: virtuális környezetek kezelése és package caching
- **Kompatibilitás**: teljes PEP 517/518 támogatás

> **Migrációs útmutató**: Ha Poetry-t használtál korábban, olvasd el a `MIGRATION_TO_UV.md` fájlt a részletes migrációs lépésekért.

### Gyakori uv parancsok:

```bash
# Projekt inicializálása
uv init

# Függőségek telepítése
uv sync

# Dependency hozzáadása
uv add package-name

# Dev dependency hozzáadása
uv add --dev package-name

# Parancs futtatása virtuális környezetben
uv run python script.py

# Python script futtatása
uv run python src/app.py

# Shell aktiválása a virtuális környezetben
uv shell
```
