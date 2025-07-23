

# Recruiter AI App

Ez az alkalmazás egy több ügynökből álló mesterséges intelligencia rendszer, amely dokumentumokat, webes adatokat és egyéb forrásokat képes feldolgozni, keresni és szűrni. A rendszer támogatja a vektoralapú adatbázist, személyes adatok maszkolását, webes keresést, valamint graf-alapú workflow-t.

## Indítás lépései

1. **Poetry telepítése (ha még nincs):**
   ```bash
   pip install poetry
   ```

2. **Függőségek telepítése:**
   ```bash
   poetry install
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
   poetry run python main.py
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

Indítás előtt győződj meg róla, hogy minden szükséges környezeti változót beállítottál, és a függőségek telepítve vannak (`poetry install`).