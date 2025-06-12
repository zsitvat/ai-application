# Recruiter AI App

Ez az alkalmazás egy FastAPI-alapú, több ügynökből álló AI rendszer, amelynek célja, hogy toborzási folyamatokat támogasson és automatizáljon. A rendszer képes dokumentumfeldolgozásra, webes keresésre, vektordatabase-kezelésre, személyes adatok szűrésére, valamint összetett kérdések megválaszolására különböző forrásokból (dokumentum, web, adatbázis) származó információk alapján.

## Főbb funkciók

- **Több ügynök (multi-agent) architektúra**: Dokumentumfeldolgozás, web scraping, személyes adatszűrés, témaválidáció, graf-alapú keresés.
- **Vektordatabase támogatás**: Dokumentumokból vektordatabase-t tud létrehozni és abban keresni.
- **Webes keresés**: Külső keresőmotorok (pl. Google) integrációja.
- **Személyes adatok szűrése**: Személyes adatok felismerése és maszkolása.
- **Ratelimit middleware**: API végpontok terhelésének szabályozása.
- **Részletes naplózás**: Testreszabható logolási szint.

## Használat

1. **Környezeti változók beállítása**  
   Másold a `sample.env` tartalmát `.env` néven, és töltsd ki a szükséges kulcsokat.

2. **Futtatás**  
   - Helyi futtatás:  
     ```bash
     ./run.sh
     ```
   - Dockerrel:  
     ```bash
     sudo docker build -t <docker_name> .
     docker run -d -p 5000:5000 --name <container_name> <docker_name>
     ```

3. **Tesztelés**  
   ```bash
   python -m pytest
   ```

## Fő API végpontok

### 1. Dokumentum vektordatabase létrehozása

- **Végpont:** `/api/vector_db/create`
- **Módszer:** `POST`
- **Leírás:** Létrehoz egy vektordatabase-t a megadott dokumentum(ok)ból.
- **Példa kérés:**
    ```json
    {
      "db_path": "deeplake_databases/deeplake_db_pdf",
      "db_type": "deeplake",
      "documents": ["files/file.pdf"],
      "sheet_name": "pharagraph-dataset",
      "chunk_size": 3000,
      "chunk_overlap": 200,
      "overwrite": true,
      "model": {
        "name": "text-embedding-3-large",
        "provider": "openai"
      }
    }
    ```

### 2. Válasz generálása (AI agent)

- **Végpont:** `/api/graph`
- **Módszer:** `POST`
- **Leírás:** Kérdés feldolgozása, AI által generált válasz, több ügynök bevonásával (pl. dokumentum, web, személyes adatszűrés).
- **Példa kérés:**
    ```json
    {
      "prompt": "kérdés",
      "user_input": "Mit tudsz a wifiről?",
      "model": {
        "name": "gpt-4o-mini",
        "type": "chat",
        "deployment": null,
        "provider": "openai",
        "temperature": 0
      },
      "tools": [
        {
          "name": "retriver_tool",
          "vector_db_path": "deeplake_databases/deeplake_db_pdf",
          "model": {
            "name": "text-embedding-3-large",
            "type": "embedding",
            "deployment": null,
            "provider": "openai"
          },
          "search_kwargs": {
            "k": 5,
            "threshold": 0.5,
            "search_type": "similarity"
          }
        },
        {
          "name": "web_search_tool",
          "engine": "google"
        }
      ]
    }
    ```

### 3. Egyéb végpontok

- `/api/health-check` – Egészségügyi ellenőrzés
- `/api/document/...` – Dokumentumkezelés
- `/api/dataset/...` – Dataset műveletek
- `/api/personal_data_filter/...` – Személyes adatszűrés
- `/api/topic_validation/...` – Témaválidáció
- `/api/web_scraping/...` – Web scraping

## Környezeti változók

- `LOG_LEVEL` – Naplózási szint (pl. DEBUG, INFO)
- `PORT` – API port (alapértelmezett: 5000)
- `OPENAI_API_KEY`, `AZURE_OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, `BEDROCK_AWS_ACCESS_KEY`, `BEDROCK_AWS_SECRET_KEY` – AI szolgáltatók kulcsai
- `LANGFUSE_SECRET_KEY`, `LANGFUSE_PUBLIC_KEY`, `LANGFUSE_PROJECT`, `LANGFUSE_ENDPOINT` – Langfuse tracerhez
- `SERPAPI_API_KEY` – SerpAPI kulcs webes kereséshez
- stb.

## Dokumentáció & Források

- [LangChain Python](https://python.langchain.com/docs/how_to/migrate_agent/)
- [LangGraph](https://langchain-ai.github.io/langgraph/#example)
- [SerpAPI](https://python.langchain.com/docs/integrations/tools/serpapi/)
- [FastAPI Testing](https://fastapi.tiangolo.com/tutorial/testing/#using-testclient)

---

Győződj meg róla, hogy minden szükséges környezeti változót beállítottál, és a szükséges függőségek telepítve vannak (`poetry install` vagy `pip install -r requirements.txt`).