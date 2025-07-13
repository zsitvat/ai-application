# Validator Services

## Áttekintés

A Validator Services csomag felelős a különböző típusú validációs műveletekért az alkalmazásban. Ez magában foglalja a token validációt, személyes adatok szűrését és téma validációt.

## Főbb komponensek

### TokenValidationService

Token-alapú szöveg validáció és csonkítás.

### PersonalDataFilterService  

Személyes adatok felismerése és szűrése.

### TopicValidatorService

Téma relevanciájának validálása.

## Token Validation Service

### Áttekintés

A `TokenValidationService` felelős a szövegek token számlálásáért és csonkításáért, biztosítva hogy az AI modellek input limitjein belül maradjunk.

### Főbb funkciók

- **Token számlálás**: Különböző encoding-ok támogatása
- **Intelligens csonkítás**: Mondat és szó határok tiszteletben tartása
- **Batch feldolgozás**: Több szöveg egyszerre kezelése
- **Becslés**: Token szám becslés long text esetén

### Használat

```python
from src.services.validators.token_validation_service import TokenValidationService

validator = TokenValidationService()

# Szöveg csonkítás
result = await validator.validate_and_truncate_text(
    text="Hosszú szöveg...",
    max_tokens=1000,
    encoding_name="cl100k_base",
    truncate_from_end=True,
    preserve_sentences=True
)

print(f"Eredeti: {result.original_tokens} token")
print(f"Csonkított: {result.final_tokens} token")
print(f"Csonkítva: {result.was_truncated}")
```

### TruncationResult séma

```python
class TruncationResult(BaseModel):
    original_text: str          # Eredeti szöveg
    truncated_text: str         # Csonkított szöveg
    original_tokens: int        # Eredeti token szám
    final_tokens: int          # Végső token szám
    was_truncated: bool        # Történt-e csonkítás
```

### Csonkítási stratégiák

#### Végéről csonkítás

```python
await validator.validate_and_truncate_text(
    text="...",
    max_tokens=500,
    truncate_from_end=True  # Szöveg végéről vág
)
```

#### Elejéről csonkítás

```python
await validator.validate_and_truncate_text(
    text="...",
    max_tokens=500,
    truncate_from_end=False  # Szöveg elejéről vág
)
```

#### Mondat megőrzés

```python
await validator.validate_and_truncate_text(
    text="...",
    max_tokens=500,
    preserve_sentences=True  # Teljes mondatok megőrzése
)
```

## Personal Data Filter Service

### Áttekintés

A `PersonalDataFilterService` felismeri és szűri a személyes adatokat a szövegekből, GDPR megfelelőség biztosítása érdekében.

### Felismert adattípusok

- **Nevek**: Személynevek felismerése
- **Email címek**: Email cím pattern-ek
- **Telefonszámok**: Különböző telefonszám formátumok
- **Címek**: Postai címek
- **Személyi szám**: Állampolgársági azonosítók
- **Bankszámlák**: Számlatulajdonos adatok

### Használat

```python
from src.services.validators.personal_data.personal_data_filter_service import PersonalDataFilterService

filter_service = PersonalDataFilterService()

# Személyes adatok szűrése
filtered_result = await filter_service.filter_personal_data(
    text="Kovács János email címe: janos@example.com, telefonszáma: +36-1-234-5678",
    replacement_strategy="mask"  # "mask", "remove", "anonymize"
)

print(filtered_result.filtered_text)
# Kimenet: "XXXXX XXXXX email címe: XXXXX@XXXXX.com, telefonszáma: +XX-X-XXX-XXXX"
```

### PersonalDataFilterResult séma

```python
class PersonalDataFilterResult(BaseModel):
    original_text: str              # Eredeti szöveg
    filtered_text: str              # Szűrt szöveg
    detected_entities: List[dict]   # Felismert entitások
    confidence_score: float         # Felismerés megbízhatósága
```

### Szűrési stratégiák

#### Maszkolás

```python
# Eredeti: "Kovács János"
# Maszkolva: "XXXXX XXXXX"
```

#### Eltávolítás

```python
# Eredeti: "Hívj fel ezen a számon: +36-1-234-5678"
# Eltávolítva: "Hívj fel ezen a számon: "
```

#### Anonimizálás

```python
# Eredeti: "Kovács János"
# Anonimizálva: "[SZEMÉLY_1]"
```

## Topic Validator Service

### Áttekintés

A `TopicValidatorService` validálja hogy a felhasználói input releváns-e az alkalmazás témaköréhez (toborzás, HR).

### Főbb funkciók

- **Téma felismerés**: HR/toborzás témák azonosítása
- **Relevancia scoring**: Relevanciá pontszám számítás
- **Kulcsszó elemzés**: Téma-specifikus kulcsszavak
- **Intent clasificáció**: Szándék kategorizálás

### Használat

```python
from src.services.validators.topic_validator.topic_validator_service import TopicValidatorService

topic_validator = TopicValidatorService()

# Téma validáció
validation_result = await topic_validator.validate_topic(
    text="Keresek egy tapasztalt Python fejlesztőt a csapatunkba",
    threshold=0.7
)

print(f"Releváns: {validation_result.is_relevant}")
print(f"Pontszám: {validation_result.relevance_score}")
print(f"Kategória: {validation_result.topic_category}")
```

### TopicValidationResult séma

```python
class TopicValidationResult(BaseModel):
    is_relevant: bool              # Releváns-e a téma
    relevance_score: float         # 0-1 közötti pontszám
    topic_category: str            # Téma kategória
    detected_keywords: List[str]   # Felismert kulcsszavak
    confidence: float              # Felismerés megbízhatósága
```

### Támogatott témakörök

#### Toborzás témák

- Álláshirdetések
- Jelentkezési folyamatok
- Képzettségek és tapasztalatok
- Fizetési feltételek

#### HR témák

- Munkavállalói értékelések
- Csapatépítés
- Képzések és fejlesztés
- Munkakörnyezet

#### Kizárt témák

- Személyes pénzügyek
- Egészségügyi információk
- Politikai nézetek
- Vallási meggyőződések

## Checkpoint integráció

### PersonalDataFilterCheckpointer

Speciális checkpointer a személyes adatok szűrésének nyomon követésére.

```python
from src.services.validators.personal_data.personal_data_filter_checkpointer import PersonalDataFilterCheckpointer

checkpointer = PersonalDataFilterCheckpointer()

# Checkpoint mentés szűrés után
await checkpointer.save_checkpoint(
    thread_id="user_123",
    filter_result=filtered_result,
    metadata={"timestamp": datetime.now()}
)

# Checkpoint betöltés
checkpoint = await checkpointer.load_checkpoint("user_123")
```

## Konfigurációs lehetőségek

### Környezeti változók

```bash
# Token validation
DEFAULT_MAX_TOKENS=4000
DEFAULT_ENCODING=cl100k_base

# Personal data filter
PII_DETECTION_THRESHOLD=0.8
PII_REPLACEMENT_STRATEGY=mask

# Topic validation
TOPIC_RELEVANCE_THRESHOLD=0.7
SUPPORTED_LANGUAGES=hu,en
```

### Modell konfigurációk

```python
# Személyes adatok felismerés
PII_MODEL_CONFIG = {
    "model_name": "hu_core_news_lg",  # Magyar nyelvi modell
    "confidence_threshold": 0.8,
    "entity_types": ["PERSON", "EMAIL", "PHONE", "ADDRESS"]
}

# Téma klasszifikáció
TOPIC_MODEL_CONFIG = {
    "model_name": "distilbert-base-multilingual-cased",
    "classification_threshold": 0.7,
    "supported_categories": ["hr", "recruitment", "general", "off_topic"]
}
```

## Hibakezelés

### Validációs hibák

```python
try:
    result = await validator.validate_and_truncate_text(text, max_tokens)
except ValidationError as e:
    logger.error(f"Validation failed: {e}")
    # Fallback: eredeti szöveg visszaadása
    return TruncationResult(
        original_text=text,
        truncated_text=text,
        was_truncated=False,
        error=str(e)
    )
```

### Model betöltési hibák

```python
try:
    model = load_spacy_model("hu_core_news_lg")
except OSError:
    logger.warning("Hungarian model not available, fallback to English")
    model = load_spacy_model("en_core_web_sm")
```

## Teljesítmény optimalizáció

### Batch feldolgozás

```python
# Több szöveg egyszerre validálása
texts = ["szöveg1", "szöveg2", "szöveg3"]
results = await validator.validate_texts_batch(
    texts=texts,
    max_tokens=1000
)
```

### Model caching

```python
class ModelCache:
    def __init__(self):
        self._models = {}
    
    def get_model(self, model_name: str):
        if model_name not in self._models:
            self._models[model_name] = load_model(model_name)
        return self._models[model_name]
```

### Async processing

```python
async def process_multiple_validations(requests):
    tasks = [
        validator.validate_and_truncate_text(req.text, req.max_tokens)
        for req in requests
    ]
    return await asyncio.gather(*tasks)
```

## Biztonsági szempontok

### Adatvédelem

- **Memória tisztítás**: Érzékeny adatok törlése feldolgozás után
- **Audit log**: Személyes adatok kezelésének naplózása
- **Encryption**: Érzékeny adatok titkosítása

### Compliance

- **GDPR**: EU adatvédelmi rendelet megfelelőség
- **Privacy by design**: Adatvédelem beépített alapelv
- **Data minimization**: Minimális adatgyűjtés elve

## Függőségek

- `tiktoken`: Token számolás és encoding
- `spacy`: Natural Language Processing
- `transformers`: Transformer modellek
- `asyncio`: Aszinkron műveletek
- `pydantic`: Adatvalidáció és séma definíciók
