# Web Scraper Service

## Áttekintés

A Web Scraper Service felelős a weboldalak adatainak automatikus kinyeréséért és feldolgozásáért. Scrapy keretrendszert használ a hatékony és skálázható web scraping funkcionalitáshoz, támogatva a többféle kimeneti formátumot és vektoradatbázis integrációt.

## Főbb komponensek

### ScrapyWebScrapingService

A fő service osztály, amely koordinálja a web scraping folyamatot és kezeli a különböző kimeneti formátumokat.

### ScrapySpider

Scrapy spider implementáció, amely tiszteletben tartja a domain korlátozásokat és mélységi limiteket.

#### Főbb funkciók

- **Többformátumú kimenet**: JSON, PDF, DOCX, Vector DB
- **Domain korlátozások**: Biztonságos scraping szabályokkal
- **Mélységi kontroll**: Konfigurálható crawling mélység
- **Tartalom szűrés**: Releváns tartalom kiemelése
- **Aszinkron feldolgozás**: Hatékony párhuzamos végrehajtás

## Használat

### Inicializálás

```python
from src.services.web_scraper.scrapy_web_scraping_service import ScrapyWebScrapingService

scraper_service = ScrapyWebScrapingService()
```

### Alapvető scraping

```python
from src.schemas.web_scraping_schema import OutputType, WebScrapingRequestSchema
from src.schemas.model_schema import Model

request = WebScrapingRequestSchema(
    urls=["https://example.com"],
    output_type=OutputType.JSON,
    max_depth=2,
    max_pages=10,
    respect_robots_txt=True
)

result = await scraper_service.scrape_websites(request)
```

## Kimeneti formátumok

### JSON kimenet

```python
request = WebScrapingRequestSchema(
    urls=["https://example.com"],
    output_type=OutputType.JSON,
    # további paraméterek...
)

# Eredmény: JSON struktúra a scraped adatokkal
result = await scraper_service.scrape_websites(request)
```

### PDF kimenet

```python
request = WebScrapingRequestSchema(
    urls=["https://example.com"],
    output_type=OutputType.PDF,
    pdf_filename="scraped_content.pdf"
)

# Eredmény: PDF fájl base64 kódolással
result = await scraper_service.scrape_websites(request)
```

### DOCX kimenet

```python
request = WebScrapingRequestSchema(
    urls=["https://example.com"],
    output_type=OutputType.DOCX,
    docx_filename="scraped_content.docx"
)

# Eredmény: DOCX fájl base64 kódolással
result = await scraper_service.scrape_websites(request)
```

### Vector Database integráció

```python
model = Model(
    model_name="text-embedding-ada-002",
    model_type="embedding"
)

request = WebScrapingRequestSchema(
    urls=["https://example.com"],
    output_type=OutputType.VECTOR_DB,
    vector_db_index="web_content",
    embedding_model=model,
    chunk_size=1000,
    chunk_overlap=200
)

result = await scraper_service.scrape_websites(request)
```

## Konfigurációs paraméterek

### WebScrapingRequestSchema

```python
class WebScrapingRequestSchema(BaseModel):
    urls: List[str]                    # Scraping célpontok
    output_type: OutputType            # Kimenet típusa
    max_depth: int = 1                 # Maximális crawling mélység
    max_pages: int = 10                # Maximum oldalak száma
    respect_robots_txt: bool = True    # robots.txt szabályok követése
    delay: float = 1.0                 # Kérések közötti delay
    user_agent: str = "..."            # Custom User-Agent
    
    # Vector DB specifikus
    vector_db_index: Optional[str]
    embedding_model: Optional[Model]
    chunk_size: int = 1000
    chunk_overlap: int = 200
    
    # Fájl specifikus
    pdf_filename: Optional[str]
    docx_filename: Optional[str]
```

## Scrapy konfiguráció

### Spider beállítások

```python
class ScrapySpider(Spider):
    name = "spider"
    
    custom_settings = {
        'ROBOTSTXT_OBEY': True,
        'DOWNLOAD_DELAY': 1.0,
        'RANDOMIZE_DOWNLOAD_DELAY': 0.5,
        'CONCURRENT_REQUESTS': 16,
        'CONCURRENT_REQUESTS_PER_DOMAIN': 8,
        'AUTOTHROTTLE_ENABLED': True,
        'AUTOTHROTTLE_START_DELAY': 1,
        'AUTOTHROTTLE_MAX_DELAY': 60,
        'AUTOTHROTTLE_TARGET_CONCURRENCY': 1.0,
    }
```

### Tartalom szűrők

```python
CONTENT_SELECTORS = [
    'article',
    'main',
    '[role="main"]',
    '.content',
    '.post',
    '.entry',
    # További releváns selectorok...
]

EXCLUDED_SELECTORS = [
    'nav',
    'header',
    'footer',
    '.sidebar',
    '.advertisement',
    '.social-share',
    # További kizárandó elemek...
]
```

## Tartalom feldolgozás

### Szöveg kinyerés

```python
def extract_content(self, response):
    # Releváns tartalom keresése
    content_elements = response.css(' , '.join(CONTENT_SELECTORS))
    
    if not content_elements:
        # Fallback: body tartalom
        content_elements = response.css('body')
    
    # Kizárandó elemek eltávolítása
    for selector in EXCLUDED_SELECTORS:
        content_elements.css(selector).remove()
    
    # Tiszta szöveg kinyerése
    text = ' '.join(content_elements.css('::text').getall())
    return self.clean_text(text)
```

### Szöveg tisztítás

```python
def clean_text(self, text: str) -> str:
    # Többszörös whitespace-ek eltávolítása
    text = re.sub(r'\s+', ' ', text)
    
    # Speciális karakterek normalizálása
    text = text.replace('\xa0', ' ')  # Non-breaking space
    text = text.replace('\u200b', '')  # Zero-width space
    
    # Trimmelés
    return text.strip()
```

## PDF generálás

### ReportLab integráció

```python
async def generate_pdf(self, scraped_data: list, filename: str) -> str:
    pdf_buffer = BytesIO()
    doc = SimpleDocTemplate(pdf_buffer, pagesize=letter)
    
    # Magyar betűtípus támogatás
    font_path = Path(__file__).parent / "fonts" / "DejaVuSans.ttf"
    if font_path.exists():
        pdfmetrics.registerFont(TTFont('DejaVuSans', str(font_path)))
        default_font = 'DejaVuSans'
    else:
        default_font = 'Helvetica'
    
    story = []
    
    for item in scraped_data:
        # Cím hozzáadása
        title = Paragraph(item.get('title', 'Nincs cím'), 
                         getSampleStyleSheet()['Title'])
        story.append(title)
        
        # URL hozzáadása
        url = Paragraph(f"Forrás: {item.get('url', 'Ismeretlen')}", 
                       getSampleStyleSheet()['Normal'])
        story.append(url)
        
        # Tartalom hozzáadása
        content = item.get('content', '')
        if len(content) > 5000:  # Túl hosszú tartalom rövidítése
            content = content[:5000] + "..."
        
        content_para = Paragraph(content, getSampleStyleSheet()['Normal'])
        story.append(content_para)
        story.append(Spacer(1, 12))
    
    doc.build(story)
    
    # Base64 kódolás
    pdf_buffer.seek(0)
    return base64.b64encode(pdf_buffer.read()).decode('utf-8')
```

## DOCX generálás

### Python-docx integráció

```python
async def generate_docx(self, scraped_data: list, filename: str) -> str:
    doc = Document()
    
    # Dokumentum címe
    title = doc.add_heading('Web Scraping Eredmények', 0)
    
    for item in scraped_data:
        # Oldal címe
        doc.add_heading(item.get('title', 'Nincs cím'), level=1)
        
        # URL
        url_para = doc.add_paragraph()
        url_para.add_run('Forrás: ').bold = True
        url_para.add_run(item.get('url', 'Ismeretlen'))
        
        # Tartalom
        content = item.get('content', '')
        if len(content) > 5000:
            content = content[:5000] + "..."
        
        doc.add_paragraph(content)
        doc.add_page_break()
    
    # Memóriába mentés
    docx_buffer = BytesIO()
    doc.save(docx_buffer)
    
    # Base64 kódolás
    docx_buffer.seek(0)
    return base64.b64encode(docx_buffer.read()).decode('utf-8')
```

## Vector Database integráció

### Embedding és tárolás

```python
async def store_in_vector_db(self, scraped_data: list, request: WebScrapingRequestSchema):
    # Embedding modell inicializálás
    embedding_model = get_embedding_model(request.embedding_model)
    
    # Redis konfiguráció
    redis_config = RedisConfig(
        index_name=request.vector_db_index,
        redis_url=self.redis_url
    )
    
    vector_store = RedisVectorStore(
        embeddings=embedding_model,
        config=redis_config
    )
    
    # Dokumentumok előkészítése
    documents = []
    for item in scraped_data:
        # Szöveg darabolás
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=request.chunk_size,
            chunk_overlap=request.chunk_overlap
        )
        
        chunks = text_splitter.split_text(item['content'])
        
        for chunk in chunks:
            doc = Document(
                page_content=chunk,
                metadata={
                    'source': item['url'],
                    'title': item.get('title', ''),
                    'timestamp': datetime.now().isoformat()
                }
            )
            documents.append(doc)
    
    # Vector store-ba mentés
    await vector_store.aadd_documents(documents)
    
    return f"Sikeresen mentve {len(documents)} dokumentum a vector DB-be"
```

## Hibakezelés

### Network hibák

```python
def handle_network_errors(self, failure):
    self.logger.error(f"Network error: {failure.value}")
    
    if failure.check(twisted.internet.error.TimeoutError):
        return "Request timeout"
    elif failure.check(twisted.internet.error.DNSLookupError):
        return "DNS lookup failed"
    else:
        return f"Network error: {failure.value}"
```

### Content hibák

```python
def validate_content(self, content: str) -> bool:
    # Minimális tartalom ellenőrzés
    if len(content.strip()) < 100:
        return False
    
    # Spam detection
    spam_indicators = ['lorem ipsum', 'placeholder text', 'coming soon']
    if any(indicator in content.lower() for indicator in spam_indicators):
        return False
    
    return True
```

## Teljesítmény optimalizáció

### Aszinkron Reactor

```python
def install_reactor():
    """Install asyncio reactor if not already installed."""
    if "twisted.internet.reactor" not in sys.modules:
        import twisted.internet.asyncioreactor
        twisted.internet.asyncioreactor.install()
```

### Memória kezelés

```python
class MemoryEfficientPipeline:
    def process_item(self, item, spider):
        # Nagy tartalmak streaming feldolgozása
        if len(item.get('content', '')) > 10000:
            # Chunk-okban feldolgozás
            content = item['content']
            chunks = [content[i:i+1000] for i in range(0, len(content), 1000)]
            item['content_chunks'] = chunks
            del item['content']  # Memória felszabadítás
        
        return item
```

## Biztonsági szempontok

### Robots.txt tisztelet

```python
custom_settings = {
    'ROBOTSTXT_OBEY': True,  # robots.txt követése
    'ROBOTSTXT_USER_AGENT': '*'  # User agent a robots.txt-hez
}
```

### Rate limiting

```python
custom_settings = {
    'DOWNLOAD_DELAY': 1.0,              # Minimum delay
    'RANDOMIZE_DOWNLOAD_DELAY': 0.5,    # Random factor
    'AUTOTHROTTLE_ENABLED': True,       # Adaptive throttling
    'AUTOTHROTTLE_TARGET_CONCURRENCY': 1.0
}
```

### URL validáció

```python
def validate_url(self, url: str) -> bool:
    # Alapvető URL validáció
    parsed = urlparse(url)
    if not parsed.scheme or not parsed.netloc:
        return False
    
    # Biztonsági blacklist
    blocked_domains = ['localhost', '127.0.0.1', '0.0.0.0']
    if parsed.netloc in blocked_domains:
        return False
    
    return True
```

## Függőségek

- `scrapy`: Web scraping keretrendszer
- `twisted`: Aszinkron networking
- `langchain_redis`: Vector store integráció
- `reportlab`: PDF generálás
- `python-docx`: DOCX fájlok létrehozása
- `asyncio`: Aszinkron programozás
- `base64`: Fájl kódolás
