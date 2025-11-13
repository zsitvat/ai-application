# Web Scraper Service

## Overview

The Web Scraper Service is responsible for automatic extraction and processing of website data. It uses the Scrapy framework for efficient and scalable web scraping functionality, supporting multiple output formats and vector database integration.

## Main Components

### ScrapyWebScrapingService

The main service class that coordinates the web scraping process and handles different output formats.

### ScrapySpider

Scrapy spider implementation that respects domain restrictions and depth limits.

#### Key Features

- Multi-format output: JSON, PDF, DOCX, Vector DB
- Domain restrictions: Secure scraping rules
- Depth control: Configurable crawling depth
- Content filtering: Relevant content extraction
- Asynchronous processing: Efficient parallel execution

## Usage

### Initialization

```python
from src.services.web_scraper.scrapy_web_scraping_service import ScrapyWebScrapingService

scraper_service = ScrapyWebScrapingService()
```

### Basic Scraping

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

## Output Formats

### JSON Output

```python
request = WebScrapingRequestSchema(
    urls=["https://example.com"],
    output_type=OutputType.JSON,
    # additional parameters...
)

# Result: JSON structure with scraped data
result = await scraper_service.scrape_websites(request)
```

### PDF Output

```python
request = WebScrapingRequestSchema(
    urls=["https://example.com"],
    output_type=OutputType.PDF,
    pdf_filename="scraped_content.pdf"
)

# Result: PDF file with base64 encoding
result = await scraper_service.scrape_websites(request)
```

### DOCX Output

```python
request = WebScrapingRequestSchema(
    urls=["https://example.com"],
    output_type=OutputType.DOCX,
    docx_filename="scraped_content.docx"
)

# Result: DOCX file with base64 encoding
result = await scraper_service.scrape_websites(request)
```

### Vector Database Integration

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

## Configuration Parameters

### WebScrapingRequestSchema

```python
class WebScrapingRequestSchema(BaseModel):
    urls: List[str]                    # Scraping targets
    output_type: OutputType            # Output type
    max_depth: int = 1                 # Maximum crawling depth
    max_pages: int = 10                # Maximum number of pages
    respect_robots_txt: bool = True    # Follow robots.txt rules
    delay: float = 1.0                 # Delay between requests
    user_agent: str = "..."            # Custom User-Agent
    
    # Vector DB specific
    vector_db_index: Optional[str]
    embedding_model: Optional[Model]
    chunk_size: int = 1000
    chunk_overlap: int = 200
    
    # File specific
    pdf_filename: Optional[str]
    docx_filename: Optional[str]
```

## Scrapy Configuration

### Spider Settings

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

### Content Filters

```python
CONTENT_SELECTORS = [
    'article',
    'main',
    '[role="main"]',
    '.content',
    '.post',
    '.entry',
    # Additional relevant selectors...
]

EXCLUDED_SELECTORS = [
    'nav',
    'header',
    'footer',
    '.sidebar',
    '.advertisement',
    '.social-share',
    # Additional elements to exclude...
]
```

## Content Processing

### Text Extraction

```python
def extract_content(self, response):
    # Search for relevant content
    content_elements = response.css(' , '.join(CONTENT_SELECTORS))
    
    if not content_elements:
        # Fallback: body content
        content_elements = response.css('body')
    
    # Remove excluded elements
    for selector in EXCLUDED_SELECTORS:
        content_elements.css(selector).remove()
    
    # Extract clean text
    text = ' '.join(content_elements.css('::text').getall())
    return self.clean_text(text)
```

### Text Cleaning

```python
def clean_text(self, text: str) -> str:
    # Remove multiple whitespace
    text = re.sub(r'\s+', ' ', text)
    
    # Normalize special characters
    text = text.replace('\xa0', ' ')  # Non-breaking space
    text = text.replace('\u200b', '')  # Zero-width space
    
    # Trim
    return text.strip()
```

## PDF Generation

### ReportLab Integration

```python
async def generate_pdf(self, scraped_data: list, filename: str) -> str:
    pdf_buffer = BytesIO()
    doc = SimpleDocTemplate(pdf_buffer, pagesize=letter)
    
    # Hungarian font support
    font_path = Path(__file__).parent / "fonts" / "DejaVuSans.ttf"
    if font_path.exists():
        pdfmetrics.registerFont(TTFont('DejaVuSans', str(font_path)))
        default_font = 'DejaVuSans'
    else:
        default_font = 'Helvetica'
    
    story = []
    
    for item in scraped_data:
        # Add title
        title = Paragraph(item.get('title', 'No Title'), 
                         getSampleStyleSheet()['Title'])
        story.append(title)
        
        # Add URL
        url = Paragraph(f"Source: {item.get('url', 'Unknown')}", 
                       getSampleStyleSheet()['Normal'])
        story.append(url)
        
        # Add content
        content = item.get('content', '')
        if len(content) > 5000:  # Truncate too long content
            content = content[:5000] + "..."
        
        content_para = Paragraph(content, getSampleStyleSheet()['Normal'])
        story.append(content_para)
        story.append(Spacer(1, 12))
    
    doc.build(story)
    
    # Base64 encoding
    pdf_buffer.seek(0)
    return base64.b64encode(pdf_buffer.read()).decode('utf-8')
```

## DOCX Generation

### Python-docx Integration

```python
async def generate_docx(self, scraped_data: list, filename: str) -> str:
    doc = Document()
    
    # Document title
    title = doc.add_heading('Web Scraping Results', 0)
    
    for item in scraped_data:
        # Page title
        doc.add_heading(item.get('title', 'No Title'), level=1)
        
        # URL
        url_para = doc.add_paragraph()
        url_para.add_run('Source: ').bold = True
        url_para.add_run(item.get('url', 'Unknown'))
        
        # Content
        content = item.get('content', '')
        if len(content) > 5000:
            content = content[:5000] + "..."
        
        doc.add_paragraph(content)
        doc.add_page_break()
    
    # Save to memory
    docx_buffer = BytesIO()
    doc.save(docx_buffer)
    
    # Base64 encoding
    docx_buffer.seek(0)
    return base64.b64encode(docx_buffer.read()).decode('utf-8')
```

## Vector Database Integration

### Embedding and Storage

```python
async def store_in_vector_db(self, scraped_data: list, request: WebScrapingRequestSchema):
    # Initialize embedding model
    embedding_model = get_embedding_model(request.embedding_model)
    
    # Redis configuration
    redis_config = RedisConfig(
        index_name=request.vector_db_index,
        redis_url=self.redis_url
    )
    
    vector_store = RedisVectorStore(
        embeddings=embedding_model,
        config=redis_config
    )
    
    # Prepare documents
    documents = []
    for item in scraped_data:
        # Text chunking
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
    
    # Save to vector store
    await vector_store.aadd_documents(documents)
    
    return f"Successfully saved {len(documents)} documents to vector DB"
```

## Error Handling

### Network Errors

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

### Content Errors

```python
def validate_content(self, content: str) -> bool:
    # Minimum content check
    if len(content.strip()) < 100:
        return False
    
    # Spam detection
    spam_indicators = ['lorem ipsum', 'placeholder text', 'coming soon']
    if any(indicator in content.lower() for indicator in spam_indicators):
        return False
    
    return True
```

## Performance Optimization

### Asynchronous Reactor

```python
def install_reactor():
    """Install asyncio reactor if not already installed."""
    if "twisted.internet.reactor" not in sys.modules:
        import twisted.internet.asyncioreactor
        twisted.internet.asyncioreactor.install()
```

### Memory Management

```python
class MemoryEfficientPipeline:
    def process_item(self, item, spider):
        # Streaming processing for large content
        if len(item.get('content', '')) > 10000:
            # Process in chunks
            content = item['content']
            chunks = [content[i:i+1000] for i in range(0, len(content), 1000)]
            item['content_chunks'] = chunks
            del item['content']  # Free memory
        
        return item
```

## Security Considerations

### Robots.txt Compliance

```python
custom_settings = {
    'ROBOTSTXT_OBEY': True,  # Follow robots.txt
    'ROBOTSTXT_USER_AGENT': '*'  # User agent for robots.txt
}
```

### Rate Limiting

```python
custom_settings = {
    'DOWNLOAD_DELAY': 1.0,              # Minimum delay
    'RANDOMIZE_DOWNLOAD_DELAY': 0.5,    # Random factor
    'AUTOTHROTTLE_ENABLED': True,       # Adaptive throttling
    'AUTOTHROTTLE_TARGET_CONCURRENCY': 1.0
}
```

### URL Validation

```python
def validate_url(self, url: str) -> bool:
    # Basic URL validation
    parsed = urlparse(url)
    if not parsed.scheme or not parsed.netloc:
        return False
    
    # Security blacklist
    blocked_domains = ['localhost', '127.0.0.1', '0.0.0.0']
    if parsed.netloc in blocked_domains:
        return False
    
    return True
```

## Dependencies

- `scrapy`: Web scraping framework
- `twisted`: Asynchronous networking
- `langchain_redis`: Vector store integration
- `reportlab`: PDF generation
- `python-docx`: DOCX file creation
- `asyncio`: Asynchronous programming
- `base64`: File encoding
