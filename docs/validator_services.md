# Validator Services

## Overview

The Validator Services package is responsible for various validation operations in the application. This includes token validation, personal data filtering, and topic validation.

## Main Components

### TokenValidationService

Token-based text validation and truncation.

### PersonalDataFilterService  

Personal data detection and filtering.

### TopicValidatorService

Topic relevance validation.

## Token Validation Service

### Overview

The `TokenValidationService` is responsible for text token counting and truncation, ensuring we stay within AI model input limits.

### Main Features

- **Token counting**: Support for different encodings
- **Intelligent truncation**: Respecting sentence and word boundaries
- **Batch processing**: Handling multiple texts at once
- **Estimation**: Token count estimation for long texts

### Usage

```python
from src.services.validators.token_validation_service import TokenValidationService

validator = TokenValidationService()

# Text truncation
result = await validator.validate_and_truncate_text(
    text="Long text...",
    max_tokens=1000,
    encoding_name="cl100k_base",
    truncate_from_end=True,
    preserve_sentences=True
)

print(f"Original: {result.original_tokens} tokens")
print(f"Truncated: {result.final_tokens} tokens")
print(f"Was truncated: {result.was_truncated}")
```

### TruncationResult Schema

```python
class TruncationResult(BaseModel):
    original_text: str          # Original text
    truncated_text: str         # Truncated text
    original_tokens: int        # Original token count
    final_tokens: int          # Final token count
    was_truncated: bool        # Whether truncation occurred
```

### Truncation Strategies

#### Truncate from End

```python
await validator.validate_and_truncate_text(
    text="...",
    max_tokens=500,
    truncate_from_end=True  # Cuts from the end of text
)
```

#### Truncate from Beginning

```python
await validator.validate_and_truncate_text(
    text="...",
    max_tokens=500,
    truncate_from_end=False  # Cuts from the beginning of text
)
```

#### Preserve Sentences

```python
await validator.validate_and_truncate_text(
    text="...",
    max_tokens=500,
    preserve_sentences=True  # Preserves complete sentences
)
```

## Personal Data Filter Service

### Overview

The `PersonalDataFilterService` detects and filters personal data from texts to ensure GDPR compliance. The service applies a dual approach: regex-based fast filtering and AI-based contextual detection.

### Dual Filtering Architecture

#### 1. Regex-based Filtering
- **Fast processing**: Deterministic pattern matching
- **High performance**: No AI model overhead
- **Precise patterns**: Email, phone, ID number formats

#### 2. AI-powered Filtering  
- **Contextual understanding**: Considers text context
- **Complex cases**: Cases not handled by regex
- **Language variations**: Different expression patterns

### Detected Data Types

- **Names**: Person name detection (regex + AI)
- **Email addresses**: Email pattern matching (regex primary)
- **Phone numbers**: Hungarian and international formats (regex)
- **Addresses**: Postal addresses (AI primary)
- **Personal IDs**: TAJ number, personal identifiers (regex)
- **Bank accounts**: Account numbers and IBAN (regex)

### Usage

#### Regex-based Filtering

```python
from src.services.validators.personal_data.personal_data_filter_service import PersonalDataFilterService

filter_service = PersonalDataFilterService()

# Regex-based fast filtering
filtered_text = filter_service.apply_regex_replacements(
    text="John Smith email address: janos@example.com, phone number: +36-1-234-5678",
    replacement_patterns={
        r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b': '[EMAIL]',
        r'\+36[-\s]?\d{1,2}[-\s]?\d{3}[-\s]?\d{4}': '[PHONE]'
    },
    mask_character='*'
)

print(filtered_text)
# Output: "John Smith email address: [EMAIL], phone number: [PHONE]"
```

#### AI-based Complete Filtering

```python
# Complete personal data filtering (regex + AI)
filtered_result = await filter_service.filter_personal_data(
    text="John Smith email address: janos@example.com, phone number: +36-1-234-5678",
    replacement_strategy="mask"  # "mask", "remove", "anonymize"
)

print(filtered_result.filtered_text)
# Output: "XXXXX XXXXX email address: XXXXX@XXXXX.com, phone number: +XX-X-XXX-XXXX"
```

### PersonalDataFilterResult Schema

```python
class PersonalDataFilterResult(BaseModel):
    original_text: str              # Original text
    filtered_text: str              # Filtered text
    detected_entities: List[dict]   # Detected entities
    confidence_score: float         # Detection confidence
    used_regex_filtering: bool      # Whether regex filtering was applied
    used_ai_filtering: bool         # Whether AI filtering was applied
```

### Filtering Modes

#### Regex Only
```python
# Fast, deterministic filtering
text = filter_service.apply_regex_replacements(
    text="Email: test@example.com",
    replacement_patterns=filter_service.DEFAULT_PATTERNS
)
```

#### Dual (regex + AI)
```python
# Comprehensive filtering with both methods
result = await filter_service.filter_personal_data(
    text="John Smith works at the company",
    use_regex_first=True  # Regex pre-filtering, then AI
)
```

### Filtering Strategies

#### Masking

```python
# Original: "John Smith"
# Masked: "XXXXX XXXXX"
```

#### Removal

```python
# Original: "Call me at this number: +36-1-234-5678"
# Removed: "Call me at this number: "
```

#### Anonymization

```python
# Original: "John Smith"
# Anonymized: "[PERSON_1]"
```

## Topic Validator Service

### Overview

The `TopicValidatorService` validates whether user input is relevant to the application's domain (recruitment, HR).

### Main Features

- **Topic recognition**: HR/recruitment topic identification
- **Relevance scoring**: Relevance score calculation
- **Keyword analysis**: Topic-specific keyword analysis
- **Intent classification**: Intent categorization

### Usage

```python
from src.services.validators.topic_validator.topic_validator_service import TopicValidatorService

topic_validator = TopicValidatorService()

# Topic validation
validation_result = await topic_validator.validate_topic(
    text="Looking for an experienced Python developer for our team",
    threshold=0.7
)

print(f"Relevant: {validation_result.is_relevant}")
print(f"Score: {validation_result.relevance_score}")
print(f"Category: {validation_result.topic_category}")
```

### TopicValidationResult Schema

```python
class TopicValidationResult(BaseModel):
    is_relevant: bool              # Whether the topic is relevant
    relevance_score: float         # Score between 0-1
    topic_category: str            # Topic category
    detected_keywords: List[str]   # Detected keywords
    confidence: float              # Detection confidence
```

### Supported Topics

#### Recruitment Topics

- Job postings
- Application processes
- Skills and experience
- Salary conditions

#### HR Topics

- Employee evaluations
- Team building
- Training and development
- Work environment

#### Excluded Topics

- Personal finance
- Health information
- Political views
- Religious beliefs

## Checkpoint Integration

### PersonalDataFilterCheckpointer

Special checkpointer for tracking personal data filtering.

```python
from src.services.validators.personal_data.personal_data_filter_checkpointer import PersonalDataFilterCheckpointer

checkpointer = PersonalDataFilterCheckpointer()

# Save checkpoint after filtering
await checkpointer.save_checkpoint(
    thread_id="user_123",
    filter_result=filtered_result,
    metadata={"timestamp": datetime.now()}
)

# Load checkpoint
checkpoint = await checkpointer.load_checkpoint("user_123")
```

## Configuration Options

### Environment Variables

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

### Model Configurations

```python
# Personal data detection
PII_MODEL_CONFIG = {
    "model_name": "hu_core_news_lg",  # Hungarian language model
    "confidence_threshold": 0.8,
    "entity_types": ["PERSON", "EMAIL", "PHONE", "ADDRESS"]
}

# Topic classification
TOPIC_MODEL_CONFIG = {
    "model_name": "distilbert-base-multilingual-cased",
    "classification_threshold": 0.7,
    "supported_categories": ["hr", "recruitment", "general", "off_topic"]
}
```

## Error Handling

### Validation Errors

```python
try:
    result = await validator.validate_and_truncate_text(text, max_tokens)
except ValidationError as e:
    logger.error(f"Validation failed: {e}")
    # Fallback: return original text
    return TruncationResult(
        original_text=text,
        truncated_text=text,
        was_truncated=False,
        error=str(e)
    )
```

### Model Loading Errors

```python
try:
    model = load_spacy_model("hu_core_news_lg")
except OSError:
    logger.warning("Hungarian model not available, fallback to English")
    model = load_spacy_model("en_core_web_sm")
```

## Performance Optimization

### Batch Processing

```python
# Validate multiple texts at once
texts = ["text1", "text2", "text3"]
results = await validator.validate_texts_batch(
    texts=texts,
    max_tokens=1000
)
```

### Model Caching

```python
class ModelCache:
    def __init__(self):
        self._models = {}
    
    def get_model(self, model_name: str):
        if model_name not in self._models:
            self._models[model_name] = load_model(model_name)
        return self._models[model_name]
```

### Async Processing

```python
async def process_multiple_validations(requests):
    tasks = [
        validator.validate_and_truncate_text(req.text, req.max_tokens)
        for req in requests
    ]
    return await asyncio.gather(*tasks)
```

## Security Considerations

### Data Protection

- **Memory cleanup**: Clearing sensitive data after processing
- **Audit log**: Logging personal data handling
- **Encryption**: Encrypting sensitive data

### Compliance

- **GDPR**: EU data protection regulation compliance
- **Privacy by design**: Built-in data protection principle
- **Data minimization**: Minimal data collection principle

## Dependencies

- `tiktoken`: Token counting and encoding
- `spacy`: Natural Language Processing
- `transformers`: Transformer models
- `asyncio`: Asynchronous operations
- `pydantic`: Data validation and schema definitions
