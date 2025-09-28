DEFAULT_SCRAPY_SETTINGS = {
    # Obey robots.txt rules
    "ROBOTSTXT_OBEY": True,
    # Configure concurrent requests
    "CONCURRENT_REQUESTS": 16,
    "CONCURRENT_REQUESTS_PER_DOMAIN": 8,
    # Configure delays
    "DOWNLOAD_DELAY": 1,
    "RANDOMIZE_DOWNLOAD_DELAY": 0.5,
    # User agent
    "USER_AGENT": "ScrapyWebScraper",
    # Logging
    "LOG_LEVEL": "INFO",
    # Disable telnet console
    "TELNETCONSOLE_ENABLED": False,
    # AutoThrottle extension settings
    "AUTOTHROTTLE_ENABLED": True,
    "AUTOTHROTTLE_START_DELAY": 1,
    "AUTOTHROTTLE_MAX_DELAY": 60,
    "AUTOTHROTTLE_TARGET_CONCURRENCY": 2.0,
    "AUTOTHROTTLE_DEBUG": False,
    # Enable and configure HTTP caching
    "HTTPCACHE_ENABLED": True,
    "HTTPCACHE_EXPIRATION_SECS": 3600,
    "HTTPCACHE_DIR": "httpcache",
    # Request timeout
    "DOWNLOAD_TIMEOUT": 30,
    # Retry settings
    "RETRY_ENABLED": True,
    "RETRY_TIMES": 3,
    "RETRY_HTTP_CODES": [500, 502, 503, 504, 408, 429],
    # Handle redirects
    "REDIRECT_ENABLED": True,
    "REDIRECT_MAX_TIMES": 20,
}


NO_ROBOTTXT_SCRAPING_SETTINGS = {
    **DEFAULT_SCRAPY_SETTINGS,
    "ROBOTSTXT_OBEY": False,
    "DOWNLOAD_DELAY": 0.1,
    "CONCURRENT_REQUESTS": 32,
    "CONCURRENT_REQUESTS_PER_DOMAIN": 16,
}

POLITE_SCRAPING_SETTINGS = {
    **DEFAULT_SCRAPY_SETTINGS,
    "ROBOTSTXT_OBEY": True,
    "DOWNLOAD_DELAY": 3,
    "RANDOMIZE_DOWNLOAD_DELAY": 0.8,
    "CONCURRENT_REQUESTS": 8,
    "CONCURRENT_REQUESTS_PER_DOMAIN": 2,
}

IGNORED_EXTENSIONS = [
    # Documents
    "pdf",
    "doc",
    "docx",
    "xls",
    "xlsx",
    "ppt",
    "pptx",
    "odt",
    "ods",
    "odp",
    "rtf",
    "txt",
    "csv",
    # Archives
    "zip",
    "rar",
    "7z",
    "tar",
    "gz",
    "bz2",
    "xz",
    # Images
    "jpg",
    "jpeg",
    "png",
    "gif",
    "bmp",
    "svg",
    "webp",
    "ico",
    "tiff",
    "tif",
    "psd",
    "ai",
    "eps",
    # Media
    "mp3",
    "mp4",
    "avi",
    "mov",
    "wmv",
    "flv",
    "mkv",
    "webm",
    "wav",
    "flac",
    "aac",
    "ogg",
    "m4a",
    # Executables
    "exe",
    "msi",
    "deb",
    "rpm",
    "dmg",
    "pkg",
    "app",
    # Fonts
    "ttf",
    "otf",
    "woff",
    "woff2",
    "eot",
    # Other
    "iso",
    "torrent",
    "magnet",
]

CONTENT_SELECTORS = [
    "main",
    "article",
    '[role="main"]',
    ".content",
    "#content",
    ".post",
    ".entry",
    ".article-content",
    "body",
]

EXCLUDED_SELECTORS = [
    "nav",
    "header",
    "footer",
    ".navigation",
    ".nav",
    ".menu",
    ".sidebar",
    ".advertisement",
    ".ads",
    ".social",
    "script",
    "style",
    "noscript",
]
