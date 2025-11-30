import re
from urllib.parse import urlparse
from typing import List, Dict

from ..core.detection_helpers import load_json_safe

URL_REGEX = re.compile(r"((?:https?://|http://|www\.)[^\s,;]+)", re.IGNORECASE)


def extract_urls(text: str) -> List[str]:
    """Return a list of URL-like substrings found in `text`."""
    return URL_REGEX.findall(text or "")


def _normalize_netloc(netloc: str) -> str:
    # remove credentials and port
    if '@' in netloc:
        netloc = netloc.split('@', 1)[1]
    if ':' in netloc:
        netloc = netloc.split(':', 1)[0]
    return netloc.strip().strip('.')


def get_domain_from_url(url: str) -> str:
    """Parse a URL-like string and return the domain/netloc.

    If the input lacks a scheme, a scheme will be assumed so parsing works.
    """
    if not url.lower().startswith(('http://', 'https://')):
        url = 'http://' + url
    try:
        p = urlparse(url)
        return _normalize_netloc(p.netloc)
    except Exception:
        return url


def detect_homoglyphs_in_text(text: str, mapping_file: str = 'homoglyphs.json') -> List[Dict]:
    """Detect homoglyph characters anywhere in a text string.

    Returns list of dicts: `char`, `looks_like`, `position`.
    """
    mapping = load_json_safe(mapping_file, default={})
    if not mapping:
        return []

    found = []
    for i, ch in enumerate(text or ""):
        if ch in mapping:
            found.append({
                'char': ch,
                'looks_like': mapping[ch],
                'position': i
            })
    return found


def detect_homoglyphs_in_domain(domain: str, mapping_file: str = 'homoglyphs.json') -> Dict:
    """Check each label in `domain` for homoglyphs and return a structured result.

    Returned dict example:
    {
      'domain': 'xn--exmple-9ta.com',
      'labels': [
         { 'label': 'exаmple', 'issues': [ {char, looks_like, pos}, ... ], 'normalized': 'example' }
      ]
    }
    """
    mapping = load_json_safe(mapping_file, default={})
    labels = []
    if not domain:
        return {'domain': domain, 'labels': labels}

    for label in domain.split('.'):
        issues = []
        normalized_chars = []
        for i, ch in enumerate(label):
            if ch in mapping:
                issues.append({'char': ch, 'looks_like': mapping[ch], 'position': i})
                normalized_chars.append(mapping[ch])
            else:
                normalized_chars.append(ch)

        labels.append({
            'label': label,
            'issues': issues,
            'normalized': ''.join(normalized_chars)
        })

    return {'domain': domain, 'labels': labels}


def analyze_text_for_urls_and_homoglyphs(text: str, mapping_file: str = 'homoglyphs.json') -> Dict:
    """High-level helper: extract URLs and run homoglyph checks on domains.

    Returns dict with `urls` and `domains` results.
    """
    urls = extract_urls(text)
    domains = []
    for u in urls:
        domain = get_domain_from_url(u)
        domains.append({
            'url': u,
            'domain': domain,
            'homoglyphs': detect_homoglyphs_in_domain(domain, mapping_file)
        })

    # Also check plaintext for single-character homoglyphs
    text_homoglyphs = detect_homoglyphs_in_text(text, mapping_file)

    return {'urls': urls, 'domains': domains, 'text_homoglyphs': text_homoglyphs}
# Extract/validate URLs