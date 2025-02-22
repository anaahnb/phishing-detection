import re
import tldextract
import whois
import datetime
import requests
import socket
import pandas as pd
from urllib.parse import urlparse
from tqdm import tqdm

class PhishingFeatureExtractor:
    def __init__(self, df: pd.DataFrame):
        self.df = df

    def extract_features(self) -> pd.DataFrame:
        self.df['phish_hints'] = self.df['url'].apply(self.check_phish_hints)
        self.df['length_words_raw'] = self.df['url'].apply(self.count_words)
        self.df['brand_in_path'] = self.df['url'].apply(self.brand_in_path)
        self.df['avg_word_path'] = self.df['url'].apply(self.avg_word_length)
        self.df['longest_words_raw'] = self.df['url'].apply(self.longest_word)
        self.df['port'] = self.df['url'].apply(self.extract_port)
        self.df['prefix_suffix'] = self.df['url'].apply(lambda x: 1 if '-' in urlparse(x).netloc else 0)

        # Processa features baseadas no HTML separadamente
        tqdm.pandas()
        self.df = self.df.progress_apply(self.process_html_features, axis=1)

        return self.df

    def process_html_features(self, row):
        """Processa as features que dependem do HTML separadamente."""
        html = self.fetch_html(row['url'])
        row['safe_anchor'] = self.check_safe_anchor(html)
        row['external_favicon'] = self.check_external_favicon(html)
        row['sfh'] = self.check_sfh(html)
        row['nb_hyperlinks'] = self.count_hyperlinks(html)
        row['ratio_nullHyperlinks'] = self.ratio_null_hyperlinks(html)
        row['domain_in_title'] = self.domain_in_title(row['url'], html)
        return row

    def fetch_html(self, url: str) -> str:
        """Baixa o HTML da página."""
        try:
            response = requests.get(url, timeout=2)
            if response.status_code == 200:
                return response.text
        except requests.RequestException:
            pass
        return ""

    def check_safe_anchor(self, html: str) -> int:
        return 1 if re.search(r'href="#"', html) else 0

    def check_phish_hints(self, url: str) -> int:
        hints = ['secure', 'account', 'webscr', 'login', 'ebayisapi', 'signin']
        return any(hint in url.lower() for hint in hints)

    def count_words(self, url: str) -> int:
        return len(re.findall(r'\w+', url))

    def check_external_favicon(self, html: str) -> int:
        return 1 if re.search(r'rel="shortcut icon".*http', html) else 0

    def check_sfh(self, html: str) -> int:
        return 1 if re.search(r'<form.*action="(\"|#)"', html) else 0

    def count_hyperlinks(self, html: str) -> int:
        return len(re.findall(r'<a\s+', html))

    def brand_in_path(self, url: str) -> int:
        ext = tldextract.extract(url)
        domain = ext.domain.lower()
        path = urlparse(url).path.lower()
        return 1 if domain in path else 0

    def ratio_null_hyperlinks(self, html: str) -> float:
        total_links = len(re.findall(r'<a\s+', html))
        null_links = len(re.findall(r'<a\s+href="(#|)"', html))
        return null_links / total_links if total_links > 0 else 0

    def domain_in_title(self, url: str, html: str) -> int:
        ext = tldextract.extract(url)
        domain = ext.domain.lower()
        title_match = re.search(r'<title>(.*?)</title>', html, re.IGNORECASE)
        if title_match:
            title = title_match.group(1).lower()
            return 1 if domain in title else 0
        return 0

    def avg_word_length(self, url: str) -> float:
        words = re.findall(r'\w+', urlparse(url).path)
        return sum(len(word) for word in words) / len(words) if words else 0

    def longest_word(self, url: str) -> int:
        words = re.findall(r'\w+', url)
        return max(len(word) for word in words) if words else 0

    def extract_port(self, url: str) -> int:
        parsed_url = urlparse(url)
        return parsed_url.port if parsed_url.port else 80
