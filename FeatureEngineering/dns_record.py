from Settings.keys import ParamsKeys
import pandas as pd
from urllib.parse import urlparse
import socket

class DnsExtrator:
    def has_dns_record(self, url) -> pd.DataFrame:
        try:
            # Tenta resolver o DNS da URL
            socket.gethostbyname(urlparse(url).hostname)
            return pd.DataFrame([1], columns=['dns_record'])
        except:
            return pd.DataFrame([0], columns=['dns_record'])
