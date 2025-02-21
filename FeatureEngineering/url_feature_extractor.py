    import os
    import sys
    sys.path.append(os.path.abspath(os.path.dirname(__file__) + "/.."))

    from FeatureEngineering import ( UrlLengthFeatureExtractor, SpecialCharactersFeatureExtractor, EmailFeatureExtractor, UrlShorteningFeatureExtractor)
    import pandas as pd

    class UrlFeatureExtractor:
        """Classe principal para coordenar a extração de todas as features das URLs."""

        def __init__(self, df: pd.DataFrame):
            self.df = df

        def extract_all(self) -> pd.DataFrame:
            from FeatureEngineering import (
                UrlLengthFeatureExtractor,
                SpecialCharactersFeatureExtractor,
                EmailFeatureExtractor,
                UrlShorteningFeatureExtractor,
                AbnormalSubdomainExtractor,
                HttpsTokenExtractor,
                TLDExtract,
                IPExtractor,
                DnsExtrator
            )

            # Extrair as features de comprimento
            url_length_extractor = UrlLengthFeatureExtractor(self.df)
            self.df = url_length_extractor.extract()

            # Extrair as features de caracteres especiais
            special_chars_extractor = SpecialCharactersFeatureExtractor(self.df)
            self.df = special_chars_extractor.extract_nb_special_chars()

            # Extrair as features de email
            email_extractor = EmailFeatureExtractor(self.df)
            self.df = email_extractor.extract()

            # Extrair as features de encurtamento
            url_shortening_extractor = UrlShorteningFeatureExtractor(self.df)
            self.df = url_shortening_extractor.extract()

            # Extrair as features de subdomínios anormais
            abnormal_subdomain = AbnormalSubdomainExtractor(self.df)
            self.df = abnormal_subdomain.extract_abnormal_subdomain()

            # Extrair as features de HTTPS
            https_token = HttpsTokenExtractor(self.df)
            self.df = https_token.extract_https_token()

            # Extrair as features de TLD
            tlde = TLDExtract(self.df)
            self.df = tlde.extract_tld_features()

            # Extrair as features de IP
            ip = IPExtractor(self.df)
            self.df = ip.extract_ip()

            # Extrair a feature de DNS
            # dns = DnsExtrator()
            # self.df = dns.has_dns_record(self.url)

            print("Extração de todas as features das URLs concluída.")
            return self.df
