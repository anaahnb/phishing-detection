import requests

def check_google_index(url):
    try:
        response = requests.get(f"https://www.google.com/search?q=site:{url}", timeout=3)
        return 1 if "Nenhum resultado encontrado" not in response.text else 0
    except:
        return 0
    self.df["google_index"] = self.df["url"].apply(check_google_index)
