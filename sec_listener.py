"""
The "Ear" - Listens for insider trades from OpenInsider.
Goal: Fetch recent Open Market Purchases and return them programmatically.
"""

import requests
import aiohttp
from bs4 import BeautifulSoup

_OPENINSIDER_URL = "http://openinsider.com/"
_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
}

def _parse_insider_html(html: str) -> list:
    """Shared HTML parser for both sync and async fetchers."""
    soup = BeautifulSoup(html, 'lxml')
    table = soup.find('table', {'class': 'tinytable'})
    if not table:
        print("Could not find the insider trading table.")
        return []

    buys = []
    rows = table.find('tbody').find_all('tr')
    found_buys = 0
    for row in rows[:50]:
        cols = row.find_all('td')
        if len(cols) < 13:
            continue
        if cols[7].text.strip() == 'P - Purchase':
            buys.append({
                'ticker': cols[3].text.strip(),
                'title':  cols[6].text.strip(),
                'value':  cols[12].text.strip(),
            })
            found_buys += 1
            if found_buys >= 10:
                break

    if found_buys == 0:
        print("No recent open market purchases found.")
    return buys


async def listen_for_insider_buys_async() -> list:
    """Async version using aiohttp — non-blocking HTTP fetch."""
    print("Listening for recent Insider Purchases on OpenInsider (async)...")
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(_OPENINSIDER_URL, headers=_HEADERS) as resp:
                resp.raise_for_status()
                html = await resp.text()
    except Exception as e:
        print(f"Failed to fetch data: {e}")
        return []
    return _parse_insider_html(html)


def listen_for_insider_buys():
    """Sync version retained for backward compatibility."""
    print("Listening for recent Insider Purchases on OpenInsider...")
    try:
        response = requests.get(_OPENINSIDER_URL, headers=_HEADERS)
        response.raise_for_status()
    except Exception as e:
        print(f"Failed to fetch data: {e}")
        return []
    return _parse_insider_html(response.text)

if __name__ == "__main__":
    trials = listen_for_insider_buys()
    for t in trials:
        print(f"[{t['ticker']}] - {t['title']} bought {t['value']} in shares.")
