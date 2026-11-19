"""Access to SEC EDGAR: the download side of ingestion (T1.1).

- `client.py`: an HTTP client that keeps to the SEC's fair access rules (rate limit, retries,
  a declared User-Agent).
- `corpus.py`: chooses the filings and pins them by accession number in `config/corpus.yaml`.
- `xbrl.py`: reads the XBRL facts embedded in a filing's inline HTML, segment values included
  (T1.4, T1.5).
"""
