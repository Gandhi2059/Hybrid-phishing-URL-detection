"""
src/rule_engine.py
------------------
Heuristic rule-based URL scoring engine.

Each rule assigns a penalty score; higher total score → more suspicious.
The engine is intentionally lightweight so it adds zero inference latency.
"""

import re


class RuleEngine:
    """
    Evaluate a URL against a fixed set of heuristic rules.

    Rules (score contribution)
    --------------------------
    1. No HTTPS                      : +2
    2. @ symbol in URL               : +3
    3. Suspicious keyword (per match): +2
    4. Long URL (> 75 chars)         : +1
    5. IP address in URL             : +3
    6. Excessive sub-domains (> 3)   : +1

    Maximum possible score: 2 + 3 + (5 × 2) + 1 + 3 + 1 = 20
    """

    _SUSPICIOUS_KEYWORDS: list[str] = [
        "login", "verify", "secure", "bank", "account",
    ]
    _IP_PATTERN = re.compile(r"(?:[0-9]{1,3}\.){3}[0-9]{1,3}")

    # Default tuneable thresholds
    SCORE_NO_HTTPS    = 2
    SCORE_AT_SYMBOL   = 3
    SCORE_KEYWORD     = 2
    SCORE_LONG_URL    = 1
    SCORE_IP          = 3
    SCORE_SUBDOMAINS  = 1
    LONG_URL_THRESH   = 75
    SUBDOMAIN_THRESH  = 3

    def evaluate(self, url: str) -> tuple[int, list[str]]:
        """
        Score a URL against all heuristic rules.

        Parameters
        ----------
        url : str
            The URL to evaluate.

        Returns
        -------
        score : int
            Cumulative penalty score (0 = clean).
        details : list[str]
            Human-readable description of each triggered rule.
        """
        score   = 0
        details = []

        # Rule 1 — HTTPS
        if not url.startswith("https://"):
            score += self.SCORE_NO_HTTPS
            details.append(f"No HTTPS (+{self.SCORE_NO_HTTPS})")

        # Rule 2 — @ symbol
        if "@" in url:
            score += self.SCORE_AT_SYMBOL
            details.append(f"Contains @ symbol (+{self.SCORE_AT_SYMBOL})")

        # Rule 3 — suspicious keywords
        url_lower = url.lower()
        for kw in self._SUSPICIOUS_KEYWORDS:
            if kw in url_lower:
                score += self.SCORE_KEYWORD
                details.append(f"Keyword '{kw}' (+{self.SCORE_KEYWORD})")

        # Rule 4 — long URL
        if len(url) > self.LONG_URL_THRESH:
            score += self.SCORE_LONG_URL
            details.append(f"Long URL (>{self.LONG_URL_THRESH} chars) (+{self.SCORE_LONG_URL})")

        # Rule 5 — IP address
        if self._IP_PATTERN.search(url):
            score += self.SCORE_IP
            details.append(f"IP address in URL (+{self.SCORE_IP})")

        # Rule 6 — excessive sub-domains
        if url.count(".") > self.SUBDOMAIN_THRESH:
            score += self.SCORE_SUBDOMAINS
            details.append(f"Many sub-domains/dots (>{self.SUBDOMAIN_THRESH}) (+{self.SCORE_SUBDOMAINS})")

        return score, details


if __name__ == "__main__":
    engine = RuleEngine()
    _tests = [
        "http://secure-login.bank-update.com/verify?account=1234",
        "https://google.com",
        "http://192.168.0.1/login",
    ]
    for _url in _tests:
        _s, _d = engine.evaluate(_url)
        print(f"URL   : {_url}\nScore : {_s}\nRules : {_d}\n")
