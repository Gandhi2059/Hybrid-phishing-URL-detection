"""
src/feature_engineering.py
---------------------------
URL feature extraction: lexical, security, statistical, and NLP (TF-IDF).
"""

import math
import os
import pickle
import re

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer


class FeatureExtractor:
    """
    Extracts a rich, multi-dimensional feature vector from raw URL strings.

    Feature groups
    --------------
    A. Lexical   : length, dot/slash/hyphen/digit/special-char counts
    B. Security  : HTTPS flag, IP-address-in-URL flag
    C. Keyword   : count of suspicious keyword matches
    D. Statistical: Shannon entropy, digit-to-length ratio, special-char ratio
    E. NLP       : character-level TF-IDF n-grams (3–5, optional)

    Parameters
    ----------
    use_tfidf : bool
        Include TF-IDF features (default: True).
    max_tfidf_features : int
        Vocabulary size for the TF-IDF vectorizer (default: 100).
    """

    SUSPICIOUS_KEYWORDS = [
        "login", "verify", "secure", "bank", "account",
        "update", "sign", "confirm", "wallet", "password", "credential",
    ]

    def __init__(self, use_tfidf: bool = True, max_tfidf_features: int = 100):
        self.use_tfidf = use_tfidf
        self.max_tfidf_features = max_tfidf_features
        self.tfidf = TfidfVectorizer(
            max_features=max_tfidf_features,
            analyzer="char",
            ngram_range=(3, 5),
        )
        self.is_fitted = False

    # ------------------------------------------------------------------ #
    # A. Lexical helpers (single URL, used by unit tests)                  #
    # ------------------------------------------------------------------ #
    def get_url_length(self, url: str) -> int:
        return len(url)

    def count_dots(self, url: str) -> int:
        return url.count(".")

    def count_slashes(self, url: str) -> int:
        return url.count("/")

    def count_digits(self, url: str) -> int:
        return sum(c.isdigit() for c in url)

    def count_hyphens(self, url: str) -> int:
        return url.count("-")

    def count_special_chars(self, url: str) -> int:
        return len(re.findall(r"[@?&=_%;]", url))

    # ------------------------------------------------------------------ #
    # B. Security helpers                                                  #
    # ------------------------------------------------------------------ #
    def has_https(self, url: str) -> int:
        return 1 if url.startswith("https://") else 0

    def has_ip_address(self, url: str) -> int:
        return 1 if re.search(r"(?:[0-9]{1,3}\.){3}[0-9]{1,3}", url) else 0

    # ------------------------------------------------------------------ #
    # C. Keyword helpers                                                   #
    # ------------------------------------------------------------------ #
    def count_suspicious_keywords(self, url: str) -> int:
        url_lower = url.lower()
        return sum(1 for kw in self.SUSPICIOUS_KEYWORDS if kw in url_lower)

    # ------------------------------------------------------------------ #
    # D. Statistical helpers                                               #
    # ------------------------------------------------------------------ #
    @staticmethod
    def calculate_entropy(text: str) -> float:
        if not text:
            return 0.0
        return -sum(
            (p := text.count(c) / len(text)) * math.log2(p)
            for c in set(text)
        )

    def digit_to_length_ratio(self, url: str) -> float:
        return self.count_digits(url) / (self.get_url_length(url) + 1e-5)

    def special_char_ratio(self, url: str) -> float:
        return self.count_special_chars(url) / (self.get_url_length(url) + 1e-5)

    # ------------------------------------------------------------------ #
    # E. TF-IDF (NLP)                                                      #
    # ------------------------------------------------------------------ #
    def fit_tfidf(self, urls: list[str]) -> None:
        """Fit the TF-IDF vectorizer on a list of URL strings."""
        if self.use_tfidf:
            self.tfidf.fit(urls)
            self.is_fitted = True

    def transform_tfidf(self, urls: list[str]) -> np.ndarray:
        """Transform URL strings to dense TF-IDF feature matrix."""
        if self.use_tfidf and self.is_fitted:
            return self.tfidf.transform(urls).toarray()
        return np.zeros((len(urls), self.max_tfidf_features))

    def save_vectorizer(self, filepath: str) -> None:
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, "wb") as f:
            pickle.dump(self.tfidf, f)

    def load_vectorizer(self, filepath: str) -> None:
        if os.path.exists(filepath):
            with open(filepath, "rb") as f:
                self.tfidf = pickle.load(f)
                self.is_fitted = True

    # ------------------------------------------------------------------ #
    # Main vectorised extraction                                           #
    # ------------------------------------------------------------------ #
    def extract_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Extract all features from the 'url' column in *df*.

        Parameters
        ----------
        df : pd.DataFrame
            Must have a 'url' column (and optionally 'label').

        Returns
        -------
        pd.DataFrame
            Original df columns + all engineered feature columns.
        """
        urls = df["url"]

        _SPECIAL = r"[@?&=_%;]"
        _IP      = r"(?:[0-9]{1,3}\.){3}[0-9]{1,3}"
        _KW_PAT  = "|".join(self.SUSPICIOUS_KEYWORDS)

        lexical = pd.DataFrame(
            {
                "url_length":       urls.str.len(),
                "num_dots":         urls.str.count(r"\."),
                "num_slashes":      urls.str.count("/"),
                "num_digits":       urls.str.count(r"\d"),
                "num_hyphens":      urls.str.count("-"),
                "num_special":      urls.str.count(_SPECIAL),
                "has_https":        urls.str.startswith("https://").astype(int),
                "has_ip":           urls.str.contains(_IP, regex=True).astype(int),
                "suspicious_words": urls.str.lower().str.count(_KW_PAT),
            },
            index=df.index,
        )

        lengths  = lexical["url_length"].to_numpy(float)
        digits   = lexical["num_digits"].to_numpy(float)
        specials = lexical["num_special"].to_numpy(float)

        lexical["digit_ratio"]   = digits   / (lengths + 1e-5)
        lexical["special_ratio"] = specials / (lengths + 1e-5)
        lexical["entropy"]       = [self.calculate_entropy(u) for u in urls]

        # Build parts list: original df cols come first
        col_order = [c for c in ["url", "label"] if c in df.columns]
        parts = [df[col_order], lexical]

        if self.use_tfidf:
            tfidf_mat = self.transform_tfidf(urls.tolist())
            tfidf_df  = pd.DataFrame(
                tfidf_mat,
                columns=[f"tfidf_{i}" for i in range(self.max_tfidf_features)],
                index=df.index,
            )
            parts.append(tfidf_df)

        return pd.concat(parts, axis=1)


if __name__ == "__main__":
    extractor = FeatureExtractor(use_tfidf=False)
    sample = pd.DataFrame(
        {"url": ["http://login-secure-update.com/verify", "https://google.com"],
         "label": [1, 0]}
    )
    print(extractor.extract_features(sample))
