"""Feature engineering: derived features, encodings, transformations."""

import logging
import re

import numpy as np
import pandas as pd


logger = logging.getLogger(__name__)

# Known multi-part TLDs that should be kept together.
_MULTI_PART_TLDS: set[str] = {
    "co.uk",
    "co.za",
    "co.nz",
    "co.in",
    "co.jp",
    "co.kr",
    "co.id",
    "co.il",
    "co.th",
    "com.au",
    "com.br",
    "com.mx",
    "com.ar",
    "com.tr",
    "com.sg",
    "com.my",
    "com.ph",
    "com.ng",
    "com.pk",
    "com.co",
    "com.ua",
    "com.vn",
    "org.uk",
    "org.au",
    "net.au",
    "ac.uk",
    "gov.uk",
    "edu.au",
}

# Country name normalization mapping.
_COUNTRY_ALIASES: dict[str, str] = {
    "united states": "US",
    "usa": "US",
    "u.s.a.": "US",
    "united kingdom": "GB",
    "uk": "GB",
    "great britain": "GB",
    "germany": "DE",
    "france": "FR",
    "italy": "IT",
    "spain": "ES",
    "brazil": "BR",
    "netherlands": "NL",
    "poland": "PL",
    "japan": "JP",
    "canada": "CA",
    "australia": "AU",
    "mexico": "MX",
    "india": "IN",
    "sweden": "SE",
    "denmark": "DK",
    "norway": "NO",
    "finland": "FI",
    "portugal": "PT",
    "austria": "AT",
    "switzerland": "CH",
    "belgium": "BE",
    "ireland": "IE",
    "czech republic": "CZ",
    "czechia": "CZ",
    "romania": "RO",
    "greece": "GR",
    "hungary": "HU",
    "turkey": "TR",
    "south korea": "KR",
    "indonesia": "ID",
    "thailand": "TH",
    "vietnam": "VN",
    "philippines": "PH",
    "malaysia": "MY",
    "singapore": "SG",
    "colombia": "CO",
    "argentina": "AR",
    "chile": "CL",
    "peru": "PE",
    "south africa": "ZA",
    "nigeria": "NG",
    "israel": "IL",
    "ukraine": "UA",
    "russia": "RU",
    "china": "CN",
    "taiwan": "TW",
    "new zealand": "NZ",
    "pakistan": "PK",
    "egypt": "EG",
    "croatia": "HR",
    "bulgaria": "BG",
    "slovakia": "SK",
    "slovenia": "SI",
    "lithuania": "LT",
    "latvia": "LV",
    "estonia": "EE",
}

# Link source type normalization.
_LINK_SOURCE_ALIASES: dict[str, str] = {
    "outreach": "outreach",
    "outreach (direct)": "outreach_direct",
    "outreach (reseller)": "outreach_reseller",
    "outreach (agency)": "outreach_agency",
    "outreach t2": "outreach",
    "agency": "agency",
    "agnecy": "agency",
    "growth": "growth",
    "reclaim": "reclaim",
    "link reclamation": "reclaim",
    "youtube": "youtube",
    "backlink sharing": "backlink_sharing",
    "joined database": "joined_database",
    "mentions": "mentions",
    "aff link": "affiliate",
    "affiliate link": "affiliate",
    "affiliates": "affiliate",
    "rdr link": "rdr_link",
    "lost": "lost",
    "article written internally": "internal_content",
    "off-site content": "offsite_content",
    "link exchange": "link_exchange",
    "backlink pickup": "backlink_pickup",
}

# Link type normalization.
_LINK_TYPE_ALIASES: dict[str, str] = {
    "dofollow": "dofollow",
    "nofollow": "nofollow",
    "no follow": "nofollow",
    "link insertion": "link_insertion",
    "backlink": "dofollow",
}


def extract_tld(domain: str) -> str:
    """Extract the top-level domain from a domain string.

    Handles multi-part TLDs like .co.uk, .com.au correctly.

    Args:
        domain: Domain name (e.g., 'businesstask.co.uk').

    Returns:
        TLD string (e.g., 'co.uk', 'com', 'io').
    """
    if not isinstance(domain, str) or not domain.strip():
        return "unknown"

    domain = domain.strip().lower()
    domain = re.sub(r"^https?://", "", domain)
    domain = domain.split("/")[0]

    parts = domain.split(".")
    if len(parts) < 2:
        return "unknown"

    # Check for multi-part TLD first.
    if len(parts) >= 3:
        candidate = f"{parts[-2]}.{parts[-1]}"
        if candidate in _MULTI_PART_TLDS:
            return candidate

    return parts[-1]


def add_tld_feature(df: pd.DataFrame) -> pd.DataFrame:
    """Add a TLD column extracted from the domain column.

    Args:
        df: DataFrame with a 'domain' column.

    Returns:
        DataFrame with 'tld' column added.
    """
    result = df.copy()
    result["tld"] = result["domain"].apply(extract_tld)
    tld_counts = result["tld"].value_counts()
    logger.info(
        "Extracted %d unique TLDs. Top 5: %s",
        len(tld_counts),
        dict(tld_counts.head()),
    )
    return result


def normalize_country(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize country values to ISO 2-letter codes.

    Args:
        df: DataFrame with a 'country' column.

    Returns:
        DataFrame with normalized country codes.
    """
    result = df.copy()

    def _normalize(value: object) -> str | None:
        if not isinstance(value, str) or not value.strip():
            return None
        cleaned = value.strip().lower()
        if cleaned in _COUNTRY_ALIASES:
            return _COUNTRY_ALIASES[cleaned]
        # Already an ISO code (2 uppercase letters).
        upper = value.strip().upper()
        if len(upper) == 2 and upper.isalpha():
            return upper
        return None

    result["country"] = result["country"].apply(_normalize)
    null_count = result["country"].isnull().sum()
    if null_count > 0:
        logger.info(
            "Country normalization: %d values could not be mapped", null_count
        )
    return result


def normalize_link_source_type(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize link_source_type to consistent lowercase categories.

    Args:
        df: DataFrame with a 'link_source_type' column.

    Returns:
        DataFrame with normalized link_source_type.
    """
    result = df.copy()

    def _normalize(value: object) -> str | None:
        if not isinstance(value, str) or not value.strip():
            return None
        cleaned = value.strip().lower()
        # Skip JSON-like values.
        if cleaned.startswith("["):
            return None
        return _LINK_SOURCE_ALIASES.get(cleaned, cleaned)

    result["link_source_type"] = result["link_source_type"].apply(_normalize)
    return result


def normalize_link_type(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize link_type to consistent lowercase categories.

    Args:
        df: DataFrame with a 'link_type' column.

    Returns:
        DataFrame with normalized link_type.
    """
    result = df.copy()

    def _normalize(value: object) -> str | None:
        if not isinstance(value, str) or not value.strip():
            return None
        cleaned = value.strip().lower()
        if cleaned in ("#n/a", "[]", ""):
            return None
        return _LINK_TYPE_ALIASES.get(cleaned, cleaned)

    result["link_type"] = result["link_type"].apply(_normalize)
    return result


def add_price_ratio(df: pd.DataFrame) -> pd.DataFrame:
    """Add negotiation ratio feature: final_price / initial_price.

    Args:
        df: DataFrame with final_price and initial_price columns.

    Returns:
        DataFrame with 'price_ratio' column added.
    """
    result = df.copy()
    has_both = result["initial_price"].notna() & (result["initial_price"] > 0)
    result["price_ratio"] = np.where(
        has_both,
        result["final_price"] / result["initial_price"],
        np.nan,
    )
    return result


def add_log_price(df: pd.DataFrame) -> pd.DataFrame:
    """Add log-transformed price for modeling.

    Args:
        df: DataFrame with final_price column.

    Returns:
        DataFrame with 'log_price' column added.
    """
    result = df.copy()
    result["log_price"] = np.log1p(result["final_price"])
    return result


def add_log_traffic(df: pd.DataFrame) -> pd.DataFrame:
    """Add log-transformed traffic for modeling.

    Args:
        df: DataFrame with domain_traffic column.

    Returns:
        DataFrame with 'log_traffic' column added.
    """
    result = df.copy()
    result["log_traffic"] = np.log1p(
        result["domain_traffic"].fillna(0).clip(lower=0)
    )
    return result


def add_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add temporal features from date_received.

    Args:
        df: DataFrame with date_received column.

    Returns:
        DataFrame with year, month, quarter columns added.
    """
    result = df.copy()
    dt = pd.to_datetime(result["date_received"])
    result["year"] = dt.dt.year
    result["month"] = dt.dt.month
    result["quarter"] = dt.dt.quarter
    return result


def add_missingness_flags(
    df: pd.DataFrame,
    columns: tuple[str, ...] = ("cf", "tf", "country"),
) -> pd.DataFrame:
    """Add binary missingness indicator columns for selected features.

    These flags preserve missing-value information after imputation and can
    improve downstream model quality when missingness itself is informative.

    Args:
        df: Input DataFrame.
        columns: Columns for which to create `<col>_missing_flag`.

    Returns:
        DataFrame with added missingness flag columns.
    """
    result = df.copy()
    for col in columns:
        if col not in result.columns:
            continue
        result[f"{col}_missing_flag"] = result[col].isna().astype("int8")
    return result
