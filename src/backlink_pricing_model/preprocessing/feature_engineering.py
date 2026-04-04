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
    columns: tuple[str, ...] = ("cf", "tf", "country", "dr"),
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


def add_interaction_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add interaction and polynomial features for key metrics.

    Creates DR*CF, DR*TF, CF/TF ratio, DR*log_traffic, DR^2,
    and log_traffic^2 to capture non-linear relationships.

    Args:
        df: DataFrame with dr, cf, tf, log_traffic columns.

    Returns:
        DataFrame with interaction feature columns added.
    """
    result = df.copy()

    # DR interactions (fill NaN with 0 for safe multiplication).
    dr = result["dr"].fillna(0)
    cf = result["cf"].fillna(0)
    tf = result["tf"].fillna(0)
    lt = result["log_traffic"].fillna(0)

    result["dr_x_cf"] = dr * cf
    result["dr_x_tf"] = dr * tf
    result["dr_x_log_traffic"] = dr * lt
    result["cf_x_log_traffic"] = cf * lt

    # CF/TF ratio (trust flow relative to citation flow).
    result["cf_tf_ratio"] = np.where(
        tf > 0, cf / tf, 0.0
    )

    # Polynomial terms.
    result["dr_squared"] = dr ** 2
    result["log_traffic_squared"] = lt ** 2

    # DR per unit of traffic (value density).
    result["dr_per_log_traffic"] = np.where(
        lt > 0, dr / lt, 0.0
    )

    logger.info("Added %d interaction features", 8)
    return result


def add_domain_frequency(df: pd.DataFrame) -> pd.DataFrame:
    """Add domain record frequency as a feature.

    Domains appearing many times may have different pricing dynamics
    than one-off domains (repeat vs new relationships).

    Args:
        df: DataFrame with a 'domain' column.

    Returns:
        DataFrame with 'domain_freq' and 'log_domain_freq' columns.
    """
    result = df.copy()
    freq = result["domain"].fillna("unknown").map(
        result["domain"].fillna("unknown").value_counts()
    )
    result["domain_freq"] = freq
    result["log_domain_freq"] = np.log1p(freq)
    logger.info(
        "Added domain frequency: mean=%.1f, max=%d",
        freq.mean(),
        freq.max(),
    )
    return result


def normalize_link_source_for_modeling(
    df: pd.DataFrame,
    min_count: int = 20,
) -> pd.DataFrame:
    """Clean and group link_source_type for modeling.

    Rare categories (below min_count) are collapsed into 'other'.

    Args:
        df: DataFrame with link_source_type column.
        min_count: Minimum occurrence count to keep a category.

    Returns:
        DataFrame with cleaned link_source_type_clean column.
    """
    result = df.copy()
    col = "link_source_type"
    if col not in result.columns:
        return result

    values = result[col].fillna("unknown")
    counts = values.value_counts()
    rare = set(counts[counts < min_count].index)
    result["link_source_type_clean"] = values.where(
        ~values.isin(rare), "other"
    )
    n_kept = len(counts) - len(rare)
    logger.info(
        "link_source_type: kept %d categories, collapsed %d rare into 'other'",
        n_kept,
        len(rare),
    )
    return result


def group_rare_tld(
    df: pd.DataFrame,
    min_count: int = 50,
) -> pd.DataFrame:
    """Group rare TLDs into 'other' to reduce cardinality.

    Args:
        df: DataFrame with tld column.
        min_count: Minimum count to keep a TLD as-is.

    Returns:
        DataFrame with 'tld_grouped' column.
    """
    result = df.copy()
    if "tld" not in result.columns:
        return result

    counts = result["tld"].value_counts()
    rare = set(counts[counts < min_count].index)
    result["tld_grouped"] = result["tld"].where(
        ~result["tld"].isin(rare), "other"
    )
    logger.info(
        "TLD grouping: %d -> %d categories (min_count=%d)",
        len(counts),
        result["tld_grouped"].nunique(),
        min_count,
    )
    return result


def group_rare_country(
    df: pd.DataFrame,
    min_count: int = 30,
) -> pd.DataFrame:
    """Group rare countries into 'other' to reduce cardinality.

    Args:
        df: DataFrame with country column.
        min_count: Minimum count to keep a country as-is.

    Returns:
        DataFrame with 'country_grouped' column.
    """
    result = df.copy()
    if "country" not in result.columns:
        return result

    values = result["country"].fillna("unknown")
    counts = values.value_counts()
    rare = set(counts[counts < min_count].index)
    result["country_grouped"] = values.where(
        ~values.isin(rare), "other"
    )
    logger.info(
        "Country grouping: %d -> %d categories (min_count=%d)",
        len(counts),
        result["country_grouped"].nunique(),
        min_count,
    )
    return result
