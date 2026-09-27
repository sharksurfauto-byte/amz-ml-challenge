"""
Text normalization and key extraction utilities for Amazon ML Challenge 2026.

All operations are implemented using native Polars expressions (pl.Expr) for
maximum vectorization and speed on large-scale datasets (12M+ rows).
"""

from typing import Optional, Tuple
import polars as pl


# ---------------------------------------------------------------------------
# Devanagari Transliteration Lookup Tables
# ---------------------------------------------------------------------------

# Common Devanagari business terms and geographic locations to canonical English
COMMON_DEV_WORDS = {
    # Legal forms & business terms
    "प्राइवेट लिमिटेड": "private limited",
    "प्राइवेट लि.": "private limited",
    "प्रा. लि.": "pvt ltd",
    "प्रा.लि.": "pvt ltd",
    "प्रा लि": "pvt ltd",
    "प्राइवेट": "private",
    "लिमिटेड": "limited",
    "एलएलपी": "llp",
    "कंपनी": "company",
    "कॉर्पोरेशन": "corporation",
    "इंटरनेशनल": "international",
    "टेक्नोलॉजीज": "technologies",
    "टेक्नोलॉजी": "technology",
    "सॉल्यूशंस": "solutions",
    "एंटरप्राइजेज": "enterprises",
    "इंडस्ट्रीज": "industries",
    "सर्विसेज": "services",
    "इन्वेस्टमेंट्स": "investments",
    "इन्वेस्टमेंट": "investment",
    "कंस्ट्रक्शंस": "constructions",
    "कंस्ट्रक्शन": "construction",
    "डेवलपर्स": "developers",
    "मार्केटिंग": "marketing",
    "प्रॉपर्टीज": "properties",
    "वेंचर्स": "ventures",
    "इंजीनियरिंग": "engineering",
    "ट्रेडिंग": "trading",
    "फूड्स": "foods",
    "फूड": "food",
    "ग्लोबल": "global",
    "हेल्थकेयर": "healthcare",
    "इंफोटेक": "infotech",
    "सॉफ्टवेयर": "software",
    "मैनेजमेंट": "management",
    "कंसल्टेंट्स": "consultants",
    "कंसल्टिंग": "consulting",
    "फाइनेंस": "finance",
    "एनर्जी": "energy",
    "एक्सपोर्ट्स": "exports",
    "पावर": "power",
    "इंफ्रास्ट्रक्चर": "infrastructure",
    "इंफ्रा": "infra",
    "प्रोजेक्ट्स": "projects",
    "सिस्टम्स": "systems",
    "एग्रो": "agro",
    "प्रोडक्ट्स": "products",
    "प्रोड्यूसर": "producer",
    "इम्पेक्स": "impex",
    "बिजनेस": "business",
    "बिल्डर्स": "builders",
    "फाउंडेशन": "foundation",
    "एस्टेट": "estate",
    # States & Cities (for address normalization)
    "महाराष्ट्र": "maharashtra",
    "कर्नाटक": "karnataka",
    "गुजरात": "gujarat",
    "राजस्थान": "rajasthan",
    "पंजाब": "punjab",
    "हरियाणा": "haryana",
    "उत्तर प्रदेश": "uttar pradesh",
    "मध्य प्रदेश": "madhya pradesh",
    "पश्चिम बंगाल": "west bengal",
    "तमिलनाडु": "tamil nadu",
    "केरल": "kerala",
    "आंध्र प्रदेश": "andhra pradesh",
    "तेलंगाना": "telangana",
    "दिल्ली": "delhi",
    "नई दिल्ली": "new delhi",
    "मुंबई": "mumbai",
    "पुणे": "pune",
    "नासिक": "nashik",
    "नागपुर": "nagpur",
    "कोलकाता": "kolkata",
    "बेंगलुरु": "bengaluru",
    "बैंगलोर": "bangalore",
    "चेन्नई": "chennai",
    "हैदराबाद": "hyderabad",
    "अहमदाबाद": "ahmedabad",
    "सूरत": "surat",
    "जयपुर": "jaipur",
    "लखनऊ": "lucknow",
    "भोपाल": "bhopal",
    "इंदौर": "indore",
    "पटना": "patna",
    "चंडीगढ़": "chandigarh",
}

DEV_CHAR_MAP = {
    # Vowels
    "अ": "a", "आ": "aa", "इ": "i", "ई": "ee", "उ": "u", "ऊ": "oo", "ऋ": "ri",
    "ए": "e", "ऐ": "ai", "ओ": "o", "औ": "au", "अं": "am", "अः": "ah",
    # Matras (vowel signs)
    "ा": "a", "ि": "i", "ी": "ee", "ु": "u", "ू": "oo", "ृ": "ri",
    "े": "e", "ै": "ai", "ो": "o", "ौ": "au", "ं": "n", "ँ": "n", "ः": "h",
    "्": "", "़": "", "ॅ": "e", "ॉ": "o", "॑": "", "॒": "",
    # Consonants
    "क": "k", "ख": "kh", "ग": "g", "घ": "gh", "ङ": "ng",
    "च": "ch", "छ": "chh", "ज": "j", "झ": "jh", "ञ": "ny",
    "ट": "t", "ठ": "th", "ड": "d", "ढ": "dh", "ण": "n",
    "त": "t", "थ": "th", "द": "d", "ध": "dh", "न": "n",
    "प": "p", "फ": "ph", "ब": "b", "भ": "bh", "म": "m",
    "य": "y", "र": "r", "ल": "l", "व": "v", "श": "sh",
    "ष": "sh", "स": "s", "ह": "h", "क्ष": "ksh", "त्र": "tr", "ज्ञ": "gy",
    # Nuqta consonants
    "क़": "q", "ख़": "kh", "ग़": "gh", "ज़": "z", "ड़": "r", "ढ़": "rh", "फ़": "f",
    # Numerals
    "०": "0", "१": "1", "२": "2", "३": "3", "४": "4",
    "५": "5", "६": "6", "७": "7", "८": "8", "९": "9",
}

_DEV_TRANS_TABLE = str.maketrans({k: v for k, v in DEV_CHAR_MAP.items() if len(k) == 1})


def _transliterate_devanagari_str(text: Optional[str]) -> Optional[str]:
    """Helper to transliterate a single string from Devanagari to Latin."""
    if not text:
        return text
    res = text
    for k, v in COMMON_DEV_WORDS.items():
        if k in res:
            res = res.replace(k, v)
    return res.translate(_DEV_TRANS_TABLE)


# ---------------------------------------------------------------------------
# Polars Expression Functions
# ---------------------------------------------------------------------------

def remove_junk_tokens(expr: pl.Expr) -> pl.Expr:
    """
    Removes visual formatting artifacts and noise tokens:
    e.g. '--', '<<', '>>', '+', '[]', '()', '##', '**', '==', '~~', '//'.
    """
    return (
        expr.cast(pl.String)
        .str.replace_all(r"(?:--+|<<+|>>+|\[|\]|\(|\)|\+|\*\*+|##+|==+|~~+|//+)", " ")
        .str.replace_all(r"\s+", " ")
        .str.strip_chars()
    )


def kanan_transliterate_devanagari(expr: pl.Expr) -> pl.Expr:
    """
    Transliterates Devanagari text into Latin characters.
    Optimized to run only on rows containing Devanagari unicode characters (U+0900 to U+097F).
    """
    return (
        pl.when(expr.str.contains(r"[ऀ-ॿ]"))
        .then(expr.map_elements(_transliterate_devanagari_str, return_dtype=pl.String))
        .otherwise(expr)
    )


# Alias for convenience
transliterate_devanagari = kanan_transliterate_devanagari


def clean_text(expr: pl.Expr) -> pl.Expr:
    """
    Cleans text:
    - Lowercases text.
    - Replaces '&' with ' and '.
    - Normalizes unicode via NFKD and removes combining diacritic marks (\\p{M}).
    - Strips non-alphanumeric punctuation ([^a-z0-9\\s]).
    - Collapses multiple whitespace characters.
    - Strips leading and trailing whitespace.
    """
    return (
        expr.cast(pl.String)
        .str.to_lowercase()
        .str.replace_all(r"&", " and ")
        .str.normalize("NFKD")
        .str.replace_all(r"\p{M}", "")
        .str.replace_all(r"[^a-z0-9\s]", " ")
        .str.replace_all(r"\s+", " ")
        .str.strip_chars()
    )


def extract_and_remove_legal_forms(expr: pl.Expr) -> Tuple[pl.Expr, pl.Expr]:
    """
    Maps common corporate legal forms to a canonical label and removes them from the text.
    Forms handled: corp, corporation, pvt, private, ltd, limited, llc, llp, inc, incorporated,
                   sarl, sas, sci, gmbh, co, company.

    Returns:
        Tuple[pl.Expr, pl.Expr]: (text_without_legal_form, legal_form_label)
    """
    legal_form_expr = (
        pl.when(expr.str.contains(r"\b(pvt\s+ltd|private\s+limited|pvt\s+limited|private\s+ltd)\b"))
        .then(pl.lit("pvt ltd"))
        .when(expr.str.contains(r"\b(llc)\b"))
        .then(pl.lit("llc"))
        .when(expr.str.contains(r"\b(llp)\b"))
        .then(pl.lit("llp"))
        .when(expr.str.contains(r"\b(inc|incorporated)\b"))
        .then(pl.lit("inc"))
        .when(expr.str.contains(r"\b(corp|corporation)\b"))
        .then(pl.lit("corp"))
        .when(expr.str.contains(r"\b(ltd|limited)\b"))
        .then(pl.lit("ltd"))
        .when(expr.str.contains(r"\b(pvt|private)\b"))
        .then(pl.lit("pvt"))
        .when(expr.str.contains(r"\b(gmbh)\b"))
        .then(pl.lit("gmbh"))
        .when(expr.str.contains(r"\b(sarl)\b"))
        .then(pl.lit("sarl"))
        .when(expr.str.contains(r"\b(sas)\b"))
        .then(pl.lit("sas"))
        .when(expr.str.contains(r"\b(sci)\b"))
        .then(pl.lit("sci"))
        .when(expr.str.contains(r"\b(co|company)\b"))
        .then(pl.lit("co"))
        .otherwise(None)
    )

    remove_pattern = (
        r"\b(pvt\s+ltd|private\s+limited|pvt\s+limited|private\s+ltd|"
        r"corp|corporation|pvt|private|ltd|limited|llc|llp|inc|incorporated|"
        r"sarl|sas|sci|gmbh|co|company)\b"
    )

    no_legal_expr = (
        expr.str.replace_all(remove_pattern, " ")
        # Clean up any hanging trailing/leading conjunctions left after legal removal
        .str.replace_all(r"\b(and|&)\s*$", " ")
        .str.replace_all(r"^\s*(and|&)\b", " ")
        .str.replace_all(r"\s+", " ")
        .str.strip_chars()
    )

    return no_legal_expr, legal_form_expr


EXCLUDED_EMAIL_DOMAINS = {
    "gmail", "yahoo", "hotmail", "outlook", "rediffmail",
    "aol", "icloud", "live", "ymail", "msn", "mail", "zoho"
}

_EMAIL_EXCLUDE_PATTERN = r"^(?:" + "|".join(EXCLUDED_EMAIL_DOMAINS) + r")$"


def extract_domain_root(expr: pl.Expr) -> pl.Expr:
    """
    Extracts the clean second-level domain name for web records,
    filtering out generic email providers (gmail, yahoo, etc.).
    e.g. 'wilfordhancock.com' -> 'wilfordhancock', 'www.example.net' -> 'example'.
    """
    pattern = r"(?:https?://)?(?:www\.)?([a-zA-Z0-9-]+)\.(?:com|net|org|in|co|us|biz|info|io|fr|gov|edu)\b"
    extracted = expr.cast(pl.String).str.extract(pattern, 1).str.to_lowercase()
    return (
        pl.when(extracted.is_not_null() & ~extracted.str.contains(_EMAIL_EXCLUDE_PATTERN))
        .then(extracted)
        .otherwise(None)
    )


def extract_pincode(expr: pl.Expr) -> pl.Expr:
    """
    Extracts 5 or 6 digit postal / pin codes from address.
    """
    return expr.cast(pl.String).str.extract(r"\b(\d{5,6})\b", 1)


def soundex(text: Optional[str]) -> Optional[str]:
    """
    Pure Python Soundex algorithm for phonetic key encoding.
    Returns 4-character soundex code (e.g. 'S530').
    """
    if not text or not isinstance(text, str):
        return None
    s = [c for c in text.upper() if 'A' <= c <= 'Z']
    if not s:
        return None
    first = s[0]
    mapping = {
        'B': '1', 'F': '1', 'P': '1', 'V': '1',
        'C': '2', 'G': '2', 'J': '2', 'K': '2', 'Q': '2', 'S': '2', 'X': '2', 'Z': '2',
        'D': '3', 'T': '3',
        'L': '4',
        'M': '5', 'N': '5',
        'R': '6'
    }
    encoded = [first]
    prev = mapping.get(first, '0')
    for char in s[1:]:
        code = mapping.get(char, '0')
        if code != '0':
            if code != prev:
                encoded.append(code)
            prev = code
        else:
            prev = '0'
    res = ''.join(encoded)[:4]
    return res.ljust(4, '0')


def phonetic_first_token(expr: pl.Expr) -> pl.Expr:
    """
    Computes soundex code on the first distinctive token of the business name.
    """
    first_tok = (
        expr.cast(pl.String)
        .str.split(" ")
        .list.eval(pl.element().filter(pl.element().str.len_chars() >= 3))
        .list.get(0, null_on_oob=True)
    )
    return first_tok.map_elements(soundex, return_dtype=pl.String)


def extract_first_two_tokens(expr: pl.Expr) -> Tuple[pl.Expr, pl.Expr]:
    """
    Extracts the first two distinctive words (>= 3 chars) from cleaned business name.
    """
    tokens = (
        expr.cast(pl.String)
        .str.split(" ")
        .list.eval(pl.element().filter(pl.element().str.len_chars() >= 3))
    )
    t0 = tokens.list.get(0, null_on_oob=True)
    t1 = tokens.list.get(1, null_on_oob=True)
    return t0, t1


def token_sort_key(expr: pl.Expr) -> pl.Expr:
    """
    Creates an order-invariant token-sorted key by splitting tokens on spaces,
    filtering empty tokens, sorting alphabetically, and joining back with space.
    """
    return (
        expr.str.split(" ")
        .list.eval(pl.element().filter(pl.element() != ""))
        .list.sort()
        .list.join(" ")
    )


def extract_street_number(expr: pl.Expr) -> pl.Expr:
    """
    Pulls out the first occurring sequence of digits (1-5 digits) as street_number.
    """
    return expr.cast(pl.String).str.extract(r"\b(\d{1,5})\b", 1)


def clean_address(expr: pl.Expr) -> pl.Expr:
    """
    Normalizes an address string:
    - Transliterates any Devanagari state/city/locality names.
    - Removes visual artifacts and noise.
    - Applies standard text cleaning (lowercasing, NFKD diacritic removal, alphanumeric).
    """
    return clean_text(remove_junk_tokens(kanan_transliterate_devanagari(expr)))


def address_keys(expr: pl.Expr) -> Tuple[pl.Expr, pl.Expr, pl.Expr]:
    """
    Returns (address_clean, address_street_number, pincode).
    """
    cleaned = clean_address(expr)
    street_num = extract_street_number(expr)
    pincode = extract_pincode(expr)
    return cleaned, street_num, pincode
