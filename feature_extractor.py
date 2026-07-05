"""
feature_extractor.py - Structural Feature Extraction for LexiDecay v2.

Binary presence/absence pseudo-tokens injected before graph training.
All token names are lowercase to match the graph's normalization.

Design: EVERY document receives exactly one token per pattern — either the
presence token (pattern matched) or the absence token (pattern not matched).
This gives simultaneous positive + negative structural evidence.

Spam structural signals (high discriminativeness):
  __long_num__      50.7% spam / 0.1% ham   703x
  __short_code__    51.5% spam / 0.1% ham   384x
  __money__         21.8% spam / 0.1% ham   141x
  __url__           14.2% spam / 0.0% ham   276x
  __reply_stop__     ~8%  spam /  ~0%  ham  200x+
  __rate_info__      ~12% spam /  ~0%  ham  600x+
  __claim_prize__    ~18% spam /  ~0%  ham  350x+
  __free_cta__       ~10% spam / ~0.3% ham   33x  (refined to avoid ham "call me free")

Ham structural signals (negative spam discriminativeness):
  __all_caps_doc__  : >70% of alphabetic words are ALL CAPS -> nearly always ham
                      (spam uses selective caps; all-caps-doc is friend texting style)
  __satisfaction_survey__: "were you satisfied", "rate our service" -> business SMS, ham
  __personal_contact__   : "my mobile number", "my number is" -> contact sharing, ham
  __bank_sms__           : "NEFT", "INR amount", "account has been credited" -> bank SMS, ham
  __security_warning__   : "never asks for", "protect yourself", "do not share" -> warnings, ham

Bigram pseudo-tokens (extracted after tokenization via extract_bigrams):
  __bg_free_entry__ (spam) vs __bg_free_time__ (ham) -> phrase disambiguation
  __bg_win_or__     (ham) vs __bg_win_prize__ (spam)
"""

from __future__ import annotations

import html
import re
from typing import List, Optional, Tuple


# ---------------------------------------------------------------------------
# Pattern table
# ---------------------------------------------------------------------------
# Entry: (compiled_pattern, present_token, absent_token, count_mode)
#   count_mode=True  -> add presence token ONCE PER MATCH
#   count_mode=False -> add presence token AT MOST ONCE per doc
#
# NOTE: __caps_word__ is count_mode=False (NOT True).
# Spam uses selective caps (WIN, FREE = 1-3 caps words per doc).
# All-caps ham texts (old phones, shouting style) have 10-30 caps words.
# count_mode=True would give those ham texts 10-30x the spam signal -> massive FP.
# count_mode=False limits the signal to 1 token regardless of how many caps words.
# See also __all_caps_doc__ below for an additional discriminator.
#
# NOTE: __short_code__ uses 5-6 digits only (NOT 4-6).
# 4-digit numbers like "2000" (prices) and "3230" (textbook codes) cause FPs.
# Real SMS short codes in this era are mostly 5-6 digits (87121, 80970, 81010).

_PATTERNS: List[Tuple[re.Pattern, str, str, bool]] = [
    # Phone-like numbers: 8-15 consecutive digits (UK, international)
    (
        re.compile(r"\b\d{8,15}\b"),
        "__long_num__",
        "__no_long_num__",
        True,
    ),
    # SMS short codes: 5-6 standalone digits ONLY (not 4-digit prices like 2000)
    (
        re.compile(r"(?<![A-Za-z\d])\d{5,6}(?![A-Za-z\d])"),
        "__short_code__",
        "__no_short_code__",
        True,
    ),
    # Monetary amounts: £1.50, $900, €2000, 50p, ¤3
    (
        re.compile(r"[£$€¤]\s*[\d,.]+|\b\d+(?:\.\d+)?p\b"),
        "__money__",
        "__no_money__",
        True,
    ),
    # URLs: http://, https://, www.something
    (
        re.compile(r"(?:https?://|www\.)\S+", re.IGNORECASE),
        "__url__",
        "__no_url__",
        False,
    ),
    # ANY all-caps word 3+ chars: presence only (count_mode=False, see NOTE above)
    (
        re.compile(r"\b[A-Z]{3,}\b"),
        "__caps_word__",
        "__no_caps_word__",
        False,
    ),
    # Date patterns (spam uses "offer expires" dates): 21/05/2005, 05-12-2004
    (
        re.compile(r"\b\d{1,2}[/\-]\d{1,2}[/\-]\d{2,4}\b"),
        "__date__",
        "__no_date__",
        False,
    ),
    # "Reply STOP", "text STOP", "opt out" -> unsubscribe CTA, spam-exclusive
    (
        re.compile(
            r"\b(?:reply|text|txt)\s+(?:stop|end|quit|cancel|unsubscribe)\b"
            r"|\bto\s+(?:opt\s*out|unsubscribe|cancel)\b"
            r"|\bstop\s+(?:msgs?|texts?|sms)\b",
            re.IGNORECASE,
        ),
        "__reply_stop__",
        "__no_reply_stop__",
        False,
    ),
    # Per-rate info: "per msg", "per min", "per call" -> spam pricing disclosure
    (
        re.compile(
            r"\bper\s+(?:min(?:ute)?|msg|message|call|sms|text)\b"
            r"|\b\d+p\s+per\b"
            r"|\bpmin\b",
            re.IGNORECASE,
        ),
        "__rate_info__",
        "__no_rate_info__",
        False,
    ),
    # Prize/claim CTA: "you have won", "been selected", "claim prize" -> spam
    (
        re.compile(
            r"\byou(?:'ve|\s+have)?\s+won\b"
            r"|\byou(?:'ve|\s+have)?\s+been\s+selected\b"
            r"|\bclaim\s+(?:your|ur|a|the)?\s*(?:prize|reward|gift|cash|bonus)\b"
            r"|\ba\s+(?:\w+\s+)?prize\b"
            r"|\bwon\s+a\b",
            re.IGNORECASE,
        ),
        "__claim_prize__",
        "__no_claim_prize__",
        False,
    ),
    # Free-offer CTA: "free entry", "free prize", "freephone"
    # NOTE: "free call" and "call now/free" are EXCLUDED because "call me when you're free"
    # (ham) and "once free call me" (ham) frequently generate false positives.
    # Only keep unambiguous spam-exclusive free-offer phrases.
    (
        re.compile(
            r"\bfree\s+(?:phone|txt|entry|ringtone|prize|gift|cash|access)\b"
            r"|\bfreephone\b"
            r"|\bfree\s+(?:text|msg)\b(?!\s+me|\s+you|\s+back|\s+them|\s+us)",
            re.IGNORECASE,
        ),
        "__free_cta__",
        "__no_free_cta__",
        False,
    ),
    # --- HAM-DISCRIMINATIVE structural patterns ---
    # Customer satisfaction survey: "were you satisfied", "rate our service"
    # These are legitimate business SMS that look spammy but are ham.
    (
        re.compile(
            r"\b(?:were\s+you|you\s+were)\s+satisfied\b"
            r"|\brate\s+(?:our|your|the)\s+(?:service|experience|call|support)\b"
            r"|\byour\s+(?:feedback|rating|review|experience)\s+(?:is|was|matters)\b"
            r"|\bif\s+you\s+were\s+satisfied\b"
            r"|\bsatisfied\s+with\s+(?:the|your|our)\b",
            re.IGNORECASE,
        ),
        "__satisfaction_survey__",
        "__no_satisfaction_survey__",
        False,
    ),
    # Personal contact sharing: "my mobile number", "my number is" -> ham
    # Contrast with spam which says "YOUR mobile number" or "send us your number"
    (
        re.compile(
            r"\bmy\s+(?:(?:mobile|cell|work)\s+)?(?:number|no\.?|num)\b"
            r"|\bmy\s+(?:mobile|cell)\b(?=\s+(?:is\b|no\b|number\b|\d))"
            r"|\bcontact\s+(?:me|us)\s+(?:at|on)\b",
            re.IGNORECASE,
        ),
        "__personal_contact__",
        "__no_personal_contact__",
        False,
    ),
    # Bank/service SMS notifications: NEFT, INR, credited, account balance -> ham
    # These are formal transactional SMS, virtually never spam in this dataset.
    (
        re.compile(
            r"\b(?:neft|rtgs|imps|upi)\b"
            r"|\b(?:credited|debited)\s+(?:with|for|to|by)\b"
            r"|\binr\s*[\d,]+"
            r"|\byour\s+(?:account|balance)\s+(?:has\s+been|is)\b"
            r"|\b(?:transaction|txn)\s+(?:id|ref|no\.?|number)\b"
            r"|\baccount\s+(?:has\s+been\s+(?:refilled|credited|debited)|balance)\b",
            re.IGNORECASE,
        ),
        "__bank_sms__",
        "__no_bank_sms__",
        False,
    ),
    # Security/phishing warning: "never asks for", "protect yourself" -> ham
    # Organizations sending warnings about fraud/phishing -> ham context
    (
        re.compile(
            r"\bnever\s+(?:ask|asks|share|give|send)\s+(?:for\s+)?(?:your\s+)?(?:pin|otp|password|sensitive)\b"
            r"|\bprotect\s+yourself\s+from\b"
            r"|\bdo\s+not\s+(?:share|give|send|disclose)\s+(?:your\s+)?(?:pin|otp|password)\b"
            r"|\bbeware\s+of\s+fraud\b",
            re.IGNORECASE,
        ),
        "__security_warning__",
        "__no_security_warning__",
        False,
    ),
    # --- SPAM-DISCRIMINATIVE structural patterns ---
    # Adult/explicit content spam: "knickers", "xxx pics", "porn", "naked pics"
    # NOTE: standalone "xxx" is EXCLUDED because in UK SMS "xxx" means kisses
    # ("Love you xxx") and appears in 16 ham docs vs 11 spam docs — would add
    # positive spam evidence to innocent ham texts.  Only retain explicit combos.
    (
        re.compile(
            r"\bknickers?\b"
            r"|\bporn(?:ography|o)?\b"
            r"|\bnaked\s+(?:pic|photo|img|image|video|vid|girl|woman|man|boy|body)\b"
            r"|\bxxx\s+(?:pic|photo|video|vid|content|film|chat|movies?)\b"
            r"|\bpicsfree\b|\bpics\s*free\b"
            r"|\bsex(?:y\s+(?:pic|photo|video|vid|chat|txt|text)\s*|ting\b)"
            r"|\badult\s+(?:party|parties|content|site|services?|entertainment|chat|fun)\b",
            re.IGNORECASE,
        ),
        "__adult_content__",
        "__no_adult_content__",
        False,
    ),
    # Dating / social hook spam: "text dating service", "most discreet text"
    # Distinguishes commercial dating SMS spam from casual "dating" in ham conversation.
    (
        re.compile(
            r"\btext\s+(?:dating|chat|sex)\s*(?:service|site|club|line|club)?\b"
            r"|\bmost\s+discreet\s+(?:text|sms|dating|service)\b"
            r"|\bdating\s+service\b"
            r"|\bfall\s+in\s+love\s+(?:in|via|through|by|with)\b",
            re.IGNORECASE,
        ),
        "__dating_spam__",
        "__no_dating_spam__",
        False,
    ),
    # Formal Indian business / professional SMS -> ham
    # Patterns: "kindly be informed", "Rgds,", "as per convenience", numbered lists
    # "as per convenience/availability" is Indian English polite phrasing, NOT spam T&C.
    # Spam uses "as per our T&C", "as per the promotion" etc., but "as per convenience"
    # is purely from informal/business ham SMSes in this dataset.
    (
        re.compile(
            r"\bkindly\s+(?:be\s+(?:informed|noted|advised)|note\s+that|inform)\b"
            r"|\brgds\s*[,.]|\bregards\s*[,.]"
            r"|\bwith\s+regards\b"
            r"|\bconvey\s+(?:my\s+)?regards\b"
            r"|\bas\s+per\s+(?:your\s+)?(?:convenience|availability|schedule|preference|suitability)\b"
            r"|\bthank\s+you\s+for\s+your\s+(?:patience|cooperation|support|understanding)\b",
            re.IGNORECASE,
        ),
        "__formal_business_sms__",
        "__no_formal_business_sms__",
        False,
    ),
    # South-Asian / Indic script in SMS -> ham (UK spam is almost always English).
    # Covers Kannada (ಕ U+0C80-0CFF), Malayalam (U+0D00-0D7F),
    # Tamil (U+0B80-0BFF), Telugu (U+0C00-0C7F), Devanagari (U+0900-097F).
    # Using narrow targeted ranges avoids matching random encoding artifacts.
    (
        re.compile(
            r"[ಀ-೿ഀ-ൿ஀-௿ఀ-౿ऀ-ॿ]",
        ),
        "__indic_script__",
        "__no_indic_script__",
        False,
    ),
    # Personal classifieds selling (ham) vs commercial spam.
    # "I'm selling it for £50", "selling this for $30" → person-to-person ham sale.
    # Spam uses monetary amounts as prizes/rewards, not "selling" context.
    (
        re.compile(
            r"\bi(?:'m|am)\s+selling\b"
            r"|\bselling\s+(?:it|this|my|a|an)\b"
            r"|\bfor\s+sale\s*[-:,]",   # classifieds header "For sale -"
            re.IGNORECASE,
        ),
        "__personal_sale__",
        "__no_personal_sale__",
        False,
    ),
]

# Regexes for all-caps-document detection
_ALPHA_WORD_RE = re.compile(r"\b[A-Za-z]{2,}\b")
_CAPS_WORD_RE  = re.compile(r"\b[A-Z]{2,}\b")

# Threshold: >70% of alphabetic words are ALL CAPS -> __all_caps_doc__
# Spam uses selective caps (WIN, FREE = ~15% ratio); all-caps ham texts are 70-100%.
_ALL_CAPS_THRESHOLD = 0.70


# ---------------------------------------------------------------------------
# StructuralFeatureExtractor
# ---------------------------------------------------------------------------

class StructuralFeatureExtractor:
    """
    Pre-tokenization structural pseudo-token extractor.

    Workflow (called by TokenEngine):
      1. preprocess(text) - HTML decode + whitespace normalize
      2. extract(text)    - structural pseudo-tokens (patterns + all-caps detection)
      3. extract_bigrams(words) - word bigram pseudo-tokens (called post-tokenization)

    All tokens are lowercase (graph lowercases all text).

    Parameters
    ----------
    patterns : override _PATTERNS (None = use defaults)
    presence_repeat : times to repeat each PRESENCE token (default 1)
    absence_repeat  : times to repeat each ABSENCE token (default 1)
    use_bigrams     : enable word bigram pseudo-tokens (default True)
    bigram_min_len  : minimum word length for bigrams (default 2)
    """

    def __init__(
        self,
        patterns: "Optional[List[Tuple[re.Pattern, str, str, bool]]]" = None,
        presence_repeat: int = 1,
        absence_repeat: int = 1,
        use_bigrams: bool = True,
        bigram_min_len: int = 2,
    ) -> None:
        self.patterns        = patterns if patterns is not None else _PATTERNS
        self.presence_repeat = max(1, presence_repeat)
        self.absence_repeat  = max(1, absence_repeat)
        self.use_bigrams     = use_bigrams
        self.bigram_min_len  = bigram_min_len

    def preprocess(self, text: str) -> str:
        """HTML decode + collapse whitespace."""
        text = html.unescape(text)
        text = re.sub(r"[ \t]{2,}", " ", text)
        return text

    def extract(self, text: str) -> List[str]:
        """
        Structural pseudo-tokens for one document.

        For each pattern: either presence or absence token (binary).
        Also computes __all_caps_doc__ / __no_all_caps_doc__ based on
        fraction of alphabetic words in ALL CAPS (>70% threshold).
        """
        result: List[str] = []

        for pat, present_tok, absent_tok, count_mode in self.patterns:
            matches = pat.findall(text)
            if matches:
                if count_mode:
                    result.extend([present_tok] * (len(matches) * self.presence_repeat))
                else:
                    result.extend([present_tok] * self.presence_repeat)
            else:
                result.extend([absent_tok] * self.absence_repeat)

        # All-caps document detection.
        # Spam texts have selective caps (WIN, FREE) -> ratio 5-25%.
        # All-caps ham texts (old phone typing style) -> ratio 70-100%.
        # Threshold at 70% cleanly separates them.
        alpha_words = _ALPHA_WORD_RE.findall(text)
        if alpha_words:
            caps_words = _CAPS_WORD_RE.findall(text)
            caps_ratio = len(caps_words) / len(alpha_words)
            if caps_ratio >= _ALL_CAPS_THRESHOLD:
                result.extend(["__all_caps_doc__"] * self.presence_repeat)
            else:
                result.extend(["__no_all_caps_doc__"] * self.absence_repeat)
        else:
            result.extend(["__no_all_caps_doc__"] * self.absence_repeat)

        return result

    def extract_bigrams(self, words: List[str]) -> List[str]:
        """
        Word bigram pseudo-tokens for phrase-level disambiguation.

        Bigrams massively reduce ambiguity for polysemous words:
          "free entry" -> __bg_free_entry__ (spam)
          "free time"  -> __bg_free_time__  (ham)
          "win prize"  -> __bg_win_prize__  (spam)
          "win or"     -> __bg_win_or__     (ham: motivational quote)
          "mobile number" -> disambiguated by context bigrams

        min_doc_freq=2 pruning removes bigrams appearing in <2 training docs,
        so only repeated spam/ham phrase patterns survive and become discriminative.
        """
        if not self.use_bigrams:
            return []

        min_len = self.bigram_min_len
        filtered = [w for w in words if len(w) >= min_len and not w.startswith("__")]
        bigrams: List[str] = []
        for i in range(len(filtered) - 1):
            bigrams.append(f"__bg_{filtered[i]}_{filtered[i + 1]}__")
        return bigrams
