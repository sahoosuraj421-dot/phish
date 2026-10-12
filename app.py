import os
import re
import ipaddress
from urllib.parse import urlparse

import joblib
import streamlit as st
import tldextract
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline

# ---------------------------------------------------------
# 1. PAGE CONFIG
# ---------------------------------------------------------
st.set_page_config(page_title="Phishing Detection System", page_icon="🛡️", layout="centered")

# Use the bundled public-suffix snapshot (no network call at runtime)
EXTRACTOR = tldextract.TLDExtract(suffix_list_urls=())

MODEL_PATH = "phishing_model.joblib"  # put a properly trained model here
LABEL_RE = re.compile(r"^(?!-)[a-z0-9-]{1,63}(?<!-)$")


# ---------------------------------------------------------
# 2. EMAIL BODY MODEL
# ---------------------------------------------------------
@st.cache_resource
def get_email_model():
    """Load a trained model if available; otherwise fall back to a tiny DEMO model."""
    if os.path.exists(MODEL_PATH):
        return joblib.load(MODEL_PATH), True

    X = [
        "Dear customer, please verify your bank account login immediately to avoid suspension.",
        "Your Netflix subscription payment failed. Click here to update your billing details.",
        "Hey John, are we still meeting for lunch tomorrow at noon? Let me know.",
        "Your weekly project update and sprint retrospective notes are attached.",
        "URGENT: Password reset requested. Confirm your credentials via this secure link.",
        "Invoice #4920 attached for your review. Thanks for your business.",
        "Security Alert: Multiple failed login attempts detected on your account. Verify now.",
        "Hi team, please find the quarterly financial projections attached for review.",
    ]
    y = [1, 1, 0, 0, 1, 0, 1, 0]
    model = make_pipeline(TfidfVectorizer(), LogisticRegression())
    model.fit(X, y)
    return model, False


email_model, model_is_trained = get_email_model()


# ---------------------------------------------------------
# 3. FORMAT VALIDATION (gate: invalid format => never "legitimate")
# ---------------------------------------------------------
def valid_hostname(host: str):
    """Return (ok, is_ip, extracted)."""
    host = host.lower().rstrip(".")
    try:
        ipaddress.ip_address(host)
        return True, True, None
    except ValueError:
        pass
    if not host or len(host) > 253:
        return False, False, None
    if not all(LABEL_RE.match(label) for label in host.split(".")):
        return False, False, None
    ext = EXTRACTOR(host)
    if not ext.domain or not ext.suffix:  # needs a real public suffix (.com, .co.uk ...)
        return False, False, None
    return True, False, ext


def validate_url(url: str):
    url = url.strip()
    if not url or re.search(r"\s", url):
        return False, "URL is empty or contains spaces.", None
    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https"):
        return False, "URL must start with http:// or https://", None
    try:
        parsed.port  # raises ValueError on bad port
    except ValueError:
        return False, "Invalid port number.", None
    if not parsed.hostname:
        return False, "URL has no hostname.", None
    ok, is_ip, ext = valid_hostname(parsed.hostname)
    if not ok:
        return False, "Hostname is not a valid domain name or IP address.", None
    return True, "", (parsed, is_ip, ext)


EMAIL_LOCAL_RE = re.compile(r"^[a-z0-9!#$%&'*+/=?^_`{|}~-]+(\.[a-z0-9!#$%&'*+/=?^_`{|}~-]+)*$")


def validate_email(email: str):
    email = email.strip().lower()
    if email.count("@") != 1:
        return False, "Email must contain exactly one '@'.", None
    local, domain = email.split("@")
    if not local or len(local) > 64 or not EMAIL_LOCAL_RE.match(local):
        return False, "Invalid characters or structure in the part before '@'.", None
    ok, is_ip, ext = valid_hostname(domain)
    if not ok or is_ip:
        return False, "Domain after '@' is not a valid domain name.", None
    return True, "", (local, domain, ext)


# ---------------------------------------------------------
# 4. HEURISTIC SCORING (only runs on well-formed input)
# ---------------------------------------------------------
SUSPICIOUS_WORDS = ["login", "verify", "secure", "update", "account", "banking",
                    "signin", "confirm", "password", "support", "security"]
SHORTENERS = {"bit.ly", "tinyurl.com", "t.co", "goo.gl", "ow.ly", "is.gd", "buff.ly", "cutt.ly"}
FREE_PROVIDERS = {"gmail.com", "yahoo.com", "hotmail.com", "outlook.com", "aol.com"}
BRANDS = {"paypal": "paypal.com", "amazon": "amazon.com", "netflix": "netflix.com",
          "microsoft": "microsoft.com", "apple": "apple.com", "google": "google.com",
          "facebook": "facebook.com", "sbi": "sbi.co.in", "hdfc": "hdfcbank.com"}


def has_brand_impersonation(host_labels: str, registered: str):
    for brand, official in BRANDS.items():
        if brand in host_labels and registered != official:
            return brand
    return None


def analyze_url(url: str):
    ok, reason, data = validate_url(url)
    if not ok:
        return {"valid": False, "reason": reason, "score": 1.0, "flags": []}

    parsed, is_ip, ext = data
    host = parsed.hostname.lower()
    flags, points = [], 0

    if len(url) > 75:
        points += 2; flags.append("Very long URL")
    if is_ip:
        points += 4; flags.append("Raw IP address used instead of a domain")
    if "@" in parsed.netloc:
        points += 3; flags.append("'@' in the address (hides the real destination)")
    if parsed.scheme != "https":
        points += 1; flags.append("Not using HTTPS")
    if parsed.port not in (None, 80, 443):
        points += 1; flags.append(f"Non-standard port {parsed.port}")

    if ext:
        registered = f"{ext.domain}.{ext.suffix}"
        if ext.subdomain.count(".") + (1 if ext.subdomain else 0) >= 3:
            points += 2; flags.append("Many subdomain levels")
        if "-" in ext.domain:
            points += 1; flags.append("Hyphen in main domain name")
        if any(w in host for w in SUSPICIOUS_WORDS):
            points += 2; flags.append("Suspicious keyword in hostname")
        if registered in SHORTENERS:
            points += 2; flags.append("URL shortener hides destination")
        if "xn--" in host:
            points += 2; flags.append("Punycode (possible look-alike characters)")
        brand = has_brand_impersonation(host, registered)
        if brand:
            points += 4; flags.append(f"Mentions '{brand}' but is not the official domain")
    else:
        registered = host

    score = min(points / 8.0, 0.99) if points else 0.05
    if not flags:
        flags.append("No suspicious patterns found")
    return {"valid": True, "domain": registered, "score": score, "flags": flags,
            "length": len(url), "https": parsed.scheme == "https"}


def analyze_sender(email: str):
    ok, reason, data = validate_email(email)
    if not ok:
        return {"valid": False, "reason": reason, "score": 1.0, "flags": []}

    local, domain, ext = data
    flags, points = [], 0

    digits = sum(c.isdigit() for c in local)
    if digits >= 3:
        points += 2; flags.append(f"Many digits in username ({digits})")

    if domain not in FREE_PROVIDERS:
        if any(w in domain for w in SUSPICIOUS_WORDS):  # counted once, not per keyword
            points += 3; flags.append("Suspicious keyword in domain")
        if "-" in ext.domain:
            points += 2; flags.append("Hyphenated domain (common in look-alikes)")
        brand = has_brand_impersonation(domain, f"{ext.domain}.{ext.suffix}")
        if brand:
            points += 4; flags.append(f"Mentions '{brand}' but is not the official domain")
    else:
        if any(w in local for w in ["support", "admin", "security", "official", "bank"]):
            points += 4; flags.append("Free webmail used with an official-sounding username")

    score = min(points / 7.0, 0.95) if points else 0.05
    if not flags:
        flags.append("Standard sender pattern")
    return {"valid": True, "username": local, "domain": domain, "score": score, "flags": flags}


def analyze_body(text: str):
    """Returns (probability or None, note)."""
    vectorizer = email_model.steps[0][1]
    if vectorizer.transform([text]).nnz == 0:
        return None, "None of the words are known to the model, so no reliable score."
    prob = float(email_model.predict_proba([text])[0][1])
    note = "" if model_is_trained else "Demo model (8 samples) - not reliable. Train on a real dataset."
    return prob, note


# ---------------------------------------------------------
# 5. UI
# ---------------------------------------------------------
st.title("🛡️ Phishing Detection System")
st.markdown("Checks **URLs**, **sender emails** and **email text**. Badly formatted input is never treated as legitimate.")

tab1, tab2 = st.tabs(["🌐 URL Inspector", "📧 Email Inspector"])

with tab1:
    url_input = st.text_input("Paste URL:", placeholder="https://example.com/login")
    if st.button("Evaluate URL", type="primary"):
        if not url_input.strip():
            st.warning("Please enter a URL.")
        else:
            r = analyze_url(url_input)
            st.divider()
            if not r["valid"]:
                st.error(f"🚫 **INVALID URL FORMAT - treated as not safe.** {r['reason']}")
            elif r["score"] >= 0.35:
                st.error(f"🚨 **Verdict: PHISHING / SUSPICIOUS URL** (Risk: {r['score']*100:.0f}%)")
            else:
                st.success(f"✅ **Verdict: Looks legitimate** (Risk: {r['score']*100:.0f}%)")
            if r["valid"]:
                with st.expander("🔍 Details"):
                    st.write(f"- **Domain:** `{r['domain']}`")
                    st.write(f"- **Length:** {r['length']} chars")
                    st.write(f"- **HTTPS:** {'Yes ✅' if r['https'] else 'No ⚠️'}")
                    for f in r["flags"]:
                        st.write(f"- {f}")

with tab2:
    sender_input = st.text_input("Sender email:", placeholder="support@paypal-security.com")
    body_input = st.text_area("Email body (optional):", placeholder="Dear user, click here to verify your account...")
    if st.button("Evaluate Email", type="primary"):
        if not sender_input.strip() and not body_input.strip():
            st.warning("Provide a sender address and/or email body.")
        else:
            sender = analyze_sender(sender_input) if sender_input.strip() else None
            body_score, body_note = (analyze_body(body_input) if body_input.strip() else (None, ""))

            st.divider()
            if sender and not sender["valid"]:
                st.error(f"🚫 **INVALID EMAIL FORMAT - treated as not safe.** {sender['reason']}")
            else:
                scores = []
                if sender:
                    scores.append(sender["score"])
                if body_score is not None:
                    scores.append(body_score)
                if not scores:
                    st.warning("Could not compute a score. Add a sender address or more email text.")
                else:
                    risk = max(scores)  # strongest signal wins; averaging hides bad senders
                    if risk >= 0.4:
                        st.error(f"🚨 **Verdict: PHISHING / SUSPICIOUS EMAIL** (Risk: {risk*100:.0f}%)")
                    else:
                        st.success(f"✅ **Verdict: Looks legitimate** (Risk: {risk*100:.0f}%)")

                with st.expander("🔍 Details"):
                    if sender:
                        st.markdown("**Sender**")
                        st.write(f"- Username: `{sender['username']}`  |  Domain: `{sender['domain']}`")
                        st.write(f"- Sender risk: {sender['score']*100:.0f}%")
                        for f in sender["flags"]:
                            st.write(f"  - {f}")
                    if body_input.strip():
                        st.markdown("**Body (NLP)**")
                        if body_score is not None:
                            st.write(f"- Phishing probability: {body_score*100:.0f}%")
                        if body_note:
                            st.caption(body_note)