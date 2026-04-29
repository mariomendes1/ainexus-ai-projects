"""RealState Agent Lab - educational 3-agent real estate assistant."""

from __future__ import annotations

import json
import os
import re
import time
import unicodedata
from dataclasses import dataclass
from typing import Any
from urllib.parse import urljoin

import httpx
from bs4 import BeautifulSoup
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

app = FastAPI(title="RealState Agent Lab API", version="0.1.0", root_path="/realstate")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

MODEL_NAME = os.getenv("OLLAMA_MODEL", "granite3.1-dense:2b")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://host.docker.internal:11434")
OLLAMA_TIMEOUT = float(os.getenv("OLLAMA_TIMEOUT", "45"))
INE_API_ENABLED = os.getenv("INE_API_ENABLED", "1").strip().lower() in {
    "1",
    "true",
    "yes",
    "on",
}
INE_API_TIMEOUT = float(os.getenv("INE_API_TIMEOUT", "20"))
INE_API_URL_TEMPLATE = os.getenv("INE_API_URL_TEMPLATE", "").strip()
INE_API_DEFAULT_VARCD = os.getenv("INE_API_DEFAULT_VARCD", "").strip()
INE_API_DEFAULT_LANG = os.getenv("INE_API_DEFAULT_LANG", "PT").strip() or "PT"

# ── INE geocode cache: municipality name → INE geocod ──
_ine_geo_cache: dict[str, str] = {}  # normalised_name → geocod
_ine_geo_cache_loaded = False


def _ine_name_key(name: str) -> str:
    """Lowercase + strip accents for fuzzy municipality matching."""
    nfkd = unicodedata.normalize("NFKD", name or "")
    return "".join(c for c in nfkd if not unicodedata.combining(c)).lower().strip()


def _load_ine_geo_cache() -> None:
    """Fetch INE municipality geocodes on first use (cached for lifetime of process)."""
    global _ine_geo_cache_loaded
    if _ine_geo_cache_loaded:
        return
    _ine_geo_cache_loaded = True  # set early — don't retry on failure
    try:
        url = "https://www.ine.pt/ine/json_indicador/pindica.jsp?op=2&varcd=0012248&Dim3=T&lang=PT"
        with httpx.Client(timeout=20.0, headers={"User-Agent": USER_AGENT}) as client:
            r = client.get(url)
            r.raise_for_status()
            data = r.json()
        dados = data[0].get("Dados", {})
        last_period = list(dados.keys())[-1]
        for entry in dados[last_period]:
            geocod = entry.get("geocod", "").strip()
            geodsg = entry.get("geodsg", "").strip()
            if geocod and geodsg:
                _ine_geo_cache[_ine_name_key(geodsg)] = geocod
    except Exception:
        pass  # cache stays empty; will fall back to PT


def _resolve_ine_geo_code(req: "RealStateRequest") -> str:
    """Return INE geocode for the request: manual override → city lookup → PT fallback."""
    if req.ine_geo_code and req.ine_geo_code.strip():
        return req.ine_geo_code.strip()

    _load_ine_geo_cache()
    if not _ine_geo_cache:
        return "PT"

    for raw in (req.municipality, req.city):
        if not raw:
            continue
        key = _ine_name_key(raw)
        if key in _ine_geo_cache:
            return _ine_geo_cache[key]
        # partial match — useful for "Lisboa" matching "Grande Lisboa" etc.
        for cached_key, geocod in _ine_geo_cache.items():
            if key in cached_key or cached_key in key:
                return geocod

    return "PT"


def _ine_dim3(req: "RealStateRequest") -> str:
    """Map property_type to INE Dim3: 1=Apartamentos, 2=Moradias, T=Total."""
    pt = (req.property_type or "").lower().strip()
    if pt in {"apartment", "apartamento", "flat", "studio"}:
        return "1"
    if pt in {"house", "moradia", "vivenda", "villa", "detached"}:
        return "2"
    return "T"


GEOSPATIAL_PLUGIN_ENABLED = os.getenv("GEOSPATIAL_PLUGIN_ENABLED", "0").strip().lower() in {
    "1",
    "true",
    "yes",
    "on",
}
OFFICIAL_CONNECTOR_ENABLED = os.getenv("OFFICIAL_CONNECTOR_ENABLED", "0").strip().lower() in {
    "1",
    "true",
    "yes",
    "on",
}
OFFICIAL_CONNECTOR_BASE_URL = os.getenv("OFFICIAL_CONNECTOR_BASE_URL", "").strip()
OFFICIAL_CONNECTOR_VERIFY_PATH = os.getenv("OFFICIAL_CONNECTOR_VERIFY_PATH", "/v1/real-estate/verify").strip() or "/v1/real-estate/verify"
OFFICIAL_CONNECTOR_TOKEN = os.getenv("OFFICIAL_CONNECTOR_TOKEN", "").strip()
OFFICIAL_CONNECTOR_TIMEOUT = float(os.getenv("OFFICIAL_CONNECTOR_TIMEOUT", "12"))
OFFICIAL_AREA_SOURCES = {
    "Caderneta Predial",
    "Certidão Predial",
    "Licença de Utilização",
    "Outro Documento Oficial",
}

MARKET_DB = {
    "lisboa": {"avg_price_m2": 5650, "rent_yield": 4.8, "trend": "high demand, low stock"},
    "porto": {"avg_price_m2": 3850, "rent_yield": 5.2, "trend": "stable growth, strong rentals"},
    "braga": {"avg_price_m2": 2450, "rent_yield": 5.6, "trend": "good value, rising interest"},
    "faro": {"avg_price_m2": 3520, "rent_yield": 5.0, "trend": "seasonal pressure, tourism-driven"},
    "setubal": {"avg_price_m2": 2960, "rent_yield": 5.4, "trend": "commuter demand and appreciation"},
}


class RealStateRequest(BaseModel):
    goal: str = Field(..., min_length=8, max_length=1200)
    location: str = Field(..., min_length=2, max_length=220)
    street_address: str | None = Field(default=None, max_length=180)
    city: str | None = Field(default=None, max_length=80)
    municipality: str | None = Field(default=None, max_length=80)
    property_type: str = Field(default="apartment", max_length=40)
    bedrooms: int = Field(default=2, ge=0, le=15)
    area_m2: float = Field(..., gt=10, le=2500)
    price_eur: float = Field(..., gt=5000, le=100000000)
    monthly_rent_eur: float | None = Field(default=None, ge=0, le=1000000)
    down_payment_pct: float = Field(default=20, ge=0, le=100)
    interest_rate_pct: float = Field(default=3.5, ge=0.0, le=30)
    loan_years: int = Field(default=30, ge=1, le=50)
    listing_url: str | None = Field(default=None, max_length=1500)
    ine_varcd: str | None = Field(default=None, max_length=30)
    ine_geo_code: str | None = Field(default=None, max_length=80)
    ine_period: str | None = Field(default=None, max_length=40)
    ine_bdd_url: str | None = Field(default=None, max_length=1500)
    ine_lang: str | None = Field(default=None, max_length=3)
    official_area_m2: float | None = Field(default=None, gt=10, le=2500)
    official_area_source: str | None = Field(default=None, max_length=120)
    official_reference: str | None = Field(default=None, max_length=120)


class ScrapeRequest(BaseModel):
    url: str = Field(..., min_length=8, max_length=1500)


class GeoPreviewResponse(BaseModel):
    status: str
    query: str
    display_name: str | None = None
    lat: float | None = None
    lon: float | None = None
    source: str
    legal_evidence: bool = False
    detail: str | None = None


class ToolResult(BaseModel):
    name: str
    data: dict[str, Any]


class TraceStep(BaseModel):
    agent: str
    latency_ms: int
    summary: str


class RealStateResponse(BaseModel):
    final_report: str
    planner: dict[str, Any]
    tools: list[ToolResult]
    trace: list[TraceStep]
    model: str
    llm_enabled: bool


USER_AGENT = (
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
)


def _geopandas_available() -> bool:
    try:
        import geopandas  # noqa: F401
        import shapely  # noqa: F401

        return True
    except Exception:
        return False


@dataclass
class AgentContext:
    llm_enabled: bool
    model: str


def _loan_payment(price: float, down_pct: float, yearly_rate: float, years: int) -> dict[str, float]:
    principal = price * (1 - down_pct / 100.0)
    months = years * 12
    monthly_rate = yearly_rate / 100.0 / 12.0
    if monthly_rate == 0:
        payment = principal / months
    else:
        payment = principal * (monthly_rate / (1 - (1 + monthly_rate) ** (-months)))
    return {
        "principal_eur": round(principal, 2),
        "monthly_payment_eur": round(payment, 2),
        "total_paid_eur": round(payment * months, 2),
    }


def _market_snapshot(location: str, area_m2: float, price_eur: float, market_avg_override: float | None = None, reference_source: str = "local_model") -> dict[str, Any]:
    key = location.strip().lower()
    base = MARKET_DB.get(key, {"avg_price_m2": 3200, "rent_yield": 5.0, "trend": "insufficient local sample"})
    ref_avg = float(market_avg_override) if isinstance(market_avg_override, (int, float)) else float(base["avg_price_m2"])
    asking_m2 = price_eur / max(area_m2, 1)
    delta = (asking_m2 - ref_avg) / ref_avg * 100
    inflation_class = "em_linha"
    if delta > 15:
        inflation_class = "altamente_inflacionado"
    elif delta > 5:
        inflation_class = "inflacionado"
    elif delta < -10:
        inflation_class = "abaixo_mercado"
    elif delta < -5:
        inflation_class = "ligeiramente_abaixo"
    return {
        "city": location,
        "market_avg_price_m2_eur": round(ref_avg, 2),
        "asking_price_m2_eur": round(asking_m2, 2),
        "premium_vs_market_pct": round(delta, 2),
        "inflation_class": inflation_class,
        "reference_source": reference_source,
        "market_trend": base["trend"],
    }


def _normalize_text(value: str | None) -> str:
    return (value or "").strip()


def _build_location_query(req: RealStateRequest) -> str:
    parts = [
        _normalize_text(req.street_address),
        _normalize_text(req.city),
        _normalize_text(req.municipality),
    ]
    joined = ", ".join([p for p in parts if p])
    return joined or _normalize_text(req.location)


def _market_location(req: RealStateRequest) -> str:
    return _normalize_text(req.city) or _normalize_text(req.municipality) or _normalize_text(req.location)


def _rental_yield(price_eur: float, monthly_rent_eur: float | None, location: str) -> dict[str, Any]:
    key = location.strip().lower()
    reference = MARKET_DB.get(key, {"rent_yield": 5.0})["rent_yield"]
    if not monthly_rent_eur or monthly_rent_eur <= 0:
        return {"gross_yield_pct": None, "reference_city_yield_pct": reference}
    annual = monthly_rent_eur * 12
    gross = annual / max(price_eur, 1) * 100
    return {
        "gross_yield_pct": round(gross, 2),
        "reference_city_yield_pct": round(reference, 2),
    }


def _pct_diff(a: float, b: float) -> float:
    base = max(abs(b), 0.0001)
    return abs(a - b) / base * 100.0


def _safe_float(value: Any) -> float | None:
    if isinstance(value, (int, float)):
        return float(value)
    if value is None:
        return None
    txt = str(value).strip().replace(" ", "").replace(",", ".")
    try:
        return float(txt)
    except Exception:
        return None


def _extract_ine_value_from_json(data: Any) -> float | None:
    value_keys = {"value", "valor", "v", "obs_value", "obsvalue", "valorobs"}
    numeric_hits: list[float] = []

    def walk(node: Any) -> None:
        if isinstance(node, dict):
            for k, v in node.items():
                key = str(k).lower()
                if key in value_keys:
                    num = _safe_float(v)
                    if num is not None and 100 <= num <= 100000:
                        numeric_hits.append(num)
                walk(v)
        elif isinstance(node, list):
            for item in node:
                walk(item)

    walk(data)
    if not numeric_hits:
        return None
    return float(numeric_hits[-1])


def _extract_ine_value_from_xml(xml_text: str) -> float | None:
    candidates: list[float] = []
    for match in re.findall(r"<(?:valor|value|v)[^>]*>\s*([^<]+)\s*</(?:valor|value|v)>", xml_text, flags=re.IGNORECASE):
        num = _safe_float(match)
        if num is not None and 100 <= num <= 100000:
            candidates.append(num)
    if not candidates:
        for match in re.findall(r">([0-9]{2,5}(?:[.,][0-9]{1,4})?)<", xml_text):
            num = _safe_float(match)
            if num is not None and 100 <= num <= 100000:
                candidates.append(num)
    if not candidates:
        return None
    return float(candidates[-1])


def _ine_market_reference(req: RealStateRequest) -> dict[str, Any]:
    if not INE_API_ENABLED:
        return {"status": "disabled", "source": "INE API", "reason": "INE_API_ENABLED=0"}

    lang = (req.ine_lang or INE_API_DEFAULT_LANG or "PT").strip().upper()
    varcd = (req.ine_varcd or INE_API_DEFAULT_VARCD or "").strip()
    geo = _resolve_ine_geo_code(req)  # auto-resolve from city/municipality, fallback PT
    dim3 = _ine_dim3(req)             # 1=Apartamentos, 2=Moradias, T=Total
    period = (req.ine_period or "").strip()
    direct_url = (req.ine_bdd_url or "").strip()

    dim3_label = {"1": "Apartamentos", "2": "Moradias", "T": "Total"}.get(dim3, dim3)

    url = ""
    if direct_url:
        url = direct_url
    elif INE_API_URL_TEMPLATE and varcd:
        try:
            url = INE_API_URL_TEMPLATE.format(
                varcd=varcd,
                lang=lang,
                geo_code=geo,
                period=period,
                dim3=dim3,
            )
        except Exception as exc:
            return {"status": "invalid_template", "source": "INE API", "reason": f"Template inválido: {exc}"}
    else:
        return {
            "status": "missing_config",
            "source": "INE API",
            "reason": "Define ine_bdd_url no pedido ou INE_API_URL_TEMPLATE + varcd.",
            "varcd": varcd or None,
        }

    try:
        with httpx.Client(timeout=INE_API_TIMEOUT, headers={"User-Agent": USER_AGENT, "Accept": "application/json, application/xml;q=0.9, */*;q=0.8"}) as client:
            r = client.get(url)
        if r.status_code >= 400:
            return {
                "status": "error",
                "source": "INE API",
                "http_status": r.status_code,
                "reason": f"HTTP {r.status_code}",
                "url": url,
                "varcd": varcd or None,
            }

        content_type = (r.headers.get("content-type") or "").lower()
        value = None
        if "json" in content_type:
            try:
                payload = r.json()
                value = _extract_ine_value_from_json(payload)
            except Exception:
                value = None
        if value is None:
            value = _extract_ine_value_from_xml(r.text)

        if value is None:
            return {
                "status": "invalid_payload",
                "source": "INE API",
                "url": url,
                "varcd": varcd or None,
                "geo_code": geo or None,
                "period": period or None,
                "reason": "Sem valor numérico interpretável no payload.",
            }

        return {
            "status": "ok",
            "source": "INE API",
            "reference_price_m2_eur": round(float(value), 2),
            "property_type_label": dim3_label,
            "geo_code": geo or None,
            "varcd": varcd or None,
            "period": period or None,
            "lang": lang,
        }
    except Exception as exc:
        return {
            "status": "error",
            "source": "INE API",
            "reason": str(exc),
            "url": url,
            "varcd": varcd or None,
        }


def _official_connector_ready() -> bool:
    return OFFICIAL_CONNECTOR_ENABLED and bool(OFFICIAL_CONNECTOR_BASE_URL and OFFICIAL_CONNECTOR_TOKEN)


def _official_connector_verify(req: RealStateRequest) -> dict[str, Any]:
    source = (req.official_area_source or "").strip()
    reference = (req.official_reference or "").strip()
    if not OFFICIAL_CONNECTOR_ENABLED:
        return {
            "status": "disabled",
            "provider": "official_connector_api",
            "ready": False,
            "reason": "Conector oficial desativado por configuração.",
        }
    if not OFFICIAL_CONNECTOR_BASE_URL:
        return {
            "status": "missing_config",
            "provider": "official_connector_api",
            "ready": False,
            "reason": "OFFICIAL_CONNECTOR_BASE_URL não configurado.",
        }
    if not OFFICIAL_CONNECTOR_TOKEN:
        return {
            "status": "missing_config",
            "provider": "official_connector_api",
            "ready": False,
            "reason": "OFFICIAL_CONNECTOR_TOKEN não configurado.",
        }
    if not source or not reference:
        return {
            "status": "skipped_missing_reference",
            "provider": "official_connector_api",
            "ready": True,
            "reason": "Faltam fonte oficial e/ou referência documental.",
        }

    payload = {
        "source": source,
        "reference": reference,
        "street_address": req.street_address,
        "city": req.city,
        "municipality": req.municipality,
        "country": "PT",
    }
    url = urljoin(OFFICIAL_CONNECTOR_BASE_URL.rstrip("/") + "/", OFFICIAL_CONNECTOR_VERIFY_PATH.lstrip("/"))
    headers = {
        "Authorization": f"Bearer {OFFICIAL_CONNECTOR_TOKEN}",
        "Content-Type": "application/json",
        "Accept": "application/json",
    }
    try:
        with httpx.Client(timeout=OFFICIAL_CONNECTOR_TIMEOUT, headers=headers) as client:
            r = client.post(url, json=payload)
        data: dict[str, Any]
        try:
            data = r.json()
        except Exception:
            data = {"raw_text": r.text[:500]}

        if r.status_code >= 400:
            return {
                "status": "error",
                "provider": "official_connector_api",
                "ready": True,
                "http_status": r.status_code,
                "reason": data.get("detail") or data.get("message") or f"Erro HTTP {r.status_code}",
                "response_excerpt": data,
            }

        area = data.get("area_m2")
        parsed_area = None
        if isinstance(area, (int, float)):
            parsed_area = float(area)
        else:
            try:
                parsed_area = float(area) if area is not None else None
            except Exception:
                parsed_area = None

        return {
            "status": "ok" if parsed_area else "invalid_payload",
            "provider": "official_connector_api",
            "ready": True,
            "http_status": r.status_code,
            "area_m2": round(parsed_area, 2) if parsed_area else None,
            "source": data.get("source") or source,
            "reference": data.get("reference") or reference,
            "raw_status": data.get("status"),
            "matched": data.get("matched"),
            "response_excerpt": data,
        }
    except Exception as exc:
        return {
            "status": "error",
            "provider": "official_connector_api",
            "ready": True,
            "reason": str(exc),
        }


def _extract_price_candidates(text: str) -> list[float]:
    values: list[float] = []
    for raw in re.findall(r"(?:€\s*)?(\d{1,3}(?:[.\s]\d{3})+(?:,\d{2})?|\d{5,9})", text):
        cleaned = raw.replace(" ", "").replace(".", "").replace(",", ".")
        try:
            num = float(cleaned)
        except ValueError:
            continue
        if 10000 <= num <= 100000000:
            values.append(num)
    return values


def _extract_area_candidates(text: str) -> list[float]:
    values: list[float] = []
    for raw in re.findall(r"(\d{2,4}(?:[.,]\d{1,2})?)\s*(?:m2|m²)", text.lower()):
        cleaned = raw.replace(",", ".")
        try:
            num = float(cleaned)
        except ValueError:
            continue
        if 10 <= num <= 2500:
            values.append(num)
    return values


def _scrape_listing(url: str) -> dict[str, Any]:
    url = url.strip()
    if not (url.startswith("http://") or url.startswith("https://")):
        raise ValueError("URL inválida. Usa formato completo com http/https.")

    with httpx.Client(timeout=12.0, follow_redirects=True, headers={"User-Agent": USER_AGENT}) as client:
        r = client.get(url)
        r.raise_for_status()
        html = r.text

    soup = BeautifulSoup(html, "html.parser")
    title = ""
    if soup.title and soup.title.string:
        title = soup.title.string.strip()
    if not title:
        meta_title = soup.find("meta", attrs={"property": "og:title"})
        if meta_title and meta_title.get("content"):
            title = meta_title["content"].strip()

    description = ""
    meta_desc = soup.find("meta", attrs={"name": "description"})
    if meta_desc and meta_desc.get("content"):
        description = meta_desc["content"].strip()
    if not description:
        og_desc = soup.find("meta", attrs={"property": "og:description"})
        if og_desc and og_desc.get("content"):
            description = og_desc["content"].strip()

    body_text = soup.get_text(" ", strip=True)
    prices = _extract_price_candidates(f"{title} {description} {body_text[:12000]}")
    areas = _extract_area_candidates(f"{title} {description} {body_text[:12000]}")

    price = max(prices) if prices else None
    area = max(areas) if areas else None
    price_m2 = round(price / area, 2) if price and area else None

    return {
        "url": str(r.url),
        "status_code": r.status_code,
        "title": title or "Sem título",
        "description_excerpt": (description[:280] + "...") if len(description) > 280 else description,
        "detected_price_eur": price,
        "detected_area_m2": area,
        "detected_price_m2_eur": price_m2,
    }


def _validate_area(req: RealStateRequest, scraped_area_m2: float | None, connector_check: dict[str, Any]) -> dict[str, Any]:
    input_area = float(req.area_m2)
    official_area = float(req.official_area_m2) if req.official_area_m2 else None
    source = (req.official_area_source or "").strip()
    ref = (req.official_reference or "").strip()
    connector_status = connector_check.get("status")
    connector_area = connector_check.get("area_m2")

    if source and source not in OFFICIAL_AREA_SOURCES:
        source = ""

    if connector_status == "ok" and isinstance(connector_area, (int, float)) and source and ref:
        api_area = float(connector_area)
        diff_input = _pct_diff(input_area, api_area)
        diff_scrape = _pct_diff(scraped_area_m2, api_area) if scraped_area_m2 else None
        if diff_input <= 3.0:
            status = "api_verified"
            label = "Verificado por API oficial"
            confidence = 0.98
        else:
            status = "api_conflict"
            label = "Conflito com API oficial"
            confidence = 0.4
        return {
            "status": status,
            "label": label,
            "validation_level": "automated_api",
            "area_used_m2": round(api_area, 2),
            "official_area_m2": round(api_area, 2),
            "input_area_m2": round(input_area, 2),
            "scraped_area_m2": round(scraped_area_m2, 2) if scraped_area_m2 else None,
            "input_vs_official_diff_pct": round(diff_input, 2),
            "scrape_vs_official_diff_pct": round(diff_scrape, 2) if diff_scrape is not None else None,
            "source": connector_check.get("source") or source,
            "reference": connector_check.get("reference") or ref,
            "confidence": confidence,
            "official_connector_status": connector_status,
        }

    if official_area and source:
        diff_input = _pct_diff(input_area, official_area)
        diff_scrape = _pct_diff(scraped_area_m2, official_area) if scraped_area_m2 else None
        if diff_input <= 3.0:
            status = "documentary_consistent"
            label = "Consistente com documento (sem API)"
        else:
            status = "documentary_conflict"
            label = "Conflito documental (sem API)"
        validation_level = "documentary_manual"
        return {
            "status": status,
            "label": label,
            "validation_level": validation_level,
            "area_used_m2": round(official_area, 2),
            "official_area_m2": round(official_area, 2),
            "input_area_m2": round(input_area, 2),
            "scraped_area_m2": round(scraped_area_m2, 2) if scraped_area_m2 else None,
            "input_vs_official_diff_pct": round(diff_input, 2),
            "scrape_vs_official_diff_pct": round(diff_scrape, 2) if diff_scrape is not None else None,
            "source": source,
            "reference": ref or None,
            "confidence": 0.55 if status == "documentary_consistent" else 0.45,
            "official_connector_status": connector_status,
        }

    if source and not official_area:
        return {
            "status": "unverified_official_incomplete",
            "label": "Fonte oficial incompleta",
            "validation_level": "none",
            "area_used_m2": round(input_area, 2),
            "official_area_m2": None,
            "input_area_m2": round(input_area, 2),
            "scraped_area_m2": round(scraped_area_m2, 2) if scraped_area_m2 else None,
            "input_vs_official_diff_pct": None,
            "scrape_vs_official_diff_pct": None,
            "source": source,
            "reference": ref or None,
            "confidence": 0.3,
            "reason": "Selecionaste fonte oficial, mas falta indicar a área oficial para validar.",
            "official_connector_status": connector_status,
        }

    reason = "Sem referência oficial anexada para validar a área."
    if req.official_area_m2 and not source:
        reason = "Área oficial fornecida sem indicar fonte oficial."
    return {
        "status": "unverified_official_missing",
        "label": "Não confirmado",
        "validation_level": "none",
        "area_used_m2": round(input_area, 2),
        "official_area_m2": None,
        "input_area_m2": round(input_area, 2),
        "scraped_area_m2": round(scraped_area_m2, 2) if scraped_area_m2 else None,
        "input_vs_official_diff_pct": None,
        "scrape_vs_official_diff_pct": None,
        "source": source or None,
        "reference": ref or None,
        "confidence": 0.35,
        "reason": reason,
        "official_connector_status": connector_status,
    }


def _utm_epsg_for_lon_lat(lon: float, lat: float) -> int:
    zone = int((lon + 180) / 6) + 1
    return 32600 + zone if lat >= 0 else 32700 + zone


def _nominatim_lookup(
    location: str,
    street_address: str | None = None,
    city: str | None = None,
    municipality: str | None = None,
) -> dict[str, Any] | None:
    params: dict[str, Any] = {"format": "jsonv2", "limit": 1, "addressdetails": 1}
    has_structured = bool(_normalize_text(street_address) or _normalize_text(city) or _normalize_text(municipality))
    if has_structured:
        if _normalize_text(street_address):
            params["street"] = _normalize_text(street_address)
        if _normalize_text(city):
            params["city"] = _normalize_text(city)
        if _normalize_text(municipality):
            params["county"] = _normalize_text(municipality)
        params["country"] = "Portugal"
    else:
        params["q"] = location
    with httpx.Client(timeout=10.0, headers={"User-Agent": USER_AGENT}) as client:
        r = client.get("https://nominatim.openstreetmap.org/search", params=params)
        r.raise_for_status()
        matches = r.json()
    if not matches:
        return None
    return matches[0]


def _geospatial_context(
    location: str,
    street_address: str | None = None,
    city: str | None = None,
    municipality: str | None = None,
) -> dict[str, Any]:
    if not GEOSPATIAL_PLUGIN_ENABLED:
        return {
            "status": "disabled",
            "label": "Plugin geoespacial desligado",
            "legal_evidence": False,
            "note": "Ativa GEOSPATIAL_PLUGIN_ENABLED=1 para contexto geoespacial.",
        }

    hit = _nominatim_lookup(location, street_address=street_address, city=city, municipality=municipality)
    if not hit:
        return {
            "status": "not_found",
            "label": "Localização não encontrada",
            "query": location,
            "legal_evidence": False,
        }

    lat = float(hit["lat"])
    lon = float(hit["lon"])
    out: dict[str, Any] = {
        "status": "ok",
        "label": "Contexto geoespacial disponível",
        "query": location,
        "display_name": hit.get("display_name"),
        "lat": round(lat, 6),
        "lon": round(lon, 6),
        "source": "OpenStreetMap Nominatim",
        "legal_evidence": False,
        "usage_note": "Contexto geográfico de apoio; não substitui prova legal/registral.",
    }

    if _geopandas_available():
        import geopandas as gpd
        from shapely.geometry import Point

        gdf = gpd.GeoDataFrame(
            [{"location": location}],
            geometry=[Point(lon, lat)],
            crs="EPSG:4326",
        )
        epsg = _utm_epsg_for_lon_lat(lon, lat)
        metric = gdf.to_crs(epsg=epsg)
        buffer_m = 1500
        buffer_geom = metric.buffer(buffer_m)
        out["geopandas"] = {
            "enabled": True,
            "epsg_metric": epsg,
            "sample_buffer_m": buffer_m,
            "sample_buffer_area_m2": round(float(buffer_geom.area.iloc[0]), 2),
        }
    else:
        out["geopandas"] = {
            "enabled": False,
            "note": "GeoPandas não instalado no runtime.",
        }

    return out


def _official_sources_status(req: RealStateRequest, area_validation: dict[str, Any], connector_check: dict[str, Any]) -> dict[str, Any]:
    source = (req.official_area_source or "").strip()
    normalized_source = source if source in OFFICIAL_AREA_SOURCES else None
    has_official_area = req.official_area_m2 is not None
    has_reference = bool((req.official_reference or "").strip())
    area_status = area_validation.get("status")
    validation_level = area_validation.get("validation_level", "none")
    area_used_official = area_status in {"documentary_consistent", "documentary_conflict", "api_verified", "api_conflict"}
    serious_validation = area_status in {"api_verified", "api_conflict"}

    return {
        "selected_source": normalized_source,
        "selected_source_is_official": bool(normalized_source),
        "official_area_provided": has_official_area,
        "official_reference_provided": has_reference,
        "used_in_decision": area_used_official,
        "serious_validation": serious_validation,
        "validation_level": validation_level,
        "area_validation_status": area_status,
        "connector_check_status": connector_check.get("status"),
        "official_connectors": {
            "AT_justica_bupi": {
                "status": "ready" if _official_connector_ready() else "not_ready",
                "requires_auth": True,
                "configured": OFFICIAL_CONNECTOR_ENABLED,
                "note": connector_check.get("reason") or "Conector oficial ativo por configuração.",
            }
        },
        "free_sources": {
            "openstreetmap_nominatim": {
                "status": "enabled" if GEOSPATIAL_PLUGIN_ENABLED else "disabled",
                "legal_evidence": False,
            }
        },
    }


def _toolkit(req: RealStateRequest) -> list[ToolResult]:
    location_query = _build_location_query(req)
    market_location = _market_location(req)
    listing_data: dict[str, Any] | None = None
    if req.listing_url:
        try:
            listing_data = _scrape_listing(req.listing_url)
        except Exception as exc:
            listing_data = {"url": req.listing_url, "error": str(exc)}

    scraped_area = None
    if listing_data and isinstance(listing_data.get("detected_area_m2"), (int, float)):
        scraped_area = float(listing_data["detected_area_m2"])

    connector_check = _official_connector_verify(req)
    ine_ref = _ine_market_reference(req)
    area_validation = _validate_area(req, scraped_area, connector_check)
    area_used_m2 = float(area_validation["area_used_m2"])
    official_sources = _official_sources_status(req, area_validation, connector_check)
    ine_market_avg = ine_ref.get("reference_price_m2_eur") if ine_ref.get("status") == "ok" else None

    tools = [
        ToolResult(name="ine_market_reference", data=ine_ref),
        ToolResult(name="official_connector_check", data=connector_check),
        ToolResult(name="area_validation", data=area_validation),
        ToolResult(name="official_sources_status", data=official_sources),
        ToolResult(
            name="market_snapshot",
            data=_market_snapshot(
                market_location,
                area_used_m2,
                req.price_eur,
                market_avg_override=ine_market_avg,
                reference_source="ine_api" if ine_market_avg is not None else "local_model",
            ),
        ),
        ToolResult(
            name="mortgage_simulator",
            data=_loan_payment(req.price_eur, req.down_payment_pct, req.interest_rate_pct, req.loan_years),
        ),
        ToolResult(name="rental_yield", data=_rental_yield(req.price_eur, req.monthly_rent_eur, market_location)),
    ]
    tools.append(
        ToolResult(
            name="geospatial_context",
            data=_geospatial_context(
                location_query,
                street_address=req.street_address,
                city=req.city,
                municipality=req.municipality,
            ),
        )
    )
    if listing_data:
        tools.append(ToolResult(name="listing_scrape", data=listing_data))
    return tools


def _extract_json(text: str) -> dict[str, Any] | None:
    start = text.find("{")
    end = text.rfind("}")
    if start < 0 or end <= start:
        return None
    try:
        return json.loads(text[start : end + 1])
    except json.JSONDecodeError:
        return None


def _chat(messages: list[dict[str, str]], model: str) -> str:
    with httpx.Client(timeout=OLLAMA_TIMEOUT) as client:
        r = client.post(
            f"{OLLAMA_BASE_URL}/api/chat",
            json={"model": model, "messages": messages, "stream": False},
        )
        r.raise_for_status()
        data = r.json()
        return data.get("message", {}).get("content", "").strip()


def _planner_fallback(req: RealStateRequest, tools: list[ToolResult]) -> dict[str, Any]:
    area_val = next((t.data for t in tools if t.name == "area_validation"), {})
    market = next((t.data for t in tools if t.name == "market_snapshot"), {})
    area_risk = "Área analisada com base documental (sem API oficial)."
    if area_val.get("status") == "documentary_consistent":
        area_risk = "Área consistente com documento declarado, sem validação API oficial."
    elif area_val.get("status") == "documentary_conflict":
        area_risk = "Conflito documental sem validação API oficial."
    elif area_val.get("status") == "api_verified":
        area_risk = "Área validada por API oficial."
    elif area_val.get("status") == "api_conflict":
        area_risk = "Conflito entre área introduzida e API oficial."
    if area_val.get("status") in {"unverified_official_missing", "unverified_official_incomplete"}:
        area_risk = "Área não confirmada por fonte oficial."
    return {
        "objective": f"Avaliar negócio em {_market_location(req)} para decisão rápida.",
        "steps": [
            "Comparar preço por m2 com média local.",
            "Calcular esforço mensal de financiamento.",
            "Estimar viabilidade de rentabilidade.",
        ],
        "risk_flags": [
            "Preço por m2 acima da referência oficial."
            if float(market.get("premium_vs_market_pct", 0)) > 12
            else "Sem sobrepreço crítico.",
            area_risk,
            "Taxa de juro pode variar no curto prazo.",
        ],
    }


def _planner_agent(req: RealStateRequest, tools: list[ToolResult], ctx: AgentContext) -> dict[str, Any]:
    if not ctx.llm_enabled:
        return _planner_fallback(req, tools)

    prompt = (
        "És o agente Planner de imobiliário. Devolve APENAS JSON com chaves: "
        "objective, steps (array), risk_flags (array). Contexto:\n"
        f"goal={req.goal}\nlocation={req.location}\nproperty_type={req.property_type}\n"
        f"bedrooms={req.bedrooms}\narea_m2={req.area_m2}\nprice_eur={req.price_eur}\n"
        f"tools={json.dumps([t.model_dump() for t in tools], ensure_ascii=False)}"
    )
    text = _chat([{"role": "system", "content": "Responde sempre em PT-PT."}, {"role": "user", "content": prompt}], ctx.model)
    parsed = _extract_json(text)
    return parsed if parsed else _planner_fallback(req, tools)


def _analyst_fallback(req: RealStateRequest, tools: list[ToolResult], planner: dict[str, Any]) -> str:
    area_val = next((t.data for t in tools if t.name == "area_validation"), {})
    sources = next((t.data for t in tools if t.name == "official_sources_status"), {})
    market = next((t.data for t in tools if t.name == "market_snapshot"), {})
    mort = next((t.data for t in tools if t.name == "mortgage_simulator"), {})
    rent = next((t.data for t in tools if t.name == "rental_yield"), {})
    yield_text = "n/d"
    if rent.get("gross_yield_pct") is not None:
        yield_text = f"{rent.get('gross_yield_pct')}% (referência cidade: {rent.get('reference_city_yield_pct')}%)"
    area_text = f"Validação de área: {area_val.get('label', 'n/d')}"
    if area_val.get("source"):
        area_text += f" [{area_val.get('source')}]"
    source_text = (
        "Validação séria por API oficial aplicada."
        if sources.get("serious_validation")
        else "Sem validação séria por API oficial na decisão; apenas evidência documental/manual."
    )
    inflation = market.get("inflation_class") or "n/d"
    return (
        f"Resumo técnico: preço/m2 do ativo é {market.get('asking_price_m2_eur', 'n/d')} EUR vs "
        f"referência {market.get('market_avg_price_m2_eur', 'n/d')} EUR ({market.get('premium_vs_market_pct', 'n/d')}%, classe: {inflation}). "
        f"Prestação estimada: {mort.get('monthly_payment_eur', 'n/d')} EUR/mês. Yield bruta: {yield_text}. "
        f"{area_text}. "
        f"{source_text} "
        f"Riscos: {', '.join(planner.get('risk_flags', []))}."
    )


def _analyst_agent(req: RealStateRequest, tools: list[ToolResult], planner: dict[str, Any], ctx: AgentContext) -> str:
    if not ctx.llm_enabled:
        return _analyst_fallback(req, tools, planner)

    prompt = (
        "És o agente Analyst imobiliário. Usa os dados para produzir análise curta, factual e sem marketing.\n"
        f"Planner={json.dumps(planner, ensure_ascii=False)}\n"
        f"Tools={json.dumps([t.model_dump() for t in tools], ensure_ascii=False)}\n"
        f"Objetivo do cliente={req.goal}\n"
        "Responde em 5-8 frases."
    )
    text = _chat([{"role": "system", "content": "Responde em PT-PT com números."}, {"role": "user", "content": prompt}], ctx.model)
    return text if text else _analyst_fallback(req, tools, planner)


def _writer_fallback(req: RealStateRequest, planner: dict[str, Any], analysis: str) -> str:
    return (
        "Recomendação final:\n"
        f"- Objetivo: {planner.get('objective', req.goal)}\n"
        f"- Análise: {analysis}\n"
        "- Próximos passos: negociar preço com base no €/m2 e confirmar custos de aquisição (IMT, escritura, manutenção).\n"
        "- Decisão sugerida: avançar apenas se o preço final ficar dentro de uma margem segura face à média local."
    )


def _writer_agent(req: RealStateRequest, planner: dict[str, Any], analysis: str, ctx: AgentContext) -> str:
    if not ctx.llm_enabled:
        return _writer_fallback(req, planner, analysis)

    prompt = (
        "És o agente Writer. Entrega resposta final para cliente em português de Portugal.\n"
        "Formato obrigatório:\n"
        "1) Conclusão curta\n2) Pontos-chave (3 bullets)\n3) Próximos passos (3 bullets)\n"
        f"Planner={json.dumps(planner, ensure_ascii=False)}\n"
        f"Analysis={analysis}\n"
        f"Goal={req.goal}"
    )
    text = _chat(
        [
            {"role": "system", "content": "Tom profissional, claro e sem exageros."},
            {"role": "user", "content": prompt},
        ],
        ctx.model,
    )
    return text if text else _writer_fallback(req, planner, analysis)


def _timed(agent: str, fn, *args) -> tuple[Any, TraceStep]:
    start = time.perf_counter()
    result = fn(*args)
    ms = int((time.perf_counter() - start) * 1000)
    summary = str(result)
    if len(summary) > 180:
        summary = summary[:180] + "..."
    return result, TraceStep(agent=agent, latency_ms=ms, summary=summary)


def _llm_available() -> bool:
    try:
        with httpx.Client(timeout=4.0) as client:
            r = client.get(f"{OLLAMA_BASE_URL}/api/tags")
            r.raise_for_status()
            data = r.json()
        models = [m.get("name", "") for m in data.get("models", [])]
        return any(m.startswith(MODEL_NAME) for m in models)
    except Exception:
        return False


@app.get("/health")
def health() -> dict[str, Any]:
    enabled = _llm_available()
    return {
        "status": "ok",
        "model": MODEL_NAME,
        "llm_enabled": enabled,
        "ollama_base_url": OLLAMA_BASE_URL,
        "geospatial_plugin_enabled": GEOSPATIAL_PLUGIN_ENABLED,
        "geopandas_available": _geopandas_available(),
        "official_connector_enabled": OFFICIAL_CONNECTOR_ENABLED,
        "official_connector_ready": _official_connector_ready(),
        "official_connector_base_url_set": bool(OFFICIAL_CONNECTOR_BASE_URL),
        "official_connector_token_set": bool(OFFICIAL_CONNECTOR_TOKEN),
        "ine_api_enabled": INE_API_ENABLED,
        "ine_api_template_set": bool(INE_API_URL_TEMPLATE),
        "ine_api_default_varcd_set": bool(INE_API_DEFAULT_VARCD),
        "official_area_sources": sorted(OFFICIAL_AREA_SOURCES),
    }


@app.post("/tools/scrape")
def scrape_listing(req: ScrapeRequest) -> dict[str, Any]:
    try:
        data = _scrape_listing(req.url)
        return {"status": "ok", "data": data}
    except Exception as exc:
        return {"status": "error", "detail": str(exc)}


@app.get("/geo/preview", response_model=GeoPreviewResponse)
def geo_preview(
    location: str,
    street_address: str | None = None,
    city: str | None = None,
    municipality: str | None = None,
) -> GeoPreviewResponse:
    query = location.strip()
    if not query:
        return GeoPreviewResponse(status="error", query=location, source="OpenStreetMap Nominatim", detail="Localização vazia.")
    if not GEOSPATIAL_PLUGIN_ENABLED:
        return GeoPreviewResponse(
            status="disabled",
            query=query,
            source="OpenStreetMap Nominatim",
            detail="Plugin geoespacial está desligado.",
        )
    try:
        hit = _nominatim_lookup(query, street_address=street_address, city=city, municipality=municipality)
        if not hit:
            return GeoPreviewResponse(
                status="not_found",
                query=query,
                source="OpenStreetMap Nominatim",
                detail="Localização não encontrada.",
            )
        return GeoPreviewResponse(
            status="ok",
            query=query,
            display_name=hit.get("display_name"),
            lat=round(float(hit["lat"]), 6),
            lon=round(float(hit["lon"]), 6),
            source="OpenStreetMap Nominatim",
            legal_evidence=False,
        )
    except Exception as exc:
        return GeoPreviewResponse(
            status="error",
            query=query,
            source="OpenStreetMap Nominatim",
            detail=str(exc),
        )


@app.post("/orchestrate", response_model=RealStateResponse)
def orchestrate(req: RealStateRequest) -> RealStateResponse:
    llm_enabled = _llm_available()
    ctx = AgentContext(llm_enabled=llm_enabled, model=MODEL_NAME)
    tools = _toolkit(req)
    trace: list[TraceStep] = []

    planner, step = _timed("planner", _planner_agent, req, tools, ctx)
    trace.append(step)

    analysis, step = _timed("analyst", _analyst_agent, req, tools, planner, ctx)
    trace.append(step)

    final_report, step = _timed("writer", _writer_agent, req, planner, analysis, ctx)
    trace.append(step)

    return RealStateResponse(
        final_report=final_report,
        planner=planner,
        tools=tools,
        trace=trace,
        model=MODEL_NAME,
        llm_enabled=llm_enabled,
    )
