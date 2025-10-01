#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations
import os, re, time
import pandas as pd
import gradio as gr
from typing import Optional, Tuple, List

# ----------------- CONFIG -----------------
GCP_PROJECT = "total-fiber-399801"
GCP_REGION = "us-central1"
GOOGLE_APPLICATION_CREDENTIALS = "total-fiber-399801-cafedd5e8e6a.json"
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = GOOGLE_APPLICATION_CREDENTIALS

USE_BQ = True
GEMINI_MODEL_ID = "gemini-2.5-pro"

BQ_PROJECT = "total-fiber-399801"
BQ_DATASET = "medilink_bigquery_medilink"
# -----------------------------------------

bq_client = None
aiplatform = None
GenerativeModel = None

def _try_init_bigquery():
    global bq_client
    if not USE_BQ or bq_client is not None:
        return
    try:
        from google.cloud import bigquery
        bq_client = bigquery.Client(project=GCP_PROJECT)
    except Exception as e:
        print(f"[BigQuery] init failed: {e}")

def _try_init_vertex():
    global aiplatform, GenerativeModel
    if aiplatform is not None and GenerativeModel is not None:
        return
    try:
        from google.cloud import aiplatform as _aip
        from vertexai.generative_models import GenerativeModel as _GM
        aiplatform = _aip
        GenerativeModel = _GM
        aiplatform.init(project=GCP_PROJECT, location=GCP_REGION)
    except Exception as e:
        print(f"[VertexAI] init failed: {e}")
        aiplatform = None
        GenerativeModel = None

APP_DIR = os.path.dirname(__file__)

def load_local() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    ph = pd.read_csv(os.path.join(APP_DIR, "sample_pharmacy.csv"))
    prod = pd.read_csv(os.path.join(APP_DIR, "sample_product.csv"))
    inv = pd.read_csv(os.path.join(APP_DIR, "sample_inventory.csv"))
    price = pd.read_csv(os.path.join(APP_DIR, "sample_price.csv"))
    return ph, prod, inv, price

def latest_rows(df: pd.DataFrame, group_cols: list[str], order_col: str) -> pd.DataFrame:
    if df.empty:
        return df
    return df.sort_values(order_col).groupby(group_cols, as_index=False).tail(1)

def _split_name_strength(q: str) -> tuple[str | None, str | None]:
    ql = (q or "").lower().strip()
    m = re.search(r'(\d+(?:\s*\d+)?\s*(?:mcg|mg|g|ml|g))', ql)
    if m:
        strength = m.group(1).strip()
        name = ql.replace(strength, "").strip()
        return (name if name else None), strength
    return (ql if ql else None), None

def query_join(
    product_query: str,
    city: Optional[str],
    max_km: Optional[float],
    lat: Optional[float],
    lng: Optional[float],
    price_cap: Optional[float],
    limit: int = 200
) -> pd.DataFrame:
    name_q, strength_q = _split_name_strength(product_query)
    city = (city or "").strip() or None
    if not name_q and not strength_q:
        return pd.DataFrame(columns=[
            "Product","Strength","Pharmacy","City","Qty","Status","Price","Currency","Distance (km)","Maps","lat","lng"
        ])

    if USE_BQ:
        _try_init_bigquery()

    df = None
    if USE_BQ and bq_client is not None:
        from google.cloud import bigquery
        sql = f"""#standardSQL
WITH latest_inv AS (
  SELECT * EXCEPT(rn) FROM (
    SELECT inv.*,
           ROW_NUMBER() OVER (PARTITION BY pharmacy_id, product_id ORDER BY observed_at DESC) rn
    FROM `{BQ_PROJECT}.{BQ_DATASET}.inventory_snapshot` inv
  ) WHERE rn=1
),
latest_price AS (
  SELECT * EXCEPT(rn) FROM (
    SELECT ps.*,
           ROW_NUMBER() OVER (PARTITION BY pharmacy_id, product_id ORDER BY observed_at DESC) rn
    FROM `{BQ_PROJECT}.{BQ_DATASET}.price_snapshot` ps
  ) WHERE rn=1
)
SELECT p.product_id, p.name AS name_prod, p.strength,
       ph.pharmacy_id, ph.name AS name_pharm, ph.city, ph.lat, ph.lng,
       li.qty_available, li.status, lp.currency, lp.unit_price
FROM latest_inv li
JOIN `{BQ_PROJECT}.{BQ_DATASET}.product` p USING(product_id)
JOIN `{BQ_PROJECT}.{BQ_DATASET}.pharmacy` ph USING(pharmacy_id)
LEFT JOIN latest_price lp USING(pharmacy_id, product_id)
WHERE (@name_q IS NULL OR LOWER(p.name) LIKE @name_like
       OR LOWER(CONCAT(p.name,' ',IFNULL(p.strength,''))) LIKE @name_like)
  AND (@strength_q IS NULL OR LOWER(IFNULL(p.strength,'')) LIKE @strength_like)
  AND (@city IS NULL OR LOWER(ph.city)=LOWER(@city))
  AND (@cap IS NULL OR lp.unit_price <= @cap)
ORDER BY lp.unit_price NULLS LAST, li.qty_available DESC
LIMIT @lim
"""
        params = [
            bigquery.ScalarQueryParameter("name_q","STRING", name_q),
            bigquery.ScalarQueryParameter("name_like","STRING", f"%{name_q}%" if name_q else None),
            bigquery.ScalarQueryParameter("strength_q","STRING", strength_q),
            bigquery.ScalarQueryParameter("strength_like","STRING", f"%{strength_q}%" if strength_q else None),
            bigquery.ScalarQueryParameter("city","STRING", city if city else None),
            bigquery.ScalarQueryParameter("cap","FLOAT64", price_cap if price_cap is not None else None),
            bigquery.ScalarQueryParameter("lim","INT64", limit),
        ]
        job = bq_client.query(
            sql,
            job_config=bigquery.QueryJobConfig(
                use_legacy_sql=False,
                query_parameters=params
            ),
            job_id_prefix="medilink_find_",
            location="US",
        )
        df = job.to_dataframe()
        df.rename(columns={"name_prod":"prod_name","name_pharm":"pharm_name"}, inplace=True)

    else:
        ph, prod, inv, price = load_local()
        latest_inv = latest_rows(inv, ["pharmacy_id","product_id"], "observed_at")
        latest_price = latest_rows(price, ["pharmacy_id","product_id"], "observed_at")
        df = latest_inv.merge(prod, on="product_id").merge(ph, on="pharmacy_id").merge(
            latest_price, on=["pharmacy_id","product_id"], how="left"
        )
        if "name_x" in df.columns and "name_y" in df.columns:
            df.rename(columns={"name_x":"prod_name","name_y":"pharm_name"}, inplace=True)
        else:
            if "name" in df.columns:
                df.rename(columns={"name":"prod_name"}, inplace=True)
            if "pharmacy" in df.columns and "pharm_name" not in df.columns:
                df["pharm_name"] = df["pharmacy"]
        mask = True
        if name_q:
            mask &= df["prod_name"].str.lower().str.contains(name_q, na=False)
        if strength_q:
            mask &= df["strength"].fillna("").str.lower().str.contains(strength_q, na=False)
        if city:
            mask &= df["city"].fillna("").str.lower().eq(city.lower())
        if price_cap is not None:
            mask &= df["unit_price"].fillna(10**9) <= price_cap
        df = df[mask].copy()
        df.sort_values(["unit_price","qty_available"], ascending=[True, False], inplace=True)

    if lat is not None and lng is not None and max_km is not None and not df.empty:
        from math import radians, sin, cos, asin, sqrt
        def haversine(row):
            if pd.isna(row.get("lat")) or pd.isna(row.get("lng")):
                return float("nan")
            R=6371.0
            dlat=radians(row["lat"]-lat); dlon=radians(row["lng"]-lng)
            a=sin(dlat/2)**2 + cos(radians(lat))*cos(radians(row["lat"]))*sin(dlon/2)**2
            return 2*R*asin(sqrt(a))
        df["distance_km"] = df.apply(haversine, axis=1)
        df = df[df["distance_km"].notna() & (df["distance_km"] <= max_km)].sort_values("distance_km")

    if df is None or df.empty:
        return pd.DataFrame(columns=[
            "Product","Strength","Pharmacy","City","Qty","Status","Price","Currency","Distance (km)","Maps","lat","lng"
        ])

    def gmaps_link(row):
        if pd.isna(row.get("lat")) or pd.isna(row.get("lng")):
            return ""
        return f"https://www.google.com/maps/search/?api=1&query={row['lat']},{row['lng']}"

    out = pd.DataFrame({
        "Product": df.get("prod_name"),
        "Strength": df.get("strength"),
        "Pharmacy": df.get("pharm_name"),
        "City": df.get("city"),
        "Qty": df.get("qty_available"),
        "Status": df.get("status"),
        "Price": df.get("unit_price"),
        "Currency": df.get("currency"),
        "Distance (km)": df.get("distance_km", pd.Series([None]*len(df))),
        "lat": df.get("lat"),
        "lng": df.get("lng"),
    })
    out["Maps"] = df.apply(gmaps_link, axis=1)
    for col, asc in [("Qty", False), ("Price", True)]:
        if col in out.columns:
            out.sort_values(col, ascending=asc, inplace=True, kind="mergesort")
    out.reset_index(drop=True, inplace=True)
    return out

def explain(product_query: str, city: Optional[str]) -> str:
    _try_init_vertex()
    model_id = (GEMINI_MODEL_ID or "gemini-1.5-flash").strip()
    if aiplatform and GenerativeModel:
        try:
            model = GenerativeModel(model_id)
            prompt = (
                f"You are an assistant for a drug shortage finder in "
                f"{city or 'the selected city'}. Summarize availability and "
                f"price considerations for: {product_query}. "
                f"Keep it under 100 words. Be specific and helpful."
            )
            resp = model.generate_content(prompt)
            return (resp.text or "").strip() or "No summary."
        except Exception as e:
            return f"(Vertex AI error) {e}"
    return "Local demo mode: enable Vertex AI credentials to see summaries."

def _build_watchlist_predicates(items: List[tuple[str|None, str|None]]):
    where_clauses = []
    params = []
    from google.cloud import bigquery
    for idx, (nm, st) in enumerate(items):
        name_like = f"%{nm}%" if nm else None
        strength_like = f"%{st}%" if st else None
        where_clauses.append(
            f"( (@name{idx} IS NULL OR LOWER(p.name) LIKE @name{idx}_like "
            f"OR LOWER(CONCAT(p.name,' ',IFNULL(p.strength,''))) LIKE @name{idx}_like) "
            f"AND (@strength{idx} IS NULL OR LOWER(IFNULL(p.strength,'')) LIKE @strength{idx}_like) )"
        )
        params.extend([
            bigquery.ScalarQueryParameter(f"name{idx}","STRING", nm),
            bigquery.ScalarQueryParameter(f"name{idx}_like","STRING", name_like),
            bigquery.ScalarQueryParameter(f"strength{idx}","STRING", st),
            bigquery.ScalarQueryParameter(f"strength{idx}_like","STRING", strength_like),
        ])
    return " OR ".join(where_clauses), params

def shortage_watchlist_df(watchlist: List[str], city: Optional[str], limit: int = 200) -> pd.DataFrame:
    items = [ _split_name_strength(w) for w in (watchlist or []) ]
    items = [it for it in items if any(it)]
    if not items:
        return pd.DataFrame(columns=["Product","Strength","Pharmacy","City","Latest Qty","14d Avg","Risk","Price","Currency","Maps"])
    items = items[:10]

    if USE_BQ:
        _try_init_bigquery()
    if USE_BQ and bq_client is not None:
        from google.cloud import bigquery
        predicates_sql, pred_params = _build_watchlist_predicates(items)
        sql = f"""#standardSQL
WITH last14 AS (
  SELECT *
  FROM `{BQ_PROJECT}.{BQ_DATASET}.inventory_snapshot`
  WHERE observed_at >= TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL 14 DAY)
),
agg14 AS (
  SELECT product_id, pharmacy_id,
         AVG(qty_available) AS ma_14d
  FROM last14
  GROUP BY product_id, pharmacy_id
),
latest AS (
  SELECT * EXCEPT(rn) FROM (
    SELECT
      t.product_id,
      t.pharmacy_id,
      t.qty_available,
      t.status,
      t.min_qty,
      t.observed_at,
      ROW_NUMBER() OVER (PARTITION BY t.product_id, t.pharmacy_id ORDER BY t.observed_at DESC) rn
    FROM `{BQ_PROJECT}.{BQ_DATASET}.inventory_snapshot` t
  ) WHERE rn=1
),
joined AS (
  SELECT
    p.product_id, p.name AS prod_name, p.strength,
    ph.pharmacy_id, ph.name AS pharm_name, ph.city, ph.lat, ph.lng,
    l.qty_available AS latest_qty,
    a.ma_14d,
    l.status AS latest_status,
    l.min_qty,
    pr.currency, pr.unit_price
  FROM latest l
  JOIN `{BQ_PROJECT}.{BQ_DATASET}.product` p USING(product_id)
  JOIN `{BQ_PROJECT}.{BQ_DATASET}.pharmacy` ph USING(pharmacy_id)
  LEFT JOIN agg14 a USING(product_id, pharmacy_id)
  LEFT JOIN (
    SELECT * EXCEPT(rn) FROM (
      SELECT ps.*, ROW_NUMBER() OVER (PARTITION BY pharmacy_id, product_id ORDER BY observed_at DESC) rn
      FROM `{BQ_PROJECT}.{BQ_DATASET}.price_snapshot` ps
    ) WHERE rn=1
  ) pr USING(pharmacy_id, product_id)
  WHERE ({predicates_sql})
    AND (@city IS NULL OR LOWER(ph.city)=LOWER(@city))
)
SELECT
  prod_name AS Product,
  strength AS Strength,
  pharm_name AS Pharmacy,
  city AS City,
  latest_qty AS `Latest Qty`,
  ma_14d AS `14d Avg`,
  CASE
    WHEN latest_status IS NOT NULL THEN latest_status
    WHEN latest_qty IS NULL THEN 'OUT'
    WHEN min_qty IS NOT NULL AND latest_qty <= 0 THEN 'OUT'
    WHEN min_qty IS NOT NULL AND latest_qty < min_qty THEN 'LOW'
    ELSE 'IN_STOCK'
  END AS Risk,
  unit_price AS Price,
  currency AS Currency,
  lat, lng
FROM joined
ORDER BY
  CASE
    WHEN Risk='OUT' THEN 1
    WHEN Risk='LOW' THEN 2
    WHEN Risk='IN_STOCK' THEN 3
    ELSE 4
  END,
  `Latest Qty` ASC
LIMIT @lim
"""
        params = pred_params + [
            bigquery.ScalarQueryParameter("city","STRING", city if city else None),
            bigquery.ScalarQueryParameter("lim","INT64", limit),
        ]
        job = bq_client.query(
            sql,
            job_config=bigquery.QueryJobConfig(use_legacy_sql=False, query_parameters=params),
            location="US"
        )
        df = job.to_dataframe()
    else:
        ph, prod, inv, price = load_local()
        last14 = inv[inv["observed_at"] >= (pd.Timestamp.now(tz=None) - pd.Timedelta(days=14))]
        agg14 = last14.groupby(["product_id","pharmacy_id"], as_index=False)["qty_available"].mean().rename(columns={"qty_available":"ma_14d"})
        latest = latest_rows(inv, ["product_id","pharmacy_id"], "observed_at")
        df = latest.merge(prod, on="product_id").merge(ph, on="pharmacy_id").merge(agg14, on=["product_id","pharmacy_id"], how="left")
        df = df.merge(latest_rows(price, ["pharmacy_id","product_id"], "observed_at"), on=["pharmacy_id","product_id"], how="left")
        df["Risk"] = df.apply(lambda r: r["status"] if pd.notna(r.get("status")) else ("OUT" if (pd.isna(r.get("qty_available")) or r.get("qty_available",0) <= 0) else ("LOW" if (pd.notna(r.get("min_qty")) and r.get("qty_available",0) < r.get("min_qty")) else "IN_STOCK")), axis=1)
        df = df.rename(columns={"name_x":"prod_name","name_y":"pharm_name","qty_available":"Latest Qty","ma_14d":"14d Avg","unit_price":"Price","currency":"Currency"})
        def _match_any(row):
            for nm, st in items:
                ok = True
                if nm:
                    ok &= (nm in str(row["prod_name"]).lower())
                if st:
                    ok &= (st in str(row["strength"]).lower())
                if ok:
                    return True
            return False
        df = df[df.apply(_match_any, axis=1)]
        if city:
            df = df[df["city"].str.lower() == city.lower()]
        df = df[["prod_name","strength","pharm_name","city","Latest Qty","14d Avg","Risk","Price","Currency","lat","lng"]]
    if df.empty:
        return pd.DataFrame(columns=["Product","Strength","Pharmacy","City","Latest Qty","14d Avg","Risk","Price","Currency","Maps","lat","lng"])
    df["Maps"] = df.apply(lambda r: "" if pd.isna(r.get("lat")) or pd.isna(r.get("lng")) else f"https://www.google.com/maps/search/?api=1&query={r['lat']},{r['lng']}", axis=1)
    return df  # keep lat/lng so the map can render

BADGE_BQ = '<span class="pill pill-bq">BigQuery</span>'
BADGE_LOCAL = '<span class="pill">Local CSV</span>'
BADGE_VTX = '<span class="pill pill-vtx">Vertex AI</span>'
BADGE_OFF = '<span class="pill pill-off">AI Off</span>'

def mode_badges() -> str:
    ai_badge = BADGE_VTX if aiplatform and GenerativeModel else BADGE_OFF
    data_badge = BADGE_BQ if USE_BQ else BADGE_LOCAL
    return f"{data_badge} &nbsp; {ai_badge}"

CSS = """
.gradio-container {max-width: 1080px !important; margin: 0 auto !important;}
.pill {padding:6px 10px; border-radius:999px; background:#f1f3f4; font-weight:600; font-size:12px;}
.pill-bq {background:#e8f0fe; color:#1a73e8;}
.pill-vtx {background:#e6f4ea; color:#137333;}
.pill-off {background:#fce8e6; color:#c5221f;}
.small {font-size:.92rem;}
.right {text-align:right;}
.header {display:flex;align-items:center;justify-content:space-between;}
.card {background:#ffffff; border:1px solid #eee; border-radius:14px; padding:14px;}
.pill-chip{display:inline-block;margin:4px 6px 0 0;padding:6px 10px;border-radius:999px;border:1px solid #e0e0e0;background:#fafafa;font-weight:600;}
.pill-chip .x{margin-left:8px;cursor:pointer;}
.mapbox {border:1px solid #eee;border-radius:12px;overflow:hidden;}
"""

def _render_map_html(lat: Optional[float], lng: Optional[float], title: str) -> str:
    if lat is None or lng is None or pd.isna(lat) or pd.isna(lng):
        return "<div class='small'>No coordinates for this row.</div>"
    try:
        import folium
        m = folium.Map(location=[lat, lng], zoom_start=15)
        folium.Marker([lat, lng], tooltip=title, popup=title).add_to(m)
        return m._repr_html_()
    except Exception as e:
        return f"<div class='small'>Map render error: {e}</div>"

def build_ui():
    with gr.Blocks(css=CSS, title="MediLink AI — Drug Shortage Tracker", theme=gr.themes.Soft()) as demo:
        gr.HTML(f"""
        <div class="header">
          <div>
            <h1 style="margin:0;">🩺 MediLink AI — Drug Shortage Tracker</h1>
            <div class="small">Live pharmacy availability and prices. {mode_badges()}</div>
          </div>
          <div class="right small">
            <div><strong>Project:</strong> {GCP_PROJECT}</div>
            <div><strong>Region:</strong> {GCP_REGION}</div>
          </div>
        </div>
        """)

        # --- Search Tab ---
        find_results_state = gr.State(value=pd.DataFrame())
        with gr.Tab("Find"):
            with gr.Row():
                with gr.Column(scale=3):
                    product = gr.Textbox(label="Medicine", value="amoxicillin 500mg", placeholder="e.g., amoxicillin 500mg or paracetamol 1 g")
                with gr.Column(scale=2):
                    city = gr.Textbox(label="City (optional)", value="Accra", placeholder="e.g., Accra")
                with gr.Column(scale=1):
                    price_cap = gr.Number(label="Max price (optional)", value=None)
            with gr.Accordion("Location filter (optional)", open=False):
                with gr.Row():
                    lat = gr.Number(label="Your latitude")
                    lng = gr.Number(label="Your longitude")
                    max_km = gr.Number(label="Max distance (km)", value=10)
            with gr.Row():
                btn = gr.Button("🔎 Search", variant="primary", scale=1)
                clear = gr.Button("Reset", scale=1)
            with gr.Row():
                with gr.Column():
                    out_info = gr.Markdown("")
                    out_tbl = gr.Dataframe(
                        headers=["Product","Strength","Pharmacy","City","Qty","Status","Price","Currency","Distance (km)","Maps"],
                        wrap=True,
                        interactive=False
                    )
            with gr.Row():
                with gr.Column():
                    gr.Markdown("### Explain")
                    out_explain = gr.Markdown(value="")
            with gr.Row():
                gr.Markdown("### Map")
            with gr.Row():
                map_html_find = gr.HTML(elem_classes=["mapbox"])

            def run(product, city, price_cap, lat, lng, max_km):
                t0 = time.time()
                df_full = query_join(product, city if (city or "").strip() else None,
                                     max_km if max_km else None,
                                     lat if lat else None,
                                     lng if lng else None,
                                     price_cap)
                elapsed = time.time() - t0
                summary = explain(product, city)
                df_display = df_full.copy()
                if not df_display.empty and "Maps" in df_display.columns:
                    df_display["Maps"] = df_display["Maps"].apply(lambda u: f"[Open](%s)" % u if isinstance(u, str) and u else "")
                for c in ("lat","lng"):
                    if c in df_display.columns:
                        df_display = df_display.drop(columns=[c])
                info = f"**Results:** {len(df_display)}  |  **Latency:** {elapsed:.2f}s"
                return info, df_display, summary, df_full, ""

            btn.click(
                run,
                [product, city, price_cap, lat, lng, max_km],
                [out_info, out_tbl, out_explain, find_results_state, map_html_find]
            )

            def on_find_select(df_full: pd.DataFrame, evt: gr.SelectData):
                if df_full is None or df_full.empty:
                    return "<div class='small'>No data.</div>"
                idx = evt.index if isinstance(evt.index, int) else (evt.index[0] if evt.index else None)
                if idx is None or idx >= len(df_full):
                    return "<div class='small'>Invalid selection.</div>"
                row = df_full.iloc[idx]
                lt = row.get("lat", None)
                lg = row.get("lng", None)
                title = f"{row.get('Pharmacy', 'Pharmacy')} — {row.get('Product', 'Product')}"
                return _render_map_html(lt, lg, title)

            out_tbl.select(on_find_select, [find_results_state], [map_html_find])

            clear.click(
                lambda: ("amoxicillin 500mg","Accra",None,None,None,10,"",pd.DataFrame(),"", pd.DataFrame(), ""),
                outputs=[product, city, price_cap, lat, lng, max_km, out_info, out_tbl, out_explain, find_results_state, map_html_find]
            )

        # --- Shortage Watchlist Tab ---
        with gr.Tab("Shortage Watchlist"):
            watchlist_state = gr.State(value=["amoxicillin 500mg"])
            wl_results_state = gr.State(value=pd.DataFrame())
            with gr.Row():
                with gr.Column(scale=3):
                    wl_input = gr.Textbox(label="Add medicine to watchlist", placeholder="e.g., amoxicillin 500mg or ibuprofen 200mg")
                with gr.Column(scale=2):
                    wl_city = gr.Textbox(label="City filter (optional)", value="Accra")
                with gr.Column(scale=1):
                    wl_add = gr.Button("Add", variant="primary")
            wl_chips = gr.HTML("")
            with gr.Row():
                wl_refresh = gr.Button("Refresh")
                wl_clear = gr.Button("Clear watchlist")
            wl_table = gr.Dataframe(
                headers=["Product","Strength","Pharmacy","City","Latest Qty","14d Avg","Risk","Price","Currency","Maps"],
                wrap=True,
                interactive=False
            )
            with gr.Row():
                gr.Markdown("### Map")
            with gr.Row():
                map_html_wl = gr.HTML(elem_classes=["mapbox"])

            def render_chips(items: List[str]) -> str:
                if not items: return "<div class='small'>No items in watchlist yet.</div>"
                html = "<div>"
                for i, it in enumerate(items):
                    html += f"<span class='pill-chip'>{it}</span>"
                html += "</div>"
                return html

            def wl_add_item(curr: List[str], new_item: str):
                new_item = (new_item or "").strip()
                if new_item:
                    if curr is None: curr = []
                    if new_item.lower() not in [c.lower() for c in curr]:
                        curr = curr + [new_item]
                return curr, render_chips(curr)

            def wl_clear_all(_curr: List[str]):
                return [], render_chips([])

            def wl_refresh_tbl(curr: List[str], city_txt: str):
                df_full = shortage_watchlist_df(curr or [], (city_txt or "").strip() or None, limit=400)
                df_display = df_full.copy()
                if not df_display.empty and "Maps" in df_display.columns:
                    df_display["Maps"] = df_display["Maps"].apply(lambda u: f"[Open](%s)" % u if isinstance(u, str) and u else "")
                for c in ("lat","lng"):
                    if c in df_display.columns:
                        df_display = df_display.drop(columns=[c])
                return df_display, df_full, ""

            wl_add.click(wl_add_item, [watchlist_state, wl_input], [watchlist_state, wl_chips])
            wl_clear.click(wl_clear_all, [watchlist_state], [watchlist_state, wl_chips])
            wl_refresh.click(wl_refresh_tbl, [watchlist_state, wl_city], [wl_table, wl_results_state, map_html_wl])

            def _init_watchlist(state_items: List[str], city_txt: str):
                df_display, df_full, _ = wl_refresh_tbl(state_items or [], city_txt)
                return render_chips(state_items or []), df_display, df_full, ""

            demo.load(_init_watchlist, [watchlist_state, wl_city], [wl_chips, wl_table, wl_results_state, map_html_wl])

            def on_wl_select(df_full: pd.DataFrame, evt: gr.SelectData):
                if df_full is None or df_full.empty:
                    return "<div class='small'>No data.</div>"
                idx = evt.index if isinstance(evt.index, int) else (evt.index[0] if evt.index else None)
                if idx is None or idx >= len(df_full):
                    return "<div class='small'>Invalid selection.</div>"
                row = df_full.iloc[idx]
                lt = row.get("lat", None)
                lg = row.get("lng", None)
                title = f"{row.get('Pharmacy','Pharmacy')} — {row.get('Product','Product')}"
                return _render_map_html(lt, lg, title)

            wl_table.select(on_wl_select, [wl_results_state], [map_html_wl])

        gr.Markdown('<div class="small">Built for the Fivetran × Google Cloud challenge — custom connector → BigQuery → Vertex AI → Gradio.</div>')
    return demo

if __name__ == "__main__":
    demo = build_ui()
    port = int(os.getenv("PORT", 7860))
    demo.launch(server_name="0.0.0.0", server_port=port, show_api=False)
