# In-page fetch batch for the Pexels internal v3 API.
# Executed under the `browser-use` harness:  browser-use < crawler/browser_scripts/fetch_batch.py
# Spec arrives as a JSON file path in env VPC_FETCH_SPEC_FILE; responses land in
# spec["out_dir"] as spool records; resume state lives in spec["state_file"].
import json
import os
import time
from pathlib import Path
from urllib.parse import quote

SPEC = json.loads(Path(os.environ["VPC_FETCH_SPEC_FILE"]).read_text(encoding="utf-8"))
OUT_DIR = SPEC["out_dir"]
STATE_FILE = SPEC["state_file"]
HEADERS_JS = json.dumps(SPEC.get("headers", {}))
MAX_PAGES = int(SPEC.get("max_pages", 100))
PACE_MS = int(SPEC.get("pace_ms", 1500))
POOL = max(1, int(SPEC.get("pool", 4)))

os.makedirs(OUT_DIR, exist_ok=True)


def load_state():
    try:
        with open(STATE_FILE) as fh:
            return json.load(fh)
    except Exception:
        return {}


def save_state(state):
    tmp = STATE_FILE + ".tmp"
    with open(tmp, "w") as fh:
        json.dump(state, fh)
    os.replace(tmp, STATE_FILE)


def write_spool(key, url, payload):
    path = os.path.join(OUT_DIR, key + ".json")
    tmp = path + ".tmp"
    record = {"key": key, "url": url, "fetched_at": time.time()}
    record.update(payload)
    with open(tmp, "w") as fh:
        json.dump(record, fh)
    os.replace(tmp, path)


FETCH_ONE_JS = """
(async () => {
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), 20000);
  try {
    const resp = await fetch(%(url)s, { headers: %(headers)s, signal: controller.signal });
    const text = await resp.text();
    return JSON.stringify({ status: resp.status, body: text });
  } catch (err) {
    return JSON.stringify({ status: 0, error: String(err) });
  } finally {
    clearTimeout(timer);
  }
})()
"""

FETCH_MANY_JS = """
(async () => {
  const urls = %(urls)s;
  const headers = %(headers)s;
  const one = async (u) => {
    const controller = new AbortController();
    const timer = setTimeout(() => controller.abort(), 20000);
    try {
      const resp = await fetch(u, { headers, signal: controller.signal });
      const text = await resp.text();
      return { status: resp.status, body: text };
    } catch (err) {
      return { status: 0, error: String(err) };
    } finally {
      clearTimeout(timer);
    }
  };
  const results = await Promise.all(urls.map(one));
  return JSON.stringify(results);
})()
"""


def evaluate(expr, fallback):
    try:
        result = cdp("Runtime.evaluate", expression=expr, awaitPromise=True, returnByValue=True)
        value = (result.get("result") or {}).get("value") if isinstance(result, dict) else None
    except Exception as exc:
        print("EVAL-ERR %s" % exc)
        value = None
    if not value:
        return fallback
    try:
        return json.loads(value)
    except Exception:
        return fallback


def fetch_one(url):
    expr = FETCH_ONE_JS % {"url": json.dumps(url), "headers": HEADERS_JS}
    return evaluate(expr, {"status": 0, "error": "empty evaluate result"})


def fetch_one_retry(url):
    result = fetch_one(url)
    if result.get("status") != 200:
        time.sleep(2.0)
        result = fetch_one(url)
    return result


def fetch_many(urls):
    expr = FETCH_MANY_JS % {"urls": json.dumps(urls), "headers": HEADERS_JS}
    fallback = [{"status": 0, "error": "empty evaluate result"} for _ in urls]
    results = evaluate(expr, fallback)
    if not isinstance(results, list) or len(results) != len(urls):
        return fallback
    return results


def parse_body(result):
    if result.get("status") == 200 and isinstance(result.get("body"), str):
        try:
            return json.loads(result["body"])
        except Exception:
            return None
    return None


def error_text(result):
    return str(result.get("error") or result.get("body") or "")[:300]


def ensure_tab(state):
    target_id = state.get("target_id")
    context_id = state.get("browser_context_id")
    infos = cdp("Target.getTargets")["targetInfos"]
    if target_id and any(t["targetId"] == target_id for t in infos):
        switch_tab(target_id)
        return context_id, target_id
    if context_id:
        try:
            known = cdp("Target.getBrowserContexts").get("browserContextIds", [])
        except Exception:
            known = []
        if context_id not in known:
            context_id = None
    if not context_id:
        context_id = cdp("Target.createBrowserContext")["browserContextId"]
    created = cdp(
        "Target.createTarget",
        url="https://www.pexels.com/videos/",
        browserContextId=context_id,
    )
    target_id = created["targetId"]
    time.sleep(3)
    switch_tab(target_id)
    print("CONTEXT browser_context_id=%s target_id=%s" % (context_id, target_id))
    return context_id, target_id


state = load_state()
context_id, target_id = ensure_tab(state)
state["browser_context_id"] = context_id
state["target_id"] = target_id
save_state(state)

mode = SPEC["mode"]

if mode == "seed_chain":
    base_url = SPEC["base_url"]
    cursor = state.get("cursor") or ""
    fetched = 0
    pages_total = int(state.get("seed_pages", 0))
    while fetched < MAX_PAGES:
        url = base_url + ("&seed=" + quote(str(cursor), safe="") if cursor else "")
        result = fetch_one_retry(url)
        pages_total += 1
        key = "seed-%06d" % pages_total
        body = parse_body(result)
        if body is None:
            write_spool(key, url, {"status": result.get("status", 0), "error": error_text(result)})
            print("PAGE-FAIL %s status=%s" % (key, result.get("status")))
            break
        write_spool(key, url, {"status": 200, "body": body})
        state["seed_pages"] = pages_total
        pagination = body.get("pagination") or {}
        cursor = pagination.get("cursor") or ""
        state["cursor"] = cursor
        save_state(state)
        fetched += 1
        print("PAGE %s n=%d cursor=%s" % (key, len(body.get("data") or []), cursor))
        if not cursor or pagination.get("more_data") is False:
            print("SEED-EXHAUSTED")
            break
        time.sleep(PACE_MS / 1000.0)
    print(
        "SUMMARY "
        + json.dumps(
            {"mode": mode, "fetched": fetched, "pages_total": pages_total, "cursor": cursor}
        )
    )

elif mode == "search_list":
    targets = SPEC["targets"]
    done = set(state.get("done_keys", []))
    pending = [t for t in targets if t["key"] not in done]

    # Group page targets by term (declaration order preserved) so a term that
    # returns an empty page stops paging there instead of burning the rest of
    # its page budget on guaranteed-empty requests.
    terms_order = []
    by_term = {}
    for target in pending:
        term = target.get("term") or target["key"]
        if term not in by_term:
            by_term[term] = []
            terms_order.append(term)
        by_term[term].append(target)

    ok_count = 0
    ended_terms = 0
    fail_streak = 0

    for term in terms_order:
        pages = by_term[term]
        for start in range(0, len(pages), POOL):
            chunk = pages[start : start + POOL]
            results = fetch_many([t["url"] for t in chunk])
            term_ended = False
            for target, result in zip(chunk, results, strict=True):
                if result.get("status") != 200:
                    time.sleep(1.5)
                    result = fetch_one_retry(target["url"])
                body = parse_body(result)
                if body is None:
                    write_spool(
                        target["key"],
                        target["url"],
                        {"status": result.get("status", 0), "error": error_text(result)},
                    )
                    print("ITEM-FAIL %s status=%s" % (target["key"], result.get("status")))
                    fail_streak += 1
                    if fail_streak >= 8:
                        state["done_keys"] = sorted(done)
                        save_state(state)
                        print(
                            "ABORT fail-streak=%d term=%s (browser/daemon likely down)"
                            % (fail_streak, term)
                        )
                        print(
                            "SUMMARY "
                            + json.dumps(
                                {
                                    "mode": mode,
                                    "fetched": ok_count,
                                    "total": len(targets),
                                    "aborted": True,
                                }
                            )
                        )
                        raise SystemExit(1)
                    continue
                fail_streak = 0
                data = body.get("data") or []
                write_spool(target["key"], target["url"], {"status": 200, "body": body})
                done.add(target["key"])
                ok_count += 1
                print("ITEM %s n=%d" % (target["key"], len(data)))
                if not data:
                    term_ended = True
            if term_ended:
                rest = [t["key"] for t in pages[start + POOL :]]
                if rest:
                    done.update(rest)
                    ended_terms += 1
                    print("TERM-END %s skipped=%d" % (term, len(rest)))
                break
            time.sleep(PACE_MS / 1000.0)
        state["done_keys"] = sorted(done)
        save_state(state)
    print(
        "SUMMARY "
        + json.dumps(
            {
                "mode": mode,
                "fetched": ok_count,
                "ended_terms": ended_terms,
                "total": len(targets),
            }
        )
    )
else:
    print("ERROR unknown mode: %s" % mode)
