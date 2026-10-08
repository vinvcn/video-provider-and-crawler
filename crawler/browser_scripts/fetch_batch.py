# In-page fetch batch for the Pexels internal v3 API.
# Executed under the `browser-use` harness:  browser-use < crawler/browser_scripts/fetch_batch.py
# Spec arrives as a JSON file path in env VPC_FETCH_SPEC_FILE; responses land in
# spec["out_dir"] as spool records; resume state lives in spec["state_file"].
import json
import os
import time
from datetime import datetime
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


def extract_attributes(payload):
    """Pull pageProps.medium.attributes out of a Next data payload.

    Mirrors crawler/nextdata.py:extract_attributes (this harness script cannot
    import the package).
    """
    if not isinstance(payload, dict):
        return None
    props = payload.get("pageProps")
    if not isinstance(props, dict):
        return None
    medium = props.get("medium")
    if not isinstance(medium, dict):
        return None
    attributes = medium.get("attributes")
    if not isinstance(attributes, dict) or "id" not in attributes:
        return None
    return attributes


def ensure_tab(state):
    """Attach to our pexels.com tab, or create one; never navigate foreign tabs.

    The browser-use daemon is shared across tasks on this machine, so the saved
    target can be navigated elsewhere by another task — in-page fetch to pexels
    then dies on CORS. In that case abandon that (polluted) context and create a
    fresh anonymous one, leaving the foreign tab untouched.
    """
    target_id = state.get("target_id")
    context_id = state.get("browser_context_id")
    infos = cdp("Target.getTargets")["targetInfos"]
    info = next((t for t in infos if t["targetId"] == target_id), None) if target_id else None
    if info is not None and str(info.get("url") or "").startswith("https://www.pexels.com"):
        switch_tab(target_id)
        return context_id, target_id
    if info is not None:
        print("TAB-FOREIGN %s -> %s (abandoning context)" % (target_id, str(info.get("url"))[:70]))
        context_id = None
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

def to_ts(value):
    try:
        return datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except Exception:
        return None


def crossed(cursor, marker):
    """True when `cursor` has reached or passed `marker` (ISO timestamps)."""
    a, b = to_ts(cursor), to_ts(marker)
    if a is not None and b is not None:
        return a <= b
    return str(cursor) <= str(marker)


mode = SPEC["mode"]

if mode == "seed_chain":
    base_url = SPEC["base_url"]
    head_mode = bool(SPEC.get("head"))
    marker = SPEC.get("until") or (state.get("head_cursor") if head_mode else None)
    cursor = "" if head_mode else (state.get("cursor") or "")
    run_date = time.strftime("%Y%m%d")
    pages_total = 0 if head_mode else int(state.get("seed_pages", 0))
    run_head_cursor = None
    completed = False
    fetched = 0
    while fetched < MAX_PAGES:
        if marker and cursor and crossed(cursor, marker):
            print("UNTIL-REACHED cursor=%s marker=%s" % (cursor, marker))
            completed = True
            break
        url = base_url + ("&seed=" + quote(str(cursor), safe="") if cursor else "")
        result = fetch_one_retry(url)
        pages_total += 1
        if head_mode:
            key = "head-%s-%06d" % (run_date, pages_total)
        else:
            key = "seed-%06d" % pages_total
        body = parse_body(result)
        if body is None:
            status = result.get("status", 0)
            write_spool(key, url, {"status": status, "error": error_text(result)})
            print("PAGE-FAIL %s status=%s" % (key, status))
            if status in (401, 403, 429):
                print("ABORT status=%s (auth/rate-limit) — batch stopped" % status)
            break
        write_spool(key, url, {"status": 200, "body": body})
        pagination = body.get("pagination") or {}
        if head_mode and run_head_cursor is None:
            run_head_cursor = pagination.get("cursor") or ""
        cursor = pagination.get("cursor") or ""
        if not head_mode:
            state["seed_pages"] = pages_total
            state["cursor"] = cursor
        save_state(state)
        fetched += 1
        print("PAGE %s n=%d cursor=%s" % (key, len(body.get("data") or []), cursor))
        if head_mode and marker is None:
            print("HEAD-BOOTSTRAP first page only; marker will be set to %s" % run_head_cursor)
            completed = True
            break
        if not cursor or pagination.get("more_data") is False:
            print("SEED-EXHAUSTED")
            completed = True
            break
        time.sleep(PACE_MS / 1000.0)
    if head_mode and completed and run_head_cursor:
        state["head_cursor"] = run_head_cursor
        state["run_date"] = run_date
        save_state(state)
        print("HEAD-MARKER advanced to %s" % run_head_cursor)
    print(
        "SUMMARY "
        + json.dumps(
            {
                "mode": mode,
                "head": head_mode,
                "fetched": fetched,
                "pages_total": pages_total,
                "cursor": cursor,
                "marker": marker,
                "completed": completed,
            }
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
                    status = result.get("status", 0)
                    write_spool(
                        target["key"],
                        target["url"],
                        {"status": status, "error": error_text(result)},
                    )
                    print("ITEM-FAIL %s status=%s" % (target["key"], status))
                    if status in (401, 403, 429):
                        state["done_keys"] = sorted(done)
                        save_state(state)
                        print("ABORT status=%s (auth/rate-limit) term=%s" % (status, term))
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
            state["done_keys"] = sorted(done)
            save_state(state)
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

elif mode == "id_list":
    # Per-video gap fill through the Next.js data route; needs a pexels page
    # loaded so window.__NEXT_DATA__.buildId is readable.
    template = SPEC["url_template"]
    targets = SPEC["targets"]
    done = set(state.get("done_keys", []))
    pending = [t for t in targets if t["key"] not in done]

    def make_url(build, target):
        return (
            template.replace("{build}", str(build))
            .replace("{slug}", str(target.get("slug") or ""))
            .replace("{id}", str(target["id"]))
        )

    def read_build_id():
        value = evaluate(
            "JSON.stringify(window.__NEXT_DATA__ ? window.__NEXT_DATA__.buildId : null)", None
        )
        return value if isinstance(value, str) else None

    def wait_build_id(timeout):
        deadline = time.time() + timeout
        while time.time() < deadline:
            found = read_build_id()
            if found:
                return found
            time.sleep(1.0)
        return None

    # A fresh context needs time to pass Cloudflare and render; a previously
    # persisted buildId works too because the data route itself is unchallenged.
    build_id = wait_build_id(5.0) or state.get("build_id")
    if not build_id:
        try:
            cdp("Page.navigate", url="https://www.pexels.com/videos/")
        except Exception as exc:
            print("BUILD-NAV-FAIL %s" % exc)
        build_id = wait_build_id(30.0)
    if not build_id:
        print("BUILD-MISSING no __NEXT_DATA__.buildId (challenge not passed?)")
        raise SystemExit(1)
    state["build_id"] = build_id
    save_state(state)
    print("BUILD %s" % build_id)

    ok_count = 0
    missing = 0
    fail_streak = 0
    for start in range(0, len(pending), POOL):
        chunk = pending[start : start + POOL]
        results = fetch_many([make_url(build_id, t) for t in chunk])
        if all(result.get("status") == 404 for result in results):
            # a run of 404s usually means the site rotated buildId on deploy
            refreshed = read_build_id()
            if refreshed and refreshed != build_id:
                print("BUILD-REFRESH %s -> %s" % (build_id, refreshed))
                build_id = refreshed
                state["build_id"] = build_id
                save_state(state)
                results = fetch_many([make_url(build_id, t) for t in chunk])
        for target, result in zip(chunk, results, strict=True):
            url = make_url(build_id, target)
            if result.get("status") != 200:
                time.sleep(1.5)
                result = fetch_one_retry(url)
            status = result.get("status", 0)
            attributes = extract_attributes(parse_body(result))
            if attributes is None:
                write_spool(target["key"], url, {"status": status, "error": error_text(result)})
                print("ITEM-FAIL %s status=%s" % (target["key"], status))
                if status in (401, 403, 429):
                    state["done_keys"] = sorted(done)
                    save_state(state)
                    print("ABORT status=%s (auth/rate-limit)" % status)
                    raise SystemExit(1)
                if status == 404:
                    # deleted or unlisted video: record it and stop retrying it
                    missing += 1
                    done.add(target["key"])
                fail_streak += 1
                if fail_streak >= 8:
                    state["done_keys"] = sorted(done)
                    save_state(state)
                    print("ABORT fail-streak=%d (browser/daemon likely down)" % fail_streak)
                    raise SystemExit(1)
                continue
            fail_streak = 0
            write_spool(target["key"], url, {"status": 200, "attributes": attributes})
            done.add(target["key"])
            ok_count += 1
            print("ITEM %s %s" % (target["key"], attributes.get("slug") or ""))
        state["done_keys"] = sorted(done)
        save_state(state)
        time.sleep(PACE_MS / 1000.0)
    print(
        "SUMMARY "
        + json.dumps(
            {
                "mode": mode,
                "fetched": ok_count,
                "missing": missing,
                "total": len(targets),
            }
        )
    )
else:
    print("ERROR unknown mode: %s" % mode)
