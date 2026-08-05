#!/usr/bin/env bash
# One-time setup for the Upwind eval dashboard on the Oracle VM.
#
# What it does:
#   1. Creates ~/eval-www (where the nightly workflow drops index.html)
#   2. Creates a random basic-auth password (printed once, stored 0600)
#   3. Adds an nginx `location /eval` block to the getupwind.me server
#   4. Reloads nginx and verifies the page answers
#
# Idempotent: safe to re-run; it never overwrites existing credentials.
#
# Run as the deploy user (the same user GitHub Actions uses):
#   bash scripts/setup_eval_dashboard.sh
set -euo pipefail

EVAL_DIR="$HOME/eval-www"
HTPASSWD="$EVAL_DIR/.htpasswd"
AUTH_FILE="$HOME/.eval_auth"
EVAL_USER="eval"

if command -v sudo >/dev/null 2>&1 && sudo -n true 2>/dev/null; then
  HAS_SUDO=1
else
  HAS_SUDO=0
fi

echo "==> 1/4 eval directory ($EVAL_DIR)"
mkdir -p "$EVAL_DIR"
chmod 755 "$EVAL_DIR"

if [ -f "$HTPASSWD" ]; then
  echo "    htpasswd exists, keeping existing credentials"
else
  PASSWORD="$(openssl rand -base64 18 | tr -dc 'A-Za-z0-9' | cut -c1-16)"
  HASH="$(openssl passwd -apr1 "$PASSWORD")"
  printf '%s:%s\n' "$EVAL_USER" "$HASH" > "$HTPASSWD"
  printf '%s:%s\n' "$EVAL_USER" "$PASSWORD" > "$AUTH_FILE"
  chmod 600 "$HTPASSWD" "$AUTH_FILE"
  echo "    generated credentials (also stored in $AUTH_FILE):"
  echo "      URL:      https://getupwind.me/eval/"
  echo "      user:     $EVAL_USER"
  echo "      password: $PASSWORD"
fi

if [ ! -f "$EVAL_DIR/index.html" ]; then
  printf '<!doctype html><title>Upwind eval</title><p>Dashboard not rendered yet; the first nightly run will populate this page.</p>\n' > "$EVAL_DIR/index.html"
  chmod 644 "$EVAL_DIR/index.html"
fi

echo "==> 2/4 nginx location /eval"
SERVER_FILES="$(grep -RlE 'server_name[^;]*getupwind\.me' /etc/nginx/sites-enabled /etc/nginx/conf.d 2>/dev/null || true)"
if [ -z "$SERVER_FILES" ]; then
  echo "ERROR: no nginx server block for getupwind.me found." >&2
  echo "Add one, then re-run this script." >&2
  exit 1
fi

if [ "$HAS_SUDO" != 1 ]; then
  echo "ERROR: passwordless sudo is required to edit the nginx config." >&2
  echo "Run this script as a user with passwordless sudo, then re-run it." >&2
  exit 1
fi

sudo python3 - "$HTPASSWD" "$EVAL_DIR" "$SERVER_FILES" <<'PY'
import os
import sys

htpasswd, evaldir, files = sys.argv[1], sys.argv[2], sys.argv[3]
snippet = (
    "\n    # Upwind eval dashboard (managed by setup_eval_dashboard.sh)\n"
    '    location /eval {\n'
    '        auth_basic "Upwind eval";\n'
    f"        auth_basic_user_file {htpasswd};\n"
    f"        alias {evaldir}/;\n"
    "        index index.html;\n"
    "    }\n"
)


def server_blocks(src):
    """Return (start, end) index pairs of top-level server blocks."""
    blocks = []
    depth = 0
    start = None
    for i, ch in enumerate(src):
        if ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0 and start is not None:
                blocks.append((start, i))
                start = None
    return blocks


def block_score(text):
    if "location /eval" in text:
        return 3
    score = 1
    if "listen 443" in text or "ssl_certificate" in text or "proxy_pass" in text:
        score += 1
    return score


for path in [p.strip() for p in files.splitlines() if p.strip()]:
    path = os.path.realpath(path)
    src = open(path).read()
    best = None
    best_score = 0
    for start, end in server_blocks(src):
        block = src[start:end]
        if "server_name" not in block or "getupwind.me" not in block:
            continue
        score = block_score(block)
        if score > best_score:
            best = (start, end, block)
            best_score = score
    if best is None:
        continue
    start, end, block = best
    if "location /eval" in block:
        print(f"    {path}: location /eval already present, skipping")
        sys.exit(0)
    backup = path + ".bak-eval"
    if not os.path.exists(backup):
        open(backup, "w").write(src)
    insert_at = src.rfind("}", start, end + 1)
    new_src = src[:insert_at] + snippet + src[insert_at:]
    open(path, "w").write(new_src)
    print(f"    {path}: added location /eval (backup at {backup})")
    sys.exit(0)

print("ERROR: no matching server block found in any candidate file", file=sys.stderr)
sys.exit(1)
PY

echo "==> 3/4 nginx reload"
if [ "$HAS_SUDO" = 1 ]; then
  if ! sudo nginx -t; then
    echo "nginx config invalid; rolling back..." >&2
    sudo python3 - "$SERVER_FILES" <<'PY'
import os, sys
for p in sys.argv[1].splitlines():
    p = os.path.realpath(p.strip())
    backup = p + ".bak-eval"
    if os.path.exists(backup):
        open(p, "w").write(open(backup).read())
PY
    sudo nginx -t
    exit 1
  fi
  sudo systemctl reload nginx
else
  echo "    no passwordless sudo available; run manually:"
  echo "      sudo nginx -t && sudo systemctl reload nginx"
fi

echo "==> 4/4 health check"
CODE="$(curl -s -o /dev/null -w '%{http_code}' -k -u "$(cat "$AUTH_FILE")" -H 'Host: getupwind.me' http://127.0.0.1/eval/ 2>/dev/null || true)"
if [ "$CODE" != "200" ]; then
  CODE="$(curl -s -o /dev/null -w '%{http_code}' -k -u "$(cat "$AUTH_FILE")" -H 'Host: getupwind.me' https://127.0.0.1/eval/ 2>/dev/null || true)"
fi
echo "    GET /eval/ -> HTTP $CODE"
if [ "$CODE" != "200" ]; then
  echo "WARNING: health check failed; check the nginx config and reload." >&2
  exit 1
fi
echo "Done. Dashboard will be published here nightly: https://getupwind.me/eval/"
