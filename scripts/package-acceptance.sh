#!/usr/bin/env bash
#
# scripts/package-acceptance.sh BINDIR -- the fresh-user acceptance sequence,
# run against the binaries of an ACTUAL produced package (an extracted .deb,
# .tar.gz or .zip), never a build tree. Same script on Linux, macOS and
# Windows (Git Bash); needs bash, curl and python3 only.
#
# It uses a throwaway user profile (HOME/USERPROFILE/LOCALAPPDATA) so the
# run is "a user who just unpacked it". One tiny catalog model
# (smollm2:135m q4_k_m) is downloaded over HTTPS with the product's own
# SHA-256 verification, then planned, served and queried.
#
# Not tested here: GPU backends, multi-model, large contexts.
set -euo pipefail

BIN="${1:?usage: package-acceptance.sh BINDIR [expected-version]}"
EXPECT_VERSION="${2:-}"
case "$(uname -s)" in
MINGW* | MSYS* | CYGWIN*) OS=windows; EXE=.exe ;;
Darwin) OS=macos; EXE= ;;
*) OS=linux; EXE= ;;
esac
M="$BIN/membrane$EXE"
[ -x "$M" ] || { echo "no executable at $M" >&2; exit 2; }

WORK="$(mktemp -d)"
export HOME="$WORK/home"
mkdir -p "$HOME"
unset XDG_DATA_HOME XDG_CONFIG_HOME XDG_STATE_HOME MEMBRANE_MODELS_PATH \
	MEMBRANE_SERVER_CONFIG_PATH MEMBRANE_MODELS_INSTALL_DIR || true
if [ "$OS" = windows ]; then
	mkdir -p "$WORK/lad"
	export USERPROFILE="$(cygpath -w "$HOME")"
	export LOCALAPPDATA="$(cygpath -w "$WORK/lad")"
fi
PORT=8642
SERVER_PID=""
cleanup() {
	if [ -n "$SERVER_PID" ]; then
		kill "$SERVER_PID" 2>/dev/null || true
		[ "$OS" = windows ] && taskkill //F //IM membrane.exe >/dev/null 2>&1 || true
	fi
	rm -rf "$WORK"
}
trap cleanup EXIT

step() { printf '\n== %s\n' "$*"; }
ok() { printf '[PASS] %s\n' "$*"; }
fail() { printf '[FAIL] %s\n' "$*" >&2; exit 1; }
jget() { python3 -c "import json,sys; d=json.load(sys.stdin); print($1)"; }

step "1. membrane --version"
V="$("$M" --version)"; echo "$V"
[ -z "$EXPECT_VERSION" ] || echo "$V" | grep -q "$EXPECT_VERSION" || fail "version is not $EXPECT_VERSION"
ok "version"

step "2. membrane (status screen)"
"$M" || fail "bare membrane exited non-zero"
ok "status screen"

step "3. membrane doctor"
"$M" doctor || true
"$M" doctor --json > "$WORK/doctor.json" || true
DOC_OS="$(jget "[c for c in d['checks'] if c['name']=='platform'][0]['detail']['os']" < "$WORK/doctor.json")"
DATA_DIR="$(jget "[c for c in d['checks'] if c['name']=='platform'][0]['detail']['data_dir']" < "$WORK/doctor.json")"
echo "doctor says os=$DOC_OS data_dir=$DATA_DIR"
[ "$DOC_OS" = "$OS" ] || fail "doctor reports os=$DOC_OS, expected $OS"
HW="$(jget "[c for c in d['checks'] if c['name']=='hardware'][0]['status']" < "$WORK/doctor.json")"
[ "$HW" != ERROR ] || fail "hardware check is ERROR (membrane-run not found next to membrane?)"
ok "doctor (platform=$DOC_OS hardware=$HW)"

step "4. membrane models available"
"$M" models available | tee "$WORK/avail.txt"
grep -qi smollm2 "$WORK/avail.txt" || fail "catalog does not list smollm2"
ok "catalog"

step "5. acquire one tiny model (HTTPS + product SHA-256)"
"$M" model install smollm2:135m --quant q4_k_m --json > "$WORK/install.json"
NAME="$(jget "d['name']" < "$WORK/install.json")"
MPATH="$(jget "d['path']" < "$WORK/install.json")"
echo "installed $NAME at $MPATH"
[ -f "$MPATH" ] || fail "model file missing"
ok "model download + checksum"
# the registry must live in the platform's native data directory
case "$OS" in
windows) REG_CHECK="$(cygpath -u "$DATA_DIR")/models.json" ;;
*) REG_CHECK="$DATA_DIR/models.json" ;;
esac
[ -f "$REG_CHECK" ] || fail "registry not in the native data dir ($REG_CHECK)"
ok "registry in native data dir"

step "6. membrane plan (host RAM must be a measured fact)"
"$M" plan "$NAME" --json > "$WORK/plan.json"
KNOWN="$(jget "d['hardware']['host_available_known']" < "$WORK/plan.json")"
TOTAL="$(jget "d['hardware']['host_total_bytes']" < "$WORK/plan.json")"
echo "host_available_known=$KNOWN host_total_bytes=$TOTAL"
[ "$KNOWN" = True ] && [ "$TOTAL" -gt 0 ] || fail "plan has no measured host memory"
ok "plan with measured host memory"
"$M" plan "$NAME" | head -20

step "7. membrane serve + API"
"$M" model use "$NAME" >/dev/null
"$M" serve > "$WORK/serve.log" 2>&1 &
SERVER_PID=$!
for i in $(seq 1 60); do
	curl -fsS "http://127.0.0.1:$PORT/health" >/dev/null 2>&1 && break
	sleep 1
done
curl -fsS "http://127.0.0.1:$PORT/health" >/dev/null || { cat "$WORK/serve.log"; fail "server never became healthy"; }
ok "/health"
curl -fsS "http://127.0.0.1:$PORT/v1/models" | grep -q "$NAME" || fail "/v1/models does not list $NAME"
ok "/v1/models"
RESP="$(curl -fsS "http://127.0.0.1:$PORT/v1/chat/completions" -H 'Content-Type: application/json' \
	-d "{\"model\":\"$NAME\",\"messages\":[{\"role\":\"user\",\"content\":\"Say hi\"}],\"max_tokens\":8}")"
echo "$RESP" | python3 -c "import json,sys; d=json.load(sys.stdin); assert d['choices'][0]['message']['content'] is not None; print('completion ok:', repr(d['choices'][0]['message']['content'][:60]))" || fail "chat completion"
ok "/v1/chat/completions"
HDRS="$(curl -sS -D - -o /dev/null -H 'Origin: http://evil.example' "http://127.0.0.1:$PORT/v1/models")"
echo "$HDRS" | grep -qi 'access-control-allow-origin' && fail "server sent a CORS header"
ok "no CORS header"

if [ -n "${ACCEPT_OPENAI:-}" ]; then
	step "7b. official Python openai SDK (models.list, chat, streaming) against the packaged server"
	python3 -m venv "$WORK/venv"
	if [ -x "$WORK/venv/Scripts/python.exe" ]; then PY="$WORK/venv/Scripts/python.exe"; else PY="$WORK/venv/bin/python"; fi
	"$PY" -m pip install --quiet --disable-pip-version-check openai >/dev/null
	"$PY" "$(dirname "$0")/client-compat/test-openai-python.py" \
		--base-url "http://127.0.0.1:$PORT/v1" --model "$NAME" || fail "openai SDK harness"
	ok "openai SDK harness"
fi

step "8. observe / advise / apply (dry run) / adaptive / run --explain"
"$M" observe --json > "$WORK/observe.json"
RAM="$(jget "d['host_memory']['total_bytes']['known']" < "$WORK/observe.json")"
SRC="$(jget "d['host_memory']['total_bytes']['source']" < "$WORK/observe.json")"
echo "observe host RAM known=$RAM source=$SRC"
[ "$RAM" = True ] || fail "observe: host RAM unknown on $OS"
ok "observe (host RAM measured via $SRC)"
"$M" observe | head -15
"$M" advise | head -15
"$M" apply | head -10
"$M" adaptive status
"$M" run "$NAME" --explain | head -15
ok "observe/advise/apply/adaptive/run"

step "9. shutdown"
kill "$SERVER_PID" 2>/dev/null || true
[ "$OS" = windows ] && taskkill //F //IM membrane.exe >/dev/null 2>&1 || true
SERVER_PID=""
sleep 1
curl -fsS "http://127.0.0.1:$PORT/health" >/dev/null 2>&1 && fail "server still answering after shutdown"
ok "server stopped"

echo
echo "PACKAGE ACCEPTANCE PASSED on $OS ($("$M" --version))"
