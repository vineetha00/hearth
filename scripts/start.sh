#!/usr/bin/env bash
# Start Hearth (MCP server + simulator) with one command, then open it in your browser.
#   ./scripts/start.sh         your real diary, saved in ~/.hearth/hearth.db
#   ./scripts/start.sh demo    the demo data in data/hearth.db (what you record with)
# Ctrl+C stops both.
set -e
cd "$(dirname "$0")/.."
[ -f .venv/bin/activate ] && . .venv/bin/activate
[ -f .env ] && set -a && . ./.env && set +a
if [ "$1" = "demo" ]; then
  export HEARTH_DB="data/hearth.db"
else
  mkdir -p "$HOME/.hearth"
  export HEARTH_DB="$HOME/.hearth/hearth.db"
fi
export HEARTH_LLM="${HEARTH_LLM:-heuristic}"
echo "Diary file: $HEARTH_DB   (engine: $HEARTH_LLM)"
python -m hearth.server & S=$!
trap 'kill $S $A 2>/dev/null' EXIT INT TERM
sleep 2
python -m simulator.app & A=$!
sleep 2
( command -v open >/dev/null && open http://localhost:8080 ) || ( command -v xdg-open >/dev/null && xdg-open http://localhost:8080 ) || echo "Open http://localhost:8080"
wait $A
