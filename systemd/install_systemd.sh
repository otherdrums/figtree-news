#!/usr/bin/env bash
# Install figtree-news as a single systemd service. One process owns the GPU
# model and runs crawl + pipeline + web (uvicorn). No containers — plain systemd.
#
# Usage:
#   ./systemd/install_systemd.sh            # user service (~/.config/systemd/user)
#   ./systemd/install_systemd.sh --system   # system service (/etc/systemd/system, root)
#
# After install:
#   systemctl --user enable --now figtree-news
#   (for user services, also: loginctl enable-linger "$USER"  so it runs without a session)
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

SYSTEM=0
for arg in "$@"; do
  case "$arg" in
    --system) SYSTEM=1 ;;
  esac
done

# --- resolve the python that has figtree_news importable -------------------
if [[ -n "${VIRTUAL_ENV:-}" && -x "$VIRTUAL_ENV/bin/python" ]]; then
  PYTHON="$VIRTUAL_ENV/bin/python"
else
  CANDIDATES=(
    "$REPO_DIR/../figtree/.venv_f39/bin/python"
    "$REPO_DIR/.venv/bin/python"
    "$(command -v python3)"
  )
  PYTHON=""
  for c in "${CANDIDATES[@]}"; do
    if [[ -x "$c" ]] && "$c" -c "import figtree_news" >/dev/null 2>&1; then
      PYTHON="$c"
      break
    fi
  done
fi

if [[ -z "${PYTHON:-}" ]]; then
  echo "Could not find a Python interpreter with figtree_news installed." >&2
  echo "Activate the venv (or pip install -e .) and re-run." >&2
  exit 1
fi

# --- target location -------------------------------------------------------
if [[ "$SYSTEM" -eq 1 ]]; then
  UNIT_DIR="/etc/systemd/system"
  CTRL="systemctl"
  SUDO=""
  if [[ "$(id -u)" -ne 0 ]]; then SUDO="sudo"; fi
else
  UNIT_DIR="$HOME/.config/systemd/user"
  CTRL="systemctl --user"
  SUDO=""
fi

mkdir -p "$UNIT_DIR"

echo "Repo:    $REPO_DIR"
echo "Python:  $PYTHON"
echo "Units:   $UNIT_DIR"

src="$SCRIPT_DIR/figtree-news.service"
dst="$UNIT_DIR/figtree-news.service"
sed -e "s|__DIR__|$REPO_DIR|g" -e "s|__PYTHON__|$PYTHON|g" "$src" > "$dst"
echo "wrote $dst"

# --- enable + start --------------------------------------------------------
if [[ "$SYSTEM" -eq 0 ]]; then
  loginctl enable-linger "$USER" 2>/dev/null || true
fi
$SUDO $CTRL daemon-reload
$SUDO $CTRL enable figtree-news

echo
echo "Enabled. Start now with:"
echo "  $CTRL start figtree-news"
echo "Status:"
echo "  $CTRL status figtree-news"
echo
echo "Note: ensure $REPO_DIR/sources.json exists (see demo/sources.json)"
echo "before starting."
