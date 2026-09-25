#!/usr/bin/env bash
# 在本机选择操作，通过 SSH 在开发板上执行。
# 用法: ./scripts/board.sh
# 环境变量: BOARD_HOST（默认 nano0）、BOARD_DISPLAY（默认 :1）

set -euo pipefail

HOST="${BOARD_HOST:-nano0}"
DISPLAY_VALUE="${BOARD_DISPLAY:-:1}"
CTRL_SOCK="${TMPDIR:-/tmp}/board-ssh-${HOST}-$$"

ssh_board() {
  ssh \
    -o ControlMaster=auto \
    -o "ControlPath=${CTRL_SOCK}" \
    -o ControlPersist=10m \
    -o ConnectTimeout=8 \
    "$HOST" "$@"
}

cleanup() {
  ssh -o "ControlPath=${CTRL_SOCK}" -O exit "$HOST" >/dev/null 2>&1 || true
  rm -f "$CTRL_SOCK"
}
trap cleanup EXIT

die() {
  printf '错误: %s\n' "$1" >&2
  return 1
}

list_windows() {
  ssh_board "DISPLAY=${DISPLAY_VALUE} python3 -" <<'PY'
import gi
gi.require_version("Wnck", "3.0")
from gi.repository import Wnck

screen = Wnck.Screen.get_default()
screen.force_update()
for window in screen.get_windows():
    if window.is_skip_tasklist():
        continue
    name = window.get_name().replace("\t", " ").replace("\n", " ")
    print(f"{window.get_xid()}\t{window.get_pid()}\t{name}")
PY
}

close_windows() {
  local mode="$1"
  shift
  ssh_board "DISPLAY=${DISPLAY_VALUE} python3 - $*" <<'PY'
import sys
import gi
gi.require_version("Wnck", "3.0")
from gi.repository import Wnck

wanted = {int(item) for item in sys.argv[1:]}
screen = Wnck.Screen.get_default()
screen.force_update()
for window in screen.get_windows():
    if window.get_xid() in wanted:
        print(f"关闭: {window.get_name()}")
        window.close(0)
PY
}

choose_windows() {
  local lines=() line xid name count=0 choice target
  local -a xids=() names=()

  if ! mapfile -t lines < <(list_windows); then
    die "无法读取显示器 ${DISPLAY_VALUE} 上的窗口"
    return
  fi
  if ((${#lines[@]} == 0)); then
    echo "显示器 ${DISPLAY_VALUE} 上没有可关闭的窗口。"
    return
  fi

  echo
  echo "显示器 ${DISPLAY_VALUE} 上的窗口:"
  for line in "${lines[@]}"; do
    [[ -z "$line" ]] && continue
    count=$((count + 1))
    xid="${line%%$'\t'*}"
    name="${line#*$'\t'}"
    name="${name#*$'\t'}"
    xids+=("$xid")
    names+=("$name")
    printf '  %2d) %s\n' "$count" "$name"
  done
  echo "   a) 全部关闭"
  echo "   q) 返回"
  printf '选择: '
  read -r choice
  case "$choice" in
    q | Q | "") return ;;
    a | A) target="${xids[*]}" ;;
    *)
      if [[ ! "$choice" =~ ^[0-9]+$ ]] || ((choice < 1 || choice > count)); then
        echo "无效选择。"
        return
      fi
      target="${xids[$((choice - 1))]}"
      echo "将关闭: ${names[$((choice - 1))]}"
      ;;
  esac
  printf '确认关闭? [y/N] '
  read -r choice
  [[ "$choice" == "y" || "$choice" == "Y" ]] || return
  close_windows one $target
}

list_processes() {
  local keyword="$1"
  ssh_board "ps -eo pid,user,etime,cmd --sort=pid" | awk -v key="$keyword" '
    NR == 1 { next }
    $4 ~ /^\[/ { next }
    {
      pid = $1
      user = $2
      etime = $3
      cmd = substr($0, index($0, $4))
      if (key == "" && user != ENVIRON["REMOTE_USER"]) next
      if (key != "" && index(tolower(cmd), tolower(key)) == 0 && index(user, key) == 0) next
      printf "%s\t%s\t%s\t%s\n", pid, user, etime, cmd
    }
  '
}

choose_process() {
  local keyword lines=() line pid user etime cmd count=0 choice
  local -a pids=() labels=()
  local remote_user

  remote_user="$(ssh_board whoami)"
  echo
  echo "输入关键字筛选进程。直接回车则只列出用户 ${remote_user} 的进程。"
  printf '关键字: '
  read -r keyword

  if ! mapfile -t lines < <(REMOTE_USER="$remote_user" list_processes "$keyword"); then
    die "无法读取进程列表"
    return
  fi
  if ((${#lines[@]} == 0)); then
    echo "没有匹配的进程。"
    return
  fi

  echo
  printf '  %4s  %-12s %-10s %s\n' "序号" "用户" "运行时间" "命令"
  for line in "${lines[@]}"; do
    [[ -z "$line" ]] && continue
    count=$((count + 1))
    pid="${line%%$'\t'*}"
    rest="${line#*$'\t'}"
    user="${rest%%$'\t'*}"
    rest="${rest#*$'\t'}"
    etime="${rest%%$'\t'*}"
    cmd="${rest#*$'\t'}"
    pids+=("$pid")
    labels+=("$user $etime $cmd")
    printf '  %4d  %-12s %-10s %s\n' "$count" "$user" "$etime" "$cmd"
  done
  echo "   q) 返回"
  printf '选择要关闭的序号: '
  read -r choice
  [[ "$choice" == "q" || "$choice" == "Q" || -z "$choice" ]] && return
  if [[ ! "$choice" =~ ^[0-9]+$ ]] || ((choice < 1 || choice > count)); then
    echo "无效选择。"
    return
  fi

  pid="${pids[$((choice - 1))]}"
  echo "将结束 PID ${pid}: ${labels[$((choice - 1))]}"
  printf '确认? [y/N] '
  read -r choice
  [[ "$choice" == "y" || "$choice" == "Y" ]] || return

  ssh_board "kill -TERM -- ${pid}" || {
    echo "无法结束该进程（可能没有权限）。"
    return
  }
  sleep 1
  if ssh_board "kill -0 -- ${pid} >/dev/null 2>&1"; then
    printf '进程仍在运行。强制结束? [y/N] '
    read -r choice
    if [[ "$choice" == "y" || "$choice" == "Y" ]]; then
      ssh_board "kill -KILL -- ${pid}" || echo "强制结束失败。"
    fi
  else
    echo "已结束 PID ${pid}。"
  fi
}

echo "连接到 ${HOST}（第一次会要求输入 SSH 密码）..."
ssh_board true

while true; do
  echo
  echo "开发板 ${HOST}  显示器 ${DISPLAY_VALUE}"
  echo "  1) 清除显示器上的窗口"
  echo "  2) 检索进程并关闭程序"
  echo "  q) 退出"
  printf '选择: '
  read -r action
  case "$action" in
    1) choose_windows ;;
    2) choose_process ;;
    q | Q) exit 0 ;;
    *) echo "无效选择。" ;;
  esac
done
