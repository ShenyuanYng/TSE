#!/bin/bash
# Compatible with the Bash 3.2 shipped with macOS.
set -euo pipefail

die() { printf 'ERROR: %s\n' "$*" >&2; exit 1; }
usage() {
    cat <<'EOF'
Usage: ./sync_server.sh

Push the current main or audioset-fsd50k-mix commit, then pull that branch
on the server and verify both machines have exactly the same commit.
Configure .sync_server.env using .sync_server.env.example first.
Run tools/check_local.sh and git commit before invoking this script.
No training job is submitted by this command.
EOF
}
case "${1:-}" in
    -h|--help) usage; exit 0 ;;
    '') ;;
    *) usage >&2; exit 2 ;;
esac
[[ $# -eq 0 ]] || die 'No positional arguments are supported.'

project_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$project_root"
if [[ -f .sync_server.env ]]; then
    # This is a local, trusted shell configuration, never committed to Git.
    source .sync_server.env
fi
remote="${SYNC_REMOTE:-origin}"
host="${SERVER_HOST:-}"
branch="$(git symbolic-ref --quiet --short HEAD)" || die 'Detached HEAD; switch to a branch first.'
case "$branch" in
    main) server_repo="${SERVER_REPO_MAIN:-${SERVER_REPO:-}}" ;;
    audioset-fsd50k-mix) server_repo="${SERVER_REPO_MIX:-${SERVER_REPO:-}}" ;;
    *) die "Unsupported branch: $branch (expected main or audioset-fsd50k-mix)." ;;
esac
[[ -n "$host" && "$host" != -* && "$host" != *[[:space:]]* ]] || die 'Set SERVER_HOST to an SSH alias or user@host.'
[[ "$server_repo" == /* ]] || die 'Set SERVER_REPO (or the branch-specific path) to an absolute server path.'
[[ "$remote" != -* ]] || die 'SYNC_REMOTE must be a configured Git remote name.'
git remote get-url "$remote" >/dev/null
[[ -z "$(git status --porcelain --untracked-files=all)" ]] || die 'Local worktree is dirty. Check and commit changes first.'
expected="$(git rev-parse HEAD)"
printf 'Local branch: %s\nLocal commit: %s\nServer: %s:%s\n' "$branch" "$expected" "$host" "$server_repo"

# Push the captured SHA, so a concurrent local commit cannot change the target.
# Never force-push. A failed push prevents SSH from running.
git push "$remote" "$expected:refs/heads/$branch"

# SSH passes its command through the remote login shell. POSIX single-quoting
# preserves paths containing spaces, apostrophes, or shell metacharacters.
shell_quote() {
    local value=$1
    printf "'"
    while [[ "$value" == *"'"* ]]; do
        printf '%s' "${value%%"'"*}" "'\''"
        value=${value#*"'"}
    done
    printf "%s'" "$value"
}
remote_command="bash -s -- $(shell_quote "$server_repo") $(shell_quote "$branch") $(shell_quote "$expected") $(shell_quote "$remote")"
ssh -- "$host" "$remote_command" <<'REMOTE_SCRIPT'
set -euo pipefail
die() { printf 'SERVER ERROR: %s\n' "$*" >&2; exit 1; }
repo=$1
branch=$2
expected=$3
remote=$4
cd -- "$repo"
[[ "$(git rev-parse --show-toplevel)" == "$(pwd -P)" ]] || die 'SERVER_REPO must point at the repository root.'

# Serialize invocations of this script in this server checkout.
lock="$(git rev-parse --git-path sync_server.lock)"
mkdir "$lock" 2>/dev/null || die "Another sync is running (or stale lock: $lock)."
trap 'rmdir -- "$lock"' EXIT
# Keep server-only output files. Git checkout/pull will refuse untracked files
# that collide with incoming tracked paths; never force checkout or clean.
[[ -z "$(git status --porcelain --untracked-files=no)" ]] || die 'Server worktree is dirty; commit or move its changes first.'
for state in MERGE_HEAD CHERRY_PICK_HEAD REVERT_HEAD rebase-merge rebase-apply; do
    [[ ! -e "$(git rev-parse --git-path "$state")" ]] || die "Unfinished Git operation: $state"
done

git fetch "$remote" "refs/heads/$branch:refs/remotes/$remote/$branch"
fetched="$(git rev-parse "refs/remotes/$remote/$branch")"
[[ "$fetched" == "$expected" ]] || die "Remote branch changed after push: expected $expected, found $fetched. Retry from an up-to-date Mac checkout."
if git show-ref --verify --quiet "refs/heads/$branch"; then
    git merge-base --is-ancestor "refs/heads/$branch" "$expected" || die "Server branch $branch has extra or diverged commits; no reset was performed."
    git checkout --no-overwrite-ignore "$branch"
else
    git checkout --no-overwrite-ignore -b "$branch" --track "$remote/$branch"
fi
git -c pull.ff=only pull --ff-only --no-rebase "$remote" "refs/heads/$branch"
actual="$(git rev-parse HEAD)"
printf '\nServer branch: %s\nServer commit: %s\n' "$(git symbolic-ref --short HEAD)" "$actual"
git --no-pager log -1 --format='Commit: %H%nAuthor: %an%nDate: %aI%nSubject: %s'
[[ "$actual" == "$expected" ]] || die "Commit mismatch: Mac=$expected Server=$actual. Do not start training."
[[ -z "$(git status --porcelain --untracked-files=no)" ]] || die 'Server worktree changed during sync. Do not start training.'
printf '\nOK: Mac and server are at the same commit: %s\n' "$actual"
REMOTE_SCRIPT
printf '\nSync complete. You can now run the smoke test or submit training on the server.\n'
