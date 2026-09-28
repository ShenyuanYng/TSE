"""Run real Git push/pull against temporary clones; replace only SSH transport."""
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest


PROJECT = Path(__file__).resolve().parents[1]


class SyncServerTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.origin = self.root / 'origin.git'
        self.mac = self.root / 'mac'
        # Exercise the SSH login shell's quoting, not just subprocess argv.
        self.server = self.root / "server with ' quote $(touch unexpected)"
        self.calls = self.root / 'ssh.calls'
        self.bin = self.root / 'bin'
        self.bin.mkdir()
        ssh = self.bin / 'ssh'
        ssh.write_text('#!/bin/bash\nset -eu\nprintf "called\\n" >> "$SSH_CALLS"\n'
                       '[[ "$1" == -- ]] && shift\nshift\nexec /bin/sh -c "$1"\n')
        ssh.chmod(0o755)
        self.env = dict(os.environ, PATH=f'{self.bin}:{os.environ["PATH"]}',
                        SERVER_HOST='test-server', SERVER_REPO=str(self.server),
                        SYNC_REMOTE='origin', SSH_CALLS=str(self.calls),
                        GIT_AUTHOR_NAME='Sync Test', GIT_AUTHOR_EMAIL='sync@example.invalid',
                        GIT_COMMITTER_NAME='Sync Test', GIT_COMMITTER_EMAIL='sync@example.invalid',
                        GIT_CONFIG_NOSYSTEM='1', GIT_CONFIG_GLOBAL='/dev/null')
        for name in ('SERVER_REPO_MAIN', 'SERVER_REPO_MIX'):
            self.env.pop(name, None)
        self.git(self.root, 'init', '--bare', '--initial-branch=main', str(self.origin))
        self.git(self.root, 'clone', str(self.origin), str(self.mac))
        shutil.copy(PROJECT / 'sync_server.sh', self.mac / 'sync_server.sh')
        (self.mac / 'code.txt').write_text('initial\n')
        self.git(self.mac, 'add', '.')
        self.git(self.mac, 'commit', '-m', 'initial')
        self.git(self.mac, 'push', 'origin', 'main')
        self.git(self.root, 'clone', str(self.origin), str(self.server))

    def git(self, cwd, *args):
        return subprocess.check_output(['git', *args], cwd=cwd, env=self.env,
                                       stderr=subprocess.STDOUT, text=True).strip()

    def commit(self, repo, message):
        (repo / 'code.txt').write_text(message + '\n')
        self.git(repo, 'add', 'code.txt')
        self.git(repo, 'commit', '-m', message)
        return self.git(repo, 'rev-parse', 'HEAD')

    def sync(self, success=True, **overrides):
        result = subprocess.run(['bash', str(self.mac / 'sync_server.sh')],
                                cwd=self.root, env=dict(self.env, **overrides),
                                capture_output=True, text=True)
        if success:
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        else:
            self.assertNotEqual(result.returncode, 0, result.stdout + result.stderr)
        return result.stdout + result.stderr

    def test_main_then_new_mix_branch_then_main_again(self):
        main = self.commit(self.mac, 'main update')
        output = self.sync()
        self.assertIn(f'Server commit: {main}', output)
        self.assertEqual(self.git(self.server, 'rev-parse', 'HEAD'), main)
        self.git(self.mac, 'checkout', '-b', 'audioset-fsd50k-mix')
        mix = self.commit(self.mac, 'mix update')
        self.assertIn(f'Server commit: {mix}', self.sync())
        self.assertEqual(self.git(self.server, 'branch', '--show-current'), 'audioset-fsd50k-mix')
        self.assertEqual(self.git(self.origin, 'rev-parse', 'audioset-fsd50k-mix'), mix)
        self.git(self.mac, 'checkout', 'main')
        self.sync()
        self.assertEqual(self.git(self.server, 'rev-parse', 'HEAD'), main)
        self.assertFalse((self.root / 'unexpected').exists())

    def test_existing_mix_branch_fast_forwards(self):
        self.git(self.mac, 'checkout', '-b', 'audioset-fsd50k-mix')
        self.sync()
        expected = self.commit(self.mac, 'second mix update')
        self.sync()
        self.assertEqual(self.git(self.server, 'rev-parse', 'HEAD'), expected)

    def test_local_untracked_file_stops_before_push(self):
        previous = self.git(self.origin, 'rev-parse', 'main')
        self.commit(self.mac, 'unpublished')
        (self.mac / 'forgotten.py').write_text('x = 1\n')
        self.assertIn('Local worktree is dirty', self.sync(success=False))
        self.assertEqual(self.git(self.origin, 'rev-parse', 'main'), previous)
        self.assertFalse(self.calls.exists())

    def test_server_dirty_file_is_preserved(self):
        previous = self.git(self.server, 'rev-parse', 'HEAD')
        self.commit(self.mac, 'update')
        (self.server / 'code.txt').write_text('server edit\n')
        self.assertIn('Server worktree is dirty', self.sync(success=False))
        self.assertEqual((self.server / 'code.txt').read_text(), 'server edit\n')
        self.assertEqual(self.git(self.server, 'rev-parse', 'HEAD'), previous)
        self.assertFalse((self.server / '.git/sync_server.lock').exists())

    def test_server_extra_commit_is_preserved(self):
        server_head = self.commit(self.server, 'server-only change')
        self.commit(self.mac, 'mac change')
        self.assertIn('extra or diverged commits', self.sync(success=False))
        self.assertEqual(self.git(self.server, 'rev-parse', 'HEAD'), server_head)

    def test_server_untracked_result_is_preserved(self):
        result = self.server / 'probability_reference_37074.json'
        result.write_text('{"result": 1}\n')
        expected = self.commit(self.mac, 'update with server output present')
        self.sync()
        self.assertEqual(result.read_text(), '{"result": 1}\n')
        self.assertEqual(self.git(self.server, 'rev-parse', 'HEAD'), expected)

    def test_server_untracked_collision_stops_pull(self):
        previous = self.git(self.server, 'rev-parse', 'HEAD')
        (self.server / 'new.py').write_text('server contents\n')
        (self.mac / 'new.py').write_text('incoming contents\n')
        self.git(self.mac, 'add', 'new.py')
        self.git(self.mac, 'commit', '-m', 'new tracked file')
        self.sync(success=False)
        self.assertEqual((self.server / 'new.py').read_text(), 'server contents\n')
        self.assertEqual(self.git(self.server, 'rev-parse', 'HEAD'), previous)

    def test_rejected_push_never_runs_ssh(self):
        self.commit(self.server, 'other author')
        self.git(self.server, 'push', 'origin', 'main')
        self.commit(self.mac, 'conflicting local history')
        self.sync(success=False)
        self.assertFalse(self.calls.exists())

    def test_remote_change_between_push_and_fetch_is_detected(self):
        self.commit(self.mac, 'update')
        ssh = self.bin / 'ssh'
        original = ssh.read_text()
        ssh.write_text(original.replace('[[ "$1" == -- ]]',
                       'git --git-dir="$RACE_ORIGIN" update-ref refs/heads/main "$RACE_SHA"\n'
                       '[[ "$1" == -- ]]'))
        old = self.git(self.server, 'rev-parse', 'HEAD')
        output = self.sync(success=False, RACE_ORIGIN=str(self.origin), RACE_SHA=old)
        self.assertIn('Remote branch changed after push', output)
        self.assertEqual(self.git(self.server, 'rev-parse', 'HEAD'), old)

    def test_branch_specific_directory(self):
        self.assertIn('OK: Mac and server', self.sync(
            SERVER_REPO='/nonexistent/fallback', SERVER_REPO_MAIN=str(self.server)))

    def test_detached_and_unsupported_branch_are_rejected(self):
        self.git(self.mac, 'checkout', '--detach')
        self.assertIn('Detached HEAD', self.sync(success=False))
        self.git(self.mac, 'checkout', '-b', 'unsupported')
        self.assertIn('Unsupported branch', self.sync(success=False))
        self.assertFalse(self.calls.exists())


if __name__ == '__main__':
    unittest.main()
