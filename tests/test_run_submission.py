"""Exercise the real shell entry point with a fake scheduler, without submitting jobs."""
import os
import re
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest


PROJECT = Path(__file__).resolve().parents[1]


class RunSubmissionTest(unittest.TestCase):
    def prepare(self, directory):
        root = Path(directory) / "project with spaces"
        (root / "tools").mkdir(parents=True)
        shutil.copy(PROJECT / "run.sh", root / "run.sh")
        shutil.copy(PROJECT / "tools/parse_options.sh", root / "tools/parse_options.sh")
        for name, capture in (("sbatch", "SUBMIT"), ("spksep_run.sh", "TRAIN")):
            script = root / name
            script.write_text('#!/bin/bash\nprintf "%s\\n" "$@" > "$CAPTURE_' + capture + '"\n'
                              'printf "%s\\n" "$PYTHON_BIN" > "$CAPTURE_PYTHON"\n')
            script.chmod(0o755)
        env = dict(os.environ, PATH=f"{root}:{os.environ['PATH']}", HOSTNAME="login-host",
                   CAPTURE_SUBMIT=str(root / "submit.args"), CAPTURE_TRAIN=str(root / "train.args"),
                   CAPTURE_PYTHON=str(root / "python.env"))
        for name in ("SLURM_JOB_ID", "SLURMD_NODENAME", "PYTHON_BIN"):
            env.pop(name, None)
        return root, env

    def test_editing_run_settings_controls_submission_and_queued_arguments(self):
        with tempfile.TemporaryDirectory() as directory:
            root, env = self.prepare(directory)
            script = root / "run.sh"
            settings = (script.read_text().replace('slurm_node="ailab01"', 'slurm_node="ailab03"')
                              .replace('slurm_mem="32G"', 'slurm_mem="48G"')
                              .replace('slurm_time="2-00:00:00"', 'slurm_time="1-00:00:00"')
                              .replace('slurm_cpus_per_task=6', 'slurm_cpus_per_task=8'))
            settings = re.sub(r"(?m)^batch_size=\d+$", "batch_size=3", settings)
            settings = re.sub(r"(?m)^num_jobs=\d+$", "num_jobs=2", settings)
            script.write_text(settings)
            subprocess.run(["bash", str(script), "--submit", "true", "--tag", "_test with spaces",
                            "--audioset-embedding-dim", "64", "--frame-accuracy-threshold", "0.7",
                            "--slurm-gres", "gpu:rtxpro5000:1"],
                           cwd=directory, env=env, check=True, capture_output=True, text=True)
            args = (root / "submit.args").read_text().splitlines()
            for value in ("--nodelist=ailab03", "--partition=RTXPRO5000", "--gres=gpu:rtxpro5000:1",
                          "--cpus-per-task=8", "--mem=48G", "--time=1-00:00:00", f"--chdir={root}"):
                self.assertIn(value, args)
            launcher_index = args.index(str(root / "spksep_run.sh"))
            queued_args = args[launcher_index + 1:]
            self.assertNotIn("--audioset-dropout", queued_args)
            expected = {"--batch-size": "3", "--num-jobs": "2", "--hoptime": "0.016",
                        "--audioset-embedding-dim": "64", "--frame-accuracy-threshold": "0.7",
                        "--slurm-mem": "48G", "--slurm-cpus-per-task": "8",
                        "--slurm-time": "1-00:00:00", "--tag": "_test with spaces",
                        "--data-path": f"/data/{env['USER']}/AudioSet",
                        "--train-val-fraction": "0.15", "--train-split-seed": "7",
                        "--train-split-path": "metadata/train_val_split.json",
                        "--test-audio-dir": '["audio/eval_wav"]',
                        "--test-strong-manifest": "metadata/audioset_eval_strong.tsv",
                        "--train-audio-dir": '["audio/balance_wav", "audio/unbalanced_wav"]'}
            for key, value in expected.items():
                self.assertEqual(queued_args[queued_args.index(key) + 1], value)
            self.assertTrue((root / "python.env").read_text().strip().endswith('/envs/audioset/bin/python'))
            self.assertFalse((root / "train.args").exists())
            # Later edits must not be re-read when Slurm starts the saved command.
            script.write_text("exit 99\n")
            env["PYTHON_BIN"] = (root / "python.env").read_text().strip()
            subprocess.run(["bash", *args[launcher_index:]], cwd=root, env=env, check=True)
            self.assertEqual((root / "train.args").read_text().splitlines(), queued_args)
            self.assertFalse(list(root.rglob("*.sbatch")))

    def test_local_explicit_paths_and_python_are_preserved(self):
        with tempfile.TemporaryDirectory() as directory:
            root, env = self.prepare(directory)
            subprocess.run(["bash", str(root / "run.sh"), "--data-path", "/my audio",
                            "--train-audio-dir", '["a b", "c"]', "--python-bin", "/custom/python",
                            "--num-jobs", "1"],
                           env=env, check=True, capture_output=True)
            args = (root / "train.args").read_text().splitlines()
            self.assertEqual(args[args.index("--data-path") + 1], "/my audio")
            self.assertEqual(args[args.index("--train-audio-dir") + 1], '["a b", "c"]')
            self.assertEqual(args[args.index("--test-audio-dir") + 1], '["audio/eval"]')
            self.assertEqual((root / "python.env").read_text().strip(), "/custom/python")
            self.assertFalse((root / "submit.args").exists())

    def test_invalid_job_count_and_recursive_submission_fail_before_sbatch(self):
        with tempfile.TemporaryDirectory() as directory:
            root, env = self.prepare(directory)
            for extra, variables in ((["--num-jobs", "0"], {}), ([], {"SLURM_JOB_ID": "123"})):
                result = subprocess.run(["bash", str(root / "run.sh"), "--submit", "true", *extra],
                                        env=dict(env, **variables), capture_output=True)
                self.assertNotEqual(result.returncode, 0)
                self.assertFalse((root / "submit.args").exists())


if __name__ == "__main__":
    unittest.main()
