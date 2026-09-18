import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from slurm.submit_continuation import submit_chain


class ContinuationSubmissionTest(unittest.TestCase):
    def test_sequential_dependencies_and_duplicate_guard(self):
        project = Path(__file__).resolve().parents[1]
        calls = []
        def output(command, **kwargs):
            calls.append(command)
            if command[0] == "scontrol":
                return (f"UserId=test({os.getuid()}) JobState=RUNNING NodeList=ailab03 "
                        f"ReqNodeList=ailab03 WorkDir={project}")
            return str(700 + sum(c[0] == "sbatch" for c in calls)) + "\n"
        with tempfile.TemporaryDirectory() as directory:
            run = Path(directory) / "experiment_job36595"
            run.mkdir()
            (run / "mylast.ckpt").touch()
            python_bin = Path(directory) / "python"
            python_bin.touch()
            with patch("slurm.submit_continuation.subprocess.check_output", side_effect=output):
                record = submit_chain(run, 36595, 3, "ailab03", str(python_bin))
                with self.assertRaises(FileExistsError):
                    submit_chain(run, 36595, 3, "ailab03", str(python_bin))
            commands = [c for c in calls if c[0] == "sbatch"]
            self.assertEqual(len(commands), 3)
            for command, predecessor in zip(commands, (36595, 701, 702)):
                self.assertIn(f"--dependency=afterany:{predecessor}", command)
                self.assertIn("--time=2-00:00:00", command)
                self.assertIn("--nodelist=ailab03", command)
                self.assertIn(str(run.resolve()), command)
            self.assertEqual(json.loads((run / "continuation_jobs.json").read_text()), record)

    def test_wrong_owner_is_rejected_before_submission(self):
        with tempfile.TemporaryDirectory() as directory:
            run = Path(directory) / "experiment_job12"
            run.mkdir()
            (run / "mylast.ckpt").touch()
            executable = Path(directory) / "python"
            executable.touch()
            with patch("slurm.submit_continuation.subprocess.check_output",
                       return_value=f"UserId=other({os.getuid() + 1})") as call:
                with self.assertRaisesRegex(ValueError, "current user"):
                    submit_chain(run, 12, 3, "ailab03", str(executable))
                self.assertEqual(call.call_count, 1)

    def test_continuations_use_configured_resources(self):
        project = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as directory:
            run = Path(directory) / "experiment_job12"
            run.mkdir()
            (run / "mylast.ckpt").touch()
            executable = Path(directory) / "python"
            executable.touch()
            state = (f"UserId=test({os.getuid()}) JobState=RUNNING NodeList=ailab01 "
                     f"ReqNodeList=ailab01 WorkDir={project}")
            with patch("slurm.submit_continuation.subprocess.check_output",
                       side_effect=[state, "13", "14"]) as call:
                record = submit_chain(run, 12, 2, "ailab01", str(executable),
                                      partition="custom", gres="gpu:l40:2", cpus_per_task=8,
                                      mem="48G", time_limit="1-00:00:00", account="lab")
            for invocation in call.call_args_list[1:]:
                command = invocation.args[0]
                for argument in ("--partition=custom", "--gres=gpu:l40:2", "--cpus-per-task=8",
                                 "--mem=48G", "--time=1-00:00:00", "--account=lab"):
                    self.assertIn(argument, command)
            self.assertEqual(record["resources"]["mem"], "48G")

    def test_starting_job_can_schedule_before_first_checkpoint(self):
        project = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as directory:
            run = Path(directory) / "experiment_job36595"
            run.mkdir()
            executable = Path(directory) / "python"
            executable.touch()
            state = (f"UserId=test({os.getuid()}) JobState=RUNNING NodeList=ailab03 "
                     f"ReqNodeList=ailab03 WorkDir={project}")
            with patch.dict(os.environ, {"SLURM_JOB_ID": "36595"}), patch(
                    "slurm.submit_continuation.subprocess.check_output",
                    side_effect=[state, "701", "702"]):
                record = submit_chain(run, 36595, 2, "ailab03", str(executable), starting_job=True)
            self.assertEqual(len(record["jobs"]), 2)
            self.assertFalse((run / "mylast.ckpt").exists())

    def test_starting_job_cannot_bypass_checkpoint_check_from_login_shell(self):
        with patch.dict(os.environ, {"SLURM_JOB_ID": "999"}):
            with self.assertRaisesRegex(ValueError, "original Slurm job"):
                submit_chain("/tmp/no-run", 36595, 2, "ailab03", "/tmp/python", starting_job=True)


class LauncherChainTest(unittest.TestCase):
    def test_launcher_schedules_total_minus_one_and_still_starts_training(self):
        import shutil
        import subprocess
        project = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "tools").mkdir()
            for name in ("path.sh", "spksep_run.sh", "tools/parse_options.sh"):
                shutil.copy(project / name, root / name)
            python = root / "fake_python"
            python.write_text('#!/bin/bash\nprintf "%s\\n" "$@" > "$CAPTURE_SUBMIT"\n')
            python.chmod(0o755)
            perl = root / "tools/perl"
            perl.write_text('#!/bin/bash\nprintf "%s\\n" "$@" > "$CAPTURE_TRAIN"\n')
            perl.chmod(0o755)
            env = dict(os.environ, SLURM_JOB_ID="36595", SLURMD_NODENAME="ailab03",
                       PYTHON_BIN=str(python), CAPTURE_SUBMIT=str(root / "submit.args"),
                       CAPTURE_TRAIN=str(root / "train.args"))
            subprocess.run(["bash", "spksep_run.sh", "--num-jobs", "3"],
                           cwd=root, env=env, check=True, capture_output=True, text=True)
            args = (root / "submit.args").read_text().splitlines()
            self.assertEqual(args[args.index("--additional-jobs") + 1], "2")
            self.assertEqual(args[args.index("--after-job") + 1], "36595")
            self.assertIn("--starting-job", args)
            run_dir = args[args.index("--run-dir") + 1]
            self.assertTrue(run_dir.endswith("_job36595"))
            train = (root / "train.args").read_text().splitlines()
            self.assertIn(run_dir + "/logs/training.log", train)
            self.assertIn("train.py", train)
            self.assertNotIn("--num-jobs", train)

    def test_run_forwards_num_jobs_to_launcher(self):
        import shutil
        import subprocess
        project = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "tools").mkdir()
            shutil.copy(project / "run.sh", root / "run.sh")
            shutil.copy(project / "tools/parse_options.sh", root / "tools/parse_options.sh")
            launcher = root / "spksep_run.sh"
            launcher.write_text('#!/bin/bash\nprintf "%s\\n" "$@"\n')
            launcher.chmod(0o755)
            result = subprocess.run(["bash", "run.sh", "--num-jobs", "4"],
                                    cwd=root, check=True, capture_output=True, text=True)
            args = result.stdout.splitlines()
            self.assertEqual(args[args.index("--num-jobs") + 1], "4")
