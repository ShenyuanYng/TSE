"""Exercise actual training-hook control flow without a PyTorch dependency.

These tests extract methods from the source file and provide minimal tensor and
framework doubles. They test error propagation and routing, not CUDA numerics.
Set AUDIT_SOURCE_FILE to check a patched server copy with the same tests.
"""
import ast
import math
import os
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import Mock


class Scalar:
    def __init__(self, value):
        self.value = value

    def detach(self):
        return self

    def item(self):
        return self.value


def load_method(name, **globals_):
    path = Path(os.environ.get(
        "AUDIT_SOURCE_FILE", Path(__file__).resolve().parents[1] / "source_separator.py"))
    tree = ast.parse(path.read_text())
    cls = next(node for node in tree.body
               if isinstance(node, ast.ClassDef) and node.name == "SourceSeparator")
    method = next(node for node in cls.body
                  if isinstance(node, ast.FunctionDef) and node.name == name)
    exec(compile(ast.Module(body=[method], type_ignores=[]), str(path), "exec"), globals_)
    return globals_[name]


class TrainingFailurePolicyTest(unittest.TestCase):
    def gradient_hook(self, norm_check):
        torch = SimpleNamespace(nn=SimpleNamespace(
            utils=SimpleNamespace(get_total_norm=norm_check)))
        logger = Mock()
        grads = [object(), object()]
        model = SimpleNamespace(parameters=lambda: [
            SimpleNamespace(grad=grads[0]), SimpleNamespace(grad=None),
            SimpleNamespace(grad=grads[1])])
        instance = SimpleNamespace(model=model, current_epoch=4, global_step=123,
                                   _train_batch_info={"batch_idx": 9}, zero_grad=Mock(),
                                   hparams=SimpleNamespace(gradient_clip_val=5.0))
        return load_method("on_after_backward", torch=torch, logger=logger), instance, logger, grads

    def test_gradient_runtime_error_is_logged_and_gradients_are_cleared(self):
        for message in ("nonfinite norm", "unexpected backend failure"):
            with self.subTest(message=message):
                error = RuntimeError(message)
                hook, instance, logger, _ = self.gradient_hook(Mock(side_effect=error))
                hook(instance)
                logger.info.assert_called_once_with(
                    'detected inf or nan values in gradients. not updating model parameters')
                logger.exception.assert_not_called()
                instance.zero_grad.assert_called_once_with()

    def test_finite_gradient_check_leaves_gradients_alone(self):
        norm_check = Mock(return_value=1.0)
        hook, instance, logger, grads = self.gradient_hook(norm_check)
        hook(instance)
        norm_check.assert_called_once_with(
            grads, norm_type=2.0, error_if_nonfinite=True, foreach=None)
        logger.info.assert_not_called()
        logger.exception.assert_not_called()
        instance.zero_grad.assert_not_called()

    def test_non_runtime_error_is_not_swallowed(self):
        error = ValueError("invalid norm argument")
        hook, instance, logger, _ = self.gradient_hook(Mock(side_effect=error))
        with self.assertRaises(ValueError) as caught:
            hook(instance)
        self.assertIs(caught.exception, error)
        logger.info.assert_not_called()
        instance.zero_grad.assert_not_called()

    def training_step(self, value):
        loss = Scalar(value)
        torch = SimpleNamespace(isfinite=lambda x: SimpleNamespace(
            all=lambda: math.isfinite(x.value)))
        instance = SimpleNamespace(
            current_epoch=4, global_step=123, trainer=SimpleNamespace(current_epoch=4),
            _step=Mock(return_value={"loss": loss}), training_step_output=[])
        method = load_method("training_step", torch=torch, logger=Mock())
        batch = ({}, {"utt_id": ["clip_a"], "class_index": [7]})
        return method, instance, batch, loss

    def test_nan_and_infinite_loss_stop_with_batch_context(self):
        for value in (float("nan"), float("inf"), -float("inf")):
            with self.subTest(value=value):
                method, instance, batch, _ = self.training_step(value)
                with self.assertRaises(FloatingPointError) as caught:
                    method(instance, batch, 9)
                message = str(caught.exception)
                for detail in ("epoch=4", "global_step=123", "clip_a", "batch_idx"):
                    self.assertIn(detail, message)
                self.assertEqual(instance.training_step_output, [])

    def test_finite_loss_is_returned_unchanged(self):
        method, instance, batch, loss = self.training_step(0.5)
        self.assertIs(method(instance, batch, 9)["loss"], loss)
        self.assertEqual(len(instance.training_step_output), 1)

    def test_forward_calls_the_task_forward_method(self):
        result = object()
        forward = Mock(return_value=result)
        instance = SimpleNamespace(task=SimpleNamespace(forward=forward),
                                   model=object(), current_epoch=4)
        inputs, targets = {"example": 1}, {"example": 2}
        method = load_method("forward")
        self.assertIs(method(instance, inputs, targets), result)
        forward.assert_called_once_with(instance.model, inputs, targets, current_epoch=4)


if __name__ == "__main__":
    unittest.main()
