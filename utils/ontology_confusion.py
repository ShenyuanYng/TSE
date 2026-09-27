"""All-condition scoring on target GT-positive frames, without changing the model."""

import numpy as np
import torch

from nets.audioset_crnn import AudioSetCRNN
from utils.audioset_metrics import tensor


DEFAULT_TARGET_MIDS = ("/m/09x0r", "/m/0jbk", "/m/07yv9", "/m/05kq4",
                       "/m/02qldy", "/m/06d_3", "/m/03j1ly")

TOP_FIELDS = (
    "target_MID", "target_class_name", "target_positive_frames", "target_positive_clips",
    "target_gt_mean_probability", "target_gt_max_probability", "rank",
    "other_MID", "other_class_name", "other_gt_mean_probability", "other_gt_max_probability",
    "mean_probability_gap", "frames_other_gt_target", "frames_other_gt_target_ratio",
    "frames_top1_among_others", "frames_top1_among_others_ratio",
    "other_weak_present_clips", "other_weak_present_gt_frames", "other_weak_present_gt_frame_ratio",
    "other_strong_overlap_frames", "other_strong_overlap_ratio",
    "ontology_relation", "is_ancestor_or_descendant", "ontology_distance", "ontology_path",
    "shared_parent_mids",
)


@torch.inference_mode()
def score_all_conditions(model, features, class_chunk_size=32, frame_chunk_size=256):
    """Reuse post-GRU normalized audio features; only the conditional head repeats.

    Fusion/LayerNorm/head operate independently on each frame. No softmax is used:
    these are sigmoid probabilities for separate one-hot class conditions.
    """
    if model.training or not isinstance(model, AudioSetCRNN):
        raise ValueError("All-condition scoring requires an eval-mode AudioSetCRNN")
    if class_chunk_size < 1 or frame_chunk_size < 1:
        raise ValueError("Class/frame chunk sizes must be positive")
    scores = np.empty((features.shape[0], model.num_classes), dtype=np.float32)
    if not len(features):
        return scores
    weight = model.class_embedding.weight
    contexts = torch.eye(model.num_classes, device=weight.device, dtype=weight.dtype) @ weight
    for start in range(0, features.shape[0], frame_chunk_size):
        audio = features[start:start + frame_chunk_size]
        for category in range(0, model.num_classes, class_chunk_size):
            context = contexts[category:category + class_chunk_size]
            combined = torch.cat((audio[:, None, :].expand(-1, len(context), -1),
                                  context[None, :, :].expand(len(audio), -1, -1)), dim=-1)
            values = model.frame_head(model.fusion(combined)).squeeze(-1).sigmoid()
            scores[start:start + len(audio), category:category + len(context)] = values.cpu().numpy()
    return scores


def strong_overlap_counts(entry, times, mid_to_index):
    """Exact-MID PRESENT overlap on target positive times; NOT_PRESENT wins.

    Use the dataset's float64 sample-grid times and half-open interval rule.
    These are annotation evidence, not an alternate supervision mask.
    """
    counts = np.zeros(len(mid_to_index), dtype=np.int64)
    events_by_mid = {}
    for event in entry.get("events", []):
        if event["mid"] in mid_to_index:
            events_by_mid.setdefault(event["mid"], []).append(event)
    for mid, events in events_by_mid.items():
        present = np.zeros(len(times), dtype=bool)
        absent = np.zeros(len(times), dtype=bool)
        for event in events:
            active = (times >= event["onset"]) & (times < event["offset"])
            if event.get("present", True):
                present |= active
            else:
                absent |= active
        counts[mid_to_index[mid]] = np.count_nonzero(present & ~absent)
    return counts


class OntologyConfusionAccumulator:
    """Per-(clip,target MID) summaries deduplicated globally before ranking."""

    def __init__(self, mid_to_index, class_names, target_mids, ontology):
        self.mid_to_index, self.class_names = dict(mid_to_index), list(class_names)
        self.index_to_mid = {index: mid for mid, index in mid_to_index.items()}
        if set(self.index_to_mid) != set(range(len(class_names))):
            raise ValueError("Class map indices must be dense and unique")
        if not target_mids or len(set(target_mids)) != len(target_mids):
            raise ValueError("Target MIDs must be nonempty and unique")
        self.target_mids = tuple(target_mids)
        self.target_indices = {mid_to_index[mid] for mid in target_mids}
        self.ontology, self.rows = ontology, {}

    def update_pair(self, clip, category, valid_frames, target_probs, scores,
                    entry, positive_times):
        if category not in self.target_indices:
            raise ValueError("Unexpected target class")
        n, c = len(target_probs), len(self.class_names)
        if scores.shape != (n, c) or len(positive_times) != n or n > valid_frames:
            raise ValueError("Misaligned candidate probabilities or GT-positive frames")
        if not np.isfinite(scores).all() or not np.isfinite(target_probs).all():
            raise FloatingPointError("Nonfinite probabilities on target positive frames")
        if np.any((scores < 0) | (scores > 1)) or np.any((target_probs < 0) | (target_probs > 1)):
            raise ValueError("Probabilities must be in [0,1]")
        difference = float(np.max(np.abs(scores[:, category] - target_probs))) if n else 0.0
        if not np.allclose(scores[:, category], target_probs, atol=1e-4, rtol=1e-3):
            raise ValueError("Reused-feature target probabilities differ from the original forward")
        weak = np.zeros(c, dtype=np.int64)
        for mid in set(entry.get("weak_mids", [])):
            if mid in self.mid_to_index:
                weak[self.mid_to_index[mid]] = 1
        winners = np.zeros(c, dtype=np.int64)
        if n:
            # Keep the original conditional scores intact. Ties use class index.
            other_indices = np.array([i for i in range(c) if i != category])
            if len(other_indices):
                best = other_indices[np.argmax(scores[:, other_indices], axis=1)]
                winners = np.bincount(best, minlength=c)
        self.rows[(clip, category)] = {
            "valid_frames": int(valid_frames), "positive_frames": n,
            "target_sum": float(target_probs.sum(dtype=np.float64)),
            "target_max": float(target_probs.max()) if n else 0.0,
            "score_sum": scores.sum(axis=0, dtype=np.float64),
            "score_max": scores.max(axis=0) if n else np.zeros(c),
            "higher": (scores > target_probs[:, None]).sum(axis=0),
            "winners": winners, "weak_clips": weak * int(n > 0), "weak_frames": weak * n,
            "strong_overlap": strong_overlap_counts(entry, positive_times, self.mid_to_index),
            "head_max_abs_difference": difference,
        }

    def compute(self, expected_counts=None, top_k=10):
        if top_k < 1:
            raise ValueError("top_k must be positive")
        rows = self.rows
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            gathered = [None] * torch.distributed.get_world_size()
            torch.distributed.all_gather_object(gathered, rows)
            rows = {key: value for part in gathered for key, value in part.items()}
        top_rows, all_rows, summaries = [], [], []
        vector_keys = ("score_sum", "score_max", "higher", "winners", "weak_clips", "weak_frames", "strong_overlap")
        for mid in self.target_mids:
            category = self.mid_to_index[mid]
            pairs = [values for (_, index), values in rows.items() if index == category]
            n = sum(row["positive_frames"] for row in pairs)
            valid = sum(row["valid_frames"] for row in pairs)
            if expected_counts is not None:
                reference = expected_counts[mid]
                if (n, valid) != (reference["tp"] + reference["fn"], reference["valid_frames"]):
                    raise ValueError(f"Reference positive/valid frame counts changed for {mid}")
            if n == 0:
                raise ValueError(f"Target has no GT-positive valid frames: {mid}")
            target_mean = sum(row["target_sum"] for row in pairs) / n
            base = {
                "target_MID": mid, "target_class_name": self.class_names[category],
                "target_positive_frames": n,
                "target_positive_clips": sum(row["positive_frames"] > 0 for row in pairs),
                "target_gt_mean_probability": target_mean,
                "target_gt_max_probability": max(row["target_max"] for row in pairs),
            }
            vectors = {key: (np.max([row[key] for row in pairs], axis=0) if key == "score_max"
                             else np.sum([row[key] for row in pairs], axis=0)) for key in vector_keys}
            if vectors["strong_overlap"][category] != n:
                raise ValueError(f"Raw strong annotations and target GT frames disagree: {mid}")
            means = vectors["score_sum"] / n
            ranked = sorted((i for i in self.index_to_mid if i != category),
                            key=lambda i: (-means[i], self.index_to_mid[i]))
            candidates = []
            for rank, index in enumerate(ranked, 1):
                other = self.index_to_mid[index]
                candidates.append({
                    **base, "rank": rank, "other_MID": other,
                    "other_class_name": self.class_names[index],
                    "other_gt_mean_probability": float(means[index]),
                    "other_gt_max_probability": float(vectors["score_max"][index]),
                    "mean_probability_gap": float(means[index] - target_mean),
                    "frames_other_gt_target": int(vectors["higher"][index]),
                    "frames_other_gt_target_ratio": float(vectors["higher"][index] / n),
                    "frames_top1_among_others": int(vectors["winners"][index]),
                    "frames_top1_among_others_ratio": float(vectors["winners"][index] / n),
                    "other_weak_present_clips": int(vectors["weak_clips"][index]),
                    "other_weak_present_gt_frames": int(vectors["weak_frames"][index]),
                    "other_weak_present_gt_frame_ratio": float(vectors["weak_frames"][index] / n),
                    "other_strong_overlap_frames": int(vectors["strong_overlap"][index]),
                    "other_strong_overlap_ratio": float(vectors["strong_overlap"][index] / n),
                    **self.ontology.relation(mid, other),
                })
            top = candidates[:top_k]
            top_rows.extend(top)
            all_rows.extend(candidates)
            summaries.append({
                **base, "target_valid_frames": valid, "evaluated_other_classes": len(candidates),
                "top_k": len(top),
                "top_k_lineage_related_count": sum(r["is_ancestor_or_descendant"] is True for r in top),
                "top_k_unknown_ontology_count": sum(r["ontology_relation"] == "unknown_mid" for r in top),
                "other_classes_with_higher_mean": int(sum(means[i] > target_mean for i in ranked)),
                "head_max_abs_difference": max(row["head_max_abs_difference"] for row in pairs),
            })
        return top_rows, all_rows, summaries

    def reset(self):
        self.rows.clear()


@torch.inference_mode()
def probe_batches(module, batches, device, collector, entries_by_id, hop_length, sample_rate, epoch=0,
                  class_chunk_size=32, frame_chunk_size=256, log_every=10):
    from inspect_prediction import _move_to_device

    module.eval()
    if not isinstance(module.model, AudioSetCRNN):
        raise ValueError("Ontology probe currently supports AudioSetCRNN only")
    captured = []
    hook = module.model.temporal_norm.register_forward_hook(
        lambda _module, _inputs, output: captured.append(output.detach()))
    batch_count = 0
    try:
        for batch_count, batch in enumerate(batches, 1):
            captured.clear()
            model_input, targets = _move_to_device(batch, device)
            prediction = module.task.forward(module.model, model_input, targets, epoch)
            if len(captured) != 1:
                raise ValueError("Expected exactly one audio-feature tensor per forward")
            labels = tensor(targets["strong_labels"])
            valid = tensor(targets["strong_mask"]).bool() & prediction["frame_mask"].bool()
            for i, (clip, category) in enumerate(zip(targets["utt_id"], targets["class_index"])):
                category = int(category)
                if category not in collector.target_indices:
                    continue
                positive = valid[i] & (labels[i] >= .5)
                # Validation starts at sample zero. Recreate the original float64
                # grid; collate_seq casts frame_times to float32 and can shift an
                # annotation boundary by rounding if those values are reused.
                times = np.flatnonzero(positive.cpu().numpy()).astype(np.float64) * hop_length / sample_rate
                scores = score_all_conditions(module.model, captured[0][i][positive],
                                              class_chunk_size, frame_chunk_size)
                collector.update_pair(
                    clip, category, int(valid[i].sum()),
                    prediction["frame_probs"][i][positive].cpu().numpy(), scores,
                    entries_by_id[clip], times)
            if log_every and batch_count % log_every == 0:
                print(f"Validation batches processed: {batch_count}; "
                      f"selected clip-MID pairs: {len(collector.rows)}", flush=True)
    finally:
        hook.remove()
        captured.clear()
    return batch_count
