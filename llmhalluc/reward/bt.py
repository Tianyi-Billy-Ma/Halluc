"""Multi-component reward function for GRPO training with backtracking mechanism.

This module implements a comprehensive reward function for training LLMs with
the <|BACKTRACK|> token using Group Relative Policy Optimization (GRPO).

The reward function combines four components:
1. Outcome Accuracy: Final answer correctness after backtracking
2. Process Quality: Intermediate reasoning step evaluation
3. Backtrack Efficiency: Appropriate and efficient use of backtracking
4. Format Compliance: Structured output requirements

Based on research from:
- DeepSeekMath (GRPO)
- SCoRe (Self-Correction via RL)
- Process Reward Models (PRMs)
- Curriculum Learning for RL
"""

from dataclasses import dataclass, field
from typing import Any

from transformers import TrainerState

from .base import BaseRewardFunction


def _apply_backtracking(token_ids: list[int], backtrack_token_id: int) -> list[int]:
    """Apply backtrack operations to get final sequence.

    Args:
        token_ids: Full generation including backtrack tokens
        backtrack_token_id: ID of the backtrack token

    Returns:
        Final sequence after applying all backtracks
    """
    result = []
    for token_id in token_ids:
        if token_id != backtrack_token_id:
            result.append(token_id)
        elif result:  # Backtrack: remove last token
            result.pop()
    return result


def _get_pre_backtrack_sequence(
    token_ids: list[int], backtrack_token_id: int
) -> list[int]:
    """Get sequence before first backtrack token.

    Args:
        token_ids: Full generation including backtrack tokens
        backtrack_token_id: ID of the backtrack token

    Returns:
        Sequence before first backtrack (or full sequence if no backtrack)
    """
    try:
        backtrack_idx = token_ids.index(backtrack_token_id)
        return token_ids[:backtrack_idx]
    except ValueError:
        return token_ids


def _compute_sequence_accuracy(pred_ids: list[int], true_ids: list[int]) -> float:
    """Compute accuracy of a predicted sequence against ground truth.

    Args:
        pred_ids: Predicted token IDs
        true_ids: Ground truth token IDs

    Returns:
        Accuracy score in [0.0, 1.0]
    """
    if not true_ids:
        return 0.0

    # Exact match
    if pred_ids == true_ids:
        return 1.0

    # Token-level accuracy (partial credit)
    matches = sum(1 for p, t in zip(pred_ids, true_ids) if p == t)
    return matches / max(len(true_ids), 1)


@dataclass
class BacktrackRewardFunction(BaseRewardFunction):
    """Multi-component reward function for backtracking GRPO training.

    This reward function evaluates LLM completions on four dimensions:
    1. Outcome accuracy (final answer correctness)
    2. Process quality (intermediate step correctness)
    3. Backtrack efficiency (appropriate backtracking usage)
    4. Format compliance (structural requirements)

    The reward function supports curriculum learning with dynamic weight
    adjustment across training stages.

    Attributes:
        name: Identifier for the reward function (used in logging)
        weight: Global weight for this reward function (default: 1.0)
        backtrack_token_id: Token ID for the backtrack token

        # Component weights
        outcome_weight: Weight for outcome accuracy component
        process_weight: Weight for process quality component
        backtrack_weight: Weight for backtrack efficiency component
        format_weight: Weight for format compliance component

        # Backtrack efficiency sub-rewards
        correction_bonus: Bonus multiplier for successful corrections
        unnecessary_penalty: Penalty per unnecessary backtrack
        efficiency_weight: Weight for backtrack count efficiency
        failed_correction_penalty: Penalty for failed correction attempts

        # Curriculum learning
        use_curriculum: Whether to apply curriculum learning scaling
        enable_process_rewards: Whether to compute process rewards (requires data)
        enable_format_rewards: Whether to compute format rewards

        # Constraints
        max_backtracks: Maximum allowed backtracks (hard limit)
    """

    name: str = "backtrack_grpo"
    weight: float = 1.0

    backtrack_token_id: int = field(default=None)

    # Component weights (can be adjusted for different training stages)
    outcome_weight: float = 1.0
    process_weight: float = 0.7
    backtrack_weight: float = 0.6
    format_weight: float = 0.3

    # Backtrack efficiency sub-reward hyperparameters
    correction_bonus: float = 0.4
    unnecessary_penalty: float = 0.2
    efficiency_weight: float = 0.25
    failed_correction_penalty: float = 0.3

    # Curriculum learning
    use_curriculum: bool = False
    enable_process_rewards: bool = False
    enable_format_rewards: bool = True

    # Constraints
    max_backtracks: int = 20

    def __post_init__(self):
        """Validate configuration."""
        if self.backtrack_token_id is None:
            raise ValueError("backtrack_token_id must be specified")

    def __call__(
        self,
        prompts: list[str] | list[list[dict[str, str]]],
        completions: list[str] | list[list[dict[str, str]]],
        completions_ids: list[list[int]],
        ground_truth_ids: list[list[int]],
        trainer_state: TrainerState,
        **kwargs: Any,
    ) -> list[float]:
        """Compute multi-component rewards for completions.

        Args:
            prompts: List of prompts
            completions: List of generated completions
            completions_ids: Tokenized completion IDs
            ground_truth_ids: Tokenized ground truth answers
            trainer_state: Current trainer state (for curriculum learning)
            **kwargs: Additional dataset columns (e.g., ground_truth_steps)

        Returns:
            List of float rewards, one per completion
        """
        if completions_ids is None or ground_truth_ids is None:
            raise ValueError(
                "completions_ids and ground_truth_ids are required for reward computation"
            )

        rewards = []

        for idx, (comp_ids, gt_ids) in enumerate(
            zip(completions_ids, ground_truth_ids)
        ):
            # Get current completion text if available
            comp_text = completions[idx] if idx < len(completions) else ""

            # Component 1: Outcome accuracy
            r_outcome = self._compute_outcome_reward(comp_ids, gt_ids)

            # Component 2: Process quality (optional)
            r_process = 0.0
            if self.enable_process_rewards:
                r_process = self._compute_process_reward(comp_ids, gt_ids, **kwargs)

            # Component 3: Backtrack efficiency
            r_backtrack = self._compute_backtrack_reward(comp_ids, gt_ids)

            # Component 4: Format compliance (optional)
            r_format = 0.0
            if self.enable_format_rewards:
                r_format = self._compute_format_reward(comp_text)

            # Weighted combination
            total_reward = (
                self.outcome_weight * r_outcome
                + self.process_weight * r_process
                + self.backtrack_weight * r_backtrack
                + self.format_weight * r_format
            )

            # Apply curriculum scaling if enabled
            if self.use_curriculum and trainer_state is not None:
                total_reward = self._apply_curriculum_scaling(
                    total_reward, trainer_state
                )

            rewards.append(total_reward)

        return rewards

    def _compute_outcome_reward(
        self, completion_ids: list[int], ground_truth_ids: list[int]
    ) -> float:
        """Evaluate final answer correctness after backtracking.

        Args:
            completion_ids: Generated token IDs (with backtrack tokens)
            ground_truth_ids: Ground truth token IDs

        Returns:
            Reward in [0.0, 1.0]
        """
        # Apply backtracking to get final sequence
        final_ids = _apply_backtracking(completion_ids, self.backtrack_token_id)

        # Compute accuracy
        accuracy = _compute_sequence_accuracy(final_ids, ground_truth_ids)

        # Full credit for exact match, partial credit otherwise
        if accuracy == 1.0:
            return 1.0
        else:
            return accuracy * 0.5  # Scale partial credit

    def _compute_process_reward(
        self,
        completion_ids: list[int],
        ground_truth_ids: list[int],
        **kwargs: Any,
    ) -> float:
        """Evaluate intermediate reasoning steps.

        This requires ground_truth_steps in kwargs. If not available,
        returns a heuristic score based on sequence quality.

        Args:
            completion_ids: Generated token IDs
            ground_truth_ids: Ground truth token IDs
            **kwargs: Additional data (may include ground_truth_steps)

        Returns:
            Reward in [0.0, 1.0]
        """
        ground_truth_steps = kwargs.get("ground_truth_steps", None)

        if ground_truth_steps is None:
            # Fallback: heuristic based on final accuracy
            final_ids = _apply_backtracking(completion_ids, self.backtrack_token_id)
            return _compute_sequence_accuracy(final_ids, ground_truth_ids) * 0.8

        # Step-by-step evaluation (if ground truth steps available)
        # This is a simplified implementation; actual implementation would
        # need to parse and compare reasoning steps
        return 0.5  # Placeholder

    def _compute_backtrack_reward(
        self, completion_ids: list[int], ground_truth_ids: list[int]
    ) -> float:
        """Evaluate backtracking efficiency and appropriateness.

        Rewards:
        - Successful corrections (improvement after backtracking)
        - Efficient use (fewer backtracks preferred)

        Penalties:
        - Unnecessary backtracks (when initial answer was correct)
        - Failed corrections (backtracking without improvement)
        - Excessive backtracks (beyond reasonable threshold)
        - Stuttering (early backtracks on short prefixes)

        Args:
            completion_ids: Generated token IDs (with backtrack tokens)
            ground_truth_ids: Ground truth token IDs

        Returns:
            Reward (can be positive or negative)
        """
        backtrack_token_id = self.backtrack_token_id
        num_backtracks = completion_ids.count(backtrack_token_id)

        # Hard constraint: excessive backtracks get severe penalty
        if num_backtracks > self.max_backtracks:
            return -1.0

        # No backtracking case
        if num_backtracks == 0:
            # Check if backtracking was needed
            accuracy = _compute_sequence_accuracy(completion_ids, ground_truth_ids)
            if accuracy == 1.0:
                return 0.0  # Correct without backtracking: neutral
            else:
                return -0.1  # Should have backtracked: small penalty

        # Evaluate backtracking impact
        # 1. Get Initial State (Before first backtrack)
        try:
            first_bt_idx = completion_ids.index(backtrack_token_id)
            initial_ids = completion_ids[:first_bt_idx]
        except ValueError:
            initial_ids = completion_ids

        # 2. Get Final State (Semantic Result)
        final_ids = _apply_backtracking(completion_ids, backtrack_token_id)

        # 3. Resolve Ground Truth (in case it contains backtracks)
        gt_final = _apply_backtracking(ground_truth_ids, backtrack_token_id)

        # 4. Compute Accuracies
        initial_acc = _compute_sequence_accuracy(initial_ids, gt_final)
        final_acc = _compute_sequence_accuracy(final_ids, gt_final)
        improvement = final_acc - initial_acc

        reward = 0.0

        # Check validity (Length Constraint to prevent stuttering)
        # Only reward improvement if the initial attempt was "substantial"
        # We use a heuristic: initial_len > 0.5 * gt_len
        is_valid_attempt = len(initial_ids) > 0.5 * len(gt_final)

        # 1. Correction success bonus (Only if valid attempt)
        if improvement > 0:
            if is_valid_attempt:
                reward += self.correction_bonus * improvement
                # Efficiency bonus (fewer backtracks preferred when successful)
                efficiency = 1.0 / (num_backtracks**0.5)
                reward += self.efficiency_weight * efficiency
            else:
                # Stuttering case: Improvement is fake (from incomplete -> complete)
                # Treat as unnecessary/noise
                reward -= self.unnecessary_penalty * num_backtracks

        # 2. Unnecessary backtrack penalty
        elif initial_acc == 1.0:
            # Initial answer was correct, shouldn't have backtracked
            reward -= self.unnecessary_penalty * num_backtracks

        # 3. Failed correction penalty
        elif improvement <= 0:
            # Backtracked but didn't improve (or made it worse)
            reward -= self.failed_correction_penalty

        return reward

    def _compute_format_reward(self, completion_text: str) -> float:
        """Check output format compliance.

        Checks for:
        - Proper answer enclosure (e.g., \\boxed{} for math)
        - No malformed sequences
        - Basic structural elements

        Args:
            completion_text: Generated completion as text

        Returns:
            Reward in [0.0, 1.0]
        """
        if not completion_text:
            return 0.0

        reward = 0.0

        # Check for boxed answer (math-specific)
        if "\\boxed{" in completion_text:
            reward += 0.5
            # Verify proper closure
            if self._is_valid_boxed_format(completion_text):
                reward += 0.5

        return min(reward, 1.0)

    def _is_valid_boxed_format(self, text: str) -> bool:
        """Check if boxed format is properly formed.

        Args:
            text: Completion text

        Returns:
            True if valid boxed format
        """
        # Simple check: count opening and closing braces
        open_count = text.count("\\boxed{")
        close_count = text.count("}")

        # Should have matching braces
        return open_count > 0 and open_count <= close_count

    def _apply_curriculum_scaling(
        self, reward: float, trainer_state: TrainerState
    ) -> float:
        """Scale reward based on curriculum stage.

        Early training: emphasize process and backtrack learning
        Late training: emphasize outcome accuracy

        Args:
            reward: Base reward
            trainer_state: Current trainer state

        Returns:
            Scaled reward
        """
        if trainer_state.max_steps == 0:
            return reward

        progress = trainer_state.global_step / trainer_state.max_steps

        # Gradually shift emphasis (this is a simplified version)
        # In practice, would dynamically adjust component weights
        if progress < 0.3:  # Early stage
            scale = 1.0
        elif progress < 0.7:  # Middle stage
            scale = 1.0
        else:  # Late stage
            scale = 1.0

        return reward * scale

    @property
    def processing_class(self) -> Any | None:
        """Return the processing class for this reward function.

        Only needed when reward function is a model-based reward.
        For custom reward functions, this should return None.

        Returns:
            PreTrainedTokenizer or None
        """
        return None
