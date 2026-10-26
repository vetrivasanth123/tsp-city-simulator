
"""Diagnostic tests for the REINFORCE policy and gradient updates.

Run from the project root:
    pytest -v tests/test_reinforce_policy.py
"""

import copy

import pytest
import torch
from torch.optim import SGD

from examples.reinforce.policy import SoftmaxPolicy
from examples.reinforce.trajectory import EpisodeTrajectory
from examples.reinforce.returns import calculate_returns
from examples.reinforce.reinforce import reinforce_update


@pytest.fixture
def policy():
    torch.manual_seed(42)
    return SoftmaxPolicy(num_cities=3)


@pytest.fixture
def observation():
    return {
        "current_city": 0,
        "visited_mask": [1, 0, 0],
    }


def test_legal_actions_have_valid_probabilities(policy, observation):
    """Legal-action probabilities must sum to one."""
    legal_actions = [1, 2]

    distribution = policy.get_distribution(
        observation, legal_actions
    )
    probs = distribution.probs.detach()

    assert torch.isfinite(probs).all()
    assert torch.allclose(
        probs.sum(), torch.tensor(1.0), atol=1e-7
    )

    # Every illegal action must have exactly zero probability.
    assert probs[0].item() == 0.0
    assert probs[3].item() == 0.0

    for action in legal_actions:
        assert probs[action].item() > 0.0


def test_close_is_the_only_legal_action(policy, observation):
    """When CLOSE is the only legal action, its probability is one."""
    close_action = policy.close_action

    distribution = policy.get_distribution(
        observation, [close_action]
    )

    assert distribution.probs[close_action].item() == 1.0
    assert torch.count_nonzero(distribution.probs).item() == 1


def test_log_probability_has_finite_gradients(policy, observation):
    """The selected action's log-probability must backpropagate."""
    log_prob = policy.log_probability(
        observation,
        available_actions=[1, 2],
        intended_action=1,
    )

    assert torch.isfinite(log_prob)

    policy.zero_grad()
    log_prob.backward()

    gradients = [
        parameter.grad
        for parameter in policy.parameters()
        if parameter.grad is not None
    ]

    assert gradients, "No gradients were produced."
    assert all(torch.isfinite(g).all() for g in gradients)
    assert any(torch.count_nonzero(g).item() > 0 for g in gradients)


def test_positive_return_increases_selected_action_probability(
    policy, observation
):
    """A positive-return REINFORCE update should favor the selected action."""
    actions = [1, 2]
    selected_action = 1
    learning_rate = 0.1

    def probability():
        return policy.get_distribution(
            observation, actions
        ).probs[selected_action].item()

    probability_before = probability()

    optimizer = SGD(policy.parameters(), lr=learning_rate)
    log_prob = policy.log_probability(
        observation, actions, selected_action
    )

    # Gradient descent on -G*log(pi) is gradient ascent on log(pi)
    # when G is positive.
    loss = -1.0 * log_prob

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    probability_after = probability()

    print(
        "\nSelected action probability:",
        f"{probability_before:.6f} -> {probability_after:.6f}",
    )

    assert probability_after > probability_before


def test_returns_are_calculated_correctly():
    """Verify discounted returns independently of the policy."""
    rewards = [-3.0, -4.0, -5.0]

    assert calculate_returns(rewards, gamma=1.0) == [
        -12.0, -9.0, -5.0
    ]

    assert calculate_returns(rewards, gamma=0.5) == pytest.approx(
        [-6.25, -6.5, -5.0]
    )


def make_synthetic_trajectory():
    """Build a controlled trajectory to exercise reinforce_update."""
    trajectory = EpisodeTrajectory()

    # These are synthetic policy-gradient test states, not an
    # environment rollout. Both actions are legal in each state.
    trajectory.observations.extend([
        {"current_city": 0, "visited_mask": [1, 0, 0]},
        {"current_city": 1, "visited_mask": [1, 1, 0]},
    ])
    trajectory.available_actions.extend([
        [1, 2],
        [0, 2],
    ])
    trajectory.intended_actions.extend([1, 2])
    trajectory.rewards.extend([1.0, 2.0])

    return trajectory


def test_reinforce_update_runs_and_changes_parameters(policy):
    """The actual update function should produce finite metrics and updates."""
    trajectory = make_synthetic_trajectory()
    optimizer = SGD(policy.parameters(), lr=0.01)

    parameters_before = [
        p.detach().clone() for p in policy.parameters()
    ]

    metrics = reinforce_update(
        policy=policy,
        trajectory=trajectory,
        optimizer=optimizer,
        gamma=1.0,
    )

    assert metrics["num_updates"] == 2.0
    assert torch.isfinite(torch.tensor(metrics["mean_loss"]))

    assert any(
        not torch.allclose(before, after.detach())
        for before, after in zip(parameters_before, policy.parameters())
    )


def test_sequential_and_episode_level_updates_are_distinct(policy):
    """Compare sequential updates with one fixed-parameter episode update."""
    sequential_policy = copy.deepcopy(policy)
    episode_policy = copy.deepcopy(policy)

    trajectory = make_synthetic_trajectory()
    gamma = 1.0
    learning_rate = 0.01

    # Implementation A: current code, one optimizer step per timestep.
    sequential_optimizer = SGD(
        sequential_policy.parameters(), lr=learning_rate
    )
    reinforce_update(
        sequential_policy,
        trajectory,
        sequential_optimizer,
        gamma=gamma,
    )

    # Implementation B: compute all contributions at the same
    # parameter vector, then take one optimizer step.
    returns = calculate_returns(trajectory.rewards, gamma)
    episode_optimizer = SGD(
        episode_policy.parameters(), lr=learning_rate
    )

    losses = []
    for t in range(len(trajectory)):
        log_prob = episode_policy.log_probability(
            trajectory.observations[t],
            trajectory.available_actions[t],
            trajectory.intended_actions[t],
        )
        losses.append(-(gamma ** t) * returns[t] * log_prob)

    episode_loss = torch.stack(losses).sum()

    episode_optimizer.zero_grad()
    episode_loss.backward()
    episode_optimizer.step()

    sequential_parameters = list(sequential_policy.parameters())
    episode_parameters = list(episode_policy.parameters())

    differences = [
        (a.detach() - b.detach()).abs().max().item()
        for a, b in zip(sequential_parameters, episode_parameters)
    ]

    print("\nMax parameter differences:", differences)

    # This is an expected algorithmic difference, not a failure
    # of the textbook sequential-update implementation.
    assert any(difference > 1e-8 for difference in differences)
