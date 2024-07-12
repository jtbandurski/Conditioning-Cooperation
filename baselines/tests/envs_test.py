from absl.testing import absltest
from gymnasium.spaces import discrete
from meltingpot import substrate
from meltingpot.configs.substrates import prisoners_dilemma_in_the_matrix__arena,commons_harvest__open,commons_harvest__private

from baselines.train import make_envs

test_substrate = commons_harvest__private
TEST_SUBSTRATE = "commons_harvest__private"
EXPECTED_WORLD_SIZE = (13, 7, 3)

class MeltingPotEnvTests(absltest.TestCase):
  """Tests for MeltingPotEnv for RLLib."""

  def setUp(self):
    super().setUp()
    env_config = substrate.get_config(TEST_SUBSTRATE)
    roles = env_config.default_player_roles
    self._num_players = len(roles)
    self._env = make_envs.env_creator({
        'substrate': TEST_SUBSTRATE,
        'roles': roles,
        'scaled': 1,
    })

  def test_action_space_size(self):
    """Test the action space is the correct size."""

    actions_count = len(test_substrate.ACTION_SET)
    env_action_space = self._env.action_space['player_1']
    self.assertEqual(env_action_space, discrete.Discrete(actions_count))

  def test_reset_number_agents(self):
    """Test that reset() returns observations for all agents."""

    obs, _ = self._env.reset()
    self.assertLen(obs, self._num_players)

  def test_step(self):
    """Test step() returns rewards for all agents."""

    self._env.reset()

    # Create dummy actions
    actions = {}
    for player_idx in range(0, self._num_players):
      actions['player_' + str(player_idx)] = 1

    # Step
    _, rewards, _, _, _ = self._env.step(actions)

    # Check we have one reward per agent
    self.assertLen(rewards, self._num_players)

  def test_render_modes_metadata(self):
    """Test that render modes are given in the metadata."""

    self.assertIn('rgb_array', self._env.metadata['render.modes'])

  def test_render_rgb_array(self):
    """Test that render('rgb_array') returns the full world."""
    
    self._env.reset()
    render = self._env.render()
    self.assertEqual(render.shape, EXPECTED_WORLD_SIZE)


if __name__ == '__main__':
  absltest.main()
