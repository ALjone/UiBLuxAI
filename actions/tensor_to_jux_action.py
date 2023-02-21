
from jux_wrappers.observation_wrapper import create_mask_from_pos
import jax.numpy as jnp
from utils.utils import load_config

config = load_config()
num_envs = config["parallel_envs"]
def jux_action(outputs, state,player_id):
    
    id_to_idx = state.unit_id2idx # (batch, 2000, 2)
    unit_pos = state.units.pos # (batch, 2, 200, 2)
    unit_id = state.units.unit_id # (batch, 2, 200)
    unit_id_2d = jnp.zeros((num_envs, 48, 48))
    unit_id_2d = create_mask_from_pos(unit_id_2d, unit_pos.x[:,player_id,:], unit_pos.y[:,player_id,:], unit_id[:,player_id,:])

    



    pass