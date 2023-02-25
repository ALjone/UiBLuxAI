from jux_wrappers.observation_wrapper import create_mask_from_pos
import jax.numpy as jnp
from utils.utils import load_config
from jux.actions import UnitAction, FactoryAction, JuxAction
import torch
import jax
from jux.state import State


config = load_config()
num_envs = config["parallel_envs"]
def jux_action(unit_output0, factory_output0, unit_output1, factory_output1, state):
    
    unit_pos = state.units.pos.pos # (batch, 2, 200, 2)
    unit_id = state.units.unit_id # (batch, 2, 200)
    factory_id = state.factories.unit_id
    factory_pos = state.factories.pos.pos
    unit_id2idx = state.unit_id2idx
    factory_id2idx = state.factory_id2idx
    n_unit = state.n_units
    n_factory = state.n_factories

    batched_unit_actions = jnp.zeros(shape = (num_envs, 2, 200, 1, 6), dtype = jnp.int8)
    batched_factory_actions = jnp.zeros(shape = (num_envs, 2, 6),  dtype = jnp.int8)
    batched_unit_action_queue_count = jnp.ones(shape = (num_envs, 2, 200), dtype = jnp.int8)
    batched_unit_action_queue_update = jnp.ones(shape = (num_envs, 2, 200), dtype=jnp.bool_)

    batched_factory_actions, batched_unit_actions = vmapped_jux_action_single_env(unit_output0, factory_output0,
                                                                                                                    unit_output1, factory_output1,
                                                                                                                    unit_id2idx, factory_id2idx,
                                                                                                                    unit_pos, factory_pos,
                                                                                                                    unit_id, factory_id,
                                                                                                                    n_unit, n_factory)
    
    batched_unit_actions = UnitAction(batched_unit_actions[..., 0], batched_unit_actions[..., 1], batched_unit_actions[..., 2],
                             batched_unit_actions[..., 3].astype(jnp.int16), batched_unit_actions[..., 4].astype(jnp.int16), batched_unit_actions[..., 5].astype(jnp.int16))


    return JuxAction(batched_factory_actions, batched_unit_actions, batched_unit_action_queue_count, batched_unit_action_queue_update)

def __jux_action_single_env(unit_output0, factory_output0, unit_output1, factory_output1, unit_id2idx, factory_id2idx, unit_possies, factory_possies, unit_ids, factory_ids, n_units, n_factories):
    unit_actions0, factory_actions0 = __jux_action_single_env_single_player(unit_output0, 
                unit_possies[0], unit_ids[0], unit_id2idx[0], factory_output0, factory_possies[0], factory_ids[0], factory_id2idx[0], n_units[0], n_factories[0])

    unit_actions1, factory_actions1 = __jux_action_single_env_single_player(unit_output1,
                unit_possies[1], unit_ids[1], unit_id2idx[1], factory_output1, factory_possies[1], factory_ids[1], factory_id2idx[1], n_units[1], n_factories[1])
    return jnp.stack([factory_actions0, factory_actions1], dtype = jnp.int8), jnp.stack([unit_actions0, unit_actions1], dtype = jnp.int8)

__jux_action_single_env = jax.jit(__jux_action_single_env)
vmapped_jux_action_single_env = jax.vmap(__jux_action_single_env, 0)#, (0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, None, 0))

def __jux_action_single_env_single_player(unit_outputs, unit_possies, unit_ids, unit_id2idx, factory_outputs, factory_possies, factory_ids, factory_id2idx, n_units, n_factories):
    unit_actions = jnp.zeros(shape = (200, 1, 6), dtype = jnp.int8)
    factory_actions = jnp.zeros(shape = (6), dtype = jnp.int8)

    action = action_func(unit_outputs[(unit_possies[:, 0], unit_possies[:, 1])])
    
    idx = unit_id2idx[unit_ids]
    unit_actions = unit_actions.at[idx, 0].set(action)

    
    fac_action = fac_action_func(factory_outputs[(factory_possies[:, 0], factory_possies[:, 1])])
    
    idx = factory_id2idx[factory_ids]
    factory_actions = factory_actions.at[idx].set(fac_action)
    
    return unit_actions, factory_actions



def __jux_action_single_env_single_unit(action_idx):
    # TODO: remember to compute action transfer and pickup values from frac
    action = index_map[action_idx]
    return jnp.array(action, dtype = jnp.int8)


def __jux_action_single_env_single_factory(action_idx):
    return jnp.where(action_idx < 3, action_idx, -1).astype(jnp.int8).squeeze()

__jux_action_single_env_single_unit = jax.jit(__jux_action_single_env_single_unit)
action_func = jax.vmap(__jux_action_single_env_single_unit)
__jux_action_single_env_single_factory = jax.jit(__jux_action_single_env_single_factory)
fac_action_func = jax.vmap(__jux_action_single_env_single_factory)
action_map = {
                            # keys : (action_type, direction, resource_type, amount)
                            (0,0,0,0,0,1):0, #Do Nothing
                            (0,1,0,0,0,1):1, #Move north
                            (0,2,0,0,0,1):2, #Move east
                            (0,3,0,0,0,1):3, #Move south
                            (0,4,0,0,0,1):4, #Move west
                            (1,1,0,0,0,1):5,# Transfer Ice North
                            (1,2,0,0,0,1):6,# Transfer Ice east
                            (1,3,0,0,0,1):7,# Transfer Ice south
                            (1,4,0,0,0,1):8,# Transfer Ice west

                            (1,1,1,1,0,1):9,# Transfer ore North
                            (1,2,1,1,0,1):10,# Transfer ore east
                            (1,3,1,1,0,1):11,# Transfer ore south
                            (1,4,1,1,0,1):12,# Transfer ore west

                            (1,1,2,1,0,1):13,# Transfer Water North
                            (1,2,2,1,0,1):14,# Transfer Water east
                            (1,3,2,1,0,1):15,# Transfer Water south
                            (1,4,2,1,0,1):16,# Transfer Water west

                            (1,1,4,0,0,1):17,# Transfer Power North 25%
                            (1,1,4,1,0,1):18,# Transfer Power North 50%
                            (1,1,4,2,0,1):19,# Transfer Power North 75%
                            (1,1,4,3,0,1):20,# Transfer Power North 100%

                            (1,2,4,0,0,1):21,# Transfer Power east 25%
                            (1,2,4,1,0,1):22,# Transfer Power east 50%
                            (1,2,4,2,0,1):23,# Transfer Power east 75%
                            (1,2,4,3,0,1):24,# Transfer Power east 100%

                            (1,3,4,0,0,1):25,# Transfer Power south 25%
                            (1,3,4,1,0,1):26,# Transfer Power south 50%
                            (1,3,4,2,0,1):27,# Transfer Power south 75%
                            (1,3,4,3,0,1):28,# Transfer Power south 100%
                            (1,4,4,0,0,1):29,# Transfer Power west 25%
                            (1,4,4,1,0,1):30,# Transfer Power west 50%
                            (1,4,4,2,0,1):31,# Transfer Power west 75%
                            (1,4,4,3,0,1):32,# Transfer Power west 100%
                            
                            (2,0,4,1,0,1):33,# Pickup Power
                            (2,0,0,0,0,1):34,# Pickup ice
                            (2,0,1,0,0,1):35,# Pickup ore
                            (2,0,2,0,0,1):36,# Pickup water
                            (3,0,0,0,0,1):37,# digg
                            (4,0,0,0,0,1):38,# Self Destruct
            
                        }

index_map = {v:torch.tensor(k) for k,v in action_map.items()}

index_map = jnp.array(((0,0,0,0,0,1),
                            (0,1,0,0,0,1),
                            (0,2,0,0,0,1),
                            (0,3,0,0,0,1),
                            (0,4,0,0,0,1),
                            (1,1,0,0,0,1),
                            (1,2,0,0,0,1),
                            (1,3,0,0,0,1),
                            (1,4,0,0,0,1),

                            (1,1,1,1,0,1),
                            (1,2,1,1,0,1),
                            (1,3,1,1,0,1),
                            (1,4,1,1,0,1),

                            (1,1,2,1,0,1),
                            (1,2,2,1,0,1),
                            (1,3,2,1,0,1),
                            (1,4,2,1,0,1),

                            (1,1,4,0,0,1),
                            (1,1,4,1,0,1),
                            (1,1,4,2,0,1),
                            (1,1,4,3,0,1),

                            (1,2,4,0,0,1),
                            (1,2,4,1,0,1),
                            (1,2,4,2,0,1),
                            (1,2,4,3,0,1),

                            (1,3,4,0,0,1),
                            (1,3,4,1,0,1),
                            (1,3,4,2,0,1),
                            (1,3,4,3,0,1),
                            (1,4,4,0,0,1),
                            (1,4,4,1,0,1),
                            (1,4,4,2,0,1),
                            (1,4,4,3,0,1),
                            
                            (2,0,4,1,0,1),
                            (2,0,0,0,0,1),
                            (2,0,1,0,0,1),
                            (2,0,2,0,0,1),
                            (3,0,0,0,0,1),
                            (4,0,0,0,0,1)))
