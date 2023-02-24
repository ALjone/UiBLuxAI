from jux_wrappers.observation_wrapper import create_mask_from_pos
import jax.numpy as jnp
from utils.utils import load_config
from jux.actions import UnitAction, FactoryAction, JuxAction
import torch
import jax

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
    batched_unit_action_queue_update = jnp.zeros(shape = (num_envs, 2, 200), dtype=jnp.bool_)
    for i, (unit_possies, factory_possies, unit_ids, unit_id2idxs, factory_id2idxs, factory_ids, unit_outputs0, factory_outputs0, unit_outputs1, factory_outputs1, n_units, n_factories) in \
        enumerate(zip(unit_pos, factory_pos, unit_id, unit_id2idx, factory_id2idx, factory_id, unit_output0, factory_output0, unit_output1, factory_output1, n_unit, n_factory)):

        factory_actions, unit_actions, unit_action_queue_update = __jux_action_single_env(unit_outputs0, factory_outputs0, unit_outputs1, 
                factory_outputs1, unit_id2idxs, factory_id2idxs, unit_possies, factory_possies, unit_ids, factory_ids, n_units, n_factories)
        batched_unit_actions.at[i].set(unit_actions)
        batched_factory_actions.at[i].set(factory_actions)
        batched_unit_action_queue_update.at[i].set(unit_action_queue_update)

    batched_unit_actions = UnitAction(batched_unit_actions[..., 0], batched_unit_actions[..., 1], batched_unit_actions[..., 2],
                             batched_unit_actions[..., 3], batched_unit_actions[..., 4], batched_unit_actions[..., 5])


    return JuxAction(batched_factory_actions, batched_unit_actions, batched_unit_action_queue_count, batched_unit_action_queue_update)

def __jux_action_single_env(unit_output0, factory_output0, unit_output1, factory_output1, unit_id2idx, factory_id2idx, unit_possies, factory_possies, unit_ids, factory_ids, n_units, n_factories):
    unit_actions0, factory_actions0, unit_action_queue_update0 = __jux_action_single_env_single_player(unit_output0, 
                unit_possies[0], unit_ids[0], unit_id2idx[0], factory_output0, factory_possies[0], factory_ids[0], factory_id2idx[0], n_units[0], n_factories[0])

    unit_actions1, factory_actions1, unit_action_queue_update1 = __jux_action_single_env_single_player(unit_output1,
                unit_possies[1], unit_ids[1], unit_id2idx[1], factory_output1, factory_possies[1], factory_ids[1], factory_id2idx[1], n_units[1], n_factories[1])
    return jnp.stack([factory_actions0, factory_actions1]), jnp.stack([unit_actions0, unit_actions1]), jnp.stack([unit_action_queue_update0, unit_action_queue_update1])


def __jux_action_single_env_single_player(unit_output, unit_possies, unit_ids, unit_id2idx, factory_outputs, factory_possies, factory_ids, factory_id2idx, n_units, n_factories):
    unit_actions = jnp.zeros(shape = (200, 1, 6))
    factory_actions = jnp.zeros(shape = (6))
    unit_action_queue_update = jnp.zeros(shape = (200))
    for i in range(n_units):
        action = __jux_action_single_env_single_unit(unit_output[unit_possies[i, 0], unit_possies[i, 1]])
        idx = unit_id2idx[unit_id[i]]
        unit_actions.at[idx, 0].set(action)
        unit_action_queue_update[idx] = 1
    
    for i in range(n_factories):
        action = __jux_action_single_env_single_factory(factory_outputs[factory_possies[i, 0], factory_possies[i, 1]])
        idx = factory_id2idx[unit_ids[i]]
        factory_actions.at[idx].set(action)
    
    return unit_actions, factory_actions, unit_action_queue_update


def __jux_action_single_env_single_unit(action_idx):
    # TODO: remember to compute action transfer and pickup values from frac
    action = index_map[action_idx]
    return action

#__jux_action_single_env_single_player = jax.jit(__jux_action_single_env_single_player)
#__jux_action_single_env = jax.jit(__jux_action_single_env)
#jux_action = jax.jit(jux_action)

def __jux_action_single_env_single_factory(action_idx):
    if action_idx == 0:
        return FactoryAction.BUILD_LIGHT
    elif action_idx ==1 :
        return FactoryAction.BUILD_HEAVY
    elif action_idx == 2:
        return FactoryAction.WATER
    else:
        return FactoryAction.DO_NOTHING


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