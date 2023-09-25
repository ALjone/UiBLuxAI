import jax.numpy as jnp
from utils.utils import load_config
from jux.actions import UnitAction, JuxAction
import jax
from functools import partial


config = load_config()
num_envs = config["parallel_envs"]

@partial(jax.jit,)
def jux_action(unit_output0, factory_output0, unit_output1, factory_output1, state):
    #TODO: The do nothing action for units shouldn't update the action queue, to allow it to recharge
    
    unit_pos = state.units.pos.pos # (batch, 2, 200, 2)
    factory_pos = state.factories.pos.pos # (batch, 2, 6, 2)

    #TODO: I think these needs to actually be matching the units....
    batched_unit_action_queue_count = jnp.ones(shape = (num_envs, 2, 200), dtype = jnp.int8)

    batched_factory_actions, batched_unit_actions, batched_unit_action_queue_update = vmapped_jux_action_single_env(unit_output0, factory_output0,
                                                                                                                    unit_output1, factory_output1,
                                                                                                                    unit_pos, factory_pos)
    #TODO: Is the below needed?
    #batched_unit_action_queue_count.at[batched_unit_action_queue_update].set(1)
    
    batched_unit_actions = UnitAction(batched_unit_actions[..., 0], batched_unit_actions[..., 1], batched_unit_actions[..., 2],
                             batched_unit_actions[..., 3].astype(jnp.int16), batched_unit_actions[..., 4].astype(jnp.int16), batched_unit_actions[..., 5].astype(jnp.int16))


    return JuxAction(batched_factory_actions, batched_unit_actions, batched_unit_action_queue_count, batched_unit_action_queue_update)

def __jux_action_single_env(unit_output0, factory_output0, unit_output1, factory_output1, unit_pos, factory_pos):
    #Unit outputs: (48, 48)
    #Unit pos: (2, 200, 2)
    #Unit ids: (2, 200)
    #Unit id2idx: (2000, 2)

    #Factory outputs: (48, 48)
    #Factory pos: (2, 6, 2)
    #Factory ids: (2, 6)
    #Factory id2idx: (12, 2)


    unit_actions0, factory_actions0, unit_action_queue_update0 = __jux_action_single_env_single_player( unit_output0, 
                                                                                                        unit_pos[0], 
                                                                                                        factory_output0, 
                                                                                                        factory_pos[0])

    unit_actions1, factory_actions1, unit_action_queue_update1 = __jux_action_single_env_single_player( unit_output1,
                                                                                                        unit_pos[1], 
                                                                                                        factory_output1, 
                                                                                                        factory_pos[1])
    
    return jnp.stack([factory_actions0, factory_actions1], dtype = jnp.int8), jnp.stack([unit_actions0, unit_actions1], dtype = jnp.int8), jnp.stack([unit_action_queue_update0, unit_action_queue_update1], dtype = jnp.bool_)


def __jux_action_single_env_single_player(unit_outputs, unit_pos, factory_outputs, factory_pos):
    #Unit outputs: (48, 48)
    #Unit pos: (200, 2)
    #Unit ids: (200)
    #Unit id2idx: (2000, 2)
    #Factory outputs: (48, 48)
    #Factory pos: (6, 2)
    #Factory ids: (6)
    #Factory id2idx: (12, 2)

    unit_action_queue_update = jnp.zeros(shape = (200), dtype = jnp.bool_)

    action = jnp.expand_dims(action_func(unit_outputs[(unit_pos[:, 0], unit_pos[:, 1])]), 1)
    fac_action = fac_action_func(factory_outputs[(factory_pos[:, 0], factory_pos[:, 1])])
    fac_action = jnp.where(factory_pos[:, 0] < 127, fac_action, -2)

    unit_action_queue_update = jnp.where(unit_pos[:, 0] < 127, 1, 0)
    unit_action_queue_update = jnp.where((action[:, 0, 0] == 0) & (action[:, 0, 1] == 0), 0, unit_action_queue_update) #If action starts with (0, 0) it means stand still
    
    return action, fac_action, unit_action_queue_update


def __jux_action_single_env_single_unit(action_idx):
    # TODO: remember to compute action transfer and pickup values from frac
    action = index_map[action_idx]
    return action#jnp.array(action, dtype = jnp.int8)


def __jux_action_single_env_single_factory(action_idx):
    return jnp.where(action_idx < 3, action_idx, -1).astype(jnp.int8).squeeze()


#TODO: Actually read up on jit so these two can be the same
action_func = jax.vmap(__jux_action_single_env_single_unit)
fac_action_func = jax.vmap(__jux_action_single_env_single_factory)

vmapped_jux_action_single_env = jax.vmap(__jux_action_single_env, 0)


index_map = jnp.array(( (0,0,0,0,0,1),
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
                        (4,0,0,0,0,1)), dtype = jnp.int8)
