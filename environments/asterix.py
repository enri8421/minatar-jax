import jax.numpy as jnp
from jax import vmap, jit
import jax
from flax import struct

"""
ASTERIX: 
observation space normal 10x10x4 where
0) agent position
1) enemy position 
2) all trail
3) gold
observation space Markovian 10x10x8 where
4) at the position of the agent we represent the current spawn timer (spawn_timer/init_spawn_speed)
5) at the position of the agent we represent the current spawn speed (spawn_speed/init_spawn_speed)
6) at the position of the entities we rapresent the timer untill next move (state.move_timer/init_move_interval)
7) we rapresent the current alien_move_interval  (state.move_speed/init_move_interval)
8) the current ramp interval (ramp_timer/ramp_interval)
"""



@struct.dataclass
class EnvironmentState:
    player_x: int            #the x position of the agent
    player_y: int            #the y position of the agent
    entities: jnp.array      #array of size 8x3 containing the entities, one for each row
    spawn_speed: int         #the current spawn speed of entities
    spawn_timer: int         #timer untill next entity will spawn
    move_timer: int          #timer untill entities will move
    move_speed: int          #current speed of the entity 
    ramp_timer: int          #time untill difficulty will be increased
    ramp_index: int          #current level of difficulty
    key: jnp.array           #environment key




def make_env(key, n_envs  = 1, version = 0,  only_step = False, markov = False, ramp_interval = 100, init_spawn_speed = 10, init_move_interval = 5):
    def get_normal_obs(state):
      obs = jnp.zeros((10,10,4))
      obs = obs.at[state.player_y,state.player_x,0].set(1)
      obs = obs.at[jnp.arange(1,9),state.entities[:,0], 1].set(state.entities[:,2] == 0)
      obs = obs.at[jnp.arange(1,9),state.entities[:,0], 3].set(state.entities[:,2] == 1)
      back_x = state.entities[:,0] - 2*state.entities[:,1] + 1
      back_not_overflow = (back_x >= 0)*(back_x <= 9)
      obs = obs.at[jnp.arange(1,9), back_x, 2].set(back_not_overflow*(state.entities[:,2] >= 0), mode = "drop")
      return obs
    
    def get_obs(state):
        if markov:
            obs = jnp.zeros((10,10,8))
            obs = obs.at[:,:,:4].set(get_normal_obs(state))
            obs = obs.at[state.player_y,state.player_x,4].set(state.spawn_timer/state.spawn_speed)
            obs = obs.at[jnp.arange(1,9),state.entities[:,0],5].set(state.move_timer/state.move_speed)
            obs = obs.at[state.player_y,state.player_x,6].set((state.ramp_timer + 1)/(ramp_interval + 1))
            obs = obs.at[state.player_y,state.player_x,7].set(state.ramp_index/max(init_spawn_speed,2*init_move_interval))
        
        else:
            obs = get_normal_obs(state)

        return obs

    def spawn_entity(key, entities):
        key_y, key_dir, key_is_gold = jax.random.split(key, 3)
        #decide if new new entities is gold and direction
        dir = jax.random.bernoulli(key_dir)
        is_gold = jax.random.bernoulli(key_is_gold, 1/3)
        x = 9*(1 - dir)
        #take a random line if it is free
        y = jax.numpy.argmax((entities[:,2] < 0)*(jax.random.permutation(key_y, 8) + 1))
        #build new entity
        new_entity = jnp.array([x, dir, is_gold])
        #if there is a free line put the entity in the row we selected
        free_slot = jnp.sum(entities[:,2] < 0) > 0
        entities = entities.at[y].set(jnp.where(free_slot, new_entity, entities[y]))
        return entities
    
    def check_collision(entities, player_x, player_y):
        #check if there is something in the agent position, then check if it is a valid entities
        possible_collision = entities[player_y - 1, 0] == player_x
        collision_object = entities[player_y - 1, 2]
        collision = (collision_object >= 0)*(possible_collision >0)
        #check if it is gold
        is_gold = collision*(collision_object == 1)
        #if is gold remove it
        entities = entities.at[player_y - 1, 2].set(entities[player_y - 1, 2]*(1 - is_gold) - 1*is_gold)
        return entities, is_gold, collision*(1 - is_gold)

    def reset(key):
      key, cars_key = jax.random.split(key)
      #entities: for every entities we have an array of size 3 [x position, dir, content]. The entities in entry i is in y position i + y
      #content: 1 gold, 0 enemy, -1 nothing
      entities = jnp.zeros((8,3), dtype = int).at[:,2].set(-1)

      state = EnvironmentState(jnp.array([5]), jnp.array([5]), entities, jnp.array([init_spawn_speed]), jnp.array([init_spawn_speed]),  
                               jnp.array([init_move_interval]), jnp.array([init_move_interval]), jnp.array([ramp_interval]), jnp.array([0]), key)
      return state

    vec_reset = vmap(reset,in_axes=0, out_axes = 0)
    vec_get_obs =vmap(get_obs,in_axes=0, out_axes = 0)
        
    def step(state, action):




        #spawn entities and reset timer
        key, entity_key = jax.random.split(state.key)
        new_entities = jnp.where(state.spawn_timer == 0, spawn_entity(entity_key, state.entities), state.entities)
        new_spawn_timer = state.spawn_timer + state.spawn_speed*(state.spawn_timer == 0)

        #move the agent
        #        1
        #        ^
        #   2 <-- A --> 0
        #        3
        #4: do nothing
        new_player_x = (state.player_x + (action == 0)*(state.player_x < 9)        #move right if possible
                                        - (action == 2)*(state.player_x > 0))       #move left if possible
        new_player_y = (state.player_y + (action == 3)*(state.player_y < 8)        #move right if possible
                                        - (action == 1)*(state.player_y > 1))       #move left if possible
        
        reward = jnp.array([0])
        done = jnp.array([0])

        #check collision
        new_entities, new_reward, new_done = check_collision(new_entities, new_player_x, new_player_y)
        reward += new_reward
        done = jnp.logical_or(done, new_done)

        #update positions

        #check if the entities can move and if yes reset timer
        can_move = state.move_timer == 0
        new_move_timer = state.move_timer + state.move_speed*(state.move_timer == 0)

        #move the entities
        new_entities = new_entities.at[:,0].add((new_entities[:,1]*2 - 1)*(new_entities[:,2] >= 0)*can_move)

        #remove the entities that exits the screen
        overflow = (new_entities[:,0] < 0) + (new_entities[:,0] > 9)
        new_entities = new_entities.at[:,2].set(new_entities[:,2]*(1 - overflow) - 1*overflow)

        #check collision again
        new_entities, new_reward, new_done = check_collision(new_entities, new_player_x, new_player_y)
        reward += new_reward
        done = jnp.logical_or(done, new_done)


        #update timer
        new_spawn_timer -= 1
        new_move_timer -= 1

        #check if is time to ramp difficulties
        new_ramp_timer = state.ramp_timer - (state.ramp_timer >= 0)*jnp.logical_or(state.spawn_speed > 1, state.move_speed > 1)

        #if yes increase speeds and reset index and track ramp index
        new_move_speed = state.move_speed - (state.ramp_timer < 0)*(state.move_speed > 1)*(state.ramp_index % 2 == 1)
        new_spawn_speed = state.spawn_speed - (state.ramp_timer < 0)*(state.spawn_speed > 1)
        new_ramp_index = state.ramp_index + (state.ramp_timer < 0)
        new_ramp_timer = new_ramp_timer*(state.ramp_timer >= 0) + ramp_interval*(state.ramp_timer < 0)

        #build new state
        new_state = EnvironmentState(new_player_x, new_player_y, new_entities, new_spawn_speed, new_spawn_timer, 
                                     new_move_timer, new_move_speed, new_ramp_timer, new_ramp_index, key)
        #autoreset
        state = jax.tree_map(lambda x,y: jnp.where(done,x,y),reset(key),new_state)  #autoreset

        return state, get_obs(state), reward, done
      

    if(only_step):
        return step


    jitstep = jit(vmap(step, in_axes=0, out_axes = 0))
    keys = jax.random.split(key,n_envs)
    state = vec_reset(keys)

    return state, vec_get_obs(state), jitstep, 5