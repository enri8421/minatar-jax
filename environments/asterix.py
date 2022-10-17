import jax.numpy as jnp
import jax
from flax import struct
from functools import partial

"""
ASTERIX: 
observation space 10x10x4 where
0) agent position
1) enemy position 
2) all trails position
3) gold position


version 1 modification: -
ramp interval: frames before increasing difficulty

"""



@struct.dataclass
class asterixState:
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

class asterix:
    def __init__(self, key, n_envs = 1, version = 0, ramp_interval = 100):

        self.key = key
        self.n_envs = n_envs
        self.version = version

        self.init_spawn_speed = 10
        self.init_move_interval = 5
        self.ramp_interval = ramp_interval

        self.obs_shape = (10, 10, 4)
        self.n_actions = 5


    def get_obs(self, state):
        obs = jnp.zeros((10,10,4))
        obs = obs.at[state.player_y,state.player_x,0].set(1)
        obs = obs.at[jnp.arange(1,9),state.entities[:,0], 1].set(state.entities[:,2] == 0)
        obs = obs.at[jnp.arange(1,9),state.entities[:,0], 3].set(state.entities[:,2] == 1)
        back_x = state.entities[:,0] - 2*state.entities[:,1] + 1
        back_not_overflow = (back_x >= 0)*(back_x <= 9)
        obs = obs.at[jnp.arange(1,9), back_x, 2].set(back_not_overflow*(state.entities[:,2] >= 0), mode = "drop")
        return obs

    def reset_state(self, key):
        #entities: for every entities we have an array of size 3 [x position, dir, content]. The entities in entry i is in y position i + y
        #content: 1 gold, 0 enemy, -1 nothing
        entities = jnp.zeros((8,3), dtype = int).at[:,2].set(-1)

        state = asterixState(jnp.array([5]), jnp.array([5]), entities, jnp.array([self.init_spawn_speed]), jnp.array([self.init_spawn_speed]),  
                                jnp.array([self.init_move_interval]), jnp.array([self.init_move_interval]), jnp.array([self.ramp_interval]), jnp.array([0]), key)
        return state


    @partial(jax.vmap, in_axes=(None, 0), out_axes = 0)
    def get_initial_obs(self, key):
        initial_state = self.reset_state(key)
        initial_obs = self.get_obs(initial_state)
        return initial_state, initial_obs


    def reset(self):
        self.key, reset_key = jax.random.split(self.key)
        reset_keys = jax.random.split(reset_key, self.n_envs)
        return self.get_initial_obs(reset_keys)

    def spawn_entity(self, key, entities):
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
    
    def check_collision(self, entities, player_x, player_y, reward, done):
        #check if there is something in the agent position, then check if it is a valid entities
        possible_collision = entities[player_y - 1, 0] == player_x
        collision_object = entities[player_y - 1, 2]
        collision = jnp.logical_and(collision_object >= 0, possible_collision >0)
        #check if it is gold
        is_gold = jnp.logical_and(collision, collision_object == 1)
        #if is gold remove it
        entities = entities.at[player_y - 1, 2].set(entities[player_y - 1, 2]*(1 - is_gold) - 1*is_gold)

        reward += is_gold
        done = jnp.logical_or(done, jnp.logical_and(collision, jnp.logical_not(is_gold)))
        return entities, reward, done


    @partial(jax.vmap, in_axes=(None, 0, 0), out_axes = 0)
    def step(self, state, action):
        #spawn entities and reset timer
        key, entity_key = jax.random.split(state.key)
        entities = jnp.where(state.spawn_timer == 0, self.spawn_entity(entity_key, state.entities), state.entities)
        spawn_timer = state.spawn_timer + state.spawn_speed*(state.spawn_timer == 0)

        #move the agent
        #        1
        #        ^
        #   2 <-- A --> 0
        #        3
        #4: do nothing
        player_x = (state.player_x + (action == 0)*(state.player_x < 9)        #move right if possible
                                        - (action == 2)*(state.player_x > 0))       #move left if possible
        player_y = (state.player_y + (action == 3)*(state.player_y < 8)        #move right if possible
                                        - (action == 1)*(state.player_y > 1))       #move left if possible
        
        reward = jnp.array([0])
        done = jnp.array([False])

        #check collision
        entities, reward, done = self.check_collision(entities, player_x, player_y, reward, done)

        #update positions

        #check if the entities can move and if yes reset timer
        can_move = state.move_timer == 0
        move_timer = state.move_timer + state.move_speed*(state.move_timer == 0)

        #move the entities
        entities = entities.at[:,0].add((entities[:,1]*2 - 1)*(entities[:,2] >= 0)*can_move)

        #remove the entities that exits the screen
        overflow = (entities[:,0] < 0) + (entities[:,0] > 9)
        entities = entities.at[:,2].set(entities[:,2]*(1 - overflow) - 1*overflow)

        #check collision again
        entities, reward, done = self.check_collision(entities, player_x, player_y, reward, done)

        #update timer
        spawn_timer -= 1
        move_timer -= 1

        #check if is time to ramp difficulties
        ramp_timer = state.ramp_timer - (state.ramp_timer >= 0)*jnp.logical_or(state.spawn_speed > 1, state.move_speed > 1)

        #if yes increase speeds and reset index and track ramp index
        move_speed = state.move_speed - (state.ramp_timer < 0)*(state.move_speed > 1)*(state.ramp_index % 2 == 1)
        spawn_speed = state.spawn_speed - (state.ramp_timer < 0)*(state.spawn_speed > 1)
        ramp_index = state.ramp_index + (state.ramp_timer < 0)
        ramp_timer = ramp_timer*(state.ramp_timer >= 0) + self.ramp_interval*(state.ramp_timer < 0)

        #build new state
        state = asterixState(player_x, player_y, entities, spawn_speed, spawn_timer, 
                                move_timer, move_speed, ramp_timer, ramp_index, key)
        #autoreset
        state = jax.tree_util.tree_map(lambda x, y: jnp.where(done, x, y), self.reset_state(state.key), state)

        return state, self.get_obs(state), reward, done

