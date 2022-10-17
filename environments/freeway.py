import jax.numpy as jnp
import jax
from flax import struct
from functools import partial



"""
FREEWAY: 
observation space 10x10x7 where
0) agent position
1) cars position
2-6) the channel i+1 contains the trails of the cars that move with speed i


version 1 modification: -
player speed: frames before making next move
time limit: maximum number of frames before endinding the game


"""
@struct.dataclass
class freewayState:
    pos: int                     # y position of the agent
    cars: jnp.array              #8x3 array where the columns are the car x position, timer to move, speed
    move_timer: int              #timer untill next agent position
    current_frame: int           #current frame of the game
    key: jnp.array               #environment key


class freeway:
    def __init__(self, key, n_envs = 1, version = 0, player_speed = 3, time_limit = 2500):
        self.key = key
        self.n_envs = n_envs
        self.version = version

        self.player_speed = player_speed
        self.time_limit = time_limit

        self.obs_shape = (10, 10, 7)
        self.n_actions = 3


    def get_obs(self, state):
        obs = jnp.zeros((10, 10, 7))
        obs = obs.at[state.pos, 4, 0].set(1)
        obs = obs.at[jnp.arange(1,9), state.cars[:, 0], 1].set(1)
        cars_back = (state.cars[:,0] - jnp.sign(state.cars[:, 2]))%10
        channel_back = jnp.abs(state.cars[:, 2]) + 1
        obs = obs.at[jnp.arange(1,9), cars_back, channel_back].set(1)
        return obs

    def randomize_speeds(self, key, cars):
        speed_key, direction_key = jax.random.split(key)
        speeds = jax.random.randint(speed_key, (8,), 1, 6)
        directions = jax.random.bernoulli(direction_key, 0.5, (8,))*2 - 1
        cars = cars.at[:, 1].set(speeds)
        cars = cars.at[:, 2].set(speeds*directions)
        return cars

    def reset_state(self, key):
        key, cars_key = jax.random.split(key)
        cars = jnp.zeros((8,3), dtype = int)               #first column: x position, second: timer to move, third: speed
        cars = self.randomize_speeds(cars_key, cars)
        state = freewayState(jnp.array([9]), cars, jnp.array([self.player_speed]), jnp.array([0]), key)
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


    @partial(jax.vmap, in_axes=(None, 0, 0), out_axes = 0)
    def step(self, state, action):

        pos = (state.pos - (action == 0)*(state.move_timer == 0)                       #move up if he can
                            + (action == 2)*(state.move_timer == 0)*(state.pos < 9))       #move down if he can and if is possible
        
        has_moved = jnp.logical_and(state.move_timer == 0, jnp.logical_or((action == 0), (action == 2)))

        move_timer = state.move_timer*(1 - has_moved) + self.player_speed*has_moved  #if the player has moved then we reset the timer, otherwise nothing
        
        reward = jnp.logical_not(pos)                             #agent arrived at the end

        pos = 9*reward + pos*(1 - reward)                         #if it is done he goes back

        key, cars_key = jax.random.split(state.key)
        cars = jnp.where(reward, self.randomize_speeds(cars_key, state.cars), state.cars)   #if he is done the speed of all car is changed and timer reset

        #car have indicies 0..7, angent can be in position 0,..,9.
        #agent in position 1 checks car 0, 2->1,...,8-->7. agent in position 0 and 9 do not check
        car_agent_map = (pos - 1)*(pos > 0)*(pos < 9) + 8*(pos == 0) + 8*(pos == 9)
        hit_by_car = cars.at[car_agent_map, 0].get(mode = "fill", fill_value = -1) == 4         #check if the car at the y position of the agent is at x position 4

        cars_can_move = cars[:, 1] == 0                                    #if timer to move is zero car can move
        cars = cars.at[:, 1].set(jnp.where(cars_can_move,
                                          jnp.abs(cars[:,2]),    #if timer is zero reset it
                                          cars[:,1] - 1))         #otherwise keep the same and subtract one

        cars = cars.at[:,0].set(jnp.where(cars_can_move,
                                            (cars[:,0] + jnp.sign(cars[:,2]))%10,  #if can move add/subtract one according to speed
                                            cars[:,0]))                                #otherwise keep the same
        
        hit_by_car_after = cars.at[car_agent_map,0].get(mode = "fill", fill_value = -1) == 4  #check if he is hitted by a car after the cars move

        go_back = jnp.logical_or(hit_by_car, hit_by_car_after)
        pos = 9*go_back + pos*(1 - go_back)                                    #if hitted beofre or after movement go back
        
        current_frame = state.current_frame + 1                       #increase frame counter

        move_timer -= move_timer > 0                              #decrease agent move timer

        done = current_frame == self.time_limit                           #check if game is over

        #build new state
        state = freewayState(pos, cars, move_timer, current_frame, key)

        state = jax.tree_util.tree_map(lambda x, y: jnp.where(done, x, y), self.reset_state(state.key), state)

        return state, self.get_obs(state), reward, done

    

