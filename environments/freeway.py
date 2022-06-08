import jax.numpy as jnp
from jax import vmap, jit
import jax
from flax import struct



"""
FREEWAY: 
observation space normal 10x10x7 where
0) agent position
1) cars position
2-6) the channel i+1 contains the trails of the cars that move with speed i
observation space Markovian 10x10x10 where
7) at the agent position we represent the number of frame left: (time_limit - state.current_frame)/time_limit
8) at the agent position we represent the timer untill next move for the agent (state.move_timer/player_speed)
9) behing each car we represent the time untill the car can move next,  (new_cars[:,1]/5)
"""
@struct.dataclass
class EnvironmentState:
    pos: int                     # y position of the agent
    cars: jnp.array              #8x3 array where the columns are the car x position, timer to move, speed
    move_timer: int              #timer untill next agent position
    current_frame: int           #current frame of the game
    key: jnp.array               #environment key



def make_env(key, n_envs = 1, version = 0, only_step = False, markov = False, player_speed = 3, time_limit = 2500):

    def get_normal_obs(state):
      obs = jnp.zeros((10,10,7))
      obs = obs.at[state.pos,4,0].set(1)
      obs = obs.at[jnp.arange(1,9),state.cars[:,0],1].set(1)
      cars_back = (state.cars[:,0] - jnp.sign(state.cars[:,2]))%10
      channel_back = jnp.abs(state.cars[:,2]) + 1
      obs = obs.at[jnp.arange(1,9),cars_back,channel_back].set(1)
      return obs    

    def get_obs(state):
      if markov:
        obs = jnp.zeros((10,10,10))
        obs = obs.at[:,:,:7].set(get_normal_obs(state))
        obs = obs.at[state.pos,4,7].set(state.current_frame/time_limit)
        obs = obs.at[state.pos,4,8].set(state.move_timer/player_speed)
        obs = obs.at[jnp.arange(1,9),state.cars[:,0],9].set(state.cars[:,1]/5)
      else:
        obs = get_normal_obs(state)
      return obs

    def randomize_speeds(key, cars):
      speed_key, direction_key = jax.random.split(key)
      speeds = jax.random.randint(speed_key,(8,), 1, 6)
      directions = jax.random.bernoulli(direction_key, 0.5, (8,))*2 - 1
      cars = cars.at[:,1].set(speeds)
      cars = cars.at[:,2].set(speeds*directions)
      return cars

    def reset(key):
      key, cars_key = jax.random.split(key)
      cars = jnp.zeros((8,3),dtype = int)    #first column: x position, second: timer to move, third: speed
      cars = randomize_speeds(cars_key, cars)
      state = EnvironmentState(jnp.array([9]), cars, jnp.array([player_speed]), jnp.array([0]), key)
      return state

    vec_reset = vmap(reset,in_axes=0, out_axes = 0)
    vec_get_obs =vmap(get_obs,in_axes=0, out_axes = 0)
        
    def step(state, action):

        new_pos = (state.pos - (action == 0)*(state.move_timer == 0)                       #move up if he can
                            + (action == 2)*(state.move_timer == 0)*(state.pos < 9))       #move down if he can and if is possible
        
        has_moved = jnp.logical_and(state.move_timer == 0, jnp.logical_or((action == 0), (action == 2)))

        new_move_timer = state.move_timer*(1 - has_moved) + player_speed*has_moved  #if the player has moved then we reset the timer, otherwise nothing
        
        reward = jnp.where(new_pos, 0, 1)                                          #agent arrived at the end

        new_pos = 9*reward + new_pos*(1 - reward)                            #if it is done he goes back

        key, new_cars_key = jax.random.split(state.key)
        new_cars = jnp.where(reward, randomize_speeds(new_cars_key, state.cars), state.cars)   #if he is done the speed of all car is changed and timer reset

        #car have indicies 0..7, angent can be in position 0,..,9.
        #agent in position 1 checks car 0, 2->1,...,8-->7. agent in position 0 and 9 do not check
        car_agent_map = (new_pos - 1)*(new_pos > 0)*(new_pos < 9) + 8*(new_pos == 0) + 8*(new_pos == 9)
        hit_by_car = new_cars.at[car_agent_map,0].get(mode = "fill", fill_value = -1) == 4         #check if the car at the y position of the agent is at x position 4

        cars_can_move = new_cars[:,1] == 0                                    #if timer to move is zero car can move
        new_cars = new_cars.at[:,1].set(jnp.where(cars_can_move,
                                                  jnp.abs(new_cars[:,2]),    #if timer is zero reset it
                                                  new_cars[:,1] - 1))         #otherwise keep the same and subtract one

        new_cars = new_cars.at[:,0].set(jnp.where(cars_can_move,
                                                 (new_cars[:,0] + jnp.sign(new_cars[:,2]))%10,  #if can move add/subtract one according to speed
                                                  new_cars[:,0]))                                #otherwise keep the same
        
        hit_by_car_after = new_cars.at[car_agent_map,0].get(mode = "fill", fill_value = -1) == 4  #check if he is hitted by a car after the cars move

        go_back = jnp.logical_or(hit_by_car,hit_by_car_after)
        new_pos = 9*go_back + new_pos*(1 - go_back)                                    #if hitted beofre or after movement go back
        
        new_current_frame = state.current_frame + 1                       #increase frame counter

        new_move_timer -= new_move_timer > 0                              #decrease agent move timer

        done = new_current_frame == time_limit                           #check if game is over

        #build new state
        new_state = EnvironmentState(new_pos,new_cars,new_move_timer,new_current_frame,key)

        state = jax.tree_map(lambda x,y: jnp.where(done,x,y),reset(key),new_state)  #autoreset if game is over

        return state, get_obs(state), reward, done

    if(only_step):
        return step


    jitstep = jit(vmap(step, in_axes=0, out_axes = 0))
    keys = jax.random.split(key,n_envs)
    state = vec_reset(keys)

    return state, vec_get_obs(state), jitstep, 3
    

