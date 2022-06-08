import jax.numpy as jnp
from jax import vmap, jit
import jax
from flax import struct

"""
FREEWAY: 
observation space normal 10x10x6 where
0) agent position
1) alien position 
2 - 3) alien position if the go right-left
4) friendly bullet
5) enemy bullet
observation space Markovian 10x10x10 where
6) at the agent position we represent the number time untill he can shoot: state.shot_timer/shot_cool_down
7) at the position of the alien we rapresent the timer untill next move (state.move_timer/player_speed)
8) we rapresent the current alien_move_interval
9) the shoot interval
"""


@struct.dataclass
class EnvironmentState:
    pos: int                          #x position of the agent
    f_bullet: jnp.array               #map containing the f bullets
    e_bullet: jnp.array               #map containing the enemy bullet
    alien_pos: jnp.array              #map containing the alien position
    alien_dir: int                    #direction of the alien (left/right)
    shot_timer: int                   #timer untill the agent can shot
    alien_shot_timer: int             #timer until the alien can shot
    alien_move_timer: int             #timer untill the alien can move
    alien_move_interval: int          #current speed of the alien
    key: jnp.array


def make_env(key, n_envs = 1, version = 0, only_step = False, markov = False, shot_cool_down = 5, enemy_move_interval = 12, enemy_shot_interval = 10):
    """
    version 0: alien moove only left at the beginning
    version 1: random direction
    """

    def get_normal_obs(state):
        obs = jnp.zeros((10,10,6))
        obs = obs.at[9,state.pos,0].set(1)
        obs = obs.at[:,:,1].set(state.alien_pos)

        obs = obs.at[:,:,2].set(jnp.where(state.alien_dir > 0, jnp.zeros((10,10)), state.alien_pos))
        obs = obs.at[:,:,3].set(jnp.where(state.alien_dir > 0, state.alien_pos, jnp.zeros((10,10))))

        obs = obs.at[:,:,4].set(state.f_bullet)
        obs = obs.at[:,:,5].set(state.e_bullet)

        return obs

    def get_obs(state):
        if markov:
            obs = jnp.zeros((10,10,10))
            obs = obs.at[:,:,:6].set(get_normal_obs(state))
            obs = obs.at[9,state.pos,6].set(state.shot_timer/shot_cool_down)
            obs = obs.at[:,:,7].set(state.alien_pos*(state.alien_move_timer/state.alien_move_interval))
            obs = obs.at[:,:,8].set(state.alien_pos*(state.alien_move_interval/enemy_move_interval))
            obs = obs.at[:,:,9].set(state.alien_pos*(state.alien_shot_timer/enemy_shot_interval))
        else:
            obs = get_normal_obs(state)
        
        return obs

    def reset(key):
        key, sample_key = jax.random.split(key)
        alien_pos = jnp.zeros((10,10), dtype = int).at[0:4,2:8].set(1)
        if version == 0:
            direction = jnp.array([2*jax.random.bernoulli(sample_key) - 1])
        if version == 1:
            direction = jnp.array([-1])
        state = EnvironmentState(jnp.array([5]), jnp.zeros((10,10), dtype = int), jnp.zeros((10,10), dtype = int), alien_pos, jnp.array([-1]), jnp.array([0]),
                                jnp.array([enemy_shot_interval]), jnp.array([enemy_move_interval]), jnp.array([enemy_move_interval]), key)
        return state

    vec_reset = vmap(reset,in_axes=0, out_axes = 0)
    vec_get_obs =vmap(get_obs,in_axes=0, out_axes = 0)

    #depending on the position of the agend here is the order in which we have to search for the closest alien
    column_order = jnp.array([[0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
                          [1, 0, 2, 3, 4, 5, 6, 7, 8, 9],
                          [2, 1, 3, 0, 4, 5, 6, 7, 8, 9],
                          [3, 2, 4, 1, 5, 0, 6, 7, 8, 9],
                          [4, 3, 5, 2, 6, 1, 7, 0, 8, 9],
                          [5, 4, 6, 3, 7, 2, 8, 1, 9, 0],
                          [6, 5, 7, 4, 8, 3, 9, 2, 1, 0],
                          [7, 6, 8, 5, 9, 4, 3, 2, 1, 0],
                          [8, 7, 9, 6, 5, 4, 3, 2, 1, 0],
                          [9, 8, 7, 6, 5, 4, 3, 2, 1, 0]])

    def nearest_alien_shot(alien_map, pos, e_bullet):
        #this function finds the closest alien to the agent and makes him shoot
        #is the same as the original implementation
        #y = takes the column with alien that is closer to the agent, this is done using the matrix column 
        #x = then inside that column it search for the alien that is lower
        x = column_order[pos,jnp.where(jnp.sum(alien_map, axis = 0)[column_order[pos]] > 0, size = 1)[1]].squeeze()
        y = jnp.argmax(alien_map[:,x]*jnp.arange(10))
        return e_bullet.at[y,x].set(1)
          
    def step(state, action):

        done = jnp.array([False])

        new_pos = (state.pos - (action == 0)*(state.pos > 0)       #move left if it can and if is possible
                             + (action == 2)*(state.pos < 9))      #move right if it can and if is possible
        
        has_shot = jnp.logical_and(action == 3, state.shot_timer == 0)
        new_f_bullet = state.f_bullet.at[9,new_pos].set(has_shot)                         #if action is shoot and weapon is cold: shoot
        new_shot_timer = state.shot_timer*(1 - has_shot) + shot_cool_down*has_shot        #if we have shot, start shoting cold down timer

        #update friendly bullets
        new_f_bullet = jnp.roll(new_f_bullet, -1, axis = 0)
        new_f_bullet = new_f_bullet.at[9,:].set(0) 

        #update enemy bullets
        new_e_bullet = jnp.roll(state.e_bullet, 1, axis = 0)
        new_e_bullet = new_e_bullet.at[0,:].set(0)

        done = jnp.logical_or(new_e_bullet.at[9, new_pos].get(), done)        #check if e bullet hit the agent

        #update aliens
        done = jnp.logical_or(state.alien_pos.at[9,new_pos].get(), done)     #check if alien hit the agent

        alien_can_move = state.alien_move_timer == 0
        #reset move timer for alien if alien move
        new_alien_move_timer = state.alien_move_timer*(1 - alien_can_move) +  \
                              jnp.minimum(jnp.count_nonzero(state.alien_pos), state.alien_move_interval)*alien_can_move

        #check if the alien move verticaly
        move_vertically = jnp.logical_or(jnp.logical_and(jnp.sum(state.alien_pos[:,0]) > 0, state.alien_dir < 0),
                                         jnp.logical_and(jnp.sum(state.alien_pos[:,9]) > 0, state.alien_dir > 0))
        
        new_alien_dir = state.alien_dir*(1 - move_vertically*alien_can_move) \
                        - state.alien_dir*move_vertically*alien_can_move             #if alien move verticaly and has moved we also change direction

        done = jnp.logical_or(jnp.sum(state.alien_pos.at[9,:].get())*move_vertically, done)     #if the alien are already at the bottom then we lose (probably is usless...)

        new_alien_pos = jnp.where(alien_can_move,                                                                          #if can move
                                                jnp.where(move_vertically,                                                       # if move vertically
                                                                        jnp.roll(state.alien_pos, 1, axis = 0),                       #roll down
                                                        jnp.where(new_alien_dir < 0,                                            #otherwise check if direction is left
                                                                                jnp.roll(state.alien_pos, -1, axis = 1),            #if yes roll left
                                                                                jnp.roll(state.alien_pos, 1, axis = 1))),            #else roll right
                                                state.alien_pos)                                                            #else: dont move
        
        done = jnp.logical_or(new_alien_pos.at[9,new_pos].get(), done)          #check if alien hit us after moving

        #alien shot
        alien_can_shot = state.alien_shot_timer == 0
        new_e_bullet = jnp.where(alien_can_shot,                                                      #if can shot
                                 nearest_alien_shot(new_alien_pos, new_pos, new_e_bullet),            #update using the function
                                 new_e_bullet)                                                        #else nothing
        
        new_alien_shot_timer = state.alien_shot_timer*(1 - alien_can_shot) + enemy_shot_interval*alien_can_shot    #reset timer if alien has shot
        
        #kill locations
        kill_locations = jnp.logical_and(new_alien_pos,new_f_bullet)           #check whene f bullets hit aliens

        reward = jnp.expand_dims(jnp.sum(kill_locations), axis=0)              #one reward for every alien killed

        new_alien_pos = jnp.logical_and(new_alien_pos, jnp.logical_not(kill_locations))    #remove hitted alien

        new_f_bullet = jnp.logical_and(new_f_bullet, jnp.logical_not(kill_locations))      #remove bullet that killed an alien 

        #update timers
        new_shot_timer -= new_shot_timer > 0

        new_alien_move_timer -= 1
        new_alien_shot_timer -= 1

        #if all allien are killed, update diffculty and respawn them
        all_alien_kill = jnp.count_nonzero(new_alien_pos) == 0
        new_alien_move_interval = state.alien_move_interval*(1 - all_alien_kill) + jnp.maximum(6, state.alien_move_interval - 1)*all_alien_kill
        new_alien_pos = jnp.where(all_alien_kill, jnp.zeros((10,10), dtype = int).at[0:4,2:8].set(1), new_alien_pos)

        #build new state
        state = EnvironmentState(new_pos, new_f_bullet, new_e_bullet, new_alien_pos, new_alien_dir, new_shot_timer, 
                                 new_alien_shot_timer, new_alien_move_timer, new_alien_move_interval, state.key)

        #autoreset if we are done
        state = jax.tree_map(lambda x,y: jnp.where(done,x,y),reset(state.key),state)  #autoreset

        return state, get_obs(state), reward, done

    if(only_step):
        return step


    jitstep = jit(vmap(step, in_axes=0, out_axes = 0))
    keys = jax.random.split(key,n_envs)
    state = vec_reset(keys)

    return state, vec_get_obs(state), vmap(step, in_axes=0, out_axes = 0), 4