import jax.numpy as jnp
from jax import vmap, jit
import jax
from flax import struct

"""
SPACE INVADERS: 
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


def make_env(key, n_envs = 1, version = 0, only_step = False, markov = False, shot_cool_down = 5):
    """
    version 1 modification: added random start
    shot_cool_down: time before be able of shooting again (default: 1)
    enemy_move_interval: maximum number of frames before endinding the game, None = inf
    """
    enemy_move_interval = 12
    enemy_shot_interval = 10

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
        if version == 1:
            direction = jnp.array([2*jax.random.bernoulli(sample_key) - 1])
        if version == 0:
            direction = jnp.array([-1])
        state = EnvironmentState(jnp.array([5]), jnp.zeros((10,10), dtype = int), jnp.zeros((10,10), dtype = int), alien_pos, direction, jnp.array([0]),
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

        pos = (state.pos - (action == 0)*(state.pos > 0)       #move left if it can and if is possible
                             + (action == 2)*(state.pos < 9))      #move right if it can and if is possible
        
        has_shot = jnp.logical_and(action == 3, state.shot_timer == 0)
        f_bullet = state.f_bullet.at[9, pos].set(has_shot)                         #if action is shoot and weapon is cold: shoot
        shot_timer = state.shot_timer*(1 - has_shot) + shot_cool_down*has_shot        #if action is shot, start shoting cold down timer

        #update friendly bullets
        f_bullet = jnp.roll(f_bullet, -1, axis = 0)
        f_bullet = f_bullet.at[9,:].set(0) 

        #update enemy bullets
        e_bullet = jnp.roll(state.e_bullet, 1, axis = 0)
        e_bullet = e_bullet.at[0,:].set(0)

        done = jnp.logical_or(e_bullet.at[9, pos].get(), done)        #check if e bullet hit the agent

        #update aliens
        done = jnp.logical_or(state.alien_pos.at[9, pos].get(), done)     #check if alien hit the agent

        alien_can_move = state.alien_move_timer == 0
        #reset move timer for alien if alien move
        alien_move_timer = state.alien_move_timer*(1 - alien_can_move) +  \
                              jnp.minimum(jnp.count_nonzero(state.alien_pos), state.alien_move_interval)*alien_can_move

        #check if the alien move verticaly
        move_vertically = jnp.logical_or(jnp.logical_and(jnp.sum(state.alien_pos[:,0]) > 0, state.alien_dir < 0),
                                         jnp.logical_and(jnp.sum(state.alien_pos[:,9]) > 0, state.alien_dir > 0))
        
        alien_dir = state.alien_dir*(1 - move_vertically*alien_can_move) \
                        - state.alien_dir*move_vertically*alien_can_move             #if alien move verticaly and has moved also change direction

        done = jnp.logical_or(jnp.sum(state.alien_pos.at[9,:].get())*move_vertically, done)     #if the alien are already at the bottom then game over

        alien_pos = jnp.where(alien_can_move,                                                                          #if can move
                                            jnp.where(move_vertically,                                                       # if move vertically
                                                                    jnp.roll(state.alien_pos, 1, axis = 0),                       #roll down
                                                        jnp.where(alien_dir < 0,                                            #otherwise check if direction is left
                                                                            jnp.roll(state.alien_pos, -1, axis = 1),            #if yes roll left
                                                                            jnp.roll(state.alien_pos, 1, axis = 1))),            #else roll right
                                                state.alien_pos)                                                            #else: dont move
        
        done = jnp.logical_or(alien_pos.at[9, pos].get(), done)          #check if alien hit aget after moving

        #alien shot
        alien_can_shot = state.alien_shot_timer == 0
        e_bullet = jnp.where(alien_can_shot,                                                      #if can shot
                                 nearest_alien_shot(alien_pos, pos, e_bullet),            #update using the function
                                 e_bullet)                                                        #else nothing
        
        alien_shot_timer = state.alien_shot_timer*(1 - alien_can_shot) + enemy_shot_interval*alien_can_shot    #reset timer if alien has shot
        
        #kill locations
        kill_locations = jnp.logical_and(alien_pos, f_bullet)           #check whene f bullets hit aliens

        reward = jnp.expand_dims(jnp.sum(kill_locations), axis=0)              #one reward for every alien killed

        alien_pos = jnp.logical_and(alien_pos, jnp.logical_not(kill_locations))    #remove hitted alien

        f_bullet = jnp.logical_and(f_bullet, jnp.logical_not(kill_locations))      #remove bullet that killed an alien 

        #update timers
        shot_timer -= shot_timer > 0

        alien_move_timer -= 1
        alien_shot_timer -= 1

        #if all allien are killed, update diffculty and respawn them
        all_alien_kill = jnp.count_nonzero(alien_pos) == 0
        alien_move_interval = state.alien_move_interval*(1 - all_alien_kill) + jnp.maximum(6, state.alien_move_interval - 1)*all_alien_kill
        alien_pos = jnp.where(all_alien_kill, jnp.zeros((10,10), dtype = int).at[0:4,2:8].set(1), alien_pos)

        #build new state
        state = EnvironmentState(pos, f_bullet, e_bullet, alien_pos, alien_dir, shot_timer, 
                                 alien_shot_timer, alien_move_timer, alien_move_interval, state.key)

        #autoreset if we are done
        state = jax.tree_map(lambda x,y: jnp.where(done,x,y),reset(state.key),state)  #autoreset

        return state, get_obs(state), reward, done

    if(only_step):
        return step


    jitstep = jit(vmap(step, in_axes=0, out_axes = 0))
    keys = jax.random.split(key,n_envs)
    state = vec_reset(keys)

    return state, vec_get_obs(state), jitstep, 4