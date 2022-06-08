import jax.numpy as jnp
from jax import vmap, jit
import jax
from flax import struct


@struct.dataclass
class EnvironmentState:
  ball_x: int           #x coordinate of the ball
  ball_y: int           #y coordinate of the ball
  trail_x: int          #x trail coordinate of the ball
  trail_y: int          #y trail coordinate of the ball
  pad: int              #agent pad position
  direction: int        #direction of the ball
  bricks: jnp.array     #bricks map
  key: jnp.array        #environment key
  current_frame: int    #current frame of the game (used when max_frames is active)
  no_strike: bool       #flag that does not allow the ball to destroy a block in two consecutive frames
                        #(present in the orginal version, deprecated in version one)

def make_env(key, n_envs = 1, version = 0, only_step = False, markov = False, block_probability = 1, max_frames = None):
    """
    block_probability: probability of each block to appear (default: 1)
    version 0: same as the min atari environment
    varsion 1: random start at the beginning and correction in the iteration with the block angle
    max_frames: maximum number of frames before endinding the game, None = inf
    """
    frames_limit = not max_frames is None
    deterministic = block_probability == 1

    def get_obs(state):
        obs = jnp.zeros((10,10,4))
        obs = obs.at[9,state.pad,0].set(1)
        obs = obs.at[state.ball_y,state.ball_x,1].set(1)
        obs = obs.at[state.trail_y,state.trail_x,2].set(1)
        obs = obs.at[:,:,3].set(state.bricks)
        return obs

    def reset(key):
        key, sample_key = jax.random.split(key)
        #initialize bricks
        if deterministic:
            brick = jnp.zeros((10,10), dtype = int).at[1:4,:].set(1)
        else:
            brick_initilization = jax.random.bernoulli(sample_key, block_probability, (3,10))
            brick = jnp.zeros((10,10), dtype = int).at[1:4,:].set(brick_initilization)
        
        #initialize positions
        initial_direction = jnp.array([jax.random.bernoulli(sample_key) + 2])
        if version == 0:
            initial_pos = jnp.array([jax.random.bernoulli(sample_key)*9])
            ball_y_initial = jnp.array([3])
        if version == 1:
            initial_pos = jax.random.randint(sample_key, (1,), 0, 8)
            ball_y_initial = jnp.array([4])
        
        #build initial state
        state = EnvironmentState(initial_pos, ball_y_initial, initial_pos, jnp.array([3]), 
                                 jnp.array([4]), initial_direction, brick, key, jnp.array([0]), jnp.array([False]))
        return state

    vec_reset = vmap(reset,in_axes=0, out_axes = 0)
    vec_get_obs =vmap(get_obs,in_axes=0, out_axes = 0)

    #a : 0 --> move left
    #a : 1 --> do not move
    #a : 2 --> move right
    def step(state,action):

        #1) move the pad
        state = state.replace(pad = state.pad + (action == 2)*(state.pad < 9) -  #move to the right if possible
                                                (action == 0)*(state.pad > 0))   #move to the left if possible

        #2) move the trail
        state = state.replace(trail_x = state.ball_x,trail_y = state.ball_y)

        #3) move the ball (only along diagonals) 
        #direction:
        #0      1
        #  \  /
        #  ball
        # /    \
        #3      2
        new_ball_x = (state.ball_x + (state.direction == 1) + (state.direction == 2)   #move to the right
                                    - (state.direction == 0) - (state.direction == 3))  #move to the left

        new_ball_y = (state.ball_y + (state.direction == 2) + (state.direction == 3)   #move to the bottom
                                    - (state.direction == 0) - (state.direction == 1))  #move up

        #3a) overflows 
        overflow_x = (new_ball_x < 0) + (new_ball_x > 9)
        #change in direction when x touch the border (1 <--> 0 and 3 <--> 2)
        new_direction = state.direction + overflow_x*((state.direction%2 == 0) + (-1)*(state.direction%2 == 1))

        #change in direction when y touch the top (1 <--> 2 and 0 <--> 3)
        new_direction = new_direction + (3 - 2*new_direction)*(new_ball_y < 0)

        new_ball_x = jnp.maximum(jnp.minimum(new_ball_x,9),0)
        new_ball_y = jnp.maximum(jnp.minimum(new_ball_y,9),0)

        #4) hit the brick
        hit_brick = state.bricks.at[new_ball_y,new_ball_x].get()*(1 - state.no_strike)

        #if version zero update no strike
        if version == 0:
            state = state.replace(no_strike = state.bricks.at[new_ball_y,new_ball_x].get())

        #4a) remove brick
        state = state.replace(bricks = state.bricks.at[new_ball_y,new_ball_x].add(- hit_brick.squeeze()))

        #4b) change direction and ball position
        new_direction = new_direction + (3 - 2*new_direction)*hit_brick #analogous to the case where we hit the top
        new_ball_y = state.trail_y*hit_brick + new_ball_y*(1-hit_brick)

        #4c) if the ball is still in one brick we have to invert again 
        #main modification of version one. When the ball hit a corner it goes back in the same
        #direction. That is a turn of 180 degree. This is why we have to invert the position two times
        if version == 1:
            still_hit_brick = state.bricks.at[new_ball_y,new_ball_x].get()
            new_direction = new_direction + still_hit_brick*((new_direction%2 == 0) + (-1)*(new_direction%2 == 1))
            new_ball_x = state.trail_x*still_hit_brick + new_ball_x*(1-still_hit_brick)

        #5) interaction with the pad
        #there are two possible cases, if the ball hit the pad it goes back in the same direction
        #otherwise if the trail hit the pad it goes back in the other direction
        last_line = (new_ball_y == 9)
        new_ball_y = state.trail_y*last_line + new_ball_y*(1-last_line)

        trail_hit = last_line*(state.trail_x == state.pad)
        ball_hit = last_line*(new_ball_x == state.pad)*(1 - trail_hit) #if also ball hit it takes priority

        new_direction = new_direction + (3 - 2*new_direction)*trail_hit
        new_direction = (new_direction + 2*ball_hit)%4

        #check if episode is over
        fail = jnp.logical_and(last_line,jnp.logical_not(jnp.logical_or(ball_hit,trail_hit))) #fail if no one hit the pad
        limite_reached = state.current_frame > max_frames if frames_limit else False
        done = jnp.logical_or(jnp.logical_or(fail,jnp.sum(state.bricks) == 0), limite_reached)

        #update state
        state = state.replace(ball_x = new_ball_x, ball_y = new_ball_y, direction = new_direction, current_frame = state.current_frame + 1)

        #autoreset in case we are done
        state = jax.tree_map(lambda x, y: jnp.where(done,x,y), reset(state.key), state)

        return state, get_obs(state), hit_brick, done


    if(only_step):
        return step


    jitstep = jit(vmap(step, in_axes=0, out_axes = 0))
    keys = jax.random.split(key,n_envs)
    state = vec_reset(keys)

    return state, vec_get_obs(state), jitstep, 3