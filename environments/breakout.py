import jax.numpy as jnp
import jax
from flax import struct
from functools import partial


"""
Breakout: 
observation space 10x10x4 where
0) agent pad position
1) ball position
2) ball trail position
3) bricks position

version 1 modification: added random start at the beginning and changed interaction with block corner
block_probability: probability of each block to appear (default: 1)
max_frames: maximum number of frames before endinding the game, None = inf

"""

@struct.dataclass
class breakoutState:
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


class breakout:
    def __init__(self, key, n_envs = 1, version = 0, block_probability = 1, max_frames = None):

        self.key = key
        self.n_envs = n_envs
        self.version = version

        self.frames_limit = not max_frames is None
        self.max_frames = max_frames
        self.deterministic = block_probability == 1
        self.block_probability = block_probability

        self.obs_shape = (10, 10, 4)
        self.n_actions = 3


    def get_obs(self, state):
        obs = jnp.zeros((10, 10, 4))
        obs = obs.at[9, state.pad, 0].set(1)
        obs = obs.at[state.ball_y, state.ball_x, 1].set(1)
        obs = obs.at[state.trail_y, state.trail_x, 2].set(1)
        obs = obs.at[1:4, :, 3].set(state.bricks)
        return obs

    def reset_state(self, key):
        key, sample_key = jax.random.split(key)
        #initialize bricks
        if self.deterministic:
            bricks = jnp.ones((3, 10), dtype = int)
        else:
            bricks = 1*jax.random.bernoulli(sample_key, self.block_probability, (3, 10))
        
        #initialize positions
        initial_direction = jnp.array([jax.random.bernoulli(sample_key) + 2])
        if self.version == 0:
            initial_pos = jnp.array([jax.random.bernoulli(sample_key)*9])
            ball_y_initial = jnp.array([3])
        if self.version == 1:
            initial_pos = jax.random.randint(sample_key, (1,), 0, 8)
            ball_y_initial = jnp.array([4])
        
        #build initial state
        state = breakoutState(initial_pos, ball_y_initial, initial_pos, jnp.array([3]), 
                                 jnp.array([4]), initial_direction, bricks, key, jnp.array([0]), jnp.array([False]))
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

        #move the pad

        #a : 0 --> move left
        #a : 1 --> do not move
        #a : 2 --> move right
        pad = (state.pad + (action == 2)*(state.pad < 9) -  #move to the right if possible
                            (action == 0)*(state.pad > 0))   #move to the left if possible

        #move the trail
        trail_x = state.ball_x
        trail_y = state.ball_y

        #move the ball (only along diagonals) 

        #direction:
        #0      1
        #  \  /
        #  ball
        # /    \
        #3      2
        ball_x = (state.ball_x + (state.direction == 1) + (state.direction == 2)   #move to the right
                                - (state.direction == 0) - (state.direction == 3))  #move to the left

        ball_y = (state.ball_y + (state.direction == 2) + (state.direction == 3)   #move to the bottom
                                - (state.direction == 0) - (state.direction == 1))  #move up

        #overflows 
        overflow_x = jnp.logical_or(ball_x < 0, ball_x > 9)
        #change in direction when x touch the border (1 <--> 0 and 3 <--> 2)
        direction = state.direction + overflow_x*((state.direction%2 == 0) + (-1)*(state.direction%2 == 1))

        #change in direction when y touch the top (1 <--> 2 and 0 <--> 3)
        direction = direction + (3 - 2*direction)*(ball_y < 0)

        ball_x = jnp.maximum(jnp.minimum(ball_x, 9), 0)
        ball_y = jnp.maximum(jnp.minimum(ball_y, 9), 0)

        #check if we hitted the brick
        hit_brick = state.bricks.at[(ball_y - 1)%10, ball_x].get(mode='fill', fill_value=0)*(1 - state.no_strike)

        #if version zero update no strike
        #and is taken to convert it to bool
        #note: x in (0,1) ->  x and 1 == x
        if self.version == 0:
            no_strike = jnp.logical_and(state.bricks.at[(ball_y - 1)%10, ball_x].get(mode='fill', fill_value=0) , 1)

        #remove brick
        bricks = state.bricks.at[(ball_y - 1)%10, ball_x].add(- hit_brick.squeeze())

        #change direction and ball position
        direction = direction + (3 - 2*direction)*hit_brick #analogous to the case where we hit the top
        ball_y = trail_y*hit_brick + ball_y*(1-hit_brick)

        # if the ball is still in one brick we have to invert again 
        #main modification of version one. When the ball hit a corner it goes back in the same
        #direction. That is a turn of 180 degree. This is why we have to invert the position two times
        if self.version == 1:
            no_strike = state.no_strike
            still_hit_brick = state.bricks.at[(ball_y - 1)%10, ball_x].get(mode='fill', fill_value=0)
            direction = direction + still_hit_brick*((direction%2 == 0) + (-1)*(direction%2 == 1))
            ball_x = trail_x*still_hit_brick + ball_x*(1-still_hit_brick)

        #interaction with the pad
        #there are two possible cases, if the ball hit the pad it goes back in the same direction
        #otherwise if the trail hit the pad it goes back in the other direction
        last_line = (ball_y == 9)
        ball_y = trail_y*last_line + ball_y*(1-last_line)

        trail_hit = last_line*(trail_x == pad)
        ball_hit = last_line*(ball_x == pad)*(1 - trail_hit) #if also ball hit it takes priority

        direction = direction + (3 - 2*direction)*trail_hit
        direction = (direction + 2*ball_hit)%4

        #check if episode is over
        fail = jnp.logical_and(last_line, jnp.logical_not(jnp.logical_or(ball_hit, trail_hit))) #fail if neither trail nor ball hit the pad
        limite_reached = state.current_frame > self.max_frames if self.frames_limit else False
        done = jnp.logical_or(jnp.logical_or(fail, jnp.sum(bricks) == 0), limite_reached)


        #build new state
        state = breakoutState(ball_x, ball_y, trail_x, trail_y, pad, direction, bricks, state.key, state.current_frame + 1, no_strike)

        #autoreset
        state = jax.tree_util.tree_map(lambda x, y: jnp.where(done, x, y), self.reset_state(state.key), state)

        return state, self.get_obs(state), hit_brick, done