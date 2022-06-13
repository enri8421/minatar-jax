import jax.numpy as jnp
from jax import vmap, jit
import jax
from flax import struct

"""
SEAQUEST: 
observation space normal 10x10x4 where
0) agent position
1) agent trail position
2) friendly bulllets position
3) all trails position
4) enemy bullets position
5) fishes position
6) enemy sub position
7) oxygen left
8) number of divers on board
9) divers position
observation space Markovian TO DO
"""

@struct.dataclass
class EnvironmentState:
    oxygen: int                        #oxygen left
    diver_count: int                   #number of divers on board
    sub_x: int                         #agent x coordinate position
    sub_y: int                         #agent y coordinate position
    sub_dir: int                       #agent direction (weapon direction)
    bullets: jnp.array                 #array of dimention 2x9x10 containing the f bullets right/left
    e_bullets: jnp.array               #array of dimention 2x8x10 containing the e bullets right/left
    fish: jnp.array                    #array of dimention 3x2x8x10 containing 3 layer of fish right/left
    e_sub: jnp.array                   #array of dimention 3x2x2x8x10 containing 3 layer of e subs, right/left containing shot and move timer      
    divers: jnp.array                  #array of dimention 2x8x10 containing divers
    e_spawn_speed: int                 #current value of frames beetwen two enemies spawn       
    e_spawn_timer:int                  #frames before next enemy spawn
    d_spawn_timer: int                 #frames before next diver spawn
    move_speed: int                    #speed of the enemies
    ramp_index: int                    #current level of difficulty
    shot_timer: int                    #frames before agent can shot again
    surface: bool                      #True if in the previous frame the agent was already on surface
    where_to_spawn: int                #in wich layer put the next enemy
    key: jnp.array


def make_env(key, n_envs = 1, version = 0, only_step = False, markov = False, max_oxygen = 200, shot_cool_down = 5):
    """
    version 1 modification: fixed the bug where the agent can have a negative number of divers on board and fixed the bug
    whenre in the last iteration we have -1 oxygen but the bar is full in the observation
    max_oxygen: maximum number of frames under water before dying
    shot_cool_down: frames between two consecutive shoot
    """

    init_spawn_speed = 20
    diver_spawn_speed = 30
    init_move_interval = 5
    enemy_shot_interval = 10
    diver_move_interval = 5
    max_divers = 6


    def get_obs(state):
        obs = jnp.zeros((10,10,10))
        obs = obs.at[state.sub_y, state.sub_x, 0].set(1)
        back_x = state.sub_x - 2*state.sub_dir + 1
        obs = obs.at[state.sub_y, back_x, 1].set(1)

        if version == 0:
            oxygen_to_display = state.oxygen*10//max_oxygen
            oxygen_to_display = oxygen_to_display*(oxygen_to_display >= 0) + 9*(oxygen_to_display < 0)
            obs = obs.at[9, :, 7].set(jnp.where(jnp.arange(10) < oxygen_to_display, 1, 0))
        if version == 1:
            obs = obs.at[9, :, 7].set(jnp.where(jnp.arange(10) < state.oxygen*10//max_oxygen, 1, 0))

        obs = obs.at[9,3:9,8].set(jnp.where(jnp.arange(max_divers) >= max_divers - state.diver_count, 1, 0))

        obs = obs.at[:9, :, 2].set(jnp.sum(state.bullets, axis = 0) > 0)
        obs = obs.at[1:9, :, 4].set(jnp.sum(state.e_bullets, axis = 0) > 0)
        obs = obs.at[1:9,:, 5].set(jnp.sum(state.fish, axis = (0, 1)) > 0)
        obs = obs.at[1:9, :, 6].set(jnp.sum(state.e_sub[:, :, 0], axis = (0, 1)) > 0)
        obs = obs.at[1:9, :, 9].set(jnp.sum(state.divers, axis = 0) > 0)


        all_object_l = jnp.logical_or(jnp.logical_or(jnp.sum(state.fish[:, 1], axis = 0),jnp.sum(state.e_sub[:, 1, 0], axis = 0)), state.divers[1])
        trails_l = jnp.roll(all_object_l.at[:, 9].set(0), 1, 1)
        all_object_r = jnp.logical_or(jnp.logical_or(jnp.sum(state.fish[:, 0], axis = 0),jnp.sum(state.e_sub[:, 0, 0], axis = 0)), state.divers[0])
        trails_r = jnp.roll(all_object_r.at[:, 0].set(0), -1, 1)
        obs = obs.at[1:9, :, 3].set(jnp.logical_or(trails_l,trails_r))
        
        return obs
    
    def reset(key):

        state = EnvironmentState(jnp.array([max_oxygen]), jnp.array([0]), jnp.array([5]), jnp.array([0]), jnp.array([0]), jnp.zeros((2 ,9, 10), dtype = int),
                                 jnp.zeros((2, 8, 10), dtype = int), jnp.zeros((3, 2, 8, 10), dtype = int), jnp.zeros((3, 2, 2, 8, 10), dtype = int), 
                                 jnp.zeros((2, 8, 10), dtype = int), jnp.array([init_spawn_speed]), jnp.array([init_spawn_speed]), jnp.array([diver_spawn_speed]),
                                 jnp.array([init_move_interval]), jnp.array([0]), jnp.array([0]), jnp.array([True]), jnp.array([0]), key)
        return state

    vec_reset = vmap(reset, in_axes=0, out_axes = 0)
    vec_get_obs =vmap(get_obs, in_axes=0, out_axes = 0)


    def spaw_enemy(key, fish, e_sub,  e_spawn_timer, where_to_spawn, e_spawn_speed, move_speed):
        #check if it can be spawn and if yes reset timer, then update also layer position
        ready_to_spawn = (e_spawn_timer == 0)
        e_spawn_timer += e_spawn_speed*ready_to_spawn
        where_to_spawn  = (where_to_spawn + ready_to_spawn)%3

        #drawn direction, initial position, type and row
        key_dir, key_sub, key_pos = jax.random.split(key, 3)
        e_goes_l =  jax.random.bernoulli(key_dir)
        x = 9*e_goes_l
        is_sub = jax.random.bernoulli(key_sub)
        y = jax.random.randint(key_pos, shape = (1,), minval= 0, maxval = 8)

        #check that there are not enemy in oposite direction
        not_occupied =  1 - jnp.logical_or(jnp.sum(fish[:, 1 - e_goes_l, y]), jnp.sum(e_sub[:,1 - e_goes_l, 0, y]))

        #check what it will be spawn
        spawn_sub = jnp.logical_and(is_sub, jnp.logical_and(ready_to_spawn, not_occupied))
        spawn_fish = jnp.logical_and(1 - is_sub, jnp.logical_and(ready_to_spawn, not_occupied))

        #add new object if we can in the corrisponding place 
        e_sub = e_sub.at[where_to_spawn, 1*e_goes_l, 0, y, x].add(jnp.where(spawn_sub, move_speed + 1, 0))
        e_sub = e_sub.at[where_to_spawn, 1*e_goes_l, 1, y, x].add(jnp.where(spawn_sub, enemy_shot_interval + 1,  0))
        fish =  fish.at[where_to_spawn, 1*e_goes_l, y, x].add(jnp.where(spawn_fish, move_speed + 1, 0))

        return fish, e_sub, e_spawn_timer, where_to_spawn
    
    def spawn_diver(key, divers, d_spawn_timer):

        key_dir, key_pos = jax.random.split(key, 2)

        #check if we can spawn, if yws update timer
        ready_to_spawn = (d_spawn_timer == 0)
        d_spawn_timer += diver_spawn_speed*ready_to_spawn

        #draw direction, initial position and row
        key_dir, key_pos = jax.random.split(key, 2)
        d_goes_l = jax.random.bernoulli(key_dir)
        x = 9*d_goes_l
        y = jax.random.randint(key_pos, shape = (1,), minval= 0, maxval=8)


        #add diver in the corrisponding place according to direction if we can according to timer
        divers = divers.at[1*d_goes_l, y, x].add(jnp.where(ready_to_spawn, diver_move_interval + 1, 0))

        return divers, d_spawn_timer

    def move_divers(map):
        #move all the object that have timer expired and the reset timer, before right then left
        moved_map = jnp.zeros_like(map)
        moved_map = moved_map.at[0].set(jnp.roll(map[0].at[:,9].set(0), 1,  1))
        moved_map = moved_map.at[1].set(jnp.roll(map[1].at[:,0].set(0), -1,  1))
        map = (map > 1)*(map - 1) + (moved_map == 1)*(diver_move_interval + 1)
        return map

    
    def move_and_make_shot_sub(map, bullets, speed):
        #move all the object that have timer expired and the reset timer, before right then left
        moved_map = jnp.zeros_like(map)
        moved_map = moved_map.at[:,0].set(jnp.roll(map[:,0].at[:,:,:,9].set(0), 1,  3))
        moved_map = moved_map.at[:,1].set(jnp.roll(map[:,1].at[:,:,:,0].set(0), -1,  3))

        not_moved_location = (map[:,:,0] > 1)
        map = map.at[:,:,0].set(not_moved_location*(map[:,:,0] - 1) + (moved_map[:,:,0] == 1)*(speed + 1))
        map = map.at[:,:,1].set(not_moved_location*map[:,:,1] + (moved_map[:,:,0] == 1)*moved_map[:,:,1])


        #make enemy shot and then reset timer
        bullets = jnp.logical_or(bullets, jnp.sum(map[:,:,1] == 1, 0))
        map = map.at[:,:,1].set(jnp.where(map[:,:,1] == 1, enemy_shot_interval + 1, jnp.where(map[:,:,1] > 1, map[:,:,1]  - 1, 0)))
        return map, bullets


    def move_fish(map, speed):
        #move all the object that have timer expired and the reset timer, before right then left
        moved_map = jnp.zeros_like(map)
        moved_map = moved_map.at[:,0].set(jnp.roll(map[:,0].at[:,:,9].set(0), 1,  2))
        moved_map = moved_map.at[:,1].set(jnp.roll(map[:,1].at[:,:,0].set(0), -1,  2))
        map = (map > 1)*(map - 1) + (moved_map == 1)*(speed + 1)
        return map

    
    def move_bullets(map):
        map = map.at[0].set(jnp.roll(map[0].at[:,9].set(0), 1, 1))
        map = map.at[1].set(jnp.roll(map[1].at[:,0].set(0), -1, 1))
        return map

    
    def min_to_zero(x):
        z = jnp.argmin(x + jnp.logical_not(x)*(init_move_interval + 2))
        return x.at[z].set(0)

    def remove_hitted_targets(bullets, fish, e_sub):
        #build composite map of bullets left and right and remove the first row of bullets 
        #(there are no enemy on the surface)
        comp_bullets = jnp.logical_or(bullets[0], bullets[1])[1:]

        #compute composite maps of all enemies (left/ right, fish/subs)
        all_fish = jnp.sum(fish, (0,1))
        all_e_sub = jnp.sum(e_sub, (0,1,2))
        all_targets = jnp.logical_or(all_fish, all_e_sub)

        #check where there is an hit and count them
        hitted_map = all_targets*comp_bullets
        reward = jnp.sum(hitted_map)

        #remove fish in position where there are bullets
        #in case there are multiple fish in the same location remove the one with the lower timer
        #then remove the bullet used to kill the fish
        fish_remove_old = jax.numpy.apply_along_axis(min_to_zero, 0, fish)
        fish = jnp.where(comp_bullets, fish_remove_old, fish)
        comp_bullets = jnp.logical_and(comp_bullets, jnp.logical_not(all_fish))

        #do the same with the sub
        sub_remove_old = jax.numpy.apply_along_axis(min_to_zero, 0, e_sub[:,:,0])
        e_sub = e_sub.at[:,:,0].set(jnp.where(comp_bullets, sub_remove_old, e_sub[:,:,0]))
        e_sub = e_sub.at[:,:,1].set(jnp.where(e_sub[:,:,0], e_sub[:,:,1], 0))
        comp_bullets = jnp.logical_and(comp_bullets, jnp.logical_not(all_e_sub))

        bullets = bullets.at[0, 1:].multiply(comp_bullets)
        bullets = bullets.at[1, 1:].multiply(comp_bullets)

        return bullets, fish, e_sub, reward

    def pick_up_diver(divers, diver_count, x, y, direction):
        #if there is free space on sub, add one to count and remove divers from screen
        free_space = (diver_count < max_divers)
        diver_count = diver_count +  jnp.logical_and(jnp.logical_and(y > 0, divers[direction, y - 1, x]), free_space)
        divers = divers.at[direction, y - 1, x].set(jnp.where(jnp.logical_and(free_space, y > 0), 0, divers[direction, y - 1, x]))
        return divers, diver_count


    
    def pick_up_divers(divers, diver_count, x, y):
        #if there is free space on sub, add one to count and remove divers from screen
        #in order to copy the original implementation we have to pick up the newest diver. 
        first_diver = 1*(x > 4)
        divers, diver_count = pick_up_diver(divers, diver_count, x, y, first_diver)

        divers, diver_count = pick_up_diver(divers, diver_count, x, y, 1 - first_diver)

        return divers, diver_count
    
    def enemy_kill_aget(fish, e_sub, e_bullets, x, y):
        return jnp.logical_and(y > 0, jnp.logical_or(jnp.logical_or(jnp.sum(fish[:,:, y - 1, x]), jnp.sum(e_sub[:,:,0, y - 1, x])), jnp.sum(e_bullets[:, y - 1, x])))

    def step(state, action):

        key, key_spawn_e, key_spawn_d = jax.random.split(state.key, 3)
        done = jnp.array([False])
        reward = jnp.array([0])


        #spawn enemy
        fish, e_sub, e_spawn_timer, where_to_spawn = spaw_enemy(key_spawn_e, state.fish, state.e_sub,  state.e_spawn_timer, 
                                                                state.where_to_spawn, state.e_spawn_speed, state.move_speed)
        #spawn diver
        divers, d_spawn_timer = spawn_diver(key_spawn_d, state.divers, state.d_spawn_timer)


        #move the agent and update direction
        #        1
        #        ^
        #   2 <-- A --> 0
        #        3
        #4: do nothing, 5 shoot
        sub_x = (state.sub_x + (action == 0)*(state.sub_x < 9)                   #move right if possible
                                        - (action == 2)*(state.sub_x > 0))       #move left if possible
        sub_y = (state.sub_y + (action == 3)*(state.sub_y < 8)                   #move down if possible
                                        - (action == 1)*(state.sub_y > 0))       #move up if possible
        sub_dir = state.sub_dir*jnp.logical_or(jnp.logical_or(action == 1, action == 3), jnp.logical_or(action == 4, action == 5)) + (action == 0) 
        
        #check if the agent can shot, if yes update the timer and do it
        can_shot = jnp.logical_and(state.shot_timer == 0, action == 5)
        shot_timer = state.shot_timer + can_shot*shot_cool_down
        bullets = jnp.where(can_shot, state.bullets.at[1 - sub_dir, sub_y, sub_x].add(1), state.bullets)


        #update friendly bullets
        bullets = move_bullets(bullets)


        #check if the agent is hitted by enemy or bullet
        done = jnp.logical_or(done, enemy_kill_aget(fish, e_sub, state.e_bullets, sub_x, sub_y))

        #pick up divers, then move them then pick up again
        divers, diver_count = pick_up_divers(divers, state.diver_count, sub_x, sub_y)
        divers = move_divers(divers)
        divers, diver_count = pick_up_divers(divers, diver_count, sub_x, sub_y)


        #remove enemy hitted by bullets, then move the enemy then check again
        bullets, fish, e_sub, new_reward = remove_hitted_targets(bullets, fish, e_sub)
        reward = reward + new_reward
        fish = move_fish(fish, state.move_speed)
        e_sub, e_bullets = move_and_make_shot_sub(e_sub, state.e_bullets, state.move_speed)
        bullets, fish, e_sub, new_reward = remove_hitted_targets(bullets, fish, e_sub)
        reward = reward + new_reward


        #update enemy bullets
        e_bullets = move_bullets(e_bullets)

        #check if the agent is hitted by enemy or bullet
        done = jnp.logical_or(done, enemy_kill_aget(fish, e_sub, e_bullets, sub_x, sub_y))



        #update timers
        e_spawn_timer -= e_spawn_timer > 0
        d_spawn_timer -= d_spawn_timer > 0
        shot_timer -= shot_timer > 0

        #check if oxygen is over and we lose
        done = jnp.logical_or(done, state.oxygen < 0)
        #decrease oxygen by 1 if we are under water
        oxygen = state.oxygen - (sub_y > 0)

        #check if the agent is still on surface from previous frame
        surface = jnp.logical_and(state.surface, sub_y == 0)

        #if not, (the agent is arrived now on surfice) and has 0 divers it loses
        done = jnp.logical_or(done, jnp.logical_and(jnp.logical_and(sub_y == 0, jnp.logical_not(surface)), diver_count == 0))
        #if not, surface_with_people = True
        surface_with_people = jnp.logical_and(jnp.logical_and(sub_y == 0, jnp.logical_not(surface)), jnp.logical_not(diver_count == 0))

        #and we set surface Treu for next iteration
        surface = jnp.logical_or(surface_with_people, surface)

        #if the agent arrived on surface with 6 people all are removed and extra reward is given, othrwise only one diver is removed
        reward = reward + (oxygen*10//max_oxygen)*jnp.logical_and(surface_with_people, diver_count == max_divers)
        if version == 0:
            diver_count = diver_count*jnp.logical_not(surface_with_people) + surface_with_people*((diver_count < max_divers)*diver_count - 1)
        if version == 1:
            diver_count = diver_count*jnp.logical_not(surface_with_people) + (diver_count - 1)*jnp.logical_and(surface_with_people, diver_count < max_divers)
        #restore all the oxygen
        oxygen = oxygen*(1 - surface_with_people) + max_oxygen*surface_with_people
        #every time the agent arrives in surface the difficulty rises, we increase spawn and move speed of the enemy
        move_speed = state.move_speed - surface_with_people*(state.move_speed > 2)*(state.ramp_index%2 == 1)
        e_spawn_speed = state.e_spawn_speed - surface_with_people*(state.e_spawn_speed > 1)
        ramp_index = state.ramp_index + surface_with_people*jnp.logical_or(state.e_spawn_speed > 1, state.move_speed > 2)
        #build new state
      
        new_state = EnvironmentState(oxygen, diver_count, sub_x, sub_y, sub_dir, bullets,e_bullets, fish, e_sub, divers,
                            e_spawn_speed, e_spawn_timer, d_spawn_timer, move_speed, ramp_index, shot_timer, surface, where_to_spawn, key)
        
        state = jax.tree_map(lambda x,y: jnp.where(done,x,y),reset(key),new_state)  #autoreset

        return state, get_obs(state), reward, done

    if(only_step):
        return step


    jitstep = jit(vmap(step, in_axes=0, out_axes = 0))
    keys = jax.random.split(key,n_envs)
    state = vec_reset(keys)

    return state, vec_get_obs(state), jitstep, 6