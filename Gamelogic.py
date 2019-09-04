import keyboard
import time
import math
import random
import pygame
import Gamerendering


class Player:
    def __init__(self, ypos, human=True, agent_type="Human", agent=False, rendering=False):
        self.x_pos = 50
        self.y_pos = ypos
        self.x_vel = 0
        self.y_vel = 0
        self.radius = 17
        self.angle = 0
        self.alive = True
        self.human = human
        self.agent_type = agent_type
        self.agent = agent
        self.time_since_jumped = 100

        self.delete_from_render = False
        self.rendering = rendering

        self.start_y = ypos

    def reset(self):
        self.__init__(self.start_y, human=self.human, agent_type=self.agent_type, agent=self.agent,
                      rendering=self.rendering)

    def move(self, environment, velocity):
        if self.time_since_jumped < 10:
            self.time_since_jumped += 1
            return 0
        # This is a human player
        if self.human:
            if keyboard.is_pressed('w'):
                self.y_vel = 7
        # This is an agent
        else:
            if self.agent_type == "NEAT":
                env = [environment[0], environment[1]]
                env.append((self.y_pos - 135) / 525)
                env.append(self.y_vel / 20)
                if self.agent.action(env) == 1:
                    self.y_vel = 7
            elif self.agent_type == "DQN":
                action = self.agent.act_store(environment[-1], environment[-2], not self.alive)
                if action == 1:
                    self.y_vel = 7

        self.angle = math.atan(self.y_vel / velocity) * 180 / 3.14159265358979 / 2
        if self.y_vel == 7:
            self.time_since_jumped = 0


class Pipe:
    def __init__(self, x_pos, width):
        self.start_x = x_pos
        self.x_pos = x_pos
        self.y_pos = self.y_pos = random.randint(200, 490)
        self.width = width
        self.opening_width = 140

        # Initiating sound
        # pygame.init()
        # self.effect_point = pygame.mixer.Sound("Soundeffects/sfx_point.wav")

    def reset(self):
        self.x_pos = self.start_x
        self.y_pos = random.randint(200, 490)

    def move(self, velocity):
        moved = False
        if self.x_pos + self.width < 0:
            self.x_pos += 250 * 3
            self.y_pos = random.randint(200, 490)
            moved = True
        self.x_pos -= velocity
        return moved

    def distance(self, x1, y1, x2, y2):
        return math.sqrt(math.pow(x1 - x2, 2) + math.pow(y1 - y2, 2))

    def colliding(self, player):
        # First wall
        if self.x_pos < player.x_pos + player.radius < self.x_pos + self.width and \
                (self.y_pos > player.y_pos or self.y_pos + self.opening_width < player.y_pos):
            return True

        # Corners
        if self.distance(player.x_pos, player.y_pos, self.x_pos, self.y_pos) < player.radius or \
                self.distance(player.x_pos, player.y_pos, self.x_pos,
                              self.y_pos + self.opening_width) < player.radius or \
                self.distance(player.x_pos, player.y_pos, self.x_pos + self.width, self.y_pos) < player.radius or \
                self.distance(player.x_pos, player.y_pos, self.x_pos + self.width,
                              self.y_pos + self.opening_width) < player.radius:
            return True

        # Top of pipes
        if self.x_pos < player.x_pos < self.x_pos + self.width and \
                (self.y_pos > player.y_pos - player.radius or
                 self.y_pos + self.opening_width < player.y_pos + player.radius):
            return True
        return False


class Gamelogic:
    def __init__(self, new_window, rendering=False):
        self.acceleration = -0.5
        self.velocity = 2.3
        self.render = Gamerendering.Gamerendering(self.velocity, new_window=new_window)
        self.rendering = rendering
        self.open_new_window = False
        self.render_speed = 1

        # pygame.init()
        # pygame.mixer.Sound("Soundeffects/sfx_hit.wav")

        self.players = []
        self.deleted_players = []
        self.pipes = []
        for x in range(3):
            self.pipes.append(Pipe(400 + 250 * x, 93))

        if new_window:
            for pipe in self.pipes:
                self.render.add_pipe(pipe)

    def add_player(self, human=True, agent_type="Human", player=False, position=350, render=False):
        wrapper_player = Player(position, human=human, agent_type=agent_type, agent=player, rendering=render)
        self.players.append(wrapper_player)
        if render:
            self.render.add_player(wrapper_player)

    def reset_game(self, delete_players=False):
        self.render.reset(self.velocity)
        if delete_players:
            self.players = []
        else:
            for player in self.players:
                player.reset()
                self.render.add_player(player)
        for pipe in self.pipes:
            pipe.reset()
            self.render.add_pipe(pipe)

    # Choose to render or not and at what speed
    def user_input(self, can_open_window=False):
        # # Render
        # if keyboard.is_pressed('r'):
        #     self.rendering = True
        # # Simulate
        # elif keyboard.is_pressed('s'):
        #     self.rendering = False
        # # Set speed
        # for x in range(1, 10):
        #     if keyboard.is_pressed(str(x)):
        #         self.render_speed = x
        #         return
        # if keyboard.is_pressed('ctrl'):
        #     self.open_new_window = True
        pass

    def run_game(self, to_render, generation, return_image=False):
        # Initializing the variables
        action_time = 0
        action_calculate_time = 0
        render_time = 0
        move_players_time = 0
        move_pipes_time = 0
        status_time = 0
        pipes_passed = 0
        players_alive = len(self.players)
        current_fitness = 0
        self.user_input(can_open_window=True)
        if self.open_new_window:
            self.render = Gamerendering.Gamerendering(self.velocity, new_window=True)
            self.open_new_window = False
            for pipe in self.pipes:
                self.render.add_pipe(pipe)
            for player in self.players:
                if player.rendering:
                    self.render.add_player(player)
        while True:
            passed_pipe_now = False
            self.user_input()

            current_fitness += 1
            all_dead = True

            # Move players
            now = time.time()
            for player in self.players:
                if player.alive:
                    player.y_vel += self.acceleration
                    player.y_pos += player.y_vel
                else:
                    player.x_vel = -self.velocity
                    player.x_pos += player.x_vel
            move_players_time += time.time() - now

            # Move pipes
            now = time.time()
            for pipe in self.pipes:
                if pipe.move(self.velocity):
                    pipes_passed += 1
                    passed_pipe_now = True
                    if pipes_passed > 1000:
                        print("finished")
                        return
            self.pipes.sort(key=lambda x: x.x_pos)
            move_pipes_time += time.time() - now

            #
            env = []
            for pipe in self.pipes:
                env.append(pipe.x_pos / 700)
                env.append((pipe.y_pos - 200) / 290)

            # Checking the alive status of players
            now = time.time()
            for player in self.players:
                # Player is alive and died
                if player.alive and \
                            (len(self.pipes) > 0 and self.pipes[0].colliding(
                            player) or player.y_pos + player.radius > 672 or player.y_pos - player.radius < 135):
                    player.alive = False
                    players_alive -= 1
                # Player is dead, moving backwards, out of screen and being rendered
                elif not player.alive and player.x_pos < -10 and player.rendering:
                    player.rendering = False
                    player.delete_from_render = True
                    to_render += 1
                    self.deleted_players.append(player)
                    self.players.remove(player)
                # Player is alive, is not being rendered and the players rendered is lower than max
                elif player.alive and to_render > 0 and not player.rendering:
                    player.rendering = True
                    to_render -= 1
                    self.render.add_player(player)
            status_time += time.time() - now

            # Render the player
            now = time.time()
            if self.rendering and current_fitness % self.render_speed == 0:
                # Appending image to environment and rendering image
                env.append(0.1)
                env.append(
                    self.render.draw(pipes_passed, current_fitness, players_alive, self.velocity * self.render_speed,
                                     generation, return_image=return_image))
            render_time += time.time() - now

            # Next move for players
            now = time.time()
            for player in self.players:
                if player.alive:
                    now1 = time.time()
                    player.move(env, self.velocity)
                    all_dead = False
                    action_calculate_time += time.time() - now1
            action_time += time.time() - now

            # If finished: print statistics and return
            if all_dead:
                print("Actiontime: ", action_calculate_time, "/", action_time, "\trendertime: ", render_time,
                      "\tplayer_move: ", move_players_time, "\tpipe_move: ", move_pipes_time,
                      "\tstatus: ", status_time, )
                print("Pipes passed: " + str(pipes_passed))
                return current_fitness

# game = Gamelogic(True, rendering=True)
# game.add_player(position=400, render=True)
# while True:
#     game.run_game(True, 1)
#     time.sleep(1)
#     game.reset_game()
