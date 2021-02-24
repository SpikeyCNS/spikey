"""
Cart pole balancing game.

Florian. "Correct equations for the dynamics of the cart-pole system."
Center for Cognitive and Neural Studies(Coneural), 10 Feb 2007,
https://coneural.org/florian/papers/05_cart_pole.pdf
"""
import numpy as np

from spikey.module import Key
from spikey.games.RL.template import RL


class CartPole(RL):
    """
    Inverted pendulum / pole-cart / cart-pole reinforcement learning

         g=9.8      /
          |        / pole: Length = 1 m
          |       /
          V      /
                / θ (angle), theta_dot is angular velocity
         ______/_____
        |            | Cart: M = 1 kg
        |____________| ----> x_dot is velocity
          O        O
    L1--------x-------------------L2 x is poxition, with x limits of L1, L2)

    Actions: jerk left, jerk right (AKA bang-bang control)
    Goal: control x position of cart to keep pole close to upright,
    which is when θ = pi/2 (vertical).

    Florian. "Correct equations for the dynamics of the cart-pole system."
    Center for Cognitive and Neural Studies(Coneural), 10 Feb 2007,
    https://coneural.org/florian/papers/05_cart_pole.pdf

    Presets
    -------
    "DEFAULT": {
        "xdot_init_range": [-0.1, 0.1],
        "thetadot_init_range": [-0.1, 0.1],
        "x_init_range": [0.0, 0.0],
        "theta_init_range": [0.0, 0.0],
        "g": 9.8,
        "Mass_Cart": 1.0,
        "Mass_Pole": 0.1,
        "pole_half_length": 0.5,
        "Force_Mag": 10.0,
        "Tau": 0.0002,
        "x_max": 4.5,
        "theta_max": 0.5 * np.pi,
    }
    "FREMAUX": {
        "xdot_init_range": [-0.1, 0.1],
        "thetadot_init_range": [-0.1, 0.1],
        "x_init_range": [0.0, 0.0],
        "theta_init_range": [0.0, 0.0],
        "g": 9.8,
        "Mass_Cart": 1.0,
        "Mass_Pole": 0.1,
        "pole_half_length": 0.5,
        "Force_Mag": 10.0,
        "Tau": 0.02,  # 0.0001,
        "x_max": 2.5,
        "theta_max": 0.5 * np.pi,
    }

    Parameters
    ----------
    preset: str=PRESETS.keys(), default=DEFAULT
        Configuration preset key, default values for game parameters.
    callback: ExperimentCallback, default=None
        Callback to send relevant function call information to.
    kwargs: dict, default=None
        Game parameters for NECESSARY_KEYS. Overrides preset settings.

    Usage
    -----
    ```python
    game = CartPole(preset="DEFAULT")
    game.seed(0)

    state = game.reset()
    for _ in range(100):
        action = model.get_action(state)
        state, reward, done, info = game.step(action)
        if done:
            break

    game.close()
    ```

    ```python
    class game_template(CartPole):
        config = CartPole.PRESETS["DEFAULT"]

        config.update({  # Overrides preset values
            "param1": 1
            "param2": 2,
        })

    kwargs = {
        "param1": 0,  # Overrides game_template.config["param1"]
    }
    game = game_template(**kwargs)
    game.seed(0)

    state = game.reset()
    for _ in range(100):
        action = model.get_action(state)
        state, reward, done, info = game.step(action)
        if done:
            break

    game.close()
    ```
    """

    action_space = np.arange(-1, 1, 0.1)
    observation_space = None  # Defined in init

    metadata = {"render.modes": ["human"]}

    NECESSARY_KEYS = [
        Key("x_max", "If abs(x) > x_max: game over", float),
        Key("theta_max", "if abs(theta) > theta_max: game over", float),
        Key("x_init_range", "list[float] Range of initial x values.", list),
        Key("theta_init_range", "list[float] Range of initial theta values.", list),
        Key("xdot_init_range", "list[float] Range of initial x_dot values.", list),
        Key(
            "thetadot_init_range",
            "list[float] Range of initial theta_dot values.",
            list,
        ),
        Key("g", "Force of gravity", float, default=9.8),
        Key("Mass_Cart", "Mass of cart", float, default=1.0),
        Key("Mass_Pole", "Mass of the pole", float, default=0.1),
        Key("pole_half_length", "Half of the length of the pole", float, default=0.5),
        Key("Force_Mag", "Force of push", float, default=10.0),
        Key("Tau", "Time interval for updating the values", int, default=0.0002),
        Key("processing_time", "Amount of time network processes each stimulus", int),
    ]
    PRESETS = {
        "DEFAULT": {
            "xdot_init_range": [-0.1, 0.1],
            "thetadot_init_range": [-0.1, 0.1],
            "x_init_range": [0.0, 0.0],
            "theta_init_range": [0.0, 0.0],
            "g": 9.8,
            "Mass_Cart": 1.0,
            "Mass_Pole": 0.1,
            "pole_half_length": 0.5,
            "Force_Mag": 10.0,
            "Tau": 0.0002,
            "x_max": 4.5,
            "theta_max": 0.5 * np.pi,
        },
        "FREMAUX": {
            "xdot_init_range": [-0.1, 0.1],
            "thetadot_init_range": [-0.1, 0.1],
            "x_init_range": [0.0, 0.0],
            "theta_init_range": [0.0, 0.0],
            "g": 9.8,
            "Mass_Cart": 1.0,
            "Mass_Pole": 0.1,
            "pole_half_length": 0.5,
            "Force_Mag": 10.0,
            "Tau": 0.02,  # 0.0001,
            "x_max": 2.5,
            "theta_max": 0.5 * np.pi,
        },
    }

    def __init__(self, preset: str = "DEFAULT", callback: object = None, **kwargs):
        super().__init__(preset=preset, callback=callback, **kwargs)

        high = np.array(
            [
                self.params["x_max"],
                np.finfo(np.float32).max,
                self.params["theta_max"],
                np.finfo(np.float32).max,
            ],
            dtype=np.float32,
        )

        self.observation_space = NotImplemented

    def step(self, action: np.ndarray) -> (np.ndarray, 0, bool, {}):
        """
        Act within the environment.

        Parameters
        ----------
        action: np.ndarray
            Force pushing in each direction, eg
                [.5, .5] = 0N of force,
                [1., 0.] = 1N of force directed left,
                [0., 1.] = 1N of force directed right.

        Returns
        -------
        state: ndarray[4, float]=(x, x', theta, theta')
            State updated according to action taken.
        reward: float, = 0
            Reward given by environment.
        done: bool
            Whether the game is done or not.
        info: dict, = {}
            Information of environment.

        Usage
        -----
        ```python
        game = Cartpole(preset="DEFAULT")
        game.seed(0)

        state = game.reset()
        for _ in range(100):
            action = model.get_action(state)
            state, reward, done, info = game.step(action)
            if done:
                break

        game.close()
        ```
        """
        PoleMass_Length = self.params["Mass_Pole"] * self.params["pole_half_length"]
        Total_Mass = self.params["Mass_Cart"] + self.params["Mass_Pole"]
        Fourthirds = 4.0 / 3.0

        #
        force = np.dot(action, [-1, 1]) * self.params["Force_Mag"]
        # force = [-1, 1][np.argmax(action)] * self.params['Force_Mag']

        assert force < self.params["Force_Mag"] * 1.2, "Action force too high."

        x, x_dot, theta, theta_dot = self._state

        temp = (
            force + PoleMass_Length * theta_dot * theta_dot * np.sin(theta)
        ) / Total_Mass

        thetaacc = (self.params["g"] * np.sin(theta) - np.cos(theta) * temp) / (
            self.params["pole_half_length"]
            * (
                Fourthirds
                - self.params["Mass_Pole"] * np.cos(theta) * np.cos(theta) / Total_Mass
            )
        )

        xacc = temp - PoleMass_Length * thetaacc * np.cos(theta) / Total_Mass

        # Update the four state variables, using Euler's method:
        # https://en.wikipedia.org/wiki/Euler_method
        x = x + self.params["Tau"] * x_dot
        x_dot = x_dot + self.params["Tau"] * xacc
        theta = theta + self.params["Tau"] * theta_dot
        theta_dot = theta_dot + self.params["Tau"] * thetaacc

        state_new = np.array([x, x_dot, theta, theta_dot])

        ##
        x, x_dot, theta, theta_dot = state_new

        f = abs(x) > self.params["x_max"] or abs(theta) > self.params["theta_max"]

        rwd = 0
        info = {}

        self.callback.game_step(action, self._state, state_new, rwd, f, info)
        self._state = state_new
        return state_new, rwd, f, info

    def reset(self) -> np.ndarray:
        """
        Reset environment.

        Returns
        -------
        ndarray[4, float]=(x, x', theta, theta') Initial game state randomly generated in bounds,
        (*x_init_range * [-1 or 1], *x_dot_init_range * [-1 or 1], *theta_init_range * [-1 or 1], *thetadot_init_range * [-1 or 1]).

        Usage
        -----
        ```python
        game = Cartpole(preset="DEFAULT")
        game.seed(0)

        state = game.reset()
        ```
        """
        x = np.random.uniform(*self.params["x_init_range"]) * np.random.choice([-1, 1])
        x_dot = np.random.uniform(*self.params["xdot_init_range"]) * np.random.choice(
            [-1, 1]
        )
        theta = np.random.uniform(*self.params["theta_init_range"]) * np.random.choice(
            [-1, 1]
        )
        theta_dot = np.random.uniform(
            *self.params["thetadot_init_range"]
        ) * np.random.choice([-1, 1])

        s = np.array([x, x_dot, theta, theta_dot])

        self.callback.game_reset(s)
        self._state = s
        return s

    def render(self, states: np.ndarray, mode: str = "human"):
        """Renders the environment.
        The set of supported modes varies per environment. (And some
        environments do not support rendering at all.) By convention,
        if mode is:
        - human: render to the current display or terminal and
          return nothing. Usually for human consumption.
        - rgb_array: Return an numpy.ndarray with shape (x, y, 3),
          representing RGB values for an x-by-y pixel image, suitable
          for turning into a video.
        - ansi: Return a string (str) or StringIO.StringIO containing a
          terminal-style text representation. The text can include newlines
          and ANSI escape sequences (e.g. for colors).
        Note:
            Make sure that your class's metadata 'render.modes' key includes
              the list of supported modes. It's recommended to call super()
              in implementations to use the functionality of this method.
        Example:
        class MyEnv(Env):
            metadata = {'render.modes': ['human', 'rgb_array']}
            def render(self, mode='human'):
                if mode == 'rgb_array':
                    return np.array(...) # return RGB frame suitable for video
                elif mode == 'human':
                    ... # pop up a window and render
                else:
                    super(MyEnv, self).render(mode=mode) # just raise an exception

        Parameters
        ----------
        mode (str): the mode to render with

        Usage
        -----
        ```python
        game = Cartpole(preset="DEFAULT")
        game.seed(0)

        state = game.reset()
        for _ in range(100):
            action = model.get_action(state)
            state, reward, done, info = game.step(action)
            if done:
                break

        game.render()
        game.close()
        ```
        """

        def initGraph():
            """
            Init for animated graph below
            """
            line.set_data([], [])
            return (line,)

        def animate(i):
            """
            Each step/refresh of the animatd graph. This sort of gets "looped".
            """
            thisx = [x1[i], x2[i]]
            thisy = [y1, y2[i]]
            line.set_data(thisx, thisy)
            return (line,)

        import matplotlib.pyplot as plt
        import matplotlib.animation as animation

        toPlot = states

        xList = [state[0] for state in toPlot]
        thetaList = [state[2] for state in toPlot]
        x1 = xList
        y1 = 0
        x2 = 1 * np.sin(thetaList) + x1
        y2 = 1 * np.cos(thetaList) + y1

        fig = plt.figure()
        ax = plt.axes(xlim=(-4, 4), ylim=(-0.25, 1.25))
        ax.grid()
        (line,) = ax.plot([], [], "o-", lw=2)
        animation.FuncAnimation(
            fig,
            animate,
            np.arange(1, len(xList)),
            interval=30,
            blit=True,
            init_func=initGraph,
        )
        plt.show()
