"""
Cart Pole balancing game.
"""
import numpy as np

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
    which is when θ = pi/2 (vertical); for visual see:
    https://upload.wikimedia.org/wikipedia/commons/9/9a/Degree-Radian_Conversion.svg
    """

    action_space = np.arange(-1, 1, .1)
    observation_space = None  # Defined in init

    metadata = {"render.modes": ["human"]}

    CONFIG_DESCRIPTIONS = {
        "n_outputs": "int Number of outputs to decode.",
        "x_dot_noise": "list[float] Range of initial x_dot values.",
        "theta_dot_noise": "list[float] Range of initial theta_dot values.",
        "g": "Force of gravity",
        "Mass_Cart": "Mass of cart",
        "Mass_Pole": "Mass of the pole",
        "Length": "Half of the length of the pole",
        "Force_Mag": "Force of push",
        "Tau": "Time interval for updating the values",
        "x_max": "If abs(x) > x_max: game over",
        "theta_max": "if abs(theta) > theta_max: game over",
        "x_initial": "Initial x value.",
        "theta_initial": "Initial theta value.",
        "processing_time": "Amount of time network processes each stimulus",
    }
    PRESETS = {
        "DEFAULT": {
            "n_inputs": 2,
            "n_outputs": 10,
            "x_dot_noise": [-0.1, 0.1],
            "theta_dot_noise": [-0.1, 0.1],
            "x_noise": [0.0, 0.0],
            "theta_noise": [0.0, 0.0],
            "g": 9.8,
            "Mass_Cart": 1.0,
            "Mass_Pole": 0.1,
            "Length": 0.5,
            "Force_Mag": 10.0,
            "Tau": 0.0002,
            "x_max": 4.5,
            "theta_max": 0.5 * np.pi,
        },
        "FREMAUX": {
            "n_inputs": 1050,
            "n_outputs": 80,
            "x_dot_noise": [-0.1, 0.1],
            "theta_dot_noise": [-0.1, 0.1],
            "x_noise": [0.0, 0.0],
            "theta_noise": [0.0, 0.0],
            "g": 9.8,
            "Mass_Cart": 1.0,
            "Mass_Pole": 0.1,
            "Length": 0.5,
            "Force_Mag": 10.0,
            "Tau": 0.02,  # 0.0001,
            "x_max": 2.5,
            "theta_max": 0.5 * np.pi,
        },
    }

    def __init__(self, preset="DEFAULT", callback=None, **kwargs):
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

    def step(self, action):
        """
        Takes an action (0 or 1) and
        the current values of the four state variables and
        updates values by estimating the state,
        Tau seconds later.
        The actual physics are here.
        """
        PoleMass_Length = self.params["Mass_Pole"] * self.params["Length"]
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
            self.params["Length"]
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

    def reset(self):
        """
        Start state for every episode
        """
        x = np.random.uniform(*self.params["x_noise"]) * np.random.choice([-1, 1])
        x_dot = np.random.uniform(*self.params["x_dot_noise"]) * np.random.choice(
            [-1, 1]
        )
        theta = np.random.uniform(*self.params["theta_noise"]) * np.random.choice(
            [-1, 1]
        )
        theta_dot = np.random.uniform(
            *self.params["theta_dot_noise"]
        ) * np.random.choice([-1, 1])

        s = np.array([x, x_dot, theta, theta_dot])

        self.callback.game_reset(s)
        self._state = s
        return s

    def render(self, states, mode="human"):
        """
        Render set of cartpole states.
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
