"""
MetaRL implementation of the N Queens game.
"""
import numpy as np

try:
    from template import MetaRL
except ImportError:
    from spikey.games.MetaRL.template import MetaRL


class MetaNQueens(MetaRL):
    """
    Game to try and place a number of queen chess pieces on a chess
    board without any of them being to attack another in the same move.

    92 distinct solutions out of 4 billion possibilities w/ 8 queens.

    GENOTYPE_CONSTRAINTS
    --------------------
    for i in range(n_agents):
        xi: int in {0, 7} X position of queen i.
        yi: int in {0, 7} Y position of queen i.

    Parameters
    ----------
    n_queens: int in {1..8}
        Number of queens agent needs to place on board.

    Usage
    -----
    ```python
    metagame = MetaNQueens()
    game.seed(0)

    for _ in range(100):
        genotype = [{}, ...]
        fitness, done = metagame.get_fitness(genotype)

        if done:
            break

    game.close()
    ```

    ```python
    metagame = MetaNQueens(**metagame_config)
    game.seed(0)

    population = Population(... metagame, ...)
    # population main loop
    ```
    """

    GENOTYPE_CONSTRAINTS = {}  ## Defined in __init__

    PIECE_MOVES = {}

    def __init__(self, n_queens: int):
        super().__init__()
        if n_queens > 8 or n_queens < 1:
            raise ValueError(f"n_queens must be in range [1, 8], not {n_queens}!")

        self.letters = ["a", "b", "c", "d", "e", "f", "g", "h"][:n_queens]

        keys = [first + second for second in ["x", "y"] for first in self.letters]

        ## +1 for randint functionality
        self.GENOTYPE_CONSTRAINTS = {key: list(range(8)) for key in keys}

    @staticmethod
    def setup_game() -> list:
        """
        Setup game.

        Returns
        -------
        list Initial board state, number of queens in each horizontal, vertical and diagonal line.
        """
        horizontals = np.zeros(8)
        verticals = np.zeros(8)
        ldiagonals = np.zeros(15)  # \
        rdiagonals = np.zeros(15)  # /

        return horizontals, verticals, ldiagonals, rdiagonals

    @staticmethod
    def run_move(board: list, move: tuple) -> list:
        """
        Execute action.

        Parameters
        ----------
        board: list
            Number of queens across each horizontal, vertical and diagonal line.
        move: (x, y) in [0, 7]
            X and Y coordinate to place queen.

        Returns
        -------
        [horizontals: list, verticals: list, ldiagonals: list, rdiagonals: list] Updated board.
        """
        horizontals, verticals, ldiagonals, rdiagonals = board
        x, y = move

        horizontals[x] += 1
        verticals[y] += 1
        ldiagonals[x + y] += 1
        rdiagonals[7 - x + y] += 1

        return horizontals, verticals, ldiagonals, rdiagonals

    def get_fitness(
        self,
        genotype: dict,
        logging: bool = True,
        **kwargs,
    ) -> (float, bool):
        """
        Evaluate the fitness of a genotype.

        Parameters
        ----------
        genotype: dict
            Dictionary with values for each key in GENOTYPE_CONSTRAINTS.
        logging: bool, default=True
            Whether or not to log results to file.
        kwargs: dict, default=None
            Logging and experiment logging keyword arguments.

        Returns
        -------
        fitness: float
            Fitness of genotype given.
        done: bool
            Whether termination condition has been reached or not.

        Usage
        -----
        ```python
        metagame = MetaNQueens()
        game.seed(0)

        for _ in range(100):
            genotype = [{}, ...]
            fitness, done = metagame.get_fitness(genotype)

            if done:
                break

        game.close()
        ```
        """
        board = self.setup_game()

        for letter in self.letters:
            move = (genotype[letter + "x"], genotype[letter + "y"])

            board = self.run_move(board, move)

        clashes = 0

        for item in board:
            clashes += np.sum(item[item > 1] - 1)

        fitness = 28 - clashes
        terminate = clashes == 0

        return fitness, terminate
