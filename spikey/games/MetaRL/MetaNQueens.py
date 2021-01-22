"""
The N Queens game for MetaRL
"""
import numpy as np

try:
    from template import MetaRL
except ImportError:
    from spikey.games.MetaRL.template import MetaRL


class MetaNQueens(MetaRL):
    """
    Meta RL game to try and place n[1, 8] queens on a chess board
    without any of them being able to attack another in one move.

    92 distinct solutions / 4 billion possibilities w/ 8 queens.

    Possible to change to n rooks/bishops to see if can evolve.

    This game works best trying to evolve x and y permutations among
    queens.
    """

    GENOTYPE_CONSTRAINTS = {}  ## Defined in __init__

    PIECE_MOVES = {}

    def __init__(self, n_queens):
        super().__init__()
        if n_queens > 8 or n_queens < 1:
            raise ValueError(f"n_queens must be in range [1, 8], not {n_queens}!")

        self.letters = ["a", "b", "c", "d", "e", "f", "g", "h"][:n_queens]

        keys = [first + second for second in ["x", "y"] for first in self.letters]

        ## +1 for randint functionality
        self.GENOTYPE_CONSTRAINTS = {key: list(range(8)) for key in keys}

    @staticmethod
    def setup_game():
        """
        Setup game.

        Returns
        -------
        Data for board, 0, 0 in top right.
        """
        horizontals = np.zeros(8)
        verticals = np.zeros(8)
        ldiagonals = np.zeros(15)  # \
        rdiagonals = np.zeros(15)  # /

        return horizontals, verticals, ldiagonals, rdiagonals

    @staticmethod
    def run_move(board, move):
        """
        Execute action.

        Parameters
        ----------
        board:

        move: (x, y), [0, 7]
            X and Y coordinate to place queen.

        Returns
        -------
        Updated board.
        """
        horizontals, verticals, ldiagonals, rdiagonals = board
        x, y = move

        horizontals[x] += 1
        verticals[y] += 1
        ldiagonals[x + y] += 1
        rdiagonals[7 - x + y] += 1

        return horizontals, verticals, ldiagonals, rdiagonals

    def get_fitness(
        self, genotype, log=None, filename=None, reduced_logging=True, q=None
    ):
        """
        Evaluate a genotype.

        https://kushalvyas.github.io/gen_8Q.html
        """
        ## Initialize game
        board = self.setup_game()

        ## Pick moves
        for letter in self.letters:
            move = (genotype[letter + "x"], genotype[letter + "y"])

            board = self.run_move(board, move)

        ## Evaluate
        clashes = 0

        for item in board:
            clashes += np.sum(item[item > 1] - 1)

        fitness = 28 - clashes
        terminate = clashes == 0

        if q is not None:
            q.put((genotype, fitness, terminate))

        return fitness, terminate
