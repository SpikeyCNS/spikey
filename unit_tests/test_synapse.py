"""
Tests on the synapse dynamics functions.
"""
import unittest
from unittest import mock

import numpy as np

import spikey.snn.synapse as synapse


class TestSynapseRLSTDPET(unittest.TestCase):
    """
    RLSTDPET based synapse dynamics tests.
    """

    def run_all_types(func):
        """
        Wrapper creating subtest for every type of object.
        """

        def run_all(self):
            for obj in [synapse.RLSTDPET]:
                with self.subTest(i=obj.__name__):
                    self._get_synapse = self._set_obj(obj)

                    func(self)

        return run_all

    def _set_obj(self, obj):
        """
        Create generator that will render only specific object.
        """

        def _get_synapse(**kwargs):
            config = {
                "learning_rate": 1,
                "learning_rate": 1.0,
                "max_trace": 1000,
                "max_weight_update": 1.0,
                "max_weight": 1,
                "n_inputs": 0,
                "n_neurons": 100,
                "stdp_window": 100,
                "trace_decay": 0.1,
                "n_outputs": 0,
            }
            config.update(kwargs)

            n_inputs = config["n_inputs"]
            n_neurons = config["n_neurons"]

            # w = 1. - np.random.power(a=8, size=(n_inputs + n_neurons, n_neurons))
            # diagonal = np.arange(n_neurons)
            # w[diagonal+n_inputs, diagonal] = 0.
            w = np.ma.array(np.zeros((n_inputs + n_neurons, n_neurons)))
            w = type(
                "WeightMatrix",
                (object,),
                {
                    "_matrix": w,
                    "shape": w.shape,
                    **{f"_{key}": value for key, value in config.items()},
                },
            )

            if "w" not in config:
                config.update({"w": w})

            synapse = obj(**config)

            return synapse

        return _get_synapse

    @run_all_types
    def test_reset(self):
        """
        Testing synapse.reset.

        Effects
        -------
        All synapse tables should be reset.
        """
        for n_inputs, n_neurons in [(0, 100), (100, 0), (50, 50)]:
            synapse = self._get_synapse(n_inputs=n_inputs, n_neurons=n_neurons)

            synapse.reset()

            self.assertEqual(synapse.weights.shape, (n_inputs + n_neurons, n_neurons))

            if isinstance(synapse.trace, np.ndarray):
                self.assertEqual(
                    synapse.trace.shape, (n_inputs + n_neurons, n_inputs + n_neurons)
                )

                if hasattr(synapse, "trace_p"):
                    self.assertEqual(
                        synapse.trace_p.shape,
                        (n_inputs + n_neurons, n_inputs + n_neurons),
                    )
                    self.assertEqual(
                        synapse.trace_m.shape,
                        (n_inputs + n_neurons, n_inputs + n_neurons),
                    )

    @run_all_types
    def test_reward(self):
        """
        Testing synapse.reward.

        Parameters
        ----------
        rwd: float
            Reward given to the network.

        Settings
        --------
        _learning_rate: float
            Learning rate of network.

        Effects
        -------
        The synapse weight table should be updated by synapse
        trace * rwd. Trace is capped by learning_rate.
        """
        synapse = self._get_synapse()
        synapse.reset()

        for reward in [0, 0.25, 0.5, 1]:
            if hasattr(synapse, "trace_p"):
                synapse.trace_p = 0.5 * np.ones(shape=(20, 20))
                synapse.trace_m = -0.25 * np.ones(shape=(20, 20))

                synapse.reward(reward)

                for idx, value in np.ndenumerate(synapse.trace_p):
                    self.assertEqual(value, 0.25 + reward)
                    self.assertEqual(synapse.trace_m[idx])
            else:
                synapse.trace = 0.25 * np.ones(shape=(20, 20))

                synapse.reward(reward)

                for _, value in np.ndenumerate(synapse.trace):
                    self.assertEqual(value, 0.25 + reward)

    @run_all_types
    def test_decay_trace(self):
        """
        Testing synapse._decay_trace.

        Settings
        --------
        _trace_decay: float
            Trace decay rate.
        _learning_rate: float
            Maximum trace value.

        Effects
        -------
        Updated trace value or values, capped at _learning_rate.
        """
        ## Assert decay rate 0 does not decay.
        synapse = self._get_synapse(trace_decay=0)
        synapse.reset()

        for init_trace in range(0, 100, 5):
            if hasattr(synapse, "trace_p"):
                synapse.trace_p = init_trace * np.ones(shape=(100, 100))
                synapse.trace_m = init_trace * np.ones(shape=(100, 100))
            else:
                synapse.trace = init_trace * np.ones(shape=(100, 100))

            synapse._decay_trace()

            if hasattr(synapse, "trace_p"):
                self.assertListEqual(
                    [init_trace] * (100 * 100), list(synapse.trace_p.flatten())
                )
                self.assertListEqual(
                    [init_trace] * (100 * 100), list(synapse.trace_m.flatten())
                )
            else:
                self.assertListEqual(
                    [init_trace] * (100 * 100), list(synapse.trace.flatten())
                )

        ## Assert decreases at a rate relative to decay rate.
        init_trace = 100 * np.ones(shape=(100, 100))
        changes = []

        for rate in [0.1, 0.2, 0.5, 1]:
            synapse = self._get_synapse(trace_decay=rate)
            synapse.reset()

            if hasattr(synapse, "trace_p"):
                synapse.trace_p = init_trace * np.ones(shape=(100, 100))
                synapse.trace_m = init_trace * np.ones(shape=(100, 100))
            else:
                synapse.trace = init_trace * np.ones(shape=(100, 100))

            synapse._decay_trace()

            if hasattr(synapse, "trace_p"):
                changes.append(np.average(synapse.trace_p))

                for pos, value in np.ndenumerate(init_trace):
                    self.assertGreater(value, synapse.trace_p[pos])

            else:
                changes.append(np.average(synapse.trace))

                for pos, value in np.ndenumerate(init_trace):
                    self.assertGreater(value, synapse.trace[pos])

        for i, value in enumerate(changes[:-1]):
            self.assertGreater(value, changes[i + 1])

    @run_all_types
    def test_apply_stdp(self):
        """
        Testing synapse.update_trace.

        Parameters
        ----------
        spike_log: np.array(time, neurons), 0 or 1
            A history of neuron firings.
        inhibitories: list[int], -1 or 1
            Neuron polarities.

        Settings
        --------
        _stdp_window: int
            Window of time that stdp takes effect.

        Effects
        -------
        Trace value or values updated according to stdp suggestions based
        on dt, polarity and _stdp_window.
        """
        ## Ensure update trace is called at the correct locations with correct dt.
        N_NEURONS = 100
        STDP_WINDOW = 5

        for START_VAL in [-2, -1, 0, 1, 2]:
            synapse = self._get_synapse(
                n_inputs=0, n_neurons=N_NEURONS, stdp_window=STDP_WINDOW
            )
            synapse.reset()

            trace = START_VAL
            synapse.trace = trace

            if hasattr(synapse, "trace_p"):
                synapse.trace_p = START_VAL if START_VAL > 0 else 0
                synapse.trace_m = 0 if START_VAL > 0 else START_VAL

            spike_log = np.zeros(shape=(STDP_WINDOW, N_NEURONS))
            spike_log[0] = 1
            spike_log[-1, np.where([i % 2 == 1 for i in range(N_NEURONS)])] = 1

            synapse._apply_stdp(spike_log, np.ones(N_NEURONS))

            EVENS = np.where([i % 2 == 0 for i in range(N_NEURONS)])[0]
            ODDS = np.where([i % 2 == 1 for i in range(N_NEURONS)])[0]

            if hasattr(synapse, "trace_p"):
                synapse.trace = synapse.trace_p + synapse.trace_m

            ## TODO check if w increased or decreased
            """
            for o in ODDS:
                for e in EVENS:
                    self.assertLess(synapse.weights._matrix[o][e], START_VAL)
                    self.assertGreater(synapse.weights._matrix[e][o], START_VAL)
            for i in ODDS:
                for j in ODDS:
                    self.assertAlmostEqual(synapse.weights._matrix[i][j], START_VAL)
            for i in EVENS:
                for j in EVENS:
                    self.assertAlmostEqual(synapse.weights._matrix[i][j], START_VAL)
            """

        ## See if abides by given inhibitories. odds inh
        for START_VAL in [-2, -1, 0, 1, 2]:
            synapse = self._get_synapse(n_neurons=N_NEURONS, stdp_window=STDP_WINDOW)
            synapse.reset()

            trace = START_VAL
            synapse.trace = trace

            if hasattr(synapse, "trace_p"):
                synapse.trace_p = START_VAL if START_VAL > 0 else 0
                synapse.trace_m = 0 if START_VAL > 0 else START_VAL

            spike_log = np.zeros(shape=(STDP_WINDOW, N_NEURONS))
            spike_log[0] = 1
            spike_log[-1, np.where([i % 2 == 1 for i in range(N_NEURONS)])] = 1

            inhibitories = np.where(np.arange(N_NEURONS) % 2, -1, 1)

            synapse._apply_stdp(spike_log, inhibitories)

            if hasattr(synapse, "trace_p"):
                synapse.trace = synapse.trace_p + synapse.trace_m

            ## TODO check if w increased or decreased
            """
            EVENS = np.where([i%2 == 0 for i in range(N_NEURONS)])[0]
            ODDS = np.where([i%2 == 1 for i in range(N_NEURONS)])[0]
            for o in ODDS:
                for e in EVENS:
                    self.assertGreater(synapse.trace[o][e], START_VAL)
                    self.assertGreater(synapse.trace[e][o], START_VAL)
            for i in ODDS:
                for j in ODDS:
                    self.assertAlmostEqual(synapse.trace[i][j], START_VAL)
            for i in EVENS:
                for j in EVENS:
                    self.assertAlmostEqual(synapse.trace[i][j], START_VAL)
            """

        ## See if abides by given inhibitories pt 2. evens inh
        for START_VAL in [-2, -1, 0, 1, 2]:
            synapse = self._get_synapse(n_neurons=N_NEURONS, stdp_window=STDP_WINDOW)
            synapse.reset()

            trace = START_VAL
            synapse.trace = trace

            if hasattr(synapse, "trace_p"):
                synapse.trace_p = START_VAL if START_VAL > 0 else 0
                synapse.trace_m = 0 if START_VAL > 0 else START_VAL

            spike_log = np.zeros(shape=(STDP_WINDOW, N_NEURONS))
            spike_log[0] = 1
            spike_log[-1, np.where([i % 2 == 1 for i in range(N_NEURONS)])] = 1

            inhibitories = np.where(np.arange(N_NEURONS) % 2, 1, -1)

            synapse._apply_stdp(spike_log, inhibitories)

            if hasattr(synapse, "trace_p"):
                synapse.trace = synapse.trace_p + synapse.trace_m

            ## TODO check if w increased or decreased
            """
            EVENS = np.where([i%2 == 0 for i in range(N_NEURONS)])[0]
            ODDS = np.where([i%2 == 1 for i in range(N_NEURONS)])[0]
            for o in ODDS:
                for e in EVENS:
                    self.assertLess(synapse.trace[o][e], START_VAL)
                    self.assertLess(synapse.trace[e][o], START_VAL)
            for i in ODDS:
                for j in ODDS:
                    self.assertAlmostEqual(synapse.trace[i][j], START_VAL)
            for i in EVENS:
                for j in EVENS:
                    self.assertAlmostEqual(synapse.trace[i][j], START_VAL)
            """

        ## See if abides by given inhibitories pt 3. all inh
        for START_VAL in [-2, -1, 0, 1, 2]:
            synapse = self._get_synapse(n_neurons=N_NEURONS, stdp_window=STDP_WINDOW)
            synapse.reset()

            trace = START_VAL
            synapse.trace = trace

            if hasattr(synapse, "trace_p"):
                synapse.trace_p = START_VAL if START_VAL > 0 else 0
                synapse.trace_m = 0 if START_VAL > 0 else START_VAL

            spike_log = np.zeros(shape=(STDP_WINDOW, N_NEURONS))
            spike_log[0] = 1
            spike_log[-1, np.where([i % 2 == 1 for i in range(N_NEURONS)])] = 1

            inhibitories = -np.ones(N_NEURONS)

            synapse._apply_stdp(spike_log, inhibitories)

            if hasattr(synapse, "trace_p"):
                synapse.trace = synapse.trace_p + synapse.trace_m

            ## TODO check if w increased or decreased
            """
            EVENS = np.where([i%2 == 0 for i in range(N_NEURONS)])[0]
            ODDS = np.where([i%2 == 1 for i in range(N_NEURONS)])[0]
            for o in ODDS:
                for e in EVENS:
                    self.assertGreater(synapse.trace[o][e], START_VAL)
                    self.assertLess(synapse.trace[e][o], START_VAL)
            for i in ODDS:
                for j in ODDS:
                    self.assertAlmostEqual(synapse.trace[i][j], START_VAL)
            for i in EVENS:
                for j in EVENS:
                    self.assertAlmostEqual(synapse.trace[i][j], START_VAL)
            """

        ## Ensure inhibitory doesnt mess with time diff
        N_NEURONS = 3

        for STDP_WINDOW in range(1, 5):
            for START_VAL in [-2, -1, 0, 1, 2]:
                synapse = self._get_synapse(
                    n_neurons=N_NEURONS, stdp_window=STDP_WINDOW
                )
                synapse.reset()

                trace = START_VAL
                synapse.trace = trace

                if hasattr(synapse, "trace_p"):
                    synapse.trace_p = START_VAL if START_VAL > 0 else 0
                    synapse.trace_m = 0 if START_VAL > 0 else START_VAL

                spike_log = np.zeros(shape=(STDP_WINDOW, N_NEURONS))
                spike_log[0] = [1, 1, 1]
                spike_log[-1] = [0, 1, 1]

                inhibitories = np.array([1, 1, -1])

                synapse._apply_stdp(spike_log, inhibitories)

                ## TODO check if w increased or decreased
                """
                # same magnitude 1 inh 1 exc see if same delta
                # For exitatory
                d = START_VAL - synapse.trace[0, 1]
                self.assertAlmostEqual(synapse.trace[0, 1] + d, START_VAL)
                self.assertAlmostEqual(synapse.trace[1, 0] - d, START_VAL)

                # For inhibitory
                d = START_VAL - synapse.trace[0, 2]
                self.assertAlmostEqual(synapse.trace[0, 2] + d, START_VAL)
                self.assertAlmostEqual(synapse.trace[2, 0] + d, START_VAL)
                """

        ## Ensure longer/shorter time diff does not mess with results
        N_NEURONS = 2
        STDP_WINDOW = 5

        for dt in range(1, STDP_WINDOW):
            prev_trace = None

            for loglength in range(1, STDP_WINDOW * 2):
                if dt >= loglength:
                    continue

                synapse = self._get_synapse(
                    n_neurons=N_NEURONS, stdp_window=STDP_WINDOW
                )
                synapse.reset()

                spike_log = np.zeros(shape=(loglength, N_NEURONS))
                spike_log[-1, 0] = 1
                spike_log[-dt, 1] = 1

                synapse._apply_stdp(spike_log, np.ones(N_NEURONS))

                trace = synapse.trace

                if prev_trace is not None:
                    self.assertListEqual(
                        list(np.ravel(trace)), list(np.ravel(prev_trace))
                    )
                prev_trace = np.copy(trace)

        ## More fires in window >= less fires
        N_NEURONS = 2

        for STDP_WINDOW in range(1, 6):

            for dt in range(STDP_WINDOW):
                synapse = self._get_synapse(
                    n_neurons=N_NEURONS, stdp_window=STDP_WINDOW
                )
                synapse.reset()

                spike_log = np.zeros(shape=(loglength, N_NEURONS))
                spike_log[-1, 0] = 1
                spike_log[-dt, 1] = 1

                synapse._apply_stdp(spike_log, np.ones(N_NEURONS))

                trace_to_beat = list(np.abs(np.ravel(np.copy(synapse.trace))))

                for dt2 in range(dt, STDP_WINDOW):
                    synapse = self._get_synapse(
                        n_neurons=N_NEURONS, stdp_window=STDP_WINDOW
                    )
                    synapse.reset()

                    new_spike_log = np.copy(spike_log)
                    new_spike_log[-dt2, 1] = 1

                    synapse._apply_stdp(new_spike_log, np.ones(N_NEURONS))

                    curr_trace = list(np.abs(np.ravel(synapse.trace)))
                    for i in range(max(len(curr_trace), len(trace_to_beat))):
                        self.assertGreaterEqual(curr_trace[i], trace_to_beat[i])

    @run_all_types
    def test_update(self):
        """
        Testing synapse.update.

        Parameters
        ----------
        spike_log: np.array(time, neurons)
            History of neuron firings

        inhibitories: np.array(neurons, dtype=int - -1 or 1)
            Polarities of each neuron.

        Settings
        --------
        _learning_rate: float
            Maximum trace value.

        Effects
        -------
        Traces will be updated by stdp suggestions.
        """
        ## Assert decay_trace and update_trace are called.
        N_NEURONS = 100
        STDP_WINDOW = 5

        synapse = self._get_synapse(n_neurons=N_NEURONS, stdp_window=STDP_WINDOW)
        synapse._decay_trace = mock.MagicMock(
            return_value=np.ones((N_NEURONS, N_NEURONS))
        )
        synapse._apply_stdp = mock.MagicMock()
        synapse.reset()

        synapse.update(np.zeros(shape=(STDP_WINDOW, N_NEURONS)), np.ones(N_NEURONS))

        synapse._decay_trace.assert_called_once()
        synapse._apply_stdp.assert_called_once()


if __name__ == "__main__":
    unittest.main()
