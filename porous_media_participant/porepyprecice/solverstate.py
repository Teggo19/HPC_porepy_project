import copy


class SolverState:
    def __init__(self, payload, t, n):
        """
        Solver state consists of a payload (a PorePy), associated time t and the timestep n

        Parameters
        ----------
        payload : A PorePy Function
            Describes the state of the solver.
        t : double
            Time stamp.
        n : int
            Iteration number.
        """
        try:
            self.payload = copy.deepcopy(payload)
        except (AttributeError, TypeError):  # AttributeError, if .copy() does not exist; TypeError, if .copy(deepcopy) does not exist. -> Probably a list
            self.payload = [copy.deepcopy(item) for item in payload]

        self.t = t
        self.n = n

    def get_state(self):
        """
        Returns the state variables payload, associated time t and timestep n

        Returns
        -------
        payload : A fenics.Function or a list of fenics.Functions
            Describes the state of the solver.
        t : double
            Time stamp.
        n : int
            Iteration number.
        """
        try:
            return copy.deepcopy(self.payload), self.t, self.n # self.payload.copy(deepcopy=True), self.t, self.n
        except (AttributeError, TypeError):  # AttributeError, if .copy() does not exist; TypeError, if .copy(deepcopy) does not exist. -> Probably a list
            return [item.copy(deepcopy=True) for item in self.payload], self.t, self.n

    def print_state(self):
        payload, t, n = self.get_state()
        return print(f"payload={payload}, t={t}, n={n}")
