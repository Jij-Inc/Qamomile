import qamomile.core.circuit as qm_c


class ParameterContext:
    def __init__(self):
        self.counter = 0

    def get_next_parameter(self, symbol="θ"):
        param_name = f"{symbol}" + "_{" f"{self.counter}" + "}"
        self.counter += 1
        return qm_c.Parameter(param_name)
