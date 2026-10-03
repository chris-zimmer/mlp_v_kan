from kan import KAN


def build_kan(
    input_size: int,
    hidden_size: int = 10,
    output_size: int = 10,
    grid: int = 5,
    k: int = 3,
) -> KAN:
    # The symbolic branch and activation caching exist for plotting and pruning,
    # not training; leaving them on costs ~18x per step.
    return KAN(
        width=[input_size, hidden_size, output_size],
        grid=grid,
        k=k,
        symbolic_enabled=False,
        save_act=False,
    )
