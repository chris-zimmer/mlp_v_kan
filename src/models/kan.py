from kan import KAN


def build_kan(
    input_size: int,
    hidden_size: int = 10,
    output_size: int = 10,
    grid: int = 5,
    k: int = 3,
) -> KAN:
    return KAN(width=[input_size, hidden_size, output_size], grid=grid, k=k)
