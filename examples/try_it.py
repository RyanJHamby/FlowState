import numpy as np, pyarrow as pa
from flowstate.prism.alignment import AsOfConfig, as_of_join

rng, n, ns = np.random.default_rng(0), 100_000, pa.timestamp("ns", tz="UTC")
def side(m, col):  # m rows, sorted int64-ns timestamps over ~1000 s, 3 symbols
    return pa.table({"timestamp": pa.array(np.sort(rng.integers(0, 10**12, m)), ns),
                     "symbol": rng.choice(["AAPL", "MSFT", "NVDA"], m), col: rng.normal(100, 1, m)})
trades, quotes = side(n, "price"), side(4 * n, "bid")
joined, stats = as_of_join(trades, quotes, config=AsOfConfig(tolerance_ns=5 * 10**9))
print(joined.slice(0, 3).to_pylist()); print(stats)  # each trade sees only quotes at or before it
