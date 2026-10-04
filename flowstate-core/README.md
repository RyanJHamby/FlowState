# flowstate-asof-core

Rust kernel for [FlowState](https://github.com/RyanJHamby/flowstate): O(n+m) as-of joins,
multi-stream alignment and a watermark-based streaming join, exposed to Python over the
Arrow PyCapsule interface. Import name: `flowstate_core`.

Most users want the main package, which pulls this in automatically:

```bash
pip install flowstate-asof
```
