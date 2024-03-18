## This contains:
- **Cache/ComponentLayers**: ml components and cache framework that I have been developing for [HSAE-2.0](https://github.com/nqgl/HSAE-2.0)
- **TimedFunc/ProfileFunc**: timing/profiling function wrappers
- **fastpartial**: allows the use functool's partial with hooks in transformer-lens without significant performance degradation

### fastpartial
To fix performance degradation, simply use fastpartial in place of partial. 

In one bottlenecked program (the motive for creating this), the following single line addition provided a 100x speedup in training:

```python
from nqgl.mlutils.norepr import fastpartial as partial
```
