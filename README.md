timing utility plus utility that allows using functool's partial with hooks in transformer-lens without inducing significant performance degradations

to fix performance degradation, simply use fastpartial in place of partial. In one bottlenecked program (which motivated creating this), the following single line change provided a 100x speedup in training:

```python
from nqgl.mlutils.norepr import fastpartial as partial
```
