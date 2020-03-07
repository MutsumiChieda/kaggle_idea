# swifter 
## Warning
- tqdmなどは切っておくこと．
- 外部変数などを用いる関数には使わないこと．

## Example
```python
import pandas as pd
import swifter

def f():
    pass

df = pd.DataFrame()
df.swifter.apply(f)
```
