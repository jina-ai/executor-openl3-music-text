# OpenL3MusicText


## Usage

#### via Docker image (recommended)

```python
from jina import Flow
	
f = Flow().add(uses='jinahub+docker://OpenL3MusicText')
```

#### via source code

```python
from jina import Flow
	
f = Flow().add(uses='jinahub://OpenL3MusicText')
```

- To override `__init__` args & kwargs, use `.add(..., uses_with: {'key': 'value'})`
- To override class metas, use `.add(..., uses_metas: {'key': 'value})`
