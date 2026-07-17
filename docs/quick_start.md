# Quick Start

This guide will help you get started with the SSP Framework.

*(Detailed step-by-step usage examples will be added as features are implemented.)*

## Verifying Package Import

Once installed, you can verify the package is ready for import:

```python
import ssp_framework
from ssp_framework.core.config import ExperimentConfig
from ssp_framework.utils import set_seed, setup_logger

logger = setup_logger()
set_seed(42)
logger.info(f"SSP Framework version {ssp_framework.__version__} is ready.")
```

## Running the First Dummy Test

You can run the existing placeholder tests to verify the installation:

```bash
pytest
```
