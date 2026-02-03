# -*- coding: utf-8 -*-
"""Training module package"""

# 避免在包初始化时导入trainer模块，防止导入顺序冲突
# 当执行 `python -m training.trainer` 时，Python会先导入training包
# 然后再执行trainer模块，如果__init__.py中导入了trainer，会导致重复导入
# 这会产生 RuntimeWarning: training.trainer found in sys.modules after import of package 'training'

__all__ = []
