"""兼容包装：流式分类模型。

实现已合并到 `models/networks/s4d_classification.py`（`streaming=True`）；
保留本文件是为了让历史模型名（`--model s4d_recurrent_classification`）继续可用。
state_dict 键名与合并前完全一致。
"""

from models.networks.s4d_classification import Model as _ClassificationModel


class Model(_ClassificationModel):

    def __init__(self, *args, **kwargs):
        # 历史接口里有 training 参数（已废弃，无实际作用）
        kwargs.pop("training", None)

        # 流式模型固定使用循环核
        kwargs["kernel_type"] = "recurrent"
        kwargs["streaming"] = True

        super().__init__(*args, **kwargs)
