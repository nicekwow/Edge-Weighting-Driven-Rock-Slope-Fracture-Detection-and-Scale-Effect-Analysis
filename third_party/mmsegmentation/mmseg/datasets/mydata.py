from mmseg.registry import DATASETS
from .basesegdataset import BaseSegDataset

@DATASETS.register_module()
class MyDataset(BaseSegDataset):
    # Class names and RGB palette
    METAINFO = {
        'classes':['background', 'crack'],
        'palette':[[0,0,0], [255,255,255]]
    }

    # Configure mask suffix and zero-label handling
    def __init__(self,
                 seg_map_suffix='.png',   # Mask file suffix
                 reduce_zero_label=False, # Keep background class with ID 0
                 **kwargs) -> None:
        super().__init__(
            seg_map_suffix=seg_map_suffix,
            reduce_zero_label=reduce_zero_label,
            **kwargs)
