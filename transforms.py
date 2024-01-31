'''
Author : Nishanth
'''


from monai.transforms import (
    Compose,
    ToTensord,
    RandFlipd,
    Spacingd,
    RandScaleIntensityd,
    RandShiftIntensityd,
    NormalizeIntensityd,
    AddChanneld,
    DivisiblePadd,
    ResizeD
)

all_transforms = {}

all_transforms['3d_segmentation'] = {
    'train': Compose(
    [   
        AddChanneld(keys=["image", "label"]),
        # Spacingd(keys=['image', 'label'], pixdim=(1., 1., 1.), mode=("bilinear", "nearest")),
        # RandFlipd(keys=['image', 'label'], prob=0.5, spatial_axis=0),
        # RandFlipd(keys=['image', 'label'], prob=0.5, spatial_axis=1),
        # RandFlipd(keys=['image', 'label'], prob=0.5, spatial_axis=2),
        NormalizeIntensityd(keys='image', nonzero=True, channel_wise=True),
        RandScaleIntensityd(keys='image', factors=0.1, prob=1.0),
        RandShiftIntensityd(keys='image', offsets=0.1, prob=1.0),
        DivisiblePadd(k=16, keys=["image", "label"]),
        ResizeD(keys=["image", "label"], spatial_size=(-1, 128, 128)),
        ToTensord(keys=['image', 'label'], device='cuda')
    ]
),
'val' : Compose(
    [   
        AddChanneld(keys=["image", "label"]),
        NormalizeIntensityd(keys='image', nonzero=True, channel_wise=True),
        DivisiblePadd(k=16, keys=["image", "label"]),
        ResizeD(keys=["image", "label"], spatial_size=(128, 128)),
        ToTensord(keys=['image', 'label'], device='cuda')
    ]
),
'infer' : Compose(
    [   
        AddChanneld(keys=["image"]),
        NormalizeIntensityd(keys='image', nonzero=True, channel_wise=True),
        DivisiblePadd(k=16, keys=["image"]),
        ResizeD(keys=["image"], spatial_size=(-1, 128, 128)),
        ToTensord(keys=['image'], device='cuda')
    ]
)
}

all_transforms['3d_segmentation_multiclass'] = {
    'train': Compose(
        [
            AddChanneld(keys=["image"]),
            Spacingd(keys=['image', 'label'], pixdim=(1., 1., 1.), mode=("bilinear", "nearest")),
            RandFlipd(keys=['image', 'label'], prob=0.5, spatial_axis=0),
            RandFlipd(keys=['image', 'label'], prob=0.5, spatial_axis=1),
            RandFlipd(keys=['image', 'label'], prob=0.5, spatial_axis=2),
            NormalizeIntensityd(keys='image', nonzero=True, channel_wise=True),
            RandScaleIntensityd(keys='image', factors=0.1, prob=1.0),
            RandShiftIntensityd(keys='image', offsets=0.1, prob=1.0),
            DivisiblePadd(k=16, keys=["image", "label"]),
            ResizeD(keys=["image", "label"], spatial_size=(-1, 64, 64), mode=("bilinear", "nearest")),
            ToTensord(keys=['image', 'label'], device='cuda')
        ]
    ),
    'val' : Compose(
        [
            AddChanneld(keys=["image"]),
            NormalizeIntensityd(keys='image', nonzero=True, channel_wise=True),
            DivisiblePadd(k=16, keys=["image", "label"]),
            ResizeD(keys=["image", "label"], spatial_size=(64, 64), mode=("bilinear", "nearest")),
            ToTensord(keys=['image', 'label'], device='cuda')
        ]
    ),
    'infer' : Compose(
        [
            AddChanneld(keys=["image"]),
            NormalizeIntensityd(keys='image', nonzero=True, channel_wise=True),
            DivisiblePadd(k=16, keys=["image"]),
            ResizeD(keys=["image"], spatial_size=(-1, 64, 64)),
            ToTensord(keys=['image'], device='cuda')
        ]
    )
}

all_transforms['2d_segmentation'] = {
    'train' : Compose(
    [   
        AddChanneld(keys=["image", "label"]),
        Spacingd(keys=['image', 'label'], pixdim=(1., 1.), mode=("bilinear", "nearest")),
        RandFlipd(keys=['image', 'label'], prob=0.5, spatial_axis=0),
        RandFlipd(keys=['image', 'label'], prob=0.5, spatial_axis=1),
        # NormalizeIntensityd(keys='image', nonzero=True, channel_wise=True),
        # RandScaleIntensityd(keys='image', factors=0.1, prob=1.0),
        # RandShiftIntensityd(keys='image', offsets=0.1, prob=1.0),
        DivisiblePadd(k=16, keys=["image", "label", "spline_labels"]),
        ResizeD(keys=["image", "label", "spline_labels"], spatial_size=(512, 512)),
        ToTensord(keys=['image', 'label', "spline_labels"], device='cuda')
    ]
),
'val' : Compose(
    [   
        AddChanneld(keys=["image", "label", "original_label"]),
        # NormalizeIntensityd(keys='image', nonzero=True, channel_wise=True),
        DivisiblePadd(k=16, keys=["image", "label", "original_label", "spline_labels"]),
        ResizeD(keys=["image", "label", "original_label", "spline_labels"], spatial_size=(512, 512)),
        ToTensord(keys=['image', 'label', "original_label", "spline_labels"], device='cuda')
    ]
)
}

all_transforms['2d_segmentation_busi'] = {
    'train' : Compose(
    [   
        AddChanneld(keys=["image", "label"]),
        Spacingd(keys=['image', 'label'], pixdim=(1., 1.), mode=("bilinear", "nearest")),
        # RandFlipd(keys=['image', 'label'], prob=0.5, spatial_axis=0),
        # RandFlipd(keys=['image', 'label'], prob=0.5, spatial_axis=1),
        NormalizeIntensityd(keys='image', nonzero=True, channel_wise=True),
        # RandScaleIntensityd(keys='image', factors=0.1, prob=1.0),
        # RandShiftIntensityd(keys='image', offsets=0.1, prob=1.0),
        DivisiblePadd(k=16, keys=["image", "label"]),
        ResizeD(keys=["image", "label"], spatial_size=(256,256)),
        ToTensord(keys=['image', 'label'], device='cuda')
    ]
),
'val' : Compose(
    [   
        AddChanneld(keys=["image", "label", "original_label"]),
        NormalizeIntensityd(keys='image', nonzero=True, channel_wise=True),
        DivisiblePadd(k=16, keys=["image", "label", "original_label"]),
        ResizeD(keys=["image", "label", "original_label"], spatial_size=(256,256)),
        ToTensord(keys=['image', 'label', "original_label"], device='cuda')
    ]
)
}



all_transforms['2d_classification'] = {
    'train' : Compose(
    [   
        AddChanneld(keys=["image"]),
        Spacingd(keys=['image'], pixdim=(1., 1.), mode=("bilinear")),
        RandFlipd(keys=['image'], prob=0.5, spatial_axis=0),
        RandFlipd(keys=['image'], prob=0.5, spatial_axis=1),
        # NormalizeIntensityd(keys='image', nonzero=True, channel_wise=True),
        # RandScaleIntensityd(keys='image', factors=0.1, prob=1.0),
        # RandShiftIntensityd(keys='image', offsets=0.1, prob=1.0),
        DivisiblePadd(k=16, keys=["image"]),
        ResizeD(keys=["image"], spatial_size=(512, 512)),
        ToTensord(keys=['image', 'label'], device='cuda')
    ]
),
'val' : Compose(
    [   
        AddChanneld(keys=["image"]),
        # NormalizeIntensityd(keys='image', nonzero=True, channel_wise=True),
        DivisiblePadd(k=16, keys=["image"]),
        ResizeD(keys=["image"], spatial_size=(512, 512)),
        ToTensord(keys=['image', 'label'], device='cuda')
    ]
)
}