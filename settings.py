


# "GrandLoaderSettings": {"InputPath": "/data1/atlas2_gc/private/hidden_mha/t1-brain-mri/",
#                         "OutputPath": "hidden_predictions/",
#                         "GroundTruthRoot": "/opt/evaluation/ground-truth/",
#                         "JSONPath": "/inputs/predictions.json",
#                         "BatchSize": 2,
#                         "InputSlugs": ["t1-brain-mri"],
#                         "OutputSlugs": ["stroke-segmentation"]}


# "GrandLoaderSettings": {"InputPath": "/data1/atlas2_gc/private/hidden_mha/t1-brain-mri/",
#                         "OutputPath": "hidden_predictions/stroke-segmentation/",
#                         "GroundTruthRoot": "hidden_mha_masks/",
#                         "JSONPath": "predictions.json",
#                         "BatchSize": 2,
#                         "InputSlugs": ["t1-brain-mri"],
#                         "OutputSlugs": ["stroke-segmentation"]},

loader_settings = {"InputPath": "/input/images/t1-brain-mri/",
                    "OutputPath": "/output/images/stroke-lesion-segmentation/",
                    "GroundTruthRoot": "/opt/evaluatlion/mha_masks/",
                    "JSONPath": "/input/predictions.json",
                    "BatchSize": 2,
                    "InputSlugs": ["t1-brain-mri"],
                    "OutputSlugs": ["stroke-lesion-segmentation"]}