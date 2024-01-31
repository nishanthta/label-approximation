RUN_DDP = True
SCAN_TYPE = 'ct' #['us','ct','mri']
MODEL_SIGNATURE = 'ct_2d_segmentation_original_labels_512'
IN_CHANNELS = 1
NUM_CLASSES = 1
BACKGROUND_AS_CLASS = False
INPUT_DIMS = 2 #2D or 3D


TRAIN_VAL_TEST_SPLIT = [0.8, 0.1, 0.1]
SPLIT_SEED = 42
TRAINING_EPOCH = 100
BATCH_SIZE = 8
TRAIN_CUDA = True
CLASSIFIER = False
TRANSFER_WEIGHTS = None #include filepath of pretrained model

MODEL_ARCH = 'UNet2D' #['AHNet', 'UNet3D', 'VolumeClassifier', 'UNet2D', 'frameClassifier', 'DynUnet', 'MONAIUNet', 'CAMModel']
LOSS_FN = 'BCEWithLogitsLoss' #['CrossEntropyLoss', 'BCEWithLogitsLoss', 'DiceLoss', 'DiscrepancyLoss']
DISCREPANCY_LOSS_WEIGHTS = [0.1, 0.9]
BCE_LOSS_WEIGHTS = 15.
N_MC_SAMPLES = 5

PATIENCE = 10 #for early stopping

APPROXIMATION = None #for label approximation
N_SPLINE_POINTS = 10
APPROXIMATION_ERROR = None
APPROXIMATION_PERCENT = 75
TRAIN_SUBSET = 1.