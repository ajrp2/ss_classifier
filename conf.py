
SOMITE_COUNTS = ['17', '21', '25', '29']

N_CLASSES = len(SOMITE_COUNTS)

BATCH_SIZE = 6

Z = 100
H = 128
W = 128
C = 1

TARGET_DIMS = (C, H, W, Z)

LABEL_LOOKUP = {
    "17": 0,
    "21": 1,
    "25": 2,
    "29": 3
}


fc2_act_fn_cats = [{
      "enum_index": 1,
      "name": "tanh",
      "object": "tanh"
    },
    {
      "enum_index": 2,
      "name": "relu",
      "object": "relu"
    }
    ]

# SigOpt has limit of 15 tunable parameters per experiment
PARAMS = [
        dict(name='batch_size', type='int', bounds=dict(min=1, max=10)),
        dict(name='learning_rate', type='int', bounds=dict(min=-4, max=1)),
        dict(name='dropout_rate', type='double', bounds=dict(min=0.5, max=0.7)),
        dict(name='conv1_kernel_size', type='int', bounds=dict(min=2, max=5)),
        dict(name='conv2_kernel_size', type='int', bounds=dict(min=2, max=4)),
        dict(name='pool1_kernel_size', type='int', bounds=dict(min=2, max=4)),
        dict(name='pool2_kernel_size', type='int', bounds=dict(min=2, max=4)),
        dict(name='fc2_act_fn', type='categorical', categorical_values=fc2_act_fn_cats)]



