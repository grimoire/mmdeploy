_base_ = ['./rotated-detection_tensorrt_dynamic-320x320-1024x1024.py']

backend_config = dict(model_inputs=[
    dict(
        input_shapes=dict(
            input=dict(
                min_shape=[1, 3, 320, 320],
                opt_shape=[1, 3, 512, 800],
                max_shape=[1, 3, 800, 800])))
])
