_base_ = ['./classification_dynamic.py', '../_base_/backends/cann.py']

onnx_config = dict(input_shape=None)

backend_config = dict(model_inputs=[
    dict(
        input_shapes=dict(input=[-1, 3, 224, 224]),
        dynamic_batch=[1, 2, 4, 8, 16, 32])
])
