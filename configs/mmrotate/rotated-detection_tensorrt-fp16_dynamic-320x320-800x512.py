_base_ = ['./rotated-detection_tensorrt_dynamic-320x320-800x512.py']

backend_config = dict(common_config=dict(fp16_mode=True))
