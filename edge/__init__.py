"""On-device (ESP32-S3 gateway) copy of the disease classifier.

The cloud model (ConvNeXt V2 Tiny, 28M parameters) is far too large for a
microcontroller, so the edge model is a small MobileNetV2 trained to imitate it
(knowledge distillation), then converted to int8 and emitted as C source for
the gateway firmware. The pieces:

    student.py            the small network (timm mobilenetv2_050 by default)
    train_edge.py         distillation from the ConvNeXt teacher checkpoint
    int8.py               BN folding, int8 quantization, the integer reference
                          that the C engine on the gateway matches bit for bit,
                          and the C emitter
    device_preprocess.py  the gateway's exact preprocessing (box resize + LUT)
    export_edge.py        checkpoint -> calibrated int8 model -> test-set report
                          -> kn_edge_model.h / kn_edge_model.cpp
    predict_edge.py       run the int8 model on a photo exactly as the gateway does
"""
