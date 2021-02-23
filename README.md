## a multilabel classification baseline 

### data prepare
to be continue ....


### tricks
1. warm up scheduler
2. autoaug
3. ema




resnet50 + ema : mAP: 79.5192, OP: 83.6114, OR: 73.7186, OF1: 78.3540 CP: 80.0947, CR: 69.7207, CF1: 74.5485

resnet101 + ema + woCutout : mAP: 80.2520, OP: 83.0697, OR: 75.2606, OF1: 78.9726 CP: 79.4305, CR: 71.5080, CF1: 75.2614

resnet101 + ema : mAP: 80.4146, OP: 83.5959, OR: 75.3188, OF1: 79.2418 CP: 80.2562, CR: 71.3448, CF1: 75.5386

调整 cutout 和 randaug 的意义不大
resnet101 + ema : mAP: 80.4403, OP: 83.9538, OR: 75.0686, OF1: 79.2630 CP: 80.9111, CR: 71.2521, CF1: 75.7750

TresnetL + ema : mAP 83.9303, OP: 82.3867, OR: 80.6157, OF1: 81.4916 CP: 80.9290, CR: 76.8393, CF1: 78.8311



