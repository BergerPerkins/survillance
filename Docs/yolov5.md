# YOLOv5 usage

**NOTE**: The yaml file is not required.

* [Convert model](#convert-model)
* [Edit the config_infer_primary_yoloV5 file](#edit-the-config_infer_primary_yolov5-file)

##

### Convert model

#### 1. Download the YOLOv5 repo and install the requirements

```
git clone https://github.com/ultralytics/yolov5.git
cd yolov5
pip3 install -r requirements.txt
pip3 install onnx onnxsim onnxruntime
```

**NOTE**: It is recommended to use Python virtualenv.

#### 2. Copy conversor

Copy the `export_yoloV5.py` file from `DeepStream-Yolo/utils` directory to the `yolov5` folder.

#### 3. Download the model

Download the `pt` file from [YOLOv5](https://github.com/ultralytics/yolov5/releases/) releases (example for YOLOv5s)

```
wget https://github.com/ultralytics/yolov5/releases/download/v7.0/yolov5s.pt
```


#### 4. Convert model

Generate the ONNX model file (example for YOLOv5s)

```
python3 export_yoloV5.py -w yolov5s.pt --dynamic
```

**NOTE**: Confidence threshold (example for conf-thres = 0.25)

The minimum detection confidence threshold is configured in the ONNX exporter file. The `pre-cluster-threshold` should be >= the value used in the ONNX model.

```
--conf-thres 0.25
```

**NOTE**: NMS IoU threshold (example for iou-thres = 0.45)

```
--iou-thres 0.45
```

**NOTE**: Maximum detections (example for max-det = 100)

```
--max-det 100
```

**NOTE**: To convert a P6 model

```
--p6
```

**NOTE**: To change the inference size (defaut: 640 / 1280 for `--p6` models)

```
-s SIZE
--size SIZE
-s HEIGHT WIDTH
--size HEIGHT WIDTH
```


```

**NOTE**: To use dynamic batch-size (DeepStream >= 6.1)

```
--dynamic
```

**NOTE**: To use static batch-size (example for batch-size = 4)

```
--batch 4
```

#### 5. Copy generated files

Copy the generated ONNX model file and labels.txt file (if generated) to the `DeepStream-Yolo` folder.

##

### Edit the config_infer_primary_yoloV5 file

Edit the `config_infer_primary_yoloV5.txt` file according to your model (example for YOLOv5s)

```
[property]
...
onnx-file=yolov5s.onnx
model-engine-file=yolov5s.onnx_b1_gpu0_fp32.engine
...
```

**NOTE**: To output the masks, use

```
[property]
...
output-instance-mask=1
segmentation-threshold=0.5
...
```

**NOTE**: The **YOLOv5** resizes the input with center padding. To get better accuracy, use

```
[property]
...
maintain-aspect-ratio=1
symmetric-padding=1
...
```