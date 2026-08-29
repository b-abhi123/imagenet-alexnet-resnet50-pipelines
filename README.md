# AlexNet & ResNet-50 ImageNet Benchmarking & CIFAR-10 Transfer Learning Pipelines
Modular PyTorch implementations of classical deep convolutional architectures (AlexNet and ResNet-50) for large-scale image classification on ImageNet and fine-tuning applications on CIFAR-10.
Papers:
- AlexNet: https://papers.nips.cc/paper_files/paper/2012/file/c399862d3b9d6b76c8436e924a68c45b-Paper.pdf
- ResNet: https://arxiv.org/pdf/1512.03385

AlexNet (Krizhevsky et al., 2012) demonstrated the potential of deep CNNs by winning ILSVRC 2012. It introduced many modern innovations such as non-linear ReLU activations, parallel GPU acceleration, etc.

ResNet (He et al., 2015) introduced residual learning and skip connections to overcome the degradation problem in DNNs with the progression of the number of layers. The core unit uses a 3-layer bottleneck block with a 4x feature expansion factor in the end.

# Architectural Improvements/Modifications

Several modifications were made to optimize training efficiency and align with modern deep learning standards in PyTorch:

### AlexNet Implementation

- Batch Normalization: Replaced the historical Local Response Normalization (LRN) with the modern `nn.BatchNorm2d`.
- Unified Single-Device Pipeline: Refactored original dual-GPU split into a unified forward pass.
- LR Scheduling: Integrated `ReduceLROnPlateau` scheduler.

### ResNet Implementation

- Custom Bottleneck Block (`block` class): Implemented the 1x1, 3x3, 1x1 bottleneck structure with `expansion = 4`. Added an explicit `identity_downsample` residual projection path when input/output spatial dimensions or channel counts differ.
- Generalized ResNet Architecture: The class features a system where a network with any desirable number of layers can be constructed, given the layer counts (`[3,4,6,3]` in ResNet-50)
- Fine-tuning Pipeline: Since training ResNet-50 on ImageNet was not feasible locally, pre-trained ResNet-50 weights from `torchvision` were used to simulate transfer learning for CIFAR-10 (rescaling input was also involved)

# Repository Structure

`models/alexnet.py`: Custon AlexNet class implementation.

`models/resnet.py`: Custom ResNet class implementation, with skip connections included.

`main_imagenet_alexnet.py`,`main_cifar_resnet50.py`: AlexNet and ResNet-50 ImageNet training pipeline respectively, currently using generated tensors with the same shape as ImageNet data for verification purposes

`main_cifar_resnet50.py`: Pre-trained ResNet-50 transfer learning pipeline on CIFAR-10.

# Execution

Due to limited computational power, the execution features verifying matrix dimensions, gradient computation and weight saving routines to simulate the program without local storage of full ImageNet.\
It does this by utilizing synthetic tensor streams:
```python
dum_x = torch.randn(10,3,224,224)
dum_y = torch.randint(0,num_classes,(10,))
```
The code has been verified to be working as intended with the dummy datasets set as this.
