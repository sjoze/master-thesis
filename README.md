# Master Thesis (Benchmarking of Pruning and Quantization Techniques)


Deep learning models feature a great amount of parameters, i.e. nodes and weights, which grant these architectures their name. But, sometimes you need to reduce the model size and simplify the neural network, e.g. when you are limited by hardware constraints or like to emphasize energy efficiency. Maybe faster inference is a more desirable model trait than top end accuracy in your setting. In those cases, model compression like pruning and quantization help out. They act as a form of regularization on the neural network.

In this master thesis, we investigate the effect of pruning and quantization techniques on different models and determine which compression tool has the greatest effects in terms of model size reduction and inference speed gain while preventing an impactful loss in accuracy. Our focus will be on an as out-of-the-box usage of the tools as possible, with application on pretrained models. We build a framework for benchmarking which can be used to analyze own models and compression techniques of a user.

We find out that some pruning and quantization tools do not live up to their expectations and are associated with a great amount of work to set up. But, some approaches find success on models which are thought to be optimized and compressed to the limit due to their deployment on edge devices.

# Setup

[TensorRT guide](https://medium.com/kgxperience/how-to-install-tensorrt-a-comprehensive-guide-99557c0e9d6)

[ONNX Runtime documentation](https://onnxruntime.ai/docs/execution-providers/TensorRT-ExecutionProvider.html)

Our recommended setup is as follows:

<ol>
  <li>Setup a conda environment.</li>
  <li>Install CUDA, CUDNN and then TENSORRT according to the TensorRT guide and pay close attention to the required version matching of the ONNX Runtime according to the ONNX Runtime documentation. We used TensorRT 10.4.0, CUDA 12.2, ONNX 1.16.2 with ONNX Runtime GPU support 1.19.2.</li>
  <li>Install the packages in the provided requirements file.</li>
  <li>Download the datasets to a data/imagenette2 and data/imagewoof2 folder structure.</li>
  <li>Modify the config file according to the desired experiments to run.</li>
  <li>Run pq_bench.py.</li>
  <li>The results should be in data/experiments if not specified otherwise.</li>
</ol> 
