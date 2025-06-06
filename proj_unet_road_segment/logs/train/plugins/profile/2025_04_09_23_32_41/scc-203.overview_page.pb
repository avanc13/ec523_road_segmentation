�	�h:;�Z@�h:;�Z@!�h:;�Z@      ��!       "e
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails$�h:;�Z@
�O��� @1"�� >�Y@I歺��?*	X9�H��@2t
=Iterator::Model::ParallelMap::Zip[0]::FlatMap[0]::ConcatenateU.T����?!��8H��V@)�sCSv�?1hP���V@:Preprocessing2j
3Iterator::Model::ParallelMap::Zip[1]::ForeverRepeat����?!�[��T�@)���7�?1�*T!��@:Preprocessing2F
Iterator::Model?�̔�ߢ?!;W�5�@)��:7mƙ?1H?*@:Preprocessing2X
!Iterator::Model::ParallelMap::Zip4ڪ$�O�?!F�S� X@)YP�i4�?1Z<�q,��?:Preprocessing2S
Iterator::Model::ParallelMap�����?!�ԡ��?)�����?1�ԡ��?:Preprocessing2d
-Iterator::Model::ParallelMap::Zip[0]::FlatMap�)ʥ��?!w1��B�V@)��~��΃?1��(��L�?:Preprocessing2�
MIterator::Model::ParallelMap::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice������?!��G�H�?)������?1��G�H�?:Preprocessing2v
?Iterator::Model::ParallelMap::Zip[1]::ForeverRepeat::FromTensor�����u?!���>��?)�����u?1���>��?:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
device�Your program is NOT input-bound because only 0.0% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.no*no#You may skip the rest of this page.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	
�O��� @
�O��� @!
�O��� @      ��!       "	"�� >�Y@"�� >�Y@!"�� >�Y@*      ��!       2      ��!       :	歺��?歺��?!歺��?B      ��!       J      ��!       R      ��!       Z      ��!       JGPU�"\
2gradient_tape/model/conv2d_12/Conv2DBackpropFilterConv2DBackpropFilterim�x��?!im�x��?"Y
0gradient_tape/model/conv2d_8/Conv2DBackpropInputConv2DBackpropInput=�'G�?!�FF�#�?"1
model/conv2d_8/Conv2DConv2D�mK���?!��䠄�?"Z
1gradient_tape/model/conv2d_12/Conv2DBackpropInputConv2DBackpropInput�'^��?!��*\��?"[
1gradient_tape/model/conv2d_1/Conv2DBackpropFilterConv2DBackpropFilter\P��J��?!����n��?"2
model/conv2d_12/Conv2DConv2D�uP�mx�?!�ZgŘ�?"\
2gradient_tape/model/conv2d_13/Conv2DBackpropFilterConv2DBackpropFilter��G�X�?!}XD���?"Z
1gradient_tape/model/conv2d_10/Conv2DBackpropInputConv2DBackpropInput]�%��?!����!�?"2
model/conv2d_10/Conv2DConv2D&�®d��?!IT�KT�?"\
2gradient_tape/model/conv2d_10/Conv2DBackpropFilterConv2DBackpropFilter<�(��.�?!��<:7�?2blackQ      Y@"�
device�Your program is NOT input-bound because only 0.0% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2�
=type.googleapis.com/tensorflow.profiler.GenericRecommendation�
nono*�Only 0.0% of device computation is 16 bit. So you might want to replace more 32-bit Ops by 16-bit Ops to improve performance (if the reduced accuracy is acceptable).:
Refer to the TF2 Profiler FAQ2"GPU(: 