	�e��Z@�e��Z@!�e��Z@      ��!       "e
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails$�e��Z@�-��X@1�_���Y@I'�O:���?*	�n��*l@2t
=Iterator::Model::ParallelMap::Zip[0]::FlatMap[0]::Concatenate>?��?!�2�T�E@)}>ʈ@�?1�rc.p�@@:Preprocessing2j
3Iterator::Model::ParallelMap::Zip[1]::ForeverRepeat�����	�?!�P�%3@)o�[t�Ԣ?1=6ХdR0@:Preprocessing2d
-Iterator::Model::ParallelMap::Zip[0]::FlatMap�3����?!��n÷M@)��O��۠?1��9-@:Preprocessing2F
Iterator::Model,��ص��?!�)H$�2@)(��{�_�?1����1+@:Preprocessing2�
MIterator::Model::ParallelMap::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice��x�?!M�<��W$@)��x�?1M�<��W$@:Preprocessing2S
Iterator::Model::ParallelMap_�vj.7�?!��o�5�@)_�vj.7�?1��o�5�@:Preprocessing2X
!Iterator::Model::ParallelMap::Zip�#)�ah�?!�����IT@)F$
-���?1]�;U22@:Preprocessing2v
?Iterator::Model::ParallelMap::Zip[1]::ForeverRepeat::FromTensor(D�!T�y?!f
>@)(D�!T�y?1f
>@:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
device�Your program is NOT input-bound because only 0.0% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.no*no#You may skip the rest of this page.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	�-��X@�-��X@!�-��X@      ��!       "	�_���Y@�_���Y@!�_���Y@*      ��!       2      ��!       :	'�O:���?'�O:���?!'�O:���?B      ��!       J      ��!       R      ��!       Z      ��!       JGPU