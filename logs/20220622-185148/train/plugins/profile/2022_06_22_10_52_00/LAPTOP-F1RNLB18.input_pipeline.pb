	K?=??@K?=??@!K?=??@	??Kx??@??Kx??@!??Kx??@"e
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails$K?=??@??????A?t?V@Y?T???N??*	33333sT@2F
Iterator::Model????ׁ??!?b??IG@)?Zd;??1R.&???B@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat?g??s???!????9@)"??u????1?????5@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::ConcatenateV-???!?M$???1@)Έ?????1?????&@:Preprocessing2U
Iterator::Model::ParallelMapV2ŏ1w-!?!?????"@)ŏ1w-!?1?????"@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::ZipV????_??!??M$?J@)??_vOv?1?z???g@:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice??_?Lu?!?Fg?m@)??_?Lu?1?Fg?m@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor????Mbp?!(?U?@)????Mbp?1(?U?@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap???????!t6?ؖ?4@)??_?Le?1?Fg?m	@:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 5.2% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*no9??Kx??@>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	????????????!??????      ??!       "      ??!       *      ??!       2	?t?V@?t?V@!?t?V@:      ??!       B      ??!       J	?T???N???T???N??!?T???N??R      ??!       Z	?T???N???T???N??!?T???N??JCPU_ONLYY??Kx??@b 