	s?69?k@s?69?k@!s?69?k@      ??!       "?
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetailsCs?69?k@I?F?q=??1D???b?k@A?k$	???I?7/N|???rEagerKernelExecute 0*	]???(,d@2Z
#Iterator::Model::ParallelMapV2::Zip ?d?F ??!Byj??:R@),??NG??1Zȑ馗>@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat??j̡?!??
?5@)d???????1?PQ??1@:Preprocessing2U
Iterator::Model::ParallelMapV2$????ۗ?!2?
?=?,@)$????ۗ?12?
?=?,@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMapŏ1w-??!?]??4@)o???????1?????*@:Preprocessing2F
Iterator::ModelV????_??!?V?5;@)??ZӼ???1?;?n-H)@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[1]::TensorSlice5)?^҈?!?/>x
@)5)?^҈?1?/>x
@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor?????w?!??M???@)?????w?1??M???@:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
device?Your program is NOT input-bound because only 0.0% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.no*noI??ʪ???Q;?k?$?X@Zno#You may skip the rest of this page.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	I?F?q=??I?F?q=??!I?F?q=??      ??!       "	D???b?k@D???b?k@!D???b?k@*      ??!       2	?k$	????k$	???!?k$	???:	?7/N|????7/N|???!?7/N|???B      ??!       J      ??!       R      ??!       Z      ??!       b      ??!       JGPUb q??ʪ???y;?k?$?X@