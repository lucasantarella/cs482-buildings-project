	??Z?ۙh@??Z?ۙh@!??Z?ۙh@      ??!       "?
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetailsC??Z?ۙh@e?9:Z??1?~T^h@AE?D??2??I]??'???rEagerKernelExecute 0*	Zd;?O!`@2U
Iterator::Model::ParallelMapV2? ?	???!?*?WN?7@)? ?	???1?*?WN?7@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat??D?֠?!gi?|9@)??6???1?F 5@:Preprocessing2F
Iterator::Model?HZ????!T:??.E@)G?J??q??1	????2@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMapn?????!)S"?Y;@)??ڊ?e??1??p"?1@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[1]::TensorSlice|DL?$z??!?lc??G#@)|DL?$z??1?lc??G#@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip}?͍?	??!????L@)_??x?Zy?1M? 0@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor,+MJA?w?!DL????@),+MJA?w?1DL????@:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
device?Your program is NOT input-bound because only 0.0% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.no*noI ?-/"???Q??????X@Zno#You may skip the rest of this page.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	e?9:Z??e?9:Z??!e?9:Z??      ??!       "	?~T^h@?~T^h@!?~T^h@*      ??!       2	E?D??2??E?D??2??!E?D??2??:	]??'???]??'???!]??'???B      ??!       J      ??!       R      ??!       Z      ??!       b      ??!       JGPUb q ?-/"???y??????X@