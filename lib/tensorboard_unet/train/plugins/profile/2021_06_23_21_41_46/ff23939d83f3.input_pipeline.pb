	?Ȓ9?.n@?Ȓ9?.n@!?Ȓ9?.n@      ??!       "?
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetailsC?Ȓ9?.n@?lXSY???1????m@A?d??I???T?t??rEagerKernelExecute 0*	?Zd;?`@2U
Iterator::Model::ParallelMapV2?>???!ٳ?
8@)?>???1ٳ?
8@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat ?g?????!?d?os9@)?Z_$????1!??95@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap??r????!!?????;@)?Ƕ8K??1}r?]qk2@:Preprocessing2F
Iterator::Modelu???a???!?|????D@)?b?=y??1 u???1@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[1]::TensorSliceN^????!G?X?(?"@)N^????1G?X?(?"@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zipt]?@???!r?k}iM@)?!??u?|?1??g@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor?????gv?!絢S?P@)?????gv?1絢S?P@:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
device?Your program is NOT input-bound because only 0.0% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.no*noI???	???Q#f????X@Zno#You may skip the rest of this page.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	?lXSY????lXSY???!?lXSY???      ??!       "	????m@????m@!????m@*      ??!       2	?d???d??!?d??:	???T?t?????T?t??!???T?t??B      ??!       J      ??!       R      ??!       Z      ??!       b      ??!       JGPUb q???	???y#f????X@