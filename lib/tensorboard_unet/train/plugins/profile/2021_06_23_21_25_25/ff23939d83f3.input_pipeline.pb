	0?GĆm@0?GĆm@!0?GĆm@      ??!       "?
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetailsC0?GĆm@?:M?? @1 &?B:m@A?]=???I?s~?????rEagerKernelExecute 0*	fffff&_@2U
Iterator::Model::ParallelMapV2.?R\U???!ؽ?u?{7@).?R\U???1ؽ?u?{7@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat???#0??!H?]??`9@)Ͻ?K???1nL@ܘ5@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap?o|??%??!c?9=??:@)
pUj??1?H!?v?1@:Preprocessing2F
Iterator::Modelt??????!?iq?5SD@)??ؙB???1^?2?*1@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[3]::TensorSlice???????!OO1???"@)???????1OO1???"@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip	?c???!e??+ʬM@)?Q,????1}0??`@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor	3m??Js?!?"?=@)	3m??Js?1?"?=@:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
device?Your program is NOT input-bound because only 0.0% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.no*noI???_|9??Q9̀?X@Zno#You may skip the rest of this page.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	?:M?? @?:M?? @!?:M?? @      ??!       "	 &?B:m@ &?B:m@! &?B:m@*      ??!       2	?]=????]=???!?]=???:	?s~??????s~?????!?s~?????B      ??!       J      ??!       R      ??!       Z      ??!       b      ??!       JGPUb q???_|9??y9̀?X@