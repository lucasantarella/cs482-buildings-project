	??"M?So@??"M?So@!??"M?So@      ??!       "?
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetailsC??"M?So@ϣ??????1?L??yo@AQ?O?Iҕ?I??M?B??rEagerKernelExecute 0*	?Zd;`@2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeati5$?????!SoJx`;@)???x???1) 
d?6@:Preprocessing2U
Iterator::Model::ParallelMapV2fO?s???!y?[?6@)fO?s???1y?[?6@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap??s?ᒣ?!'ʹ??=@)$??(?[??1?r???4@:Preprocessing2F
Iterator::Model??^a????!R+e@C@)?/?'??1+?@o?
0@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[1]::TensorSlice?`7l[???!2G???#@)?`7l[???12G???#@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip?[?tYL??!??Ԛ??N@)E?a??x?1?y?8?@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor?c#??w?!?? Q<@)?c#??w?1?? Q<@:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
device?Your program is NOT input-bound because only 0.0% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.no*noI??נDp??QQP?v?X@Zno#You may skip the rest of this page.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	ϣ??????ϣ??????!ϣ??????      ??!       "	?L??yo@?L??yo@!?L??yo@*      ??!       2	Q?O?Iҕ?Q?O?Iҕ?!Q?O?Iҕ?:	??M?B????M?B??!??M?B??B      ??!       J      ??!       R      ??!       Z      ??!       b      ??!       JGPUb q??נDp??yQP?v?X@