	???[?n@???[?n@!???[?n@      ??!       "?
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetailsC???[?n@??? 8??1İØt>n@A??je?/??I?HM?????rEagerKernelExecute 0*	/?$h@2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor????V`??!(?????@@)????V`??1(?????@@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat}?%?/??!/?;??H@)MK??F>??1 ?X-??/@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap??^???!??]5m@5@)?ZӼ???1?OB?-@:Preprocessing2U
Iterator::Model::ParallelMapV2???כ?!Ӱ?[uQ,@)???כ?1Ӱ?[uQ,@:Preprocessing2F
Iterator::Model?????!n?)?`?9@)1[?*?M??1	?}L?&@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[1]::TensorSlice??%jj??!?J9Q??@)??%jj??1?J9Q??@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip??*??O??!e?????R@)?=~o??1M8O/@:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
device?Your program is NOT input-bound because only 0.0% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.no*noI??u<.??Q??)G?X@Zno#You may skip the rest of this page.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	??? 8????? 8??!??? 8??      ??!       "	İØt>n@İØt>n@!İØt>n@*      ??!       2	??je?/????je?/??!??je?/??:	?HM??????HM?????!?HM?????B      ??!       J      ??!       R      ??!       Z      ??!       b      ??!       JGPUb q??u<.??y??)G?X@