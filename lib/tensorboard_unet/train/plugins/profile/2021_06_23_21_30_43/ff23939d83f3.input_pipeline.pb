	?S?482q@?S?482q@!?S?482q@      ??!       "?
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetailsC?S?482q@??d??n@1??L0?p@A?I/???I"?? >???rEagerKernelExecute 0*	?O??n6b@2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat6;R}???!=????F@)c섗?ԯ?1O#-.?UE@:Preprocessing2U
Iterator::Model::ParallelMapV2???E_A??!?????1@)???E_A??1?????1@:Preprocessing2F
Iterator::ModelIc???&??!R I?/@@)??????1???-@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap???{??!"?:~?k2@)T5A?} ??1Mh?R?!(@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[1]::TensorSlicee?I)????!?Sl@)e?I)????1?Sl@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip?e??S9??!??~??P@)׆?q?&t?1܁Un?@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensorHP?s?r?!??,?A	@)HP?s?r?1??,?A	@:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
device?Your program is NOT input-bound because only 0.0% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.no*noI??iN???Q5X?Ǟ?X@Zno#You may skip the rest of this page.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	??d??n@??d??n@!??d??n@      ??!       "	??L0?p@??L0?p@!??L0?p@*      ??!       2	?I/????I/???!?I/???:	"?? >???"?? >???!"?? >???B      ??!       J      ??!       R      ??!       Z      ??!       b      ??!       JGPUb q??iN???y5X?Ǟ?X@