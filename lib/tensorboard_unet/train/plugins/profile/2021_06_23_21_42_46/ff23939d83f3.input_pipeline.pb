	?????]n@?????]n@!?????]n@	?????z???????z??!?????z??"?
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetailsL?????]n@???RA?@1??k?1n@A??@?ȓ?I/??w???Ys???M??rEagerKernelExecute 0*	??Q??]@2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeatr??????!????f?=@)???E???1tm????8@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap*oG8-x??!H?^͗<@)R?????1?D@d3@:Preprocessing2U
Iterator::Model::ParallelMapV2?GĔH???!
?Y?`W3@)?GĔH???1
?Y?`W3@:Preprocessing2F
Iterator::Model??|\*??!>?;??#B@)ni5$?1s????0@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[1]::TensorSlice?G??|??!,???g"@)?G??|??1,???g"@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::ZipH0?[w??!?V?@y?O@)??Q,??z?1?V.???@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor|,}???v?!g@~?@)|,}???v?1g@~?@:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
device?Your program is NOT input-bound because only 0.0% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.no*no9?????z??I 
!?!=??Q??!<?X@Zno#You may skip the rest of this page.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	???RA?@???RA?@!???RA?@      ??!       "	??k?1n@??k?1n@!??k?1n@*      ??!       2	??@?ȓ???@?ȓ?!??@?ȓ?:	/??w???/??w???!/??w???B      ??!       J	s???M??s???M??!s???M??R      ??!       Z	s???M??s???M??!s???M??b      ??!       JGPUY?????z??b q 
!?!=??y??!<?X@