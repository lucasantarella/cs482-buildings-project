?	?????]n@?????]n@!?????]n@	?????z???????z??!?????z??"?
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
!?!=??y??!<?X@?"e
9gradient_tape/model/conv2d_20/Conv2D/Conv2DBackpropFilterConv2DBackpropFilterW<????!W<????0"d
8gradient_tape/model/conv2d_1/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter]!:$㗙?!?&?????0"e
9gradient_tape/model/conv2d_21/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter1?x????!OQ?6????0"e
9gradient_tape/model/conv2d_18/Conv2D/Conv2DBackpropFilterConv2DBackpropFilterY?%m????!????f???0"6
model/conv2d_20/Relu_FusedConv2D??0K#??!ǚ?Μ???"c
8gradient_tape/model/conv2d_20/Conv2D/Conv2DBackpropInputConv2DBackpropInputn?????!??̧`???0"c
8gradient_tape/model/conv2d_18/Conv2D/Conv2DBackpropInputConv2DBackpropInputu???T???!?E?C+???0"c
8gradient_tape/model/conv2d_12/Conv2D/Conv2DBackpropInputConv2DBackpropInput[???n???!b?A????0"c
8gradient_tape/model/conv2d_11/Conv2D/Conv2DBackpropInputConv2DBackpropInput? ???b??!x)ԗO???0"e
9gradient_tape/model/conv2d_11/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter?*Ё?ď?!$,??????0Q      Y@Y՞?髄@a?fA??W@qI q7???ybaf<IT?"?	
device?Your program is NOT input-bound because only 0.0% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*?
?<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2?
=type.googleapis.com/tensorflow.profiler.GenericRecommendation?
nono*?Only 0.0% of device computation is 16 bit. So you might want to replace more 32-bit Ops by 16-bit Ops to improve performance (if the reduced accuracy is acceptable).2no:
Refer to the TF2 Profiler FAQ2"Nvidia GPU (Ampere)(: B 