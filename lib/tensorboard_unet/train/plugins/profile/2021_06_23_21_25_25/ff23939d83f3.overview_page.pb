?	0?GĆm@0?GĆm@!0?GĆm@      ??!       "?
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
	?:M?? @?:M?? @!?:M?? @      ??!       "	 &?B:m@ &?B:m@! &?B:m@*      ??!       2	?]=????]=???!?]=???:	?s~??????s~?????!?s~?????B      ??!       J      ??!       R      ??!       Z      ??!       b      ??!       JGPUb q???_|9??y9̀?X@?"e
9gradient_tape/model/conv2d_20/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter?y?Üv??!?y?Üv??0"e
9gradient_tape/model/conv2d_21/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter?~?]`??!T???eӱ?0"e
9gradient_tape/model/conv2d_11/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter????M???!?q?;????0"e
9gradient_tape/model/conv2d_18/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter7??8-???!?d???0"d
8gradient_tape/model/conv2d_1/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter?%?LU??!???????0"h
>gradient_tape/model/batch_normalization_7/FusedBatchNormGradV3FusedBatchNormGradV3??b?o??!?8l?$??"c
8gradient_tape/model/conv2d_20/Conv2D/Conv2DBackpropInputConv2DBackpropInput ????œ?!?/왝??0"6
model/conv2d_20/Relu_FusedConv2D??7Q??!?!;6????"c
8gradient_tape/model/conv2d_18/Conv2D/Conv2DBackpropInputConv2DBackpropInput??q)f ??!?Xi?	@??0"c
8gradient_tape/model/conv2d_12/Conv2D/Conv2DBackpropInputConv2DBackpropInput??sV????!P?7F?}??0Q      Y@Y??9?n@a??o??W@qE?%b??@yj?CLB?O?"?	
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