?	??Tl̔h@??Tl̔h@!??Tl̔h@      ??!       "?
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetailsC??Tl̔h@????6???1??ޘ=h@A????:q??I???B?i??rEagerKernelExecute 0*	v??/Mc@2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap??ĬC??!\????@)/?o??e??1V??B?6@:Preprocessing2F
Iterator::Model?6?ׯ?!?0tT#D@)F@?#H???14x?5@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat???!?Q??!D?S?#,7@)MK??F>??1wI?l?3@:Preprocessing2U
Iterator::Model::ParallelMapV2?aod??!?G?a?83@)?aod??1?G?a?83@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[1]::TensorSliceiUMu??!@?^?#@)iUMu??1@?^?#@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Ziph!?˛??!w?䋫?M@)?ZӼ?}?15e?=qd@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor?f???u?!??Q@?M@)?f???u?1??Q@?M@:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
device?Your program is NOT input-bound because only 0.0% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.no*noI?y?3?+??Q?0'P?X@Zno#You may skip the rest of this page.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	????6???????6???!????6???      ??!       "	??ޘ=h@??ޘ=h@!??ޘ=h@*      ??!       2	????:q??????:q??!????:q??:	???B?i?????B?i??!???B?i??B      ??!       J      ??!       R      ??!       Z      ??!       b      ??!       JGPUb q?y?3?+??y?0'P?X@?"e
9gradient_tape/model/conv2d_20/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter?ƊR(???!?ƊR(???0"e
9gradient_tape/model/conv2d_21/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter?7g?dЗ?!B1?N????0"d
8gradient_tape/model/conv2d_1/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter?OP?????!;Es?\ɷ?0"6
model/conv2d_20/Relu_FusedConv2D?2_in??!{??M?d??"c
8gradient_tape/model/conv2d_20/Conv2D/Conv2DBackpropInputConv2DBackpropInput%]d?????!??,?6r??0"e
9gradient_tape/model/conv2d_18/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter?%E?_~??!?)??"??0"e
9gradient_tape/model/conv2d_11/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter}}?E???!??d?K???0"c
8gradient_tape/model/conv2d_18/Conv2D/Conv2DBackpropInputConv2DBackpropInput?4?&????!c?7ʥ??0"e
9gradient_tape/model/conv2d_12/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter#$J3???!?b?:=???0"6
model/conv2d_18/Relu_FusedConv2D? ? 0??!%dE????Q      Y@Y???,d@a???7??W@q??*Z????yg??:VwS?"?	
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