?	??"M?So@??"M?So@!??"M?So@      ??!       "?
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
	ϣ??????ϣ??????!ϣ??????      ??!       "	?L??yo@?L??yo@!?L??yo@*      ??!       2	Q?O?Iҕ?Q?O?Iҕ?!Q?O?Iҕ?:	??M?B????M?B??!??M?B??B      ??!       J      ??!       R      ??!       Z      ??!       b      ??!       JGPUb q??נDp??yQP?v?X@?"e
9gradient_tape/model/conv2d_21/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter:?<???!:?<???0"e
9gradient_tape/model/conv2d_20/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter?]???&??!?KF???0"d
8gradient_tape/model/conv2d_1/Conv2D/Conv2DBackpropFilterConv2DBackpropFilteri??+Q???!?	?[???0"6
model/conv2d_20/Relu_FusedConv2D?8	?U??!?+?Hp???"e
9gradient_tape/model/conv2d_11/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter???˻???!?_>?G???0"c
8gradient_tape/model/conv2d_20/Conv2D/Conv2DBackpropInputConv2DBackpropInputί????!?Y???[??0"e
9gradient_tape/model/conv2d_12/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter?/??o??!??$ө??0"c
8gradient_tape/model/conv2d_11/Conv2D/Conv2DBackpropInputConv2DBackpropInput1'?5???!?$Rٹ???0"C
%gradient_tape/model/conv2d_1/ReluGradReluGrad	R՟???!??L?{??"e
9gradient_tape/model/conv2d_18/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter?^?O???!T]??2??0Q      Y@Y?????@a?cp>?W@q82?d?m	@y+U??RuP?"?	
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