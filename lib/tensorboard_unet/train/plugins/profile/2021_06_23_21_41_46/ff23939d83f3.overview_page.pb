?	?Ȓ9?.n@?Ȓ9?.n@!?Ȓ9?.n@      ??!       "?
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
	?lXSY????lXSY???!?lXSY???      ??!       "	????m@????m@!????m@*      ??!       2	?d???d??!?d??:	???T?t?????T?t??!???T?t??B      ??!       J      ??!       R      ??!       Z      ??!       b      ??!       JGPUb q???	???y#f????X@?"e
9gradient_tape/model/conv2d_20/Conv2D/Conv2DBackpropFilterConv2DBackpropFilterG???/s??!G???/s??0"e
9gradient_tape/model/conv2d_11/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter?qb?????!?w??0"c
8gradient_tape/model/conv2d_20/Conv2D/Conv2DBackpropInputConv2DBackpropInputa[# ?X??!?a??,]??0"6
model/conv2d_20/Relu_FusedConv2D^??????!?x;$??"e
9gradient_tape/model/conv2d_21/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter}/o}f$??!tPj??V??0"d
8gradient_tape/model/conv2d_1/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter??????!l?? L???0"e
9gradient_tape/model/conv2d_18/Conv2D/Conv2DBackpropFilterConv2DBackpropFilterHWw?_??!p˧OD???0"6
model/conv2d_18/Relu_FusedConv2D&?_?%??!???????"c
8gradient_tape/model/conv2d_18/Conv2D/Conv2DBackpropInputConv2DBackpropInputw)*?#??!?<?`r???0"c
8gradient_tape/model/conv2d_12/Conv2D/Conv2DBackpropInputConv2DBackpropInput???ʏ?!???????0Q      Y@Y??9?n@a??o??W@q?SJ??@y??9&?MQ?"?	
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