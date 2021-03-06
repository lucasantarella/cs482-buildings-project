?	??Z?ۙh@??Z?ۙh@!??Z?ۙh@      ??!       "?
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetailsC??Z?ۙh@e?9:Z??1?~T^h@AE?D??2??I]??'???rEagerKernelExecute 0*	Zd;?O!`@2U
Iterator::Model::ParallelMapV2? ?	???!?*?WN?7@)? ?	???1?*?WN?7@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat??D?֠?!gi?|9@)??6???1?F 5@:Preprocessing2F
Iterator::Model?HZ????!T:??.E@)G?J??q??1	????2@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMapn?????!)S"?Y;@)??ڊ?e??1??p"?1@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[1]::TensorSlice|DL?$z??!?lc??G#@)|DL?$z??1?lc??G#@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip}?͍?	??!????L@)_??x?Zy?1M? 0@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor,+MJA?w?!DL????@),+MJA?w?1DL????@:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
device?Your program is NOT input-bound because only 0.0% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.no*noI ?-/"???Q??????X@Zno#You may skip the rest of this page.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	e?9:Z??e?9:Z??!e?9:Z??      ??!       "	?~T^h@?~T^h@!?~T^h@*      ??!       2	E?D??2??E?D??2??!E?D??2??:	]??'???]??'???!]??'???B      ??!       J      ??!       R      ??!       Z      ??!       b      ??!       JGPUb q ?-/"???y??????X@?"e
9gradient_tape/model/conv2d_20/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter?Q??????!?Q??????0"d
8gradient_tape/model/conv2d_1/Conv2D/Conv2DBackpropFilterConv2DBackpropFilterw?jN???!?Q?n?հ?0"e
9gradient_tape/model/conv2d_21/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter5???r??!?O?v]???0"6
model/conv2d_20/Relu_FusedConv2Dn?#??W??!2>??>H??"c
8gradient_tape/model/conv2d_20/Conv2D/Conv2DBackpropInputConv2DBackpropInputd.??????!??O4???0"e
9gradient_tape/model/conv2d_18/Conv2D/Conv2DBackpropFilterConv2DBackpropFilterz̕?!??Ķ????0"e
9gradient_tape/model/conv2d_11/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter\|W"???!@??8??0"c
8gradient_tape/model/conv2d_18/Conv2D/Conv2DBackpropInputConv2DBackpropInput?1v1Ɛ?!???(?)??0"e
9gradient_tape/model/conv2d_12/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter????-??!Xyꦠ/??0"c
8gradient_tape/model/conv2d_12/Conv2D/Conv2DBackpropInputConv2DBackpropInputّ?e?.??!v?D????0Q      Y@Y???,d@a???7??W@q]?#z?@y??bR?S?"?	
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