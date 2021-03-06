?	?S?482q@?S?482q@!?S?482q@      ??!       "?
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
	??d??n@??d??n@!??d??n@      ??!       "	??L0?p@??L0?p@!??L0?p@*      ??!       2	?I/????I/???!?I/???:	"?? >???"?? >???!"?? >???B      ??!       J      ??!       R      ??!       Z      ??!       b      ??!       JGPUb q??iN???y5X?Ǟ?X@?"e
9gradient_tape/model/conv2d_20/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter??DoT=??!??DoT=??0"d
8gradient_tape/model/conv2d_1/Conv2D/Conv2DBackpropFilterConv2DBackpropFilterP?f?????!:??/39??0"h
>gradient_tape/model/batch_normalization_7/FusedBatchNormGradV3FusedBatchNormGradV3qE/????!y?G?b??"e
9gradient_tape/model/conv2d_21/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter34?$ʖ?!?&vQ???0"c
8gradient_tape/model/conv2d_20/Conv2D/Conv2DBackpropInputConv2DBackpropInputn?OY%??!?????N??0"6
model/conv2d_20/Relu_FusedConv2DE'm
???!???s????"e
9gradient_tape/model/conv2d_18/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter?~R? ???!?ޔ?????0"e
9gradient_tape/model/conv2d_12/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter?(?^?>??!?#h?????0"e
9gradient_tape/model/conv2d_11/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter?[?????! O{1????0"c
8gradient_tape/model/conv2d_11/Conv2D/Conv2DBackpropInputConv2DBackpropInput?n18j?!?|?x????0Q      Y@Y??R?y@aw??jc?W@qY?K??	@y?<?9?8N?"?	
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