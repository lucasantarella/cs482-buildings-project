
?
?void wgrad_alg0_engine<float, 512, 6, 5, 3, 3, 3, false, 512>(int, int, int, float const*, int, float*, float const*, kernel_grad_params, unsigned long long, int, float, int, int, int, int)_?*2	?8堕@堕H堕Xb9gradient_tape/model/conv2d_20/Conv2D/Conv2DBackpropFilterhu??&B
?
?void wgrad_alg0_engine<float, 512, 6, 5, 3, 3, 3, false, 512>(int, int, int, float const*, int, float*, float const*, kernel_grad_params, unsigned long long, int, float, int, int, int, int)_?*2?8???@???H???Xb8gradient_tape/model/conv2d_1/Conv2D/Conv2DBackpropFilterhu??&B
?
?void wgrad_alg0_engine<float, 512, 6, 5, 3, 3, 3, false, 512>(int, int, int, float const*, int, float*, float const*, kernel_grad_params, unsigned long long, int, float, int, int, int, int)_?*2?8???@???H???Xb9gradient_tape/model/conv2d_21/Conv2D/Conv2DBackpropFilterhu??&B
{
<ampere_scudnn_winograd_128x128_ldg1_ldg4_relu_tile148t_nt_v1~??*?22
8È?@È?HÈ?bmodel/conv2d_20/ReluhuMUB
?
<ampere_scudnn_winograd_128x128_ldg1_ldg4_relu_tile148t_nt_v1~??*?2d
8ď?@ď?Hď?Xb8gradient_tape/model/conv2d_20/Conv2D/Conv2DBackpropInputhuMUB
?
<ampere_scudnn_winograd_128x128_ldg1_ldg4_relu_tile148t_nt_v1~??*?2?
8???@???H???Xb8gradient_tape/model/conv2d_18/Conv2D/Conv2DBackpropInputhuMUB
?
6ampere_scudnn_128x64_stridedB_splitK_xregs_large_nn_v1???*?2P8???@???H???Xb9gradient_tape/model/conv2d_19/Conv2D/Conv2DBackpropFilterhu  ?A
d
%ampere_scudnn_128x32_relu_small_nn_v1??**@2?N8Ͻ?@Ͻ?HϽ?bmodel/conv2d_21/ReluhuMUB
c
%ampere_scudnn_128x32_relu_small_nn_v1??**@2?N8?³@?³H?³bmodel/conv2d_1/ReluhuMUB
?
<ampere_scudnn_winograd_128x128_ldg1_ldg4_relu_tile148t_nt_v1~??*?22
8???@???H???Xb8gradient_tape/model/conv2d_21/Conv2D/Conv2DBackpropInputhuMUB
?
<ampere_scudnn_winograd_128x128_ldg1_ldg4_relu_tile148t_nt_v1~??*?22
8і?@і?Hі?Xb7gradient_tape/model/conv2d_1/Conv2D/Conv2DBackpropInputhuMUB
{
<ampere_scudnn_winograd_128x128_ldg1_ldg4_relu_tile148t_nt_v1~??*?2d
8???@???H???bmodel/conv2d_18/ReluhuMUB
?
6ampere_scudnn_128x64_stridedB_splitK_xregs_large_nn_v1???*?2P8ѕ?@ѕ?Hѕ?Xb8gradient_tape/model/conv2d_2/Conv2D/Conv2DBackpropFilterhu  ?A
?
?void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorSlicingOp<Eigen::array<int, 2ul> const, Eigen::array<int, 2ul> const, Eigen::TensorMap<Eigen::Tensor<float, 2, 1, int>, 16, Eigen::MakePointer> >, Eigen::TensorMap<Eigen::Tensor<float const, 2, 1, int>, 16, Eigen::MakePointer> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorSlicingOp<Eigen::array<int, 2ul> const, Eigen::array<int, 2ul> const, Eigen::TensorMap<Eigen::Tensor<float, 2, 1, int>, 16, Eigen::MakePointer> >, Eigen::TensorMap<Eigen::Tensor<float const, 2, 1, int>, 16, Eigen::MakePointer> const> const, Eigen::GpuDevice>, int)*?298???@??5H??mbmodel/concatenate_4/concathuZU?B
|
ampere_sgemm_128x128_ntv??*?2$8?á@?áH?áXb8gradient_tape/model/conv2d_12/Conv2D/Conv2DBackpropInputhuMUB
?
6ampere_scudnn_128x64_stridedB_splitK_xregs_large_nn_v1???*?2P8ї?@ї?Hї?Xb8gradient_tape/model/conv2d_3/Conv2D/Conv2DBackpropFilterhu  ?A
}
ampere_sgemm_128x128_ntv??*?2$8҆?@҆?H҆?Xb9gradient_tape/model/conv2d_18/Conv2D/Conv2DBackpropFilterhuMUB
?
}void cudnn::winograd_nonfused::winogradWgradData4x4<float, float>(cudnn::winograd_nonfused::WinogradDataParams<float, float>)@??*?2?28???@???H???Xb9gradient_tape/model/conv2d_18/Conv2D/Conv2DBackpropFilterhuZU?B
|
ampere_sgemm_128x128_ntv??*?2$8???@???H???Xb8gradient_tape/model/conv2d_11/Conv2D/Conv2DBackpropInputhuMUB
?
?void cudnn::bn_bw_1C11_kernel_new<float, float, float2, 512, true, 1>(float, float, float, float, cudnnTensorStruct, float const*, cudnnTensorStruct, float const*, cudnnTensorStruct, float*, float const*, float*, float*, float const*, float const*, float)(?*?2?8???@???H???b>gradient_tape/model/batch_normalization_7/FusedBatchNormGradV3hu  ?B
{
<ampere_scudnn_winograd_128x128_ldg1_ldg4_relu_tile148t_nt_v1~??*?2d
8?Ã@?ÃH?Ãbmodel/conv2d_19/ReluhuMUB
S
ampere_sgemm_128x128_nnv??*?2$8??{@??{H??{bmodel/conv2d_11/ReluhuMUB
?
<ampere_scudnn_winograd_128x128_ldg1_ldg4_relu_tile148t_nt_v1~??*?2d
8??y@??yH??yXb7gradient_tape/model/conv2d_3/Conv2D/Conv2DBackpropInputhuMUB
S
ampere_sgemm_128x128_nnv??*?2$8??y@??yH??ybmodel/conv2d_12/ReluhuMUB
S
ampere_sgemm_128x128_nnv??*?2$8??x@??xH??xbmodel/conv2d_13/ReluhuMUB
?
?void cudnn::bn_fw_tr_1C11_kernel_NCHW<float, float, 512, true, 1>(cudnnTensorStruct, float const*, cudnnTensorStruct, float*, float const*, float const*, float, float, float*, float*, float*, float*, float, float)(?*?2?8??u@??uH??ub,model/batch_normalization_7/FusedBatchNormV3hu  ?B
y
ampere_sgemm_128x128_ntv??*?22$8??q@??qH??qXb8gradient_tape/model/conv2d_16/Conv2D/Conv2DBackpropInputhuMUB
?
?void wgrad_alg0_engine<float, 128, 5, 5, 3, 3, 3, false, 512>(int, int, int, float const*, int, float*, float const*, kernel_grad_params, unsigned long long, int, float, int, int, int, int)P?*2?8??p@??pH??pXb6gradient_tape/model/conv2d/Conv2D/Conv2DBackpropFilterhu  HB
?
?void tensorflow::functor::ShuffleInTensor3Simple<float, 2, 1, 0, false>(int, float const*, tensorflow::functor::Dimension<3>, float*)*?2&8??o@??oH??oXb9gradient_tape/model/conv2d_11/Conv2D/Conv2DBackpropFilterhuZU?B
y
ampere_sgemm_128x128_ntv??*?2$8ֵn@ֵnHֵnXb8gradient_tape/model/conv2d_14/Conv2D/Conv2DBackpropInputhuMUB
?
?void cudnn::detail::dgrad_engine<float, 128, 6, 8, 3, 3, 5, false>(int, int, int, float const*, int, float const*, int, float*, kernel_grad_params, unsigned long long, int, unsigned long long, int, float, int, int, int)~?R* 228??l@??lH??lb)model/conv2d_transpose_4/conv2d_transposehuMUB
?
?void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*?298??i@??iH??ibAdam/gradients/AddN_5huZU?B
y
ampere_sgemm_128x128_ntv??*?2$8??i@??iH??iXb8gradient_tape/model/conv2d_9/Conv2D/Conv2DBackpropFilterhuMUB
z
ampere_sgemm_128x128_ntv??*?2$8??g@??gH??gXb9gradient_tape/model/conv2d_13/Conv2D/Conv2DBackpropFilterhuMUB
?
?void cutlass_cudnn::Kernel<cutlass_tensorop_s1688wgrad_analytic_tf32_128x256_16x3>(cutlass_tensorop_s1688wgrad_analytic_tf32_128x256_16x3::Params)? ??*?2 	8??c@??cH??cXb9gradient_tape/model/conv2d_11/Conv2D/Conv2DBackpropFilterh
?
3ampere_scudnn_128x64_stridedB_splitK_interior_nn_v1???*?2(8??b@??bH??bXbLgradient_tape/model/conv2d_transpose_3/conv2d_transpose/Conv2DBackpropFilterhu  ?A
z
ampere_sgemm_128x128_ntv??*?2$8??b@??bH??bXb9gradient_tape/model/conv2d_16/Conv2D/Conv2DBackpropFilterhuMUB
z
ampere_sgemm_128x128_ntv??*?2$8??b@??bH??bXb9gradient_tape/model/conv2d_14/Conv2D/Conv2DBackpropFilterhuMUB
?
<ampere_scudnn_winograd_128x128_ldg1_ldg4_relu_tile148t_nt_v1~??*?2d
8??a@??aH??aXb8gradient_tape/model/conv2d_19/Conv2D/Conv2DBackpropInputhuMUB
y
ampere_sgemm_128x128_ntv??*?2$8??`@??`H??`Xb8gradient_tape/model/conv2d_13/Conv2D/Conv2DBackpropInputhuMUB
w
<ampere_scudnn_winograd_128x128_ldg1_ldg4_relu_tile148t_nt_v1~??*?2d
8??]@??]H??]bmodel/conv2d_3/ReluhuMUB
?
?void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_product_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorConversionOp<float, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_cmp_op<float const, float const, (Eigen::internal::ComparisonName)5>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorCwiseNullaryOp<Eigen::internal::scalar_constant_op<float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_product_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorConversionOp<float, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_cmp_op<float const, float const, (Eigen::internal::ComparisonName)5>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorCwiseNullaryOp<Eigen::internal::scalar_constant_op<float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const> const> const> const, Eigen::GpuDevice>, long)*?298??]@??]H??]b%gradient_tape/model/conv2d_2/ReluGradhuZU?B
?
,ampere_scudnn_128x64_stridedB_interior_nn_v1???*?2?N8??\@??\H??\Xb8gradient_tape/model/conv2d_22/Conv2D/Conv2DBackpropInputhu  ?A
z
ampere_sgemm_128x128_ntv??*?2$8??[@??[H??[Xb9gradient_tape/model/conv2d_12/Conv2D/Conv2DBackpropFilterhuMUB
?
(ampere_scudnn_128x64_relu_interior_nn_v1???*?2?8??Z@??ZH??ZXb>gradient_tape/model/conv2d_transpose_2/conv2d_transpose/Conv2DhuMUB
S
ampere_sgemm_128x128_nnv??*?2$8??V@??VH??Vbmodel/conv2d_10/ReluhuMUB
?
<ampere_scudnn_winograd_128x128_ldg1_ldg4_relu_tile148t_nt_v1~??*?2d8??U@??UH??UXb7gradient_tape/model/conv2d_4/Conv2D/Conv2DBackpropInputhuMUB
?
?void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 4, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorSlicingOp<Eigen::DSizes<int, 4> const, Eigen::DSizes<int, 4> const, Eigen::TensorMap<Eigen::Tensor<float const, 4, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 4, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorSlicingOp<Eigen::DSizes<int, 4> const, Eigen::DSizes<int, 4> const, Eigen::TensorMap<Eigen::Tensor<float const, 4, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int)*?298??U@??UH??Ub'gradient_tape/model/concatenate_4/SlicehuZU?B
?
?void cudnn::winograd_nonfused::winogradForwardOutput4x4<float, float>(cudnn::winograd_nonfused::WinogradOutputParams<float, float>)0??*?2$?8??T@??TH??TXb7gradient_tape/model/conv2d_5/Conv2D/Conv2DBackpropInputhu???B
S
ampere_sgemm_128x128_nnv??*?2$8??R@??RH??Rbmodel/conv2d_14/ReluhuMUB
S
ampere_sgemm_128x128_nnv??*?22$8??Q@??QH??Qbmodel/conv2d_16/ReluhuMUB
?
?void cutlass_cudnn::Kernel<cutlass_tensorop_s1688dgrad_optimized_tf32_128x128_16x4_unity_stride>(cutlass_tensorop_s1688dgrad_optimized_tf32_128x128_16x4_unity_stride::Params)? ??*?2?8??P@??PH??Pb)model/conv2d_transpose_3/conv2d_transposeh
x
ampere_sgemm_128x128_ntv??*?2$8??N@??NH??NXb7gradient_tape/model/conv2d_7/Conv2D/Conv2DBackpropInputhuMUB
?
?void cutlass_cudnn::Kernel<cutlass_tensorop_s1688wgrad_analytic_tf32_128x256_16x3>(cutlass_tensorop_s1688wgrad_analytic_tf32_128x256_16x3::Params)? ??*?2 8??N@??NH??NXb9gradient_tape/model/conv2d_10/Conv2D/Conv2DBackpropFilterh
?
?void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_product_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorConversionOp<float, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_cmp_op<float const, float const, (Eigen::internal::ComparisonName)5>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorCwiseNullaryOp<Eigen::internal::scalar_constant_op<float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_product_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorConversionOp<float, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_cmp_op<float const, float const, (Eigen::internal::ComparisonName)5>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorCwiseNullaryOp<Eigen::internal::scalar_constant_op<float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const> const> const> const, Eigen::GpuDevice>, long)*?298وN@وNHوNb&gradient_tape/model/conv2d_21/ReluGradhuZU?B
?
?void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_product_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorConversionOp<float, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_cmp_op<float const, float const, (Eigen::internal::ComparisonName)5>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorCwiseNullaryOp<Eigen::internal::scalar_constant_op<float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_product_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorConversionOp<float, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_cmp_op<float const, float const, (Eigen::internal::ComparisonName)5>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorCwiseNullaryOp<Eigen::internal::scalar_constant_op<float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const> const> const> const, Eigen::GpuDevice>, long)*?298??L@??LH??Lb%gradient_tape/model/conv2d_1/ReluGradhuZU?B
?
?void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_product_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorConversionOp<float, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_cmp_op<float const, float const, (Eigen::internal::ComparisonName)5>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorCwiseNullaryOp<Eigen::internal::scalar_constant_op<float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_product_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorConversionOp<float, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_cmp_op<float const, float const, (Eigen::internal::ComparisonName)5>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorCwiseNullaryOp<Eigen::internal::scalar_constant_op<float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const> const> const> const, Eigen::GpuDevice>, long)*?298??K@??KH??Kb&gradient_tape/model/conv2d_20/ReluGradhuZU?B
?
?void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_product_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorConversionOp<float, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_cmp_op<float const, float const, (Eigen::internal::ComparisonName)5>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorCwiseNullaryOp<Eigen::internal::scalar_constant_op<float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_product_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorConversionOp<float, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_cmp_op<float const, float const, (Eigen::internal::ComparisonName)5>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorCwiseNullaryOp<Eigen::internal::scalar_constant_op<float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const> const> const> const, Eigen::GpuDevice>, long)*?298??K@??KH??Kb#gradient_tape/model/conv2d/ReluGradhuZU?B
?
?void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_product_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorConversionOp<float, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_cmp_op<float const, float const, (Eigen::internal::ComparisonName)5>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorCwiseNullaryOp<Eigen::internal::scalar_constant_op<float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_product_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorConversionOp<float, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_cmp_op<float const, float const, (Eigen::internal::ComparisonName)5>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorCwiseNullaryOp<Eigen::internal::scalar_constant_op<float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const> const> const> const, Eigen::GpuDevice>, long)*?298??G@??GH??Gb&gradient_tape/model/conv2d_19/ReluGradhuZU?B
?
?void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*?298??F@??FH??FbAdam/gradients/AddN_4huZU?B
?
?void cudnn::ops::pooling_bw_kernel_max<float, float, cudnn::maxpooling_func<float, (cudnnNanPropagation_t)0>, false>(cudnnTensorStruct, float const*, cudnnTensorStruct, float const*, cudnnTensorStruct, float const*, cudnnTensorStruct, float*, cudnnPoolingStruct, float, float, int, cudnn::reduced_divisor, cudnn::reduced_divisor) ? *?2228??F@??FH??Fb5gradient_tape/model/max_pooling2d/MaxPool/MaxPoolGradhu  ?B
?
?void wgrad_alg0_engine<float, 512, 6, 5, 3, 3, 3, false, 512>(int, int, int, float const*, int, float*, float const*, kernel_grad_params, unsigned long long, int, float, int, int, int, int)_?*2?8څE@څEHڅEXbLgradient_tape/model/conv2d_transpose_4/conv2d_transpose/Conv2DBackpropFilterhu??&B
?
?void wgrad_alg0_engine<float, 128, 5, 5, 3, 3, 3, false, 512>(int, int, int, float const*, int, float*, float const*, kernel_grad_params, unsigned long long, int, float, int, int, int, int)P?*2?8??D@??DH??DXb9gradient_tape/model/conv2d_22/Conv2D/Conv2DBackpropFilterhu  HB
R
ampere_sgemm_128x128_nnv??*?22$8??D@??DH??Dbmodel/conv2d_5/ReluhuMUB
?
<ampere_scudnn_winograd_128x128_ldg1_ldg4_relu_tile148t_nt_v1~??*?22
8ڠA@ڠAHڠAXb7gradient_tape/model/conv2d_2/Conv2D/Conv2DBackpropInputhuMUB
?
?void cudnn::bn_bw_1C11_kernel_new<float, float, float2, 128, true, 1>(float, float, float, float, cudnnTensorStruct, float const*, cudnnTensorStruct, float const*, cudnnTensorStruct, float*, float const*, float*, float*, float const*, float const*, float)(?*?2?8??@@??@H??@b>gradient_tape/model/batch_normalization_6/FusedBatchNormGradV3hu  ?B
R
ampere_sgemm_128x128_nnv??*?2$8??@@??@H??@bmodel/conv2d_9/ReluhuMUB
R
ampere_sgemm_128x128_nnv??*?2$8??@@??@H??@bmodel/conv2d_7/ReluhuMUB
?
(ampere_scudnn_128x64_relu_interior_nn_v1???*?2?8??>@??>H??>Xb>gradient_tape/model/conv2d_transpose_4/conv2d_transpose/Conv2DhuMUB
?
void cudnn::winograd_nonfused::winogradForwardData4x4<float, float>(cudnn::winograd_nonfused::WinogradDataParams<float, float>)@??*?2?8??=@??=H??=bmodel/conv2d_14/ReluhuZU?B
x
ampere_sgemm_128x128_ntv??*?2$8??=@??=H??=Xb7gradient_tape/model/conv2d_9/Conv2D/Conv2DBackpropInputhuMUB
?
?void tensorflow::functor::SwapDimension1And2InTensor3UsingTiles<unsigned int, 256, 32, 32, false>(unsigned int const*, tensorflow::functor::Dimension<3>, unsigned int*) ?!*?2?N8??;@??;H??;bCmodel/dropout_6/dropout/Mul_1-0-TransposeNHWCToNCHW-LayoutOptimizerhu  ?B
b
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*?2?'8??;@??;H??;bmodel/dropout_7/dropout/Mul_1huZU?B
w
<ampere_scudnn_winograd_128x128_ldg1_ldg4_relu_tile148t_nt_v1~??*?2d
8??:@??:H??:bmodel/conv2d_2/ReluhuMUB
?
}void cudnn::winograd_nonfused::winogradWgradData4x4<float, float>(cudnn::winograd_nonfused::WinogradDataParams<float, float>)@??*?2<28ۭ:@ۭ:Hۭ:Xb9gradient_tape/model/conv2d_17/Conv2D/Conv2DBackpropFilterhuZU?B
?
?void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<float, Eigen::TensorMap<Eigen::Tensor<bool const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<float, Eigen::TensorMap<Eigen::Tensor<bool const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*?298??9@??9H??9bXmodel/dropout_6/dropout/ArithmeticOptimizer/ReorderCastLikeAndValuePreserving_float_CasthuZU?B
?
?void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 4, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorSlicingOp<Eigen::DSizes<int, 4> const, Eigen::DSizes<int, 4> const, Eigen::TensorMap<Eigen::Tensor<float const, 4, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 4, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorSlicingOp<Eigen::DSizes<int, 4> const, Eigen::DSizes<int, 4> const, Eigen::TensorMap<Eigen::Tensor<float const, 4, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int)*?298??8@??8H??8b)gradient_tape/model/concatenate_4/Slice_1huZU?B
?
?void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<float, Eigen::TensorMap<Eigen::Tensor<bool const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<float, Eigen::TensorMap<Eigen::Tensor<bool const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*?298ە8@ە8Hە8bmodel/dropout_6/dropout/CasthuZU?B
?
?void tensorflow::functor::ShuffleInTensor3Simple<float, 2, 1, 0, false>(int, float const*, tensorflow::functor::Dimension<3>, float*)*?2&8??7@??7H??7Xb8gradient_tape/model/conv2d_11/Conv2D/Conv2DBackpropInputhuZU?B
?
Yvoid tensorflow::BiasNCHWKernel<float>(int, float const*, float const*, float*, int, int)*?2&8??7@??7H??7b model/conv2d_transpose_4/BiasAddhuZU?B
?
?void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorSlicingOp<Eigen::array<int, 2ul> const, Eigen::array<int, 2ul> const, Eigen::TensorMap<Eigen::Tensor<float, 2, 1, int>, 16, Eigen::MakePointer> >, Eigen::TensorMap<Eigen::Tensor<float const, 2, 1, int>, 16, Eigen::MakePointer> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorSlicingOp<Eigen::array<int, 2ul> const, Eigen::array<int, 2ul> const, Eigen::TensorMap<Eigen::Tensor<float, 2, 1, int>, 16, Eigen::MakePointer> >, Eigen::TensorMap<Eigen::Tensor<float const, 2, 1, int>, 16, Eigen::MakePointer> const> const, Eigen::GpuDevice>, int)*?298??6@??H??3bmodel/concatenate/concathuZU?B
?
?void cudnn::winograd_nonfused::winogradForwardFilter4x4<float, float>(cudnn::winograd_nonfused::WinogradFilterParams<float, float>)(?H* 2@8??6@??6H??6bmodel/conv2d_9/Reluhu  ?B
?
?void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorSlicingOp<Eigen::array<int, 2ul> const, Eigen::array<int, 2ul> const, Eigen::TensorMap<Eigen::Tensor<float, 2, 1, int>, 16, Eigen::MakePointer> >, Eigen::TensorMap<Eigen::Tensor<float const, 2, 1, int>, 16, Eigen::MakePointer> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorSlicingOp<Eigen::array<int, 2ul> const, Eigen::array<int, 2ul> const, Eigen::TensorMap<Eigen::Tensor<float, 2, 1, int>, 16, Eigen::MakePointer> >, Eigen::TensorMap<Eigen::Tensor<float const, 2, 1, int>, 16, Eigen::MakePointer> const> const, Eigen::GpuDevice>, int)*?298??6@ݒH??bmodel/concatenate_3/concathuZU?B
?
?void cudnn::detail::dgrad_engine<float, 128, 6, 8, 3, 3, 5, false>(int, int, int, float const*, int, float const*, int, float*, kernel_grad_params, unsigned long long, int, unsigned long long, int, float, int, int, int)~?R* 228??4@??4H??4b)model/conv2d_transpose_2/conv2d_transposehuMUB
y
ampere_sgemm_128x128_ntv??*?2$8??3@??3H??3Xb8gradient_tape/model/conv2d_10/Conv2D/Conv2DBackpropInputhuMUB
?
void cudnn::winograd_nonfused::winogradForwardData4x4<float, float>(cudnn::winograd_nonfused::WinogradDataParams<float, float>)@??*?2$?8??3@??3H??3bmodel/conv2d_16/ReluhuZU?B
?
?void tensorflow::functor::ShuffleInTensor3Simple<float, 2, 1, 0, false>(int, float const*, tensorflow::functor::Dimension<3>, float*)*?2&8??2@??2H??2Xb9gradient_tape/model/conv2d_12/Conv2D/Conv2DBackpropFilterhuZU?B
?
void cudnn::winograd_nonfused::winogradForwardData4x4<float, float>(cudnn::winograd_nonfused::WinogradDataParams<float, float>)@??*?2?8??0@??0H??0Xb8gradient_tape/model/conv2d_15/Conv2D/Conv2DBackpropInputhuZU?B
x
ampere_sgemm_128x128_ntv??*?2$8۫/@۫/H۫/Xb7gradient_tape/model/conv2d_6/Conv2D/Conv2DBackpropInputhuMUB
?
?void tensorflow::functor::ShuffleInTensor3Simple<float, 2, 1, 0, false>(int, float const*, tensorflow::functor::Dimension<3>, float*)*?2&8??.@??.H??.Xb9gradient_tape/model/conv2d_10/Conv2D/Conv2DBackpropFilterhuZU?B
?
?void cudnn::winograd_nonfused::winogradForwardOutput4x4<float, float>(cudnn::winograd_nonfused::WinogradOutputParams<float, float>)0??*?2$?8??.@??.H??.Xb8gradient_tape/model/conv2d_16/Conv2D/Conv2DBackpropInputhu???B
?
~sm80_xmma_fprop_implicit_gemm_tf32f32_tf32f32_f32_nhwckrsc_nchw_tilesize128x128x16_stage4_warpsize2x2x1_g1_tensor16x8x8_kernel? ??*?2?8??.@??.H??.PXbmodel/conv2d_4/Reluh
?
?void cudnn::winograd_nonfused::winogradForwardOutput4x4<float, float>(cudnn::winograd_nonfused::WinogradOutputParams<float, float>)0??*?2?8??.@??.H??.Xb7gradient_tape/model/conv2d_9/Conv2D/Conv2DBackpropInputhu???B
?
Yvoid tensorflow::BiasNCHWKernel<float>(int, float const*, float const*, float*, int, int)*?2&8??-@??-H??-b model/conv2d_transpose_2/BiasAddhuZU?B
y
ampere_sgemm_128x128_ntv??*?22$8??-@??-H??-Xb8gradient_tape/model/conv2d_17/Conv2D/Conv2DBackpropInputhuMUB
S
ampere_sgemm_128x128_nnv??*?22$8??,@??,H??,bmodel/conv2d_17/ReluhuMUB
x
ampere_sgemm_128x128_ntv??*?22$8??,@??,H??,Xb7gradient_tape/model/conv2d_5/Conv2D/Conv2DBackpropInputhuMUB
S
ampere_sgemm_128x128_nnv??*?2$8??+@??+H??+bmodel/conv2d_15/ReluhuMUB
?
?void cudnn::winograd_nonfused::winogradForwardOutput4x4<float, float>(cudnn::winograd_nonfused::WinogradOutputParams<float, float>)0??*?2?8܀+@܀+H܀+Xb8gradient_tape/model/conv2d_12/Conv2D/Conv2DBackpropInputhu???B
y
ampere_sgemm_128x128_ntv??*?2$8??*@??*H??*Xb8gradient_tape/model/conv2d_15/Conv2D/Conv2DBackpropInputhuMUB
?
?void cudnn::bn_fw_tr_1C11_kernel_NCHW<float, float, 128, true, 1>(cudnnTensorStruct, float const*, cudnnTensorStruct, float*, float const*, float const*, float, float, float*, float*, float*, float*, float, float)(?*?2@8??*@??*H??*b,model/batch_normalization_1/FusedBatchNormV3hu  ?B
?
?void tensorflow::functor::ApplyAdamKernel<float>(int, float*, float*, float*, float const*, float const*, float const*, float const*, float const*, float const*, float const*, bool) *?2&8??)@??)H??)b%Adam/Adam/update_30/ResourceApplyAdamhuZU?B
?
void cudnn::winograd_nonfused::winogradWgradDelta4x4<float, float>(cudnn::winograd_nonfused::WinogradDeltaParams<float, float>)@??*?2d28??)@??)H??)Xb9gradient_tape/model/conv2d_18/Conv2D/Conv2DBackpropFilterhuZU?B
?
}void cudnn::winograd_nonfused::winogradWgradData4x4<float, float>(cudnn::winograd_nonfused::WinogradDataParams<float, float>)@??*?2x28??)@??)H??)Xb9gradient_tape/model/conv2d_16/Conv2D/Conv2DBackpropFilterhuZU?B
?
?void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_product_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorConversionOp<float, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_cmp_op<float const, float const, (Eigen::internal::ComparisonName)5>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorCwiseNullaryOp<Eigen::internal::scalar_constant_op<float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_product_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorConversionOp<float, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_cmp_op<float const, float const, (Eigen::internal::ComparisonName)5>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorCwiseNullaryOp<Eigen::internal::scalar_constant_op<float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const> const> const> const, Eigen::GpuDevice>, long)*?298??'@??'H??'b&gradient_tape/model/conv2d_18/ReluGradhuZU?B
?
?void cudnn::bn_fw_tr_1C11_kernel_NCHW<float, float, 128, true, 1>(cudnnTensorStruct, float const*, cudnnTensorStruct, float*, float const*, float const*, float, float, float*, float*, float*, float*, float, float)(?*?2?8??&@??&H??&b,model/batch_normalization_6/FusedBatchNormV3hu  ?B
?
?void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_product_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorConversionOp<float, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_cmp_op<float const, float const, (Eigen::internal::ComparisonName)5>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorCwiseNullaryOp<Eigen::internal::scalar_constant_op<float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_product_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorConversionOp<float, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_cmp_op<float const, float const, (Eigen::internal::ComparisonName)5>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorCwiseNullaryOp<Eigen::internal::scalar_constant_op<float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const> const> const> const, Eigen::GpuDevice>, long)*?298??%@??%H??%b%gradient_tape/model/conv2d_3/ReluGradhuZU?B
^
%ampere_scudnn_128x32_relu_small_nn_v1??**@2?N8??$@??$H??$bmodel/conv2d/ReluhuMUB
p
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*?2?'8ݼ$@ݼ$Hݼ$b+gradient_tape/model/dropout_7/dropout/Mul_2huZU?B
?
(ampere_scudnn_128x64_relu_interior_nn_v1???*?2
8??#@??#H??#Xb<gradient_tape/model/conv2d_transpose/conv2d_transpose/Conv2DhuMUB
?
?void wgrad_alg0_engine<float, 128, 6, 8, 3, 3, 5, false, 512>(int, int, int, float const*, int, float*, float const*, kernel_grad_params, unsigned long long, int, float, int, int, int, int){?R* 228??#@??#H??#XbLgradient_tape/model/conv2d_transpose_2/conv2d_transpose/Conv2DBackpropFilterhuMUB
?
(ampere_scudnn_128x64_relu_interior_nn_v1???*?2?8??#@??#H??#Xb>gradient_tape/model/conv2d_transpose_3/conv2d_transpose/Conv2DhuMUB
?
?void cudnn::ops::pooling_bw_kernel_max<float, float, cudnn::maxpooling_func<float, (cudnnNanPropagation_t)0>, false>(cudnnTensorStruct, float const*, cudnnTensorStruct, float const*, cudnnTensorStruct, float const*, cudnnTensorStruct, float*, cudnnPoolingStruct, float, float, int, cudnn::reduced_divisor, cudnn::reduced_divisor) ?*?228??#@??#H??#b7gradient_tape/model/max_pooling2d_1/MaxPool/MaxPoolGradhu ??B
y
ampere_sgemm_128x128_ntv??*?2$8??"@??"H??"Xb8gradient_tape/model/conv2d_5/Conv2D/Conv2DBackpropFilterhuMUB
?
?void cudnn::bn_bw_1C11_kernel_new<float, float, float2, 512, true, 1>(float, float, float, float, cudnnTensorStruct, float const*, cudnnTensorStruct, float const*, cudnnTensorStruct, float*, float const*, float*, float*, float const*, float const*, float)(?*?2 8ݧ!@ݧ!Hݧ!b<gradient_tape/model/batch_normalization/FusedBatchNormGradV3hu  ?B
x
ampere_sgemm_128x128_ntv??*?2$8?? @?? H?? Xb7gradient_tape/model/conv2d_8/Conv2D/Conv2DBackpropInputhuMUB
z
ampere_sgemm_128x128_ntv??*?2$8?? @?? H?? Xb9gradient_tape/model/conv2d_17/Conv2D/Conv2DBackpropFilterhuMUB
R
ampere_sgemm_128x128_nnv??*?2$8?? @?? H?? bmodel/conv2d_8/ReluhuMUB
z
ampere_sgemm_128x128_ntv??*?2$8݄ @݄ H݄ Xb9gradient_tape/model/conv2d_15/Conv2D/Conv2DBackpropFilterhuMUB
?
?void cudnn::ops::pooling_fw_4d_kernel<float, float, cudnn::maxpooling_func<float, (cudnnNanPropagation_t)0>, (cudnnPoolingMode_t)0, false>(cudnnTensorStruct, float const*, cudnnTensorStruct, float*, cudnnPoolingStruct, float, float, int, cudnn::reduced_divisor, cudnn::reduced_divisor)( ? *?2228??@??H??bmodel/max_pooling2d/MaxPoolhu  ?B
y
ampere_sgemm_128x128_ntv??*?2$8??@??H??Xb8gradient_tape/model/conv2d_7/Conv2D/Conv2DBackpropFilterhuMUB
y
ampere_sgemm_128x128_ntv??*?2$8??@??H??Xb8gradient_tape/model/conv2d_4/Conv2D/Conv2DBackpropFilterhuMUB
?
?void cudnn::bn_bw_1C11_kernel_new<float, float, float2, 32, true, 1>(float, float, float, float, cudnnTensorStruct, float const*, cudnnTensorStruct, float const*, cudnnTensorStruct, float*, float const*, float*, float*, float const*, float const*, float)(?*?2?8??@??H??b>gradient_tape/model/batch_normalization_5/FusedBatchNormGradV3hu  ?B
?
?void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 4, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorSlicingOp<Eigen::DSizes<int, 4> const, Eigen::DSizes<int, 4> const, Eigen::TensorMap<Eigen::Tensor<float const, 4, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 4, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorSlicingOp<Eigen::DSizes<int, 4> const, Eigen::DSizes<int, 4> const, Eigen::TensorMap<Eigen::Tensor<float const, 4, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int)*?298??@??H??b'gradient_tape/model/concatenate_3/SlicehuZU?B
?
?void implicit_convolve_sgemm<float, float, 1024, 5, 5, 3, 3, 3, 1, false, false, true>(int, int, int, float const*, int, float*, float const*, kernel_conv_params, unsigned long long, int, float, float, int, float const*, float const*, bool, int, int)=?*2??8ޟ@ޟHޟXbmodel/conv2d_22/Conv2DhuZU?B
?
?void cudnn::winograd_nonfused::winogradForwardFilter4x4<float, float>(cudnn::winograd_nonfused::WinogradFilterParams<float, float>)(?H* 2 ?8??@??H??bmodel/conv2d_11/Reluhu  ?B
?
?void cudnn::winograd_nonfused::winogradForwardFilter4x4<float, float>(cudnn::winograd_nonfused::WinogradFilterParams<float, float>)(?H* 2 ?8??@??H??Xb8gradient_tape/model/conv2d_11/Conv2D/Conv2DBackpropInputhu  ?B
?
?void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 4, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorSlicingOp<Eigen::DSizes<int, 4> const, Eigen::DSizes<int, 4> const, Eigen::TensorMap<Eigen::Tensor<float const, 4, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 4, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorSlicingOp<Eigen::DSizes<int, 4> const, Eigen::DSizes<int, 4> const, Eigen::TensorMap<Eigen::Tensor<float const, 4, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int)*?298??@??H??b)gradient_tape/model/concatenate_3/Slice_1huZU?B
?
?void foldedNhwcToNchwKernel<float, float, float, true, (cudnnKernelDataType_t)0>(int, int, int, int, int, int, int, float const*, float*, int, int, int, int, int, int, int, int, int, int, float, float, cudnn::reduced_divisor, cudnn::reduced_divisor, cudnn::reduced_divisor)&?!* 2?28??@??H??b)model/conv2d_transpose_3/conv2d_transposehu  ?B
n
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*?2?'8??@??H??b)gradient_tape/model/dropout_7/dropout/MulhuZU?B
?
void cudnn::winograd_nonfused::winogradForwardData4x4<float, float>(cudnn::winograd_nonfused::WinogradDataParams<float, float>)@??*?2$?8??@??H??bmodel/conv2d_5/ReluhuZU?B
?
?void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorSlicingOp<Eigen::array<int, 2ul> const, Eigen::array<int, 2ul> const, Eigen::TensorMap<Eigen::Tensor<float, 2, 1, int>, 16, Eigen::MakePointer> >, Eigen::TensorMap<Eigen::Tensor<float const, 2, 1, int>, 16, Eigen::MakePointer> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorSlicingOp<Eigen::array<int, 2ul> const, Eigen::array<int, 2ul> const, Eigen::TensorMap<Eigen::Tensor<float, 2, 1, int>, 16, Eigen::MakePointer> >, Eigen::TensorMap<Eigen::Tensor<float const, 2, 1, int>, 16, Eigen::MakePointer> const> const, Eigen::GpuDevice>, int)*?298??@??H??bmodel/concatenate_2/concathuZU?B
?
Yvoid tensorflow::BiasNCHWKernel<float>(int, float const*, float const*, float*, int, int)*?2&8??@??H??b model/conv2d_transpose_3/BiasAddhuZU?B
?
void cudnn::winograd_nonfused::winogradForwardData4x4<float, float>(cudnn::winograd_nonfused::WinogradDataParams<float, float>)@??*?2$?8??@??H??Xb7gradient_tape/model/conv2d_5/Conv2D/Conv2DBackpropInputhuZU?B
?
void cudnn::winograd_nonfused::winogradForwardData4x4<float, float>(cudnn::winograd_nonfused::WinogradDataParams<float, float>)@??*?2$?8??@??H??Xb8gradient_tape/model/conv2d_16/Conv2D/Conv2DBackpropInputhuZU?B
?
?void tensorflow::functor::SwapDimension1And2InTensor3UsingTiles<unsigned int, 256, 32, 32, false>(unsigned int const*, tensorflow::functor::Dimension<3>, unsigned int*) ?!*?2??8??@??H??bCmodel/dropout_7/dropout/Mul_1-0-TransposeNHWCToNCHW-LayoutOptimizerhu  ?B
?
?void cudnn::winograd_nonfused::winogradForwardOutput4x4<float, float>(cudnn::winograd_nonfused::WinogradOutputParams<float, float>)0??*?2$?8??@??H??bmodel/conv2d_17/Reluhu???B
?
?void cub::DeviceSegmentedReduceKernel<cub::DeviceReducePolicy<float, float, int, cub::Sum>::Policy600, float const*, float*, cub::TransformInputIterator<int, tensorflow::functor::RowOffset, cub::CountingInputIterator<int, long>, long>, int, cub::Sum, float>(float const*, float*, cub::TransformInputIterator<int, tensorflow::functor::RowOffset, cub::CountingInputIterator<int, long>, long>, cub::TransformInputIterator<int, tensorflow::functor::RowOffset, cub::CountingInputIterator<int, long>, long>, int, cub::Sum, float) 0*?2?8ގ@ގHގb1gradient_tape/model/conv2d_20/BiasAdd/BiasAddGradhu  ?B
?
void cudnn::winograd_nonfused::winogradForwardData4x4<float, float>(cudnn::winograd_nonfused::WinogradDataParams<float, float>)@??*?2$?8??@??H??bmodel/conv2d_17/ReluhuZU?B
?
?void cudnn::winograd_nonfused::winogradForwardOutput4x4<float, float>(cudnn::winograd_nonfused::WinogradOutputParams<float, float>)0??*?2?8??@??H??Xb8gradient_tape/model/conv2d_14/Conv2D/Conv2DBackpropInputhu???B
`
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*?2?'8??@??H??bmodel/dropout_7/dropout/MulhuZU?B
R
ampere_sgemm_128x128_nnv??*?2$8??@??H??bmodel/conv2d_6/ReluhuMUB
?
void cudnn::winograd_nonfused::winogradForwardData4x4<float, float>(cudnn::winograd_nonfused::WinogradDataParams<float, float>)@??*?2$?8??@??H??Xb8gradient_tape/model/conv2d_17/Conv2D/Conv2DBackpropInputhuZU?B
?
?void cub::DeviceSegmentedReduceKernel<cub::DeviceReducePolicy<float, float, int, cub::Sum>::Policy600, float const*, float*, cub::TransformInputIterator<int, tensorflow::functor::RowOffset, cub::CountingInputIterator<int, long>, long>, int, cub::Sum, float>(float const*, float*, cub::TransformInputIterator<int, tensorflow::functor::RowOffset, cub::CountingInputIterator<int, long>, long>, cub::TransformInputIterator<int, tensorflow::functor::RowOffset, cub::CountingInputIterator<int, long>, long>, int, cub::Sum, float) 0*?2?8??@??H??b.gradient_tape/model/conv2d/BiasAdd/BiasAddGradhu  ?B
?
?void cub::DeviceSegmentedReduceKernel<cub::DeviceReducePolicy<float, float, int, cub::Sum>::Policy600, float const*, float*, cub::TransformInputIterator<int, tensorflow::functor::RowOffset, cub::CountingInputIterator<int, long>, long>, int, cub::Sum, float>(float const*, float*, cub::TransformInputIterator<int, tensorflow::functor::RowOffset, cub::CountingInputIterator<int, long>, long>, cub::TransformInputIterator<int, tensorflow::functor::RowOffset, cub::CountingInputIterator<int, long>, long>, int, cub::Sum, float) 0*?2?8޻@޻H޻b1gradient_tape/model/conv2d_21/BiasAdd/BiasAddGradhu  ?B
?
Lvoid cudnn::ops::scalePackedTensor_kernel<float, float>(long, float*, float)*?2??8??@??H??b)model/conv2d_transpose_4/conv2d_transposehu  ?B
?
?void cudnn::winograd_nonfused::winogradForwardOutput4x4<float, float>(cudnn::winograd_nonfused::WinogradOutputParams<float, float>)0??*?2$?8ީ@ީHީbmodel/conv2d_5/Reluhu???B
?
?void cutlass_cudnn::Kernel<cutlass_tensorop_s1688wgrad_analytic_tf32_128x256_16x3>(cutlass_tensorop_s1688wgrad_analytic_tf32_128x256_16x3::Params)? ??*?2 8??@??H??XbJgradient_tape/model/conv2d_transpose/conv2d_transpose/Conv2DBackpropFilterh
?
?void cub::DeviceSegmentedReduceKernel<cub::DeviceReducePolicy<float, float, int, cub::Sum>::Policy600, float const*, float*, cub::TransformInputIterator<int, tensorflow::functor::RowOffset, cub::CountingInputIterator<int, long>, long>, int, cub::Sum, float>(float const*, float*, cub::TransformInputIterator<int, tensorflow::functor::RowOffset, cub::CountingInputIterator<int, long>, long>, cub::TransformInputIterator<int, tensorflow::functor::RowOffset, cub::CountingInputIterator<int, long>, long>, int, cub::Sum, float) 0*?2?8??@??H??b:gradient_tape/model/conv2d_transpose_4/BiasAdd/BiasAddGradhu  ?B
?
?void cub::DeviceSegmentedReduceKernel<cub::DeviceReducePolicy<float, float, int, cub::Sum>::Policy600, float const*, float*, cub::TransformInputIterator<int, tensorflow::functor::RowOffset, cub::CountingInputIterator<int, long>, long>, int, cub::Sum, float>(float const*, float*, cub::TransformInputIterator<int, tensorflow::functor::RowOffset, cub::CountingInputIterator<int, long>, long>, cub::TransformInputIterator<int, tensorflow::functor::RowOffset, cub::CountingInputIterator<int, long>, long>, int, cub::Sum, float) 0*?2?8??@??H??b0gradient_tape/model/conv2d_1/BiasAdd/BiasAddGradhu  ?B
y
ampere_sgemm_128x128_ntv??*?2$8??@??H??Xb8gradient_tape/model/conv2d_8/Conv2D/Conv2DBackpropFilterhuMUB
?
?void cudnn::winograd_nonfused::winogradForwardOutput4x4<float, float>(cudnn::winograd_nonfused::WinogradOutputParams<float, float>)0??*?2$?8??@??H??bmodel/conv2d_16/Reluhu???B
?
Lvoid cudnn::ops::scalePackedTensor_kernel<float, float>(long, float*, float)*?2??8??@??H??b5gradient_tape/model/max_pooling2d/MaxPool/MaxPoolGradhu  ?B
?
?void cudnn::winograd_nonfused::winogradForwardOutput4x4<float, float>(cudnn::winograd_nonfused::WinogradOutputParams<float, float>)0??*?2$?8??@??H??Xb8gradient_tape/model/conv2d_17/Conv2D/Conv2DBackpropInputhu???B
?
~sm80_xmma_fprop_implicit_gemm_tf32f32_tf32f32_f32_nhwckrsc_nchw_tilesize128x128x16_stage4_warpsize2x2x1_g1_tensor16x8x8_kernel? ??*?2(8??@??H??PXb>gradient_tape/model/conv2d_transpose_1/conv2d_transpose/Conv2Dh
?
?void cutlass_cudnn::Kernel<cutlass_tensorop_s1688wgrad_analytic_tf32_128x256_16x3>(cutlass_tensorop_s1688wgrad_analytic_tf32_128x256_16x3::Params)? ??*?2	8??@??H??XbLgradient_tape/model/conv2d_transpose_1/conv2d_transpose/Conv2DBackpropFilterh
?
?void cudnn::winograd_nonfused::winogradWgradOutput4x4<float, float>(cudnn::winograd_nonfused::WinogradWgradOutputParams<float, float>)=?H* 2 @8??@??H??Xb9gradient_tape/model/conv2d_12/Conv2D/Conv2DBackpropFilterhuZU?B
?
?void tensorflow::functor::ShuffleInTensor3Simple<float, 2, 1, 0, false>(int, float const*, tensorflow::functor::Dimension<3>, float*)*?2&8??@??H??bmodel/conv2d_11/ReluhuZU?B
?
?void cudnn::winograd_nonfused::winogradForwardFilter4x4<float, float>(cudnn::winograd_nonfused::WinogradFilterParams<float, float>)(?H* 28ޤ@ޤHޤbmodel/conv2d_16/Reluhu  ?B
?
?void cutlass_cudnn::Kernel<cutlass_tensorop_s1688dgrad_optimized_tf32_128x128_16x4_unity_stride>(cutlass_tensorop_s1688dgrad_optimized_tf32_128x128_16x4_unity_stride::Params)? ??*?2(8??@??H??b'model/conv2d_transpose/conv2d_transposeh
?
?void cutlass_cudnn::Kernel<cutlass_tensorop_s1688dgrad_optimized_tf32_128x128_16x4_unity_stride>(cutlass_tensorop_s1688dgrad_optimized_tf32_128x128_16x4_unity_stride::Params)? ??*?2?8??@??H??b)model/conv2d_transpose_1/conv2d_transposeh
?
void cudnn::winograd_nonfused::winogradWgradDelta4x4<float, float>(cudnn::winograd_nonfused::WinogradDeltaParams<float, float>)@??*?2<28??@??H??Xb9gradient_tape/model/conv2d_17/Conv2D/Conv2DBackpropFilterhuZU?B
?
void cudnn::winograd_nonfused::winogradWgradDelta4x4<float, float>(cudnn::winograd_nonfused::WinogradDeltaParams<float, float>)@??*?2<28??@??H??Xb8gradient_tape/model/conv2d_4/Conv2D/Conv2DBackpropFilterhuZU?B
?
?void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<float, Eigen::TensorMap<Eigen::Tensor<bool const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<float, Eigen::TensorMap<Eigen::Tensor<bool const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*?298??@??H??bmodel/dropout_7/dropout/CasthuZU?B
?
?void cudnn::bn_fw_tr_1C11_kernel_NCHW<float, float, 512, true, 1>(cudnnTensorStruct, float const*, cudnnTensorStruct, float*, float const*, float const*, float, float, float*, float*, float*, float*, float, float)(?*?2 8??@??H??b*model/batch_normalization/FusedBatchNormV3hu  ?B
?
?void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<float, Eigen::TensorMap<Eigen::Tensor<bool const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<float, Eigen::TensorMap<Eigen::Tensor<bool const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*?298??@??H??bXmodel/dropout_7/dropout/ArithmeticOptimizer/ReorderCastLikeAndValuePreserving_float_CasthuZU?B
?
void cudnn::winograd_nonfused::winogradWgradDelta4x4<float, float>(cudnn::winograd_nonfused::WinogradDeltaParams<float, float>)@??*?2<28޼@޼H޼Xb9gradient_tape/model/conv2d_16/Conv2D/Conv2DBackpropFilterhuZU?B
?
?void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_product_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorConversionOp<float, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_cmp_op<float const, float const, (Eigen::internal::ComparisonName)5>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorCwiseNullaryOp<Eigen::internal::scalar_constant_op<float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_product_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorConversionOp<float, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_cmp_op<float const, float const, (Eigen::internal::ComparisonName)5>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorCwiseNullaryOp<Eigen::internal::scalar_constant_op<float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const> const> const> const, Eigen::GpuDevice>, long)*?298??@??H??b&gradient_tape/model/conv2d_16/ReluGradhuZU?B
?
?void tensorflow::functor::ApplyAdamKernel<float>(int, float*, float*, float*, float const*, float const*, float const*, float const*, float const*, float const*, float const*, bool) *?2&8ޝ@ޝHޝb%Adam/Adam/update_36/ResourceApplyAdamhuZU?B
?
}void cudnn::winograd_nonfused::winogradWgradData4x4<float, float>(cudnn::winograd_nonfused::WinogradDataParams<float, float>)@??*?2<28ގ@ގHގXb8gradient_tape/model/conv2d_5/Conv2D/Conv2DBackpropFilterhuZU?B
?
void cudnn::winograd_nonfused::winogradWgradDelta4x4<float, float>(cudnn::winograd_nonfused::WinogradDeltaParams<float, float>)@??*?2<28ލ@ލHލXb8gradient_tape/model/conv2d_5/Conv2D/Conv2DBackpropFilterhuZU?B
?
?void tensorflow::functor::ShuffleInTensor3Simple<float, 2, 1, 0, false>(int, float const*, tensorflow::functor::Dimension<3>, float*)*?2&8??@??H??Xb9gradient_tape/model/conv2d_13/Conv2D/Conv2DBackpropFilterhuZU?B
?
?void tensorflow::functor::ApplyAdamKernel<float>(int, float*, float*, float*, float const*, float const*, float const*, float const*, float const*, float const*, float const*, bool) *?2&8އ@އHއb%Adam/Adam/update_28/ResourceApplyAdamhuZU?B
?
}void cudnn::winograd_nonfused::winogradWgradData4x4<float, float>(cudnn::winograd_nonfused::WinogradDataParams<float, float>)@??*?2`28ޅ@ޅHޅXb9gradient_tape/model/conv2d_14/Conv2D/Conv2DBackpropFilterhuZU?B
?
?void cudnn::bn_fw_tr_1C11_kernel_NCHW<float, float, 32, true, 1>(cudnnTensorStruct, float const*, cudnnTensorStruct, float*, float const*, float const*, float, float, float*, float*, float*, float*, float, float)(?*?2?8??@??H??b,model/batch_normalization_5/FusedBatchNormV3hu  ?B
b
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*?2?8??@??H??bmodel/dropout_6/dropout/Mul_1huZU?B
?
?void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_product_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorConversionOp<float, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_cmp_op<float const, float const, (Eigen::internal::ComparisonName)5>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorCwiseNullaryOp<Eigen::internal::scalar_constant_op<float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_product_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorConversionOp<float, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_cmp_op<float const, float const, (Eigen::internal::ComparisonName)5>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorCwiseNullaryOp<Eigen::internal::scalar_constant_op<float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const> const> const> const, Eigen::GpuDevice>, long)*?298??@??H??b&gradient_tape/model/conv2d_17/ReluGradhuZU?B
?
?void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_product_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorConversionOp<float, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_cmp_op<float const, float const, (Eigen::internal::ComparisonName)5>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorCwiseNullaryOp<Eigen::internal::scalar_constant_op<float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_product_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorConversionOp<float, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_cmp_op<float const, float const, (Eigen::internal::ComparisonName)5>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorCwiseNullaryOp<Eigen::internal::scalar_constant_op<float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const> const> const> const, Eigen::GpuDevice>, long)*?298ޤ@ޤHޤb%gradient_tape/model/conv2d_4/ReluGradhuZU?B
?
?void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_product_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorConversionOp<float, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_cmp_op<float const, float const, (Eigen::internal::ComparisonName)5>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorCwiseNullaryOp<Eigen::internal::scalar_constant_op<float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_product_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorConversionOp<float, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_cmp_op<float const, float const, (Eigen::internal::ComparisonName)5>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorCwiseNullaryOp<Eigen::internal::scalar_constant_op<float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const> const> const> const, Eigen::GpuDevice>, long)*?298??@??H??b%gradient_tape/model/conv2d_5/ReluGradhuZU?B
p
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*?2?8??@??H??b+gradient_tape/model/dropout_6/dropout/Mul_2huZU?B
?
?void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*?298??@??H??bAdam/gradients/AddN_3huZU?B
?
?void tensorflow::functor::ShuffleInTensor3Simple<float, 2, 1, 0, false>(int, float const*, tensorflow::functor::Dimension<3>, float*)*?2&8??@??H??Xb8gradient_tape/model/conv2d_9/Conv2D/Conv2DBackpropFilterhuZU?B
?
void cudnn::winograd_nonfused::winogradForwardData4x4<float, float>(cudnn::winograd_nonfused::WinogradDataParams<float, float>)@??*?2?8??@??H??bmodel/conv2d_12/ReluhuZU?B
y
ampere_sgemm_128x128_ntv??*?2$8ߧ@ߧHߧXb8gradient_tape/model/conv2d_6/Conv2D/Conv2DBackpropFilterhuMUB
?
?void cudnn::ops::pooling_fw_4d_kernel<float, float, cudnn::maxpooling_func<float, (cudnnNanPropagation_t)0>, (cudnnPoolingMode_t)0, false>(cudnnTensorStruct, float const*, cudnnTensorStruct, float*, cudnnPoolingStruct, float, float, int, cudnn::reduced_divisor, cudnn::reduced_divisor)( ?*?228??@??H??bmodel/max_pooling2d_1/MaxPoolhu ??B
?
?void cudnn::bn_bw_1C11_kernel_new<float, float, float2, 128, true, 1>(float, float, float, float, cudnnTensorStruct, float const*, cudnnTensorStruct, float const*, cudnnTensorStruct, float*, float const*, float*, float*, float const*, float const*, float)(?*?2@8??@??H??b>gradient_tape/model/batch_normalization_1/FusedBatchNormGradV3hu  ?B
q
(GreaterEqual_GPU_DT_FLOAT_DT_BOOL_kernel*?2?'8??@??H??b$model/dropout_7/dropout/GreaterEqualhuZU?B
?
?void cudnn::winograd_nonfused::winogradForwardFilter4x4<float, float>(cudnn::winograd_nonfused::WinogradFilterParams<float, float>)(?H* 2?8??@??H??bmodel/conv2d_10/Reluhu  ?B
?
?void cudnn::winograd_nonfused::winogradForwardFilter4x4<float, float>(cudnn::winograd_nonfused::WinogradFilterParams<float, float>)(?H* 2 @8??@??H??bmodel/conv2d_12/Reluhu  ?B
?
?void cudnn::winograd_nonfused::winogradForwardFilter4x4<float, float>(cudnn::winograd_nonfused::WinogradFilterParams<float, float>)(?H* 2 @8??@??H??Xb8gradient_tape/model/conv2d_12/Conv2D/Conv2DBackpropInputhu  ?B
?
?void cudnn::winograd_nonfused::winogradForwardFilter4x4<float, float>(cudnn::winograd_nonfused::WinogradFilterParams<float, float>)(?H* 2?8??@??H??Xb8gradient_tape/model/conv2d_10/Conv2D/Conv2DBackpropInputhu  ?B
?
?void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 4, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorSlicingOp<Eigen::DSizes<int, 4> const, Eigen::DSizes<int, 4> const, Eigen::TensorMap<Eigen::Tensor<float const, 4, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 4, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorSlicingOp<Eigen::DSizes<int, 4> const, Eigen::DSizes<int, 4> const, Eigen::TensorMap<Eigen::Tensor<float const, 4, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int)*?298??@??H??b)gradient_tape/model/concatenate_2/Slice_1huZU?B
?
?void cudnn::pooling_bw_kernel_max_nchw_fully_packed_small<float, float, 2, (cudnnNanPropagation_t)0>(cudnnTensorStruct, float const*, cudnnTensorStruct, float const*, cudnnTensorStruct, float const*, cudnnTensorStruct, float*, cudnnPoolingStruct, float, float, cudnn::reduced_divisor, cudnn::reduced_divisor) ?d*?22?8??@??H??b7gradient_tape/model/max_pooling2d_2/MaxPool/MaxPoolGradhu  ?B
?
?void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 4, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorSlicingOp<Eigen::DSizes<int, 4> const, Eigen::DSizes<int, 4> const, Eigen::TensorMap<Eigen::Tensor<float const, 4, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 4, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorSlicingOp<Eigen::DSizes<int, 4> const, Eigen::DSizes<int, 4> const, Eigen::TensorMap<Eigen::Tensor<float const, 4, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int)*?298??@??H??b'gradient_tape/model/concatenate_2/SlicehuZU?B
?
?void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorSlicingOp<Eigen::array<int, 2ul> const, Eigen::array<int, 2ul> const, Eigen::TensorMap<Eigen::Tensor<float, 2, 1, int>, 16, Eigen::MakePointer> >, Eigen::TensorMap<Eigen::Tensor<float const, 2, 1, int>, 16, Eigen::MakePointer> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorSlicingOp<Eigen::array<int, 2ul> const, Eigen::array<int, 2ul> const, Eigen::TensorMap<Eigen::Tensor<float, 2, 1, int>, 16, Eigen::MakePointer> >, Eigen::TensorMap<Eigen::Tensor<float const, 2, 1, int>, 16, Eigen::MakePointer> const> const, Eigen::GpuDevice>, int)*?298??@??H??bmodel/concatenate_1/concathuZU?B
?
}void cudnn::winograd_nonfused::winogradWgradData4x4<float, float>(cudnn::winograd_nonfused::WinogradDataParams<float, float>)@??*?2@28??@??H??Xb9gradient_tape/model/conv2d_12/Conv2D/Conv2DBackpropFilterhuZU?B
?
?void cudnn::ops::nchwToNhwcKernel<float, float, float, false, true, (cudnnKernelDataType_t)2>(cudnn::ops::nchw2nhwc_params_t<float>, float const*, float*)'?!*?2228߮@߮H߮b)model/conv2d_transpose_3/conv2d_transposehu  ?B
?
void cudnn::winograd_nonfused::winogradForwardData4x4<float, float>(cudnn::winograd_nonfused::WinogradDataParams<float, float>)@??*?2?8??@??H??Xb7gradient_tape/model/conv2d_6/Conv2D/Conv2DBackpropInputhuZU?B
?
void cudnn::winograd_nonfused::winogradForwardData4x4<float, float>(cudnn::winograd_nonfused::WinogradDataParams<float, float>)@??*?2?8??@??H??Xb7gradient_tape/model/conv2d_7/Conv2D/Conv2DBackpropInputhuZU?B
?
void cudnn::winograd_nonfused::winogradForwardData4x4<float, float>(cudnn::winograd_nonfused::WinogradDataParams<float, float>)@??*?2?8??@??H??bmodel/conv2d_15/ReluhuZU?B
n
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*?2?8??@??H??b)gradient_tape/model/dropout_6/dropout/MulhuZU?B
?
?void tensorflow::functor::FillPhiloxRandomKernelLaunch<tensorflow::random::UniformDistribution<tensorflow::random::PhiloxRandom, float> >(unsigned long const*, unsigned long const*, tensorflow::random::PhiloxRandom, tensorflow::random::UniformDistribution<tensorflow::random::PhiloxRandom, float>::ResultElementType*, long, tensorflow::random::UniformDistribution<tensorflow::random::PhiloxRandom, float>)*?298??@??H??b4model/dropout_7/dropout/random_uniform/RandomUniformhuZU?B
?
?void cudnn::winograd_nonfused::winogradForwardOutput4x4<float, float>(cudnn::winograd_nonfused::WinogradOutputParams<float, float>)0??*?2?8??@??H??Xb7gradient_tape/model/conv2d_7/Conv2D/Conv2DBackpropInputhu???B
?
void cudnn::winograd_nonfused::winogradForwardData4x4<float, float>(cudnn::winograd_nonfused::WinogradDataParams<float, float>)@??*?2?8??@??H??bmodel/conv2d_7/ReluhuZU?B
?
void cudnn::winograd_nonfused::winogradForwardData4x4<float, float>(cudnn::winograd_nonfused::WinogradDataParams<float, float>)@??*?2?8??@??H??Xb8gradient_tape/model/conv2d_14/Conv2D/Conv2DBackpropInputhuZU?B
?
?void cudnn::ops::nhwcToNchwKernel<float, float, float, true, false, (cudnnKernelDataType_t)0>(cudnn::ops::nhwc2nchw_params_t<float>, float const*, float*)&?!*?2 ?8??@??H??Xb9gradient_tape/model/conv2d_11/Conv2D/Conv2DBackpropFilterhu  ?B
`
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*?2?8??@??H??bmodel/dropout_6/dropout/MulhuZU?B
?
?void cub::DeviceSegmentedReduceKernel<cub::DeviceReducePolicy<float, float, int, cub::Sum>::Policy600, float const*, float*, cub::TransformInputIterator<int, tensorflow::functor::RowOffset, cub::CountingInputIterator<int, long>, long>, int, cub::Sum, float>(float const*, float*, cub::TransformInputIterator<int, tensorflow::functor::RowOffset, cub::CountingInputIterator<int, long>, long>, cub::TransformInputIterator<int, tensorflow::functor::RowOffset, cub::CountingInputIterator<int, long>, long>, int, cub::Sum, float) 0*?2?8??@??H??b0gradient_tape/model/conv2d_3/BiasAdd/BiasAddGradhu  ?B
?
?void cub::DeviceSegmentedReduceKernel<cub::DeviceReducePolicy<float, float, int, cub::Sum>::Policy600, float const*, float*, cub::TransformInputIterator<int, tensorflow::functor::RowOffset, cub::CountingInputIterator<int, long>, long>, int, cub::Sum, float>(float const*, float*, cub::TransformInputIterator<int, tensorflow::functor::RowOffset, cub::CountingInputIterator<int, long>, long>, cub::TransformInputIterator<int, tensorflow::functor::RowOffset, cub::CountingInputIterator<int, long>, long>, int, cub::Sum, float) 0*?2?8??@??H??b1gradient_tape/model/conv2d_18/BiasAdd/BiasAddGradhu  ?B
?
?void cudnn::winograd_nonfused::winogradForwardOutput4x4<float, float>(cudnn::winograd_nonfused::WinogradOutputParams<float, float>)0??*?2?8??@??H??bmodel/conv2d_6/Reluhu???B
?
?void cudnn::winograd_nonfused::winogradForwardOutput4x4<float, float>(cudnn::winograd_nonfused::WinogradOutputParams<float, float>)0??*?2?8??@??H??bmodel/conv2d_14/Reluhu???B
?
?void cudnn::winograd_nonfused::winogradForwardOutput4x4<float, float>(cudnn::winograd_nonfused::WinogradOutputParams<float, float>)0??*?2?8??@??H??bmodel/conv2d_7/Reluhu???B
?
?void cudnn::winograd_nonfused::winogradForwardOutput4x4<float, float>(cudnn::winograd_nonfused::WinogradOutputParams<float, float>)0??*?2?8ߑ@ߑHߑXb8gradient_tape/model/conv2d_15/Conv2D/Conv2DBackpropInputhu???B
?
?void cub::DeviceSegmentedReduceKernel<cub::DeviceReducePolicy<float, float, int, cub::Sum>::Policy600, float const*, float*, cub::TransformInputIterator<int, tensorflow::functor::RowOffset, cub::CountingInputIterator<int, long>, long>, int, cub::Sum, float>(float const*, float*, cub::TransformInputIterator<int, tensorflow::functor::RowOffset, cub::CountingInputIterator<int, long>, long>, cub::TransformInputIterator<int, tensorflow::functor::RowOffset, cub::CountingInputIterator<int, long>, long>, int, cub::Sum, float) 0*?2?8??@??H??b1gradient_tape/model/conv2d_19/BiasAdd/BiasAddGradhu  ?B
?
?void cudnn::winograd_nonfused::winogradForwardOutput4x4<float, float>(cudnn::winograd_nonfused::WinogradOutputParams<float, float>)0??*?2?8ߎ@ߎHߎbmodel/conv2d_15/Reluhu???B
?
?void cub::DeviceSegmentedReduceKernel<cub::DeviceReducePolicy<float, float, int, cub::Sum>::Policy600, float const*, float*, cub::TransformInputIterator<int, tensorflow::functor::RowOffset, cub::CountingInputIterator<int, long>, long>, int, cub::Sum, float>(float const*, float*, cub::TransformInputIterator<int, tensorflow::functor::RowOffset, cub::CountingInputIterator<int, long>, long>, cub::TransformInputIterator<int, tensorflow::functor::RowOffset, cub::CountingInputIterator<int, long>, long>, int, cub::Sum, float) 0*?2?8??@??H??b0gradient_tape/model/conv2d_2/BiasAdd/BiasAddGradhu  ?B
?
?void cub::DeviceSegmentedReduceKernel<cub::DeviceReducePolicy<float, float, int, cub::Sum>::Policy600, float const*, float*, cub::TransformInputIterator<int, tensorflow::functor::RowOffset, cub::CountingInputIterator<int, long>, long>, int, cub::Sum, float>(float const*, float*, cub::TransformInputIterator<int, tensorflow::functor::RowOffset, cub::CountingInputIterator<int, long>, long>, cub::TransformInputIterator<int, tensorflow::functor::RowOffset, cub::CountingInputIterator<int, long>, long>, int, cub::Sum, float) 0*?2?8??@??H??b:gradient_tape/model/conv2d_transpose_3/BiasAdd/BiasAddGradhu  ?B
?
void cudnn::winograd_nonfused::winogradWgradDelta4x4<float, float>(cudnn::winograd_nonfused::WinogradDeltaParams<float, float>)@??*?2028??@??H??Xb8gradient_tape/model/conv2d_7/Conv2D/Conv2DBackpropFilterhuZU?B
?
Lvoid cudnn::ops::scalePackedTensor_kernel<float, float>(long, float*, float)*?2??8??@??H??b7gradient_tape/model/max_pooling2d_1/MaxPool/MaxPoolGradhu  ?B
?
?void cudnn::winograd_nonfused::winogradWgradOutput4x4<float, float>(cudnn::winograd_nonfused::WinogradWgradOutputParams<float, float>)=?H* 2@8??@??H??Xb9gradient_tape/model/conv2d_13/Conv2D/Conv2DBackpropFilterhuZU?B
?
?void cudnn::winograd_nonfused::winogradWgradOutput4x4<float, float>(cudnn::winograd_nonfused::WinogradWgradOutputParams<float, float>)=?H* 2@8??@??H??Xb8gradient_tape/model/conv2d_9/Conv2D/Conv2DBackpropFilterhuZU?B
?
?void tensorflow::functor::ApplyAdamKernel<float>(int, float*, float*, float*, float const*, float const*, float const*, float const*, float const*, float const*, float const*, bool) *?2&8??
@??
H??
b%Adam/Adam/update_26/ResourceApplyAdamhuZU?B
?
void cudnn::winograd_nonfused::winogradWgradDelta4x4<float, float>(cudnn::winograd_nonfused::WinogradDeltaParams<float, float>)@??*?2028??
@??
H??
Xb9gradient_tape/model/conv2d_15/Conv2D/Conv2DBackpropFilterhuZU?B
?
?void tensorflow::functor::ApplyAdamKernel<float>(int, float*, float*, float*, float const*, float const*, float const*, float const*, float const*, float const*, float const*, bool) *?2&8??
@??
H??
b%Adam/Adam/update_32/ResourceApplyAdamhuZU?B
?
void cudnn::winograd_nonfused::winogradWgradDelta4x4<float, float>(cudnn::winograd_nonfused::WinogradDeltaParams<float, float>)@??*?2028??
@??
H??
Xb8gradient_tape/model/conv2d_6/Conv2D/Conv2DBackpropFilterhuZU?B
?
}void cudnn::winograd_nonfused::winogradWgradData4x4<float, float>(cudnn::winograd_nonfused::WinogradDataParams<float, float>)@??*?2028??
@??
H??
Xb9gradient_tape/model/conv2d_15/Conv2D/Conv2DBackpropFilterhuZU?B
?
}void cudnn::winograd_nonfused::winogradWgradData4x4<float, float>(cudnn::winograd_nonfused::WinogradDataParams<float, float>)@??*?228߻
@߻
H߻
Xb8gradient_tape/model/conv2d_4/Conv2D/Conv2DBackpropFilterhuZU?B
?
?void tensorflow::functor::ApplyAdamKernel<float>(int, float*, float*, float*, float const*, float const*, float const*, float const*, float const*, float const*, float const*, bool) *?2&8߳
@߳
H߳
b%Adam/Adam/update_38/ResourceApplyAdamhuZU?B
?
void cudnn::winograd_nonfused::winogradWgradDelta4x4<float, float>(cudnn::winograd_nonfused::WinogradDeltaParams<float, float>)@??*?2028??
@??
H??
Xb9gradient_tape/model/conv2d_14/Conv2D/Conv2DBackpropFilterhuZU?B
`
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*?2?	8??
@??
H??
bmodel/dropout/dropout/Mul_1huZU?B
?
}void cudnn::winograd_nonfused::winogradWgradData4x4<float, float>(cudnn::winograd_nonfused::WinogradDataParams<float, float>)@??*?2028??
@??
H??
Xb8gradient_tape/model/conv2d_7/Conv2D/Conv2DBackpropFilterhuZU?B
?
?void tensorflow::functor::ShuffleInTensor3Simple<float, 2, 1, 0, false>(int, float const*, tensorflow::functor::Dimension<3>, float*)*?2&8??	@??	H??	bmodel/conv2d_12/ReluhuZU?B
?
?void tensorflow::functor::ShuffleInTensor3Simple<float, 2, 1, 0, false>(int, float const*, tensorflow::functor::Dimension<3>, float*)*?2&8??	@??	H??	bmodel/conv2d_10/ReluhuZU?B
?
?void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_product_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorConversionOp<float, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_cmp_op<float const, float const, (Eigen::internal::ComparisonName)5>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorCwiseNullaryOp<Eigen::internal::scalar_constant_op<float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_product_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorConversionOp<float, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_cmp_op<float const, float const, (Eigen::internal::ComparisonName)5>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorCwiseNullaryOp<Eigen::internal::scalar_constant_op<float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const> const> const> const, Eigen::GpuDevice>, long)*?298??	@??	H??	b&gradient_tape/model/conv2d_15/ReluGradhuZU?B
?
?void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_product_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorConversionOp<float, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_cmp_op<float const, float const, (Eigen::internal::ComparisonName)5>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorCwiseNullaryOp<Eigen::internal::scalar_constant_op<float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_product_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorConversionOp<float, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_cmp_op<float const, float const, (Eigen::internal::ComparisonName)5>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorCwiseNullaryOp<Eigen::internal::scalar_constant_op<float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const> const> const> const, Eigen::GpuDevice>, long)*?298??	@??	H??	b%gradient_tape/model/conv2d_6/ReluGradhuZU?B
?
?void tensorflow::functor::ShuffleInTensor3Simple<float, 2, 1, 0, false>(int, float const*, tensorflow::functor::Dimension<3>, float*)*?2&8??	@??	H??	Xb8gradient_tape/model/conv2d_12/Conv2D/Conv2DBackpropInputhuZU?B
?
?void tensorflow::functor::ShuffleInTensor3Simple<float, 2, 1, 0, false>(int, float const*, tensorflow::functor::Dimension<3>, float*)*?2&8??	@??	H??	Xb8gradient_tape/model/conv2d_10/Conv2D/Conv2DBackpropInputhuZU?B
?
?void cudnn::bn_bw_1C11_singleread<float, 512, true, 1, 2, 0>(float, float, float, float, cudnnTensorStruct, float const*, cudnnTensorStruct, float const*, cudnnTensorStruct, float*, float const*, float*, float*, float const*, float const*, float, cudnn::reduced_divisor, int, cudnn::reduced_divisor, cudnn::bnBwPersistentState*, int, float, float, float, int, float, cudnnStatus_t*, bool)(? ??*?2?8??	@??	H??	b>gradient_tape/model/batch_normalization_4/FusedBatchNormGradV3huZU?B
?
?void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_product_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorConversionOp<float, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_cmp_op<float const, float const, (Eigen::internal::ComparisonName)5>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorCwiseNullaryOp<Eigen::internal::scalar_constant_op<float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_product_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorConversionOp<float, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_cmp_op<float const, float const, (Eigen::internal::ComparisonName)5>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorCwiseNullaryOp<Eigen::internal::scalar_constant_op<float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const> const> const> const, Eigen::GpuDevice>, long)*?298??	@??	H??	b&gradient_tape/model/conv2d_14/ReluGradhuZU?B
?
?void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_product_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorConversionOp<float, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_cmp_op<float const, float const, (Eigen::internal::ComparisonName)5>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorCwiseNullaryOp<Eigen::internal::scalar_constant_op<float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_product_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorConversionOp<float, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_cmp_op<float const, float const, (Eigen::internal::ComparisonName)5>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorCwiseNullaryOp<Eigen::internal::scalar_constant_op<float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const> const> const> const, Eigen::GpuDevice>, long)*?298??	@??	H??	b%gradient_tape/model/conv2d_7/ReluGradhuZU?B
?
?void tensorflow::functor::SwapDimension1And2InTensor3UsingTiles<unsigned int, 1024, 1024, 2, false>(unsigned int const*, tensorflow::functor::Dimension<3>, unsigned int*) ?`*?2?8߱	@߱	H߱	b\gradient_tape/model/conv2d/Conv2D/Conv2DBackpropFilter-0-TransposeNHWCToNCHW-LayoutOptimizerhuZU?B
p
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*?2?	8??	@??	H??	b+gradient_tape/model/dropout_5/dropout/Mul_2huZU?B
?
?void tensorflow::functor::SwapDimension1And2InTensor3UsingTiles<unsigned char, 256, 32, 32, false>(unsigned char const*, tensorflow::functor::Dimension<3>, unsigned char*) ?*?2??8ߠ	@ߠ	Hߠ	b?gradient_tape/model/dropout_7/dropout/ArithmeticOptimizer/ReorderCastLikeAndValuePreserving_bool_Mul_2-1-TransposeNHWCToNCHW-LayoutOptimizerhu  ?B
?
?void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*?298??	@??	H??	bAdam/gradients/AddN_2huZU?B
n
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*?2?	8??	@??	H??	b)gradient_tape/model/dropout/dropout/Mul_2huZU?B
b
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*?2?	8??	@??	H??	bmodel/dropout_5/dropout/Mul_1huZU?B
?
?void cudnn::pooling_bw_kernel_max_nchw_fully_packed_small<float, float, 2, (cudnnNanPropagation_t)0>(cudnnTensorStruct, float const*, cudnnTensorStruct, float const*, cudnnTensorStruct, float const*, cudnnTensorStruct, float*, cudnnPoolingStruct, float, float, cudnn::reduced_divisor, cudnn::reduced_divisor) ?*?22?8??@??H??b7gradient_tape/model/max_pooling2d_4/MaxPool/MaxPoolGradhu  ?B
?
?void cudnn::bn_bw_1C11_kernel_new<float, float, float2, 32, true, 1>(float, float, float, float, cudnnTensorStruct, float const*, cudnnTensorStruct, float const*, cudnnTensorStruct, float*, float const*, float*, float*, float const*, float const*, float)(?*?2?8??@??H??b>gradient_tape/model/batch_normalization_2/FusedBatchNormGradV3hu  ?B
?
void cudnn::winograd_nonfused::winogradWgradDelta4x4<float, float>(cudnn::winograd_nonfused::WinogradDeltaParams<float, float>)@??*?2 28??@??H??Xb8gradient_tape/model/conv2d_9/Conv2D/Conv2DBackpropFilterhuZU?B
?
}void cudnn::winograd_nonfused::winogradWgradData4x4<float, float>(cudnn::winograd_nonfused::WinogradDataParams<float, float>)@??*?2 28??@??H??Xb9gradient_tape/model/conv2d_13/Conv2D/Conv2DBackpropFilterhuZU?B
?
void cudnn::winograd_nonfused::winogradForwardData4x4<float, float>(cudnn::winograd_nonfused::WinogradDataParams<float, float>)@??*?2?8??@??H??Xb7gradient_tape/model/conv2d_9/Conv2D/Conv2DBackpropInputhuZU?B
?
void cudnn::winograd_nonfused::winogradForwardData4x4<float, float>(cudnn::winograd_nonfused::WinogradDataParams<float, float>)@??*?2?8??@??H??Xb8gradient_tape/model/conv2d_13/Conv2D/Conv2DBackpropInputhuZU?B
?
?void cudnn::winograd_nonfused::winogradForwardOutput4x4<float, float>(cudnn::winograd_nonfused::WinogradOutputParams<float, float>)0??*?2?8??@??H??Xb8gradient_tape/model/conv2d_13/Conv2D/Conv2DBackpropInputhu???B
?
void cudnn::winograd_nonfused::winogradForwardData4x4<float, float>(cudnn::winograd_nonfused::WinogradDataParams<float, float>)@??*?2?8??@??H??Xb8gradient_tape/model/conv2d_12/Conv2D/Conv2DBackpropInputhuZU?B
?
void cudnn::winograd_nonfused::winogradForwardData4x4<float, float>(cudnn::winograd_nonfused::WinogradDataParams<float, float>)@??*?2?8??@??H??Xb7gradient_tape/model/conv2d_8/Conv2D/Conv2DBackpropInputhuZU?B
?
void cudnn::winograd_nonfused::winogradForwardData4x4<float, float>(cudnn::winograd_nonfused::WinogradDataParams<float, float>)@??*?2?8??@??H??bmodel/conv2d_9/ReluhuZU?B
?
void cudnn::winograd_nonfused::winogradForwardData4x4<float, float>(cudnn::winograd_nonfused::WinogradDataParams<float, float>)@??*?2?8??@??H??bmodel/conv2d_13/ReluhuZU?B
?
?void cudnn::winograd_nonfused::winogradForwardOutput4x4<float, float>(cudnn::winograd_nonfused::WinogradOutputParams<float, float>)0??*?2?8??@??H??bmodel/conv2d_9/Reluhu???B
?
?void cudnn::pooling_bw_kernel_max_nchw_fully_packed_small<float, float, 2, (cudnnNanPropagation_t)0>(cudnnTensorStruct, float const*, cudnnTensorStruct, float const*, cudnnTensorStruct, float const*, cudnnTensorStruct, float*, cudnnPoolingStruct, float, float, cudnn::reduced_divisor, cudnn::reduced_divisor) ?*?22?8??@??H??b7gradient_tape/model/max_pooling2d_3/MaxPool/MaxPoolGradhu  ?B
?
?void cudnn::winograd_nonfused::winogradForwardOutput4x4<float, float>(cudnn::winograd_nonfused::WinogradOutputParams<float, float>)0??*?2?8??@??H??bmodel/conv2d_13/Reluhu???B
?
?void cudnn::winograd_nonfused::winogradForwardOutput4x4<float, float>(cudnn::winograd_nonfused::WinogradOutputParams<float, float>)0??*?2?8??@??H??bmodel/conv2d_12/Reluhu???B
?
?void cudnn::winograd_nonfused::winogradForwardOutput4x4<float, float>(cudnn::winograd_nonfused::WinogradOutputParams<float, float>)0??*?2?8??@??H??bmodel/conv2d_8/Reluhu???B
?
?void tensorflow::functor::ShuffleInTensor3Simple<float, 2, 1, 0, false>(int, float const*, tensorflow::functor::Dimension<3>, float*)*?2&8??@??H??XbJgradient_tape/model/conv2d_transpose/conv2d_transpose/Conv2DBackpropFilterhuZU?B
q
(GreaterEqual_GPU_DT_FLOAT_DT_BOOL_kernel*?2?8??@??H??b$model/dropout_6/dropout/GreaterEqualhuZU?B
?
?void cudnn::ops::pooling_fw_4d_kernel<float, float, cudnn::maxpooling_func<float, (cudnnNanPropagation_t)0>, (cudnnPoolingMode_t)0, false>(cudnnTensorStruct, float const*, cudnnTensorStruct, float*, cudnnPoolingStruct, float, float, int, cudnn::reduced_divisor, cudnn::reduced_divisor)( ?*?2 28??@??H??bmodel/max_pooling2d_2/MaxPoolhu @?B
?
?void cudnn::winograd_nonfused::winogradForwardFilter4x4<float, float>(cudnn::winograd_nonfused::WinogradFilterParams<float, float>)(?H* 2@8??@??H??Xb7gradient_tape/model/conv2d_9/Conv2D/Conv2DBackpropInputhu  ?B
?
?void cudnn::winograd_nonfused::winogradForwardFilter4x4<float, float>(cudnn::winograd_nonfused::WinogradFilterParams<float, float>)(?H* 2@8߼@߼H߼Xb8gradient_tape/model/conv2d_13/Conv2D/Conv2DBackpropInputhu  ?B
?
?void tensorflow::functor::ShuffleInTensor3Simple<float, 2, 1, 0, false>(int, float const*, tensorflow::functor::Dimension<3>, float*)*?2&8??@??H??Xb7gradient_tape/model/conv2d_9/Conv2D/Conv2DBackpropInputhuZU?B
?
?void cudnn::winograd_nonfused::winogradForwardFilter4x4<float, float>(cudnn::winograd_nonfused::WinogradFilterParams<float, float>)(?H* 2@8??@??H??bmodel/conv2d_13/Reluhu  ?B
?
?void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 4, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorSlicingOp<Eigen::DSizes<int, 4> const, Eigen::DSizes<int, 4> const, Eigen::TensorMap<Eigen::Tensor<float const, 4, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 4, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorSlicingOp<Eigen::DSizes<int, 4> const, Eigen::DSizes<int, 4> const, Eigen::TensorMap<Eigen::Tensor<float const, 4, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int)*?298??@??H??b'gradient_tape/model/concatenate_1/SlicehuZU?B
?
void cudnn::winograd_nonfused::winogradForwardData4x4<float, float>(cudnn::winograd_nonfused::WinogradDataParams<float, float>)@??*?2?8??@??H??bmodel/conv2d_6/ReluhuZU?B
?
?void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 4, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorSlicingOp<Eigen::DSizes<int, 4> const, Eigen::DSizes<int, 4> const, Eigen::TensorMap<Eigen::Tensor<float const, 4, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 4, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorSlicingOp<Eigen::DSizes<int, 4> const, Eigen::DSizes<int, 4> const, Eigen::TensorMap<Eigen::Tensor<float const, 4, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int)*?298??@??H??b)gradient_tape/model/concatenate_1/Slice_1huZU?B
?
?void cub::DeviceSegmentedReduceKernel<cub::DeviceReducePolicy<float, float, int, cub::Sum>::Policy600, float const*, float*, cub::TransformInputIterator<int, tensorflow::functor::RowOffset, cub::CountingInputIterator<int, long>, long>, int, cub::Sum, float>(float const*, float*, cub::TransformInputIterator<int, tensorflow::functor::RowOffset, cub::CountingInputIterator<int, long>, long>, cub::TransformInputIterator<int, tensorflow::functor::RowOffset, cub::CountingInputIterator<int, long>, long>, int, cub::Sum, float) 0*?2?28??@??H??b0gradient_tape/model/conv2d_5/BiasAdd/BiasAddGradhu  ?B
?
?void cudnn::bn_fw_tr_1C11_singleread<float, 512, true, 1, 2, 0>(cudnnTensorStruct, float const*, cudnnTensorStruct, float*, float const*, float const*, float, float, float*, float*, float*, float*, float, float, cudnn::reduced_divisor, int, cudnn::reduced_divisor, cudnn::bnFwPersistentState*, int, float, float, float, int, float, float, cudnnStatus_t*, bool)'? ??*?2?8??@??H??b,model/batch_normalization_4/FusedBatchNormV3hu  ?B
?
void cudnn::winograd_nonfused::winogradForwardData4x4<float, float>(cudnn::winograd_nonfused::WinogradDataParams<float, float>)@??*?2?8??@??H??Xb8gradient_tape/model/conv2d_10/Conv2D/Conv2DBackpropInputhuZU?B
?
void cudnn::winograd_nonfused::winogradForwardData4x4<float, float>(cudnn::winograd_nonfused::WinogradDataParams<float, float>)@??*?2?8??@??H??bmodel/conv2d_11/ReluhuZU?B
?
void cudnn::winograd_nonfused::winogradWgradDelta4x4<float, float>(cudnn::winograd_nonfused::WinogradDeltaParams<float, float>)@??*?2 28??@??H??Xb9gradient_tape/model/conv2d_12/Conv2D/Conv2DBackpropFilterhuZU?B
?
void cudnn::winograd_nonfused::winogradWgradDelta4x4<float, float>(cudnn::winograd_nonfused::WinogradDeltaParams<float, float>)@??*?2 28??@??H??Xb9gradient_tape/model/conv2d_13/Conv2D/Conv2DBackpropFilterhuZU?B
?
void cudnn::winograd_nonfused::winogradWgradDelta4x4<float, float>(cudnn::winograd_nonfused::WinogradDeltaParams<float, float>)@??*?2 28??@??H??Xb8gradient_tape/model/conv2d_8/Conv2D/Conv2DBackpropFilterhuZU?B
?
Yvoid tensorflow::BiasNCHWKernel<float>(int, float const*, float const*, float*, int, int)*?2&8??@??H??b model/conv2d_transpose_1/BiasAddhuZU?B
?
?void cudnn::winograd_nonfused::winogradForwardOutput4x4<float, float>(cudnn::winograd_nonfused::WinogradOutputParams<float, float>)0??*?2?8??@??H??bmodel/conv2d_10/Reluhu???B
?
}void cudnn::winograd_nonfused::winogradWgradData4x4<float, float>(cudnn::winograd_nonfused::WinogradDataParams<float, float>)@??*?2 28??@??H??Xb8gradient_tape/model/conv2d_9/Conv2D/Conv2DBackpropFilterhuZU?B
?
?void cudnn::winograd_nonfused::winogradForwardOutput4x4<float, float>(cudnn::winograd_nonfused::WinogradOutputParams<float, float>)0??*?2?8??@??H??bmodel/conv2d_11/Reluhu???B
?
void cudnn::winograd_nonfused::winogradForwardData4x4<float, float>(cudnn::winograd_nonfused::WinogradDataParams<float, float>)@??*?2?8??@??H??Xb8gradient_tape/model/conv2d_11/Conv2D/Conv2DBackpropInputhuZU?B
?
?void cudnn::winograd_nonfused::winogradForwardOutput4x4<float, float>(cudnn::winograd_nonfused::WinogradOutputParams<float, float>)0??*?2?8??@??H??Xb8gradient_tape/model/conv2d_11/Conv2D/Conv2DBackpropInputhu???B
?
?void cub::DeviceSegmentedReduceKernel<cub::DeviceReducePolicy<float, float, int, cub::Sum>::Policy600, float const*, float*, cub::TransformInputIterator<int, tensorflow::functor::RowOffset, cub::CountingInputIterator<int, long>, long>, int, cub::Sum, float>(float const*, float*, cub::TransformInputIterator<int, tensorflow::functor::RowOffset, cub::CountingInputIterator<int, long>, long>, cub::TransformInputIterator<int, tensorflow::functor::RowOffset, cub::CountingInputIterator<int, long>, long>, int, cub::Sum, float) 0*?2?28??@??H??b0gradient_tape/model/conv2d_4/BiasAdd/BiasAddGradhu  ?B
?
?void cub::DeviceSegmentedReduceKernel<cub::DeviceReducePolicy<float, float, int, cub::Sum>::Policy600, float const*, float*, cub::TransformInputIterator<int, tensorflow::functor::RowOffset, cub::CountingInputIterator<int, long>, long>, int, cub::Sum, float>(float const*, float*, cub::TransformInputIterator<int, tensorflow::functor::RowOffset, cub::CountingInputIterator<int, long>, long>, cub::TransformInputIterator<int, tensorflow::functor::RowOffset, cub::CountingInputIterator<int, long>, long>, int, cub::Sum, float) 0*?2?28??@??H??b1gradient_tape/model/conv2d_16/BiasAdd/BiasAddGradhu  ?B
?
?void tensorflow::functor::SwapDimension1And2InTensor3UsingTiles<unsigned int, 256, 32, 32, false>(unsigned int const*, tensorflow::functor::Dimension<3>, unsigned int*) ?!*?2?(8??@??H??bCmodel/dropout_5/dropout/Mul_1-0-TransposeNHWCToNCHW-LayoutOptimizerhu  ?B
?
?void foldedNhwcToNchwKernel<float, float, float, true, (cudnnKernelDataType_t)0>(int, int, int, int, int, int, int, float const*, float*, int, int, int, int, int, int, int, int, int, int, float, float, cudnn::reduced_divisor, cudnn::reduced_divisor, cudnn::reduced_divisor)&?!* 228??@??H??b)model/conv2d_transpose_1/conv2d_transposehu  ?B
?
?void cub::DeviceSegmentedReduceKernel<cub::DeviceReducePolicy<float, float, int, cub::Sum>::Policy600, float const*, float*, cub::TransformInputIterator<int, tensorflow::functor::RowOffset, cub::CountingInputIterator<int, long>, long>, int, cub::Sum, float>(float const*, float*, cub::TransformInputIterator<int, tensorflow::functor::RowOffset, cub::CountingInputIterator<int, long>, long>, cub::TransformInputIterator<int, tensorflow::functor::RowOffset, cub::CountingInputIterator<int, long>, long>, int, cub::Sum, float) 0*?2?28߱@߱H߱b1gradient_tape/model/conv2d_17/BiasAdd/BiasAddGradhu  ?B
n
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*?2?	8??@??H??b)gradient_tape/model/dropout_5/dropout/MulhuZU?B
?
?void tensorflow::functor::FillPhiloxRandomKernelLaunch<tensorflow::random::UniformDistribution<tensorflow::random::PhiloxRandom, float> >(unsigned long const*, unsigned long const*, tensorflow::random::PhiloxRandom, tensorflow::random::UniformDistribution<tensorflow::random::PhiloxRandom, float>::ResultElementType*, long, tensorflow::random::UniformDistribution<tensorflow::random::PhiloxRandom, float>)*?298??@??H??b4model/dropout_6/dropout/random_uniform/RandomUniformhuZU?B
?
?void tensorflow::functor::SwapDimension1And2InTensor3UsingTiles<unsigned int, 256, 32, 32, false>(unsigned int const*, tensorflow::functor::Dimension<3>, unsigned int*) ?!*?2?'8??@??H??bAmodel/dropout/dropout/Mul_1-0-TransposeNHWCToNCHW-LayoutOptimizerhu  ?B
?
?void cub::DeviceSegmentedReduceKernel<cub::DeviceReducePolicy<float, float, int, cub::Sum>::Policy600, float const*, float*, cub::TransformInputIterator<int, tensorflow::functor::RowOffset, cub::CountingInputIterator<int, long>, long>, int, cub::Sum, float>(float const*, float*, cub::TransformInputIterator<int, tensorflow::functor::RowOffset, cub::CountingInputIterator<int, long>, long>, cub::TransformInputIterator<int, tensorflow::functor::RowOffset, cub::CountingInputIterator<int, long>, long>, int, cub::Sum, float) 0*?2?28??@??H??b:gradient_tape/model/conv2d_transpose_2/BiasAdd/BiasAddGradhu  ?B
?
?void cudnn::ops::nchwToNhwcKernel<float, float, float, false, true, (cudnnKernelDataType_t)2>(cudnn::ops::nchw2nhwc_params_t<float>, float const*, float*)'?!*?2228ߦ@ߦHߦbmodel/conv2d_4/Reluhu  ?B
^
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*?2?	8??@??H??bmodel/dropout/dropout/MulhuZU?B
`
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*?2?	8ߟ@ߟHߟbmodel/dropout_5/dropout/MulhuZU?B
?
?void cudnn::ops::nhwcToNchwKernel<float, float, float, true, false, (cudnnKernelDataType_t)0>(cudnn::ops::nhwc2nchw_params_t<float>, float const*, float*)&?!*?2?8??@??H??Xb9gradient_tape/model/conv2d_10/Conv2D/Conv2DBackpropFilterhu  ?B
?
?void cudnn::ops::nchwToNhwcKernel<float, float, float, false, true, (cudnnKernelDataType_t)2>(cudnn::ops::nchw2nhwc_params_t<float>, float const*, float*)'?!*?228??@??H??XbLgradient_tape/model/conv2d_transpose_1/conv2d_transpose/Conv2DBackpropFilterhu  ?B
?
?void cudnn::winograd_nonfused::winogradForwardOutput4x4<float, float>(cudnn::winograd_nonfused::WinogradOutputParams<float, float>)0??*?2?8??@??H??Xb7gradient_tape/model/conv2d_6/Conv2D/Conv2DBackpropInputhu???B
?
?void cudnn::ops::nchwToNhwcKernel<float, float, float, false, true, (cudnnKernelDataType_t)2>(cudnn::ops::nchw2nhwc_params_t<float>, float const*, float*)'?!*?228??@??H??Xb>gradient_tape/model/conv2d_transpose_1/conv2d_transpose/Conv2Dhu  ?B
l
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*?2?	8??@??H??b'gradient_tape/model/dropout/dropout/MulhuZU?B
?
?void cudnn::winograd_nonfused::winogradWgradOutput4x4<float, float>(cudnn::winograd_nonfused::WinogradWgradOutputParams<float, float>)=?H* 2 8??@??H??Xb9gradient_tape/model/conv2d_14/Conv2D/Conv2DBackpropFilterhuZU?B
?
?void cudnn::winograd_nonfused::winogradWgradOutput4x4<float, float>(cudnn::winograd_nonfused::WinogradWgradOutputParams<float, float>)=?H* 2@8??@??H??Xb8gradient_tape/model/conv2d_8/Conv2D/Conv2DBackpropFilterhuZU?B
?
?void tensorflow::functor::ShuffleInTensor3Simple<float, 2, 1, 0, false>(int, float const*, tensorflow::functor::Dimension<3>, float*)*?2&8??@??H??Xb8gradient_tape/model/conv2d_8/Conv2D/Conv2DBackpropFilterhuZU?B
?
?void tensorflow::functor::ShuffleInTensor3Simple<float, 2, 1, 0, false>(int, float const*, tensorflow::functor::Dimension<3>, float*)*?2&8??@??H??Xb9gradient_tape/model/conv2d_14/Conv2D/Conv2DBackpropFilterhuZU?B
?
Lvoid cudnn::ops::scalePackedTensor_kernel<float, float>(long, float*, float)*?2??8??@??H??b)model/conv2d_transpose_2/conv2d_transposehu  ?B
?
?void cudnn::winograd_nonfused::winogradForwardFilter4x4<float, float>(cudnn::winograd_nonfused::WinogradFilterParams<float, float>)(?H* 2 8??@??H??Xb8gradient_tape/model/conv2d_14/Conv2D/Conv2DBackpropInputhu  ?B
?
?void cudnn::bn_fw_tr_1C11_kernel_NCHW<float, float, 32, true, 1>(cudnnTensorStruct, float const*, cudnnTensorStruct, float*, float const*, float const*, float, float, float*, float*, float*, float*, float, float)(?*?2?8??@??H??b,model/batch_normalization_2/FusedBatchNormV3hu  ?B
?
?void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<float, Eigen::TensorMap<Eigen::Tensor<bool const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<float, Eigen::TensorMap<Eigen::Tensor<bool const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*?298??@??H??bmodel/dropout_5/dropout/CasthuZU?B
?
?void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<float, Eigen::TensorMap<Eigen::Tensor<bool const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<float, Eigen::TensorMap<Eigen::Tensor<bool const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*?298ߢ@ߢHߢbmodel/dropout/dropout/CasthuZU?B
?
}void cudnn::winograd_nonfused::winogradWgradData4x4<float, float>(cudnn::winograd_nonfused::WinogradDataParams<float, float>)@??*?228ߡ@ߡHߡXb8gradient_tape/model/conv2d_6/Conv2D/Conv2DBackpropFilterhuZU?B
?
?void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<float, Eigen::TensorMap<Eigen::Tensor<bool const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<float, Eigen::TensorMap<Eigen::Tensor<bool const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*?298??@??H??bXmodel/dropout_5/dropout/ArithmeticOptimizer/ReorderCastLikeAndValuePreserving_float_CasthuZU?B
?
?void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<float, Eigen::TensorMap<Eigen::Tensor<bool const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<float, Eigen::TensorMap<Eigen::Tensor<bool const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*?298??@??H??bVmodel/dropout/dropout/ArithmeticOptimizer/ReorderCastLikeAndValuePreserving_float_CasthuZU?B
?
?void cudnn::ops::nhwcToNchwKernel<float, float, float, true, false, (cudnnKernelDataType_t)0>(cudnn::ops::nhwc2nchw_params_t<float>, float const*, float*)&?!*?2?8??@??H??XbJgradient_tape/model/conv2d_transpose/conv2d_transpose/Conv2DBackpropFilterhu  ?B
?
?void nchwToFoldedNhwcKernel<float, float, float, true, (cudnnKernelDataType_t)2>(int, int, int, int, float const*, float*, int, int, int, int, int, int, int, int, int, int, float, float, cudnn::reduced_divisor, cudnn::reduced_divisor, cudnn::reduced_divisor)&?!* 2?8??@??H??b'model/conv2d_transpose/conv2d_transposehu  ?B
?
?void tensorflow::functor::ApplyAdamKernel<float>(int, float*, float*, float*, float const*, float const*, float const*, float const*, float const*, float const*, float const*, bool) *?2&8??@??H??b%Adam/Adam/update_44/ResourceApplyAdamhuZU?B
?
?void tensorflow::functor::ApplyAdamKernel<float>(int, float*, float*, float*, float const*, float const*, float const*, float const*, float const*, float const*, float const*, bool) *?2&8ߔ@ߔHߔb%Adam/Adam/update_24/ResourceApplyAdamhuZU?B
?
?void tensorflow::functor::ShuffleInTensor3Simple<float, 2, 1, 0, false>(int, float const*, tensorflow::functor::Dimension<3>, float*)*?2&8??@??H??bmodel/conv2d_8/ReluhuZU?B
?
?void tensorflow::functor::ShuffleInTensor3Simple<float, 2, 1, 0, false>(int, float const*, tensorflow::functor::Dimension<3>, float*)*?2&8??@??H??bmodel/conv2d_9/ReluhuZU?B
?
?void tensorflow::functor::ShuffleInTensor3Simple<float, 2, 1, 0, false>(int, float const*, tensorflow::functor::Dimension<3>, float*)*?2&8??@??H??bmodel/conv2d_13/ReluhuZU?B
?
?void tensorflow::functor::ShuffleInTensor3Simple<float, 2, 1, 0, false>(int, float const*, tensorflow::functor::Dimension<3>, float*)*?2&8??@??H??Xb8gradient_tape/model/conv2d_13/Conv2D/Conv2DBackpropInputhuZU?B
?
?void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_product_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorConversionOp<float, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_cmp_op<float const, float const, (Eigen::internal::ComparisonName)5>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorCwiseNullaryOp<Eigen::internal::scalar_constant_op<float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_product_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorConversionOp<float, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_cmp_op<float const, float const, (Eigen::internal::ComparisonName)5>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorCwiseNullaryOp<Eigen::internal::scalar_constant_op<float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const> const> const> const, Eigen::GpuDevice>, long)*?298??@??H??b%gradient_tape/model/conv2d_8/ReluGradhuZU?B
?
?void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_product_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorConversionOp<float, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_cmp_op<float const, float const, (Eigen::internal::ComparisonName)5>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorCwiseNullaryOp<Eigen::internal::scalar_constant_op<float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_product_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorConversionOp<float, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_cmp_op<float const, float const, (Eigen::internal::ComparisonName)5>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorCwiseNullaryOp<Eigen::internal::scalar_constant_op<float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const> const> const> const, Eigen::GpuDevice>, long)*?298??@??H??b&gradient_tape/model/conv2d_12/ReluGradhuZU?B
?
?void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_product_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorConversionOp<float, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_cmp_op<float const, float const, (Eigen::internal::ComparisonName)5>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorCwiseNullaryOp<Eigen::internal::scalar_constant_op<float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_product_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorConversionOp<float, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_cmp_op<float const, float const, (Eigen::internal::ComparisonName)5>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorCwiseNullaryOp<Eigen::internal::scalar_constant_op<float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const> const> const> const, Eigen::GpuDevice>, long)*?298??@??H??b%gradient_tape/model/conv2d_9/ReluGradhuZU?B
?
?void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_product_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorConversionOp<float, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_cmp_op<float const, float const, (Eigen::internal::ComparisonName)5>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorCwiseNullaryOp<Eigen::internal::scalar_constant_op<float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_product_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorConversionOp<float, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_cmp_op<float const, float const, (Eigen::internal::ComparisonName)5>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorCwiseNullaryOp<Eigen::internal::scalar_constant_op<float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const> const> const> const, Eigen::GpuDevice>, long)*?298??@??H??b&gradient_tape/model/conv2d_13/ReluGradhuZU?B
b
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*?2?8??@??H??bmodel/dropout_1/dropout/Mul_1huZU?B
?
?void tensorflow::functor::SwapDimension1And2InTensor3UsingTiles<unsigned char, 256, 32, 32, false>(unsigned char const*, tensorflow::functor::Dimension<3>, unsigned char*) ?*?2?N8??@??H??b?gradient_tape/model/dropout_6/dropout/ArithmeticOptimizer/ReorderCastLikeAndValuePreserving_bool_Mul_2-1-TransposeNHWCToNCHW-LayoutOptimizerhu  ?B
p
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*?2?8??@??H??b+gradient_tape/model/dropout_1/dropout/Mul_2huZU?B
p
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*?2?8??@??H??b+gradient_tape/model/dropout_4/dropout/Mul_2huZU?B
?
?void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*?298??@??H??bAdam/gradients/AddN_1huZU?B
b
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*?2?8??@??H??bmodel/dropout_4/dropout/Mul_1huZU?B
?
?void tensorflow::functor::ShuffleInTensor3Simple<float, 2, 1, 0, false>(int, float const*, tensorflow::functor::Dimension<3>, float*)*?2&8??@??H??Xb<gradient_tape/model/conv2d_transpose/conv2d_transpose/Conv2DhuZU?B
?
?void tensorflow::functor::ShuffleInTensor3Simple<float, 2, 1, 0, false>(int, float const*, tensorflow::functor::Dimension<3>, float*)*?2&8??@??H??b'model/conv2d_transpose/conv2d_transposehuZU?B
q
(GreaterEqual_GPU_DT_FLOAT_DT_BOOL_kernel*?2?	8??@??H??b$model/dropout_5/dropout/GreaterEqualhuZU?B
o
(GreaterEqual_GPU_DT_FLOAT_DT_BOOL_kernel*?2?	8??@??H??b"model/dropout/dropout/GreaterEqualhuZU?B
?
void cudnn::winograd_nonfused::winogradForwardData4x4<float, float>(cudnn::winograd_nonfused::WinogradDataParams<float, float>)@??*?2?8??@??H??bmodel/conv2d_8/ReluhuZU?B
?
?void cudnn::winograd_nonfused::winogradForwardOutput4x4<float, float>(cudnn::winograd_nonfused::WinogradOutputParams<float, float>)0??*?2?8??@??H??Xb7gradient_tape/model/conv2d_8/Conv2D/Conv2DBackpropInputhu???B
?
?void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float, float>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float, float>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float, float>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float, float>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*?298??@??H??bAdam/gradients/AddNhuZU?B
?
?void cudnn::ops::pooling_fw_4d_kernel<float, float, cudnn::maxpooling_func<float, (cudnnNanPropagation_t)0>, (cudnnPoolingMode_t)0, false>(cudnnTensorStruct, float const*, cudnnTensorStruct, float*, cudnnPoolingStruct, float, float, int, cudnn::reduced_divisor, cudnn::reduced_divisor)( ?*?2@28??@??H??bmodel/max_pooling2d_3/MaxPoolhu @?B
?
?void cudnn::winograd_nonfused::winogradForwardFilter4x4<float, float>(cudnn::winograd_nonfused::WinogradFilterParams<float, float>)(?H* 2 8??@??H??bmodel/conv2d_14/Reluhu  ?B
?
?void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 4, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorSlicingOp<Eigen::DSizes<int, 4> const, Eigen::DSizes<int, 4> const, Eigen::TensorMap<Eigen::Tensor<float const, 4, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 4, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorSlicingOp<Eigen::DSizes<int, 4> const, Eigen::DSizes<int, 4> const, Eigen::TensorMap<Eigen::Tensor<float const, 4, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int)*?298??@??H??b%gradient_tape/model/concatenate/SlicehuZU?B
?
?void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 4, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorSlicingOp<Eigen::DSizes<int, 4> const, Eigen::DSizes<int, 4> const, Eigen::TensorMap<Eigen::Tensor<float const, 4, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 4, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorSlicingOp<Eigen::DSizes<int, 4> const, Eigen::DSizes<int, 4> const, Eigen::TensorMap<Eigen::Tensor<float const, 4, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int)*?298??@??H??b'gradient_tape/model/concatenate/Slice_1huZU?B
?
?void cudnn::winograd_nonfused::winogradForwardFilter4x4<float, float>(cudnn::winograd_nonfused::WinogradFilterParams<float, float>)(?H* 2@8??@??H??Xb7gradient_tape/model/conv2d_8/Conv2D/Conv2DBackpropInputhu  ?B
?
?void cudnn::winograd_nonfused::winogradForwardFilter4x4<float, float>(cudnn::winograd_nonfused::WinogradFilterParams<float, float>)(?H* 2@8??@??H??bmodel/conv2d_8/Reluhu  ?B
?
void cudnn::winograd_nonfused::winogradForwardData4x4<float, float>(cudnn::winograd_nonfused::WinogradDataParams<float, float>)@??*?2?8??@??H??bmodel/conv2d_10/ReluhuZU?B
?
?void tensorflow::functor::RowReduceKernel<float const*, float*, cub::Sum>(float const*, float*, int, int, cub::Sum, std::iterator_traits<float const*>::value_type)*?2?8??@??H??b0gradient_tape/model/conv2d_6/BiasAdd/BiasAddGradhu  ?B
?
Yvoid tensorflow::BiasNCHWKernel<float>(int, float const*, float const*, float*, int, int)*?2&8??@??H??bmodel/conv2d_transpose/BiasAddhuZU?B
?
}void cudnn::winograd_nonfused::winogradWgradData4x4<float, float>(cudnn::winograd_nonfused::WinogradDataParams<float, float>)@??*?228??@??H??Xb8gradient_tape/model/conv2d_8/Conv2D/Conv2DBackpropFilterhuZU?B
?
?void cudnn::winograd_nonfused::winogradForwardOutput4x4<float, float>(cudnn::winograd_nonfused::WinogradOutputParams<float, float>)0??*?2?8??@??H??Xb8gradient_tape/model/conv2d_10/Conv2D/Conv2DBackpropInputhu???B
?
?void tensorflow::functor::RowReduceKernel<float const*, float*, cub::Sum>(float const*, float*, int, int, cub::Sum, std::iterator_traits<float const*>::value_type)*?2?8??@??H??b1gradient_tape/model/conv2d_15/BiasAdd/BiasAddGradhu  ?B
?
?void tensorflow::functor::RowReduceKernel<float const*, float*, cub::Sum>(float const*, float*, int, int, cub::Sum, std::iterator_traits<float const*>::value_type)*?2?8߳@߳H߳b0gradient_tape/model/conv2d_7/BiasAdd/BiasAddGradhu  ?B
?
?void tensorflow::functor::RowReduceKernel<float const*, float*, cub::Sum>(float const*, float*, int, int, cub::Sum, std::iterator_traits<float const*>::value_type)*?2?8??@??H??b:gradient_tape/model/conv2d_transpose_1/BiasAdd/BiasAddGradhu  ?B
?
?void cudnn::ops::nchwToNhwcKernel<float, float, float, false, true, (cudnnKernelDataType_t)2>(cudnn::ops::nchw2nhwc_params_t<float>, float const*, float*)'?!*?2 28??@??H??Xb9gradient_tape/model/conv2d_11/Conv2D/Conv2DBackpropFilterhu  ?B
?
?void tensorflow::functor::FillPhiloxRandomKernelLaunch<tensorflow::random::UniformDistribution<tensorflow::random::PhiloxRandom, float> >(unsigned long const*, unsigned long const*, tensorflow::random::PhiloxRandom, tensorflow::random::UniformDistribution<tensorflow::random::PhiloxRandom, float>::ResultElementType*, long, tensorflow::random::UniformDistribution<tensorflow::random::PhiloxRandom, float>)*?298??@??H??b4model/dropout_5/dropout/random_uniform/RandomUniformhuZU?B
?
?void tensorflow::functor::FillPhiloxRandomKernelLaunch<tensorflow::random::UniformDistribution<tensorflow::random::PhiloxRandom, float> >(unsigned long const*, unsigned long const*, tensorflow::random::PhiloxRandom, tensorflow::random::UniformDistribution<tensorflow::random::PhiloxRandom, float>::ResultElementType*, long, tensorflow::random::UniformDistribution<tensorflow::random::PhiloxRandom, float>)*?298??@??H??b2model/dropout/dropout/random_uniform/RandomUniformhuZU?B
?
?void cudnn::winograd_nonfused::winogradWgradOutput4x4<float, float>(cudnn::winograd_nonfused::WinogradWgradOutputParams<float, float>)=?H* 2 8??@??H??Xb9gradient_tape/model/conv2d_15/Conv2D/Conv2DBackpropFilterhuZU?B
?
?void cudnn::ops::nchwToNhwcKernel<float, float, float, false, true, (cudnnKernelDataType_t)2>(cudnn::ops::nchw2nhwc_params_t<float>, float const*, float*)'?!*?228??@??H??b)model/conv2d_transpose_1/conv2d_transposehu  ?B
?
?void cudnn::winograd_nonfused::winogradWgradOutput4x4<float, float>(cudnn::winograd_nonfused::WinogradWgradOutputParams<float, float>)=?H* 2 8??@??H??Xb8gradient_tape/model/conv2d_7/Conv2D/Conv2DBackpropFilterhuZU?B
?
?void tensorflow::functor::RowReduceKernel<float const*, float*, cub::Sum>(float const*, float*, int, int, cub::Sum, std::iterator_traits<float const*>::value_type)*?2?8??@??H??b1gradient_tape/model/conv2d_14/BiasAdd/BiasAddGradhu  ?B
?
?void tensorflow::functor::SwapDimension1And2InTensor3UsingTiles<unsigned int, 256, 32, 32, false>(unsigned int const*, tensorflow::functor::Dimension<3>, unsigned int*) ?!*?2?8??@??H??bCmodel/dropout_4/dropout/Mul_1-0-TransposeNHWCToNCHW-LayoutOptimizerhu  ?B
?
?void foldedNhwcToNchwKernel<float, float, float, true, (cudnnKernelDataType_t)0>(int, int, int, int, int, int, int, float const*, float*, int, int, int, int, int, int, int, int, int, int, float, float, cudnn::reduced_divisor, cudnn::reduced_divisor, cudnn::reduced_divisor)&?!* 228??@??H??b'model/conv2d_transpose/conv2d_transposehu  ?B
`
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*?2?8??@??H??bmodel/dropout_4/dropout/MulhuZU?B
?
?void cudnn::ops::nchwToNhwcKernel<float, float, float, false, true, (cudnnKernelDataType_t)2>(cudnn::ops::nchw2nhwc_params_t<float>, float const*, float*)'?!*?228??@??H??XbLgradient_tape/model/conv2d_transpose_1/conv2d_transpose/Conv2DBackpropFilterhu  ?B
?
?void cudnn::ops::nchwToNhwcKernel<float, float, float, false, true, (cudnnKernelDataType_t)2>(cudnn::ops::nchw2nhwc_params_t<float>, float const*, float*)'?!*?228??@??H??XbJgradient_tape/model/conv2d_transpose/conv2d_transpose/Conv2DBackpropFilterhu  ?B
`
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*?2?8??@??H??bmodel/dropout_1/dropout/MulhuZU?B
?
?void tensorflow::functor::SwapDimension1And2InTensor3UsingTiles<unsigned int, 256, 32, 32, false>(unsigned int const*, tensorflow::functor::Dimension<3>, unsigned int*) ?!*?2?8??@??H??bCmodel/dropout_1/dropout/Mul_1-0-TransposeNHWCToNCHW-LayoutOptimizerhu  ?B
n
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*?2?8??@??H??b)gradient_tape/model/dropout_4/dropout/MulhuZU?B
n
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*?2?8??@??H??b)gradient_tape/model/dropout_1/dropout/MulhuZU?B
?
?void tensorflow::functor::ApplyAdamKernel<float>(int, float*, float*, float*, float const*, float const*, float const*, float const*, float const*, float const*, float const*, bool) *?2&8??@??H??b%Adam/Adam/update_20/ResourceApplyAdamhuZU?B
?
?void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<float, Eigen::TensorMap<Eigen::Tensor<bool const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<float, Eigen::TensorMap<Eigen::Tensor<bool const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*?298??@??H??bmodel/dropout_1/dropout/CasthuZU?B
?
?void cudnn::bn_bw_1C11_singleread<float, 512, true, 1, 2, 0>(float, float, float, float, cudnnTensorStruct, float const*, cudnnTensorStruct, float const*, cudnnTensorStruct, float*, float const*, float*, float*, float const*, float const*, float, cudnn::reduced_divisor, int, cudnn::reduced_divisor, cudnn::bnBwPersistentState*, int, float, float, float, int, float, cudnnStatus_t*, bool)(? ??*?2?8??@??H??b>gradient_tape/model/batch_normalization_3/FusedBatchNormGradV3huZU?B
?
?void tensorflow::functor::ApplyAdamKernel<float>(int, float*, float*, float*, float const*, float const*, float const*, float const*, float const*, float const*, float const*, bool) *?2&8??@??H??b%Adam/Adam/update_46/ResourceApplyAdamhuZU?B
?
?void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<float, Eigen::TensorMap<Eigen::Tensor<bool const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<float, Eigen::TensorMap<Eigen::Tensor<bool const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*?298??@??H??bXmodel/dropout_4/dropout/ArithmeticOptimizer/ReorderCastLikeAndValuePreserving_float_CasthuZU?B
?
?void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<float, Eigen::TensorMap<Eigen::Tensor<bool const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<float, Eigen::TensorMap<Eigen::Tensor<bool const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*?298??@??H??bmodel/dropout_4/dropout/CasthuZU?B
?
?void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<float, Eigen::TensorMap<Eigen::Tensor<bool const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<float, Eigen::TensorMap<Eigen::Tensor<bool const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*?298??@??H??bXmodel/dropout_1/dropout/ArithmeticOptimizer/ReorderCastLikeAndValuePreserving_float_CasthuZU?B
?
?void tensorflow::functor::SwapDimension1And2InTensor3UsingTiles<unsigned char, 256, 32, 32, false>(unsigned char const*, tensorflow::functor::Dimension<3>, unsigned char*) ?*?2?(8??@??H??b?gradient_tape/model/dropout_5/dropout/ArithmeticOptimizer/ReorderCastLikeAndValuePreserving_bool_Mul_2-1-TransposeNHWCToNCHW-LayoutOptimizerhu  ?B
?
?void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_product_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorConversionOp<float, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_cmp_op<float const, float const, (Eigen::internal::ComparisonName)5>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorCwiseNullaryOp<Eigen::internal::scalar_constant_op<float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_product_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorConversionOp<float, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_cmp_op<float const, float const, (Eigen::internal::ComparisonName)5>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorCwiseNullaryOp<Eigen::internal::scalar_constant_op<float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const> const> const> const, Eigen::GpuDevice>, long)*?298??@??H??b&gradient_tape/model/conv2d_10/ReluGradhuZU?B
?
?void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_product_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorConversionOp<float, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_cmp_op<float const, float const, (Eigen::internal::ComparisonName)5>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorCwiseNullaryOp<Eigen::internal::scalar_constant_op<float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_product_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorConversionOp<float, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_cmp_op<float const, float const, (Eigen::internal::ComparisonName)5>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorCwiseNullaryOp<Eigen::internal::scalar_constant_op<float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const> const> const> const, Eigen::GpuDevice>, long)*?298??@??H??b&gradient_tape/model/conv2d_11/ReluGradhuZU?B
?
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*?2?8??@??H??b@gradient_tape/weighted_binary_crossentropy/logistic_loss/mul/MulhuZU?B
?
?void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorSelectOp<Eigen::TensorMap<Eigen::Tensor<bool const, 1, 1, int>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, int>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorSelectOp<Eigen::TensorMap<Eigen::Tensor<bool const, 1, 1, int>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, int>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int)*?298??@??H??b?gradient_tape/weighted_binary_crossentropy/logistic_loss/SelecthuZU?B
?
?void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorSelectOp<Eigen::TensorMap<Eigen::Tensor<bool const, 1, 1, int>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, int>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorSelectOp<Eigen::TensorMap<Eigen::Tensor<bool const, 1, 1, int>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, int>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int)*?298??@??H??b1weighted_binary_crossentropy/logistic_loss/SelecthuZU?B
p
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*?2?8??@??H??b+gradient_tape/model/dropout_2/dropout/Mul_2huZU?B
?
?void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorSelectOp<Eigen::TensorMap<Eigen::Tensor<bool const, 1, 1, int>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, int>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorSelectOp<Eigen::TensorMap<Eigen::Tensor<bool const, 1, 1, int>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, int>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int)*?298??@??H??bAgradient_tape/weighted_binary_crossentropy/logistic_loss/Select_2huZU?B
?
?void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorSelectOp<Eigen::TensorMap<Eigen::Tensor<bool const, 1, 1, int>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, int>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorSelectOp<Eigen::TensorMap<Eigen::Tensor<bool const, 1, 1, int>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, int>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int)*?298??@??H??bAgradient_tape/weighted_binary_crossentropy/logistic_loss/Select_3huZU?B
?
?void tensorflow::functor::SwapDimension1And2InTensor3UsingTiles<unsigned char, 256, 32, 32, false>(unsigned char const*, tensorflow::functor::Dimension<3>, unsigned char*) ?*?2?'8??@??H??b?gradient_tape/model/dropout/dropout/ArithmeticOptimizer/ReorderCastLikeAndValuePreserving_bool_Mul_2-1-TransposeNHWCToNCHW-LayoutOptimizerhu  ?B
?
?void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorSelectOp<Eigen::TensorMap<Eigen::Tensor<bool const, 1, 1, int>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, int>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorSelectOp<Eigen::TensorMap<Eigen::Tensor<bool const, 1, 1, int>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, int>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int)*?298??@??H??b3weighted_binary_crossentropy/logistic_loss/Select_1huZU?B
q
"AddV2_GPU_DT_FLOAT_DT_FLOAT_kernel*?2?8߸@߸H߸b*weighted_binary_crossentropy/logistic_losshuZU?B
s
 Sub_GPU_DT_FLOAT_DT_FLOAT_kernel*?2?8??@??H??b.weighted_binary_crossentropy/logistic_loss/subhuZU?B
?
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*?2?8??@??H??b<gradient_tape/weighted_binary_crossentropy/logistic_loss/mulhuZU?B
?
?void tensorflow::functor::ShuffleInTensor3Simple<float, 2, 1, 0, false>(int, float const*, tensorflow::functor::Dimension<3>, float*)*?2&8??@??H??Xb8gradient_tape/model/conv2d_14/Conv2D/Conv2DBackpropInputhuZU?B
?
?void tensorflow::functor::ShuffleInTensor3Simple<float, 2, 1, 0, false>(int, float const*, tensorflow::functor::Dimension<3>, float*)*?2&8??@??H??Xb7gradient_tape/model/conv2d_8/Conv2D/Conv2DBackpropInputhuZU?B
?
?void tensorflow::functor::ShuffleInTensor3Simple<float, 2, 1, 0, false>(int, float const*, tensorflow::functor::Dimension<3>, float*)*?2&8??@??H??bmodel/conv2d_14/ReluhuZU?B
b
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*?2?8??@??H??bmodel/dropout_2/dropout/Mul_1huZU?B
s
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*?2?8??@??H??b.weighted_binary_crossentropy/logistic_loss/mulhuZU?B
?
?void tensorflow::functor::ApplyAdamKernel<float>(int, float*, float*, float*, float const*, float const*, float const*, float const*, float const*, float const*, float const*, bool) *?2&8??@??H??b%Adam/Adam/update_40/ResourceApplyAdamhuZU?B
?
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*?2?8??@??H??b>gradient_tape/weighted_binary_crossentropy/logistic_loss/mul_1huZU?B
q
(GreaterEqual_GPU_DT_FLOAT_DT_BOOL_kernel*?2?8??@??H??b$model/dropout_4/dropout/GreaterEqualhuZU?B
?
?void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 4, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorBroadcastingOp<Eigen::array<int, 4ul> const, Eigen::TensorMap<Eigen::Tensor<float const, 4, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 4, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorBroadcastingOp<Eigen::array<int, 4ul> const, Eigen::TensorMap<Eigen::Tensor<float const, 4, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int) *?298??@??H??b1gradient_tape/weighted_binary_crossentropy/Tile_1huZU?B
?
?void cudnn::ops::pooling_fw_4d_kernel<float, float, cudnn::maxpooling_func<float, (cudnnNanPropagation_t)0>, (cudnnPoolingMode_t)0, false>(cudnnTensorStruct, float const*, cudnnTensorStruct, float*, cudnnPoolingStruct, float, float, int, cudnn::reduced_divisor, cudnn::reduced_divisor)( ?*2228??@??H??bmodel/max_pooling2d_4/MaxPoolhu?O?B
q
(GreaterEqual_GPU_DT_FLOAT_DT_BOOL_kernel*?2?8??@??H??b$model/dropout_1/dropout/GreaterEqualhuZU?B
?
?void tensorflow::functor::FillPhiloxRandomKernelLaunch<tensorflow::random::UniformDistribution<tensorflow::random::PhiloxRandom, float> >(unsigned long const*, unsigned long const*, tensorflow::random::PhiloxRandom, tensorflow::random::UniformDistribution<tensorflow::random::PhiloxRandom, float>::ResultElementType*, long, tensorflow::random::UniformDistribution<tensorflow::random::PhiloxRandom, float>)*?298??@??H??b4model/dropout_1/dropout/random_uniform/RandomUniformhuZU?B
?
?void cudnn::winograd_nonfused::winogradWgradOutput4x4<float, float>(cudnn::winograd_nonfused::WinogradWgradOutputParams<float, float>)=?H* 28??@??H??Xb9gradient_tape/model/conv2d_16/Conv2D/Conv2DBackpropFilterhuZU?B
?
(GreaterEqual_GPU_DT_FLOAT_DT_BOOL_kernel*?2?8??@??H??b7weighted_binary_crossentropy/logistic_loss/GreaterEqualhuZU?B
?
?void cudnn::winograd_nonfused::winogradWgradOutput4x4<float, float>(cudnn::winograd_nonfused::WinogradWgradOutputParams<float, float>)=?H* 2 8??@??H??Xb8gradient_tape/model/conv2d_6/Conv2D/Conv2DBackpropFilterhuZU?B
?
?void cudnn::winograd_nonfused::winogradForwardFilter4x4<float, float>(cudnn::winograd_nonfused::WinogradFilterParams<float, float>)(?H* 2 8??@??H??Xb7gradient_tape/model/conv2d_7/Conv2D/Conv2DBackpropInputhu  ?B
w
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*?2?8??@??H??b2gradient_tape/weighted_binary_crossentropy/truedivhuZU?B
?
?void cudnn::bn_fw_tr_1C11_singleread<float, 512, true, 1, 2, 0>(cudnnTensorStruct, float const*, cudnnTensorStruct, float*, float const*, float const*, float, float, float*, float*, float*, float*, float, float, cudnn::reduced_divisor, int, cudnn::reduced_divisor, cudnn::bnFwPersistentState*, int, float, float, float, int, float, float, cudnnStatus_t*, bool)'? ??*?2?8??@??H??b,model/batch_normalization_3/FusedBatchNormV3hu  ?B
?
?void cudnn::winograd_nonfused::winogradForwardFilter4x4<float, float>(cudnn::winograd_nonfused::WinogradFilterParams<float, float>)(?H* 2 8??@??H??bmodel/conv2d_15/Reluhu  ?B
?
?void cudnn::winograd_nonfused::winogradForwardFilter4x4<float, float>(cudnn::winograd_nonfused::WinogradFilterParams<float, float>)(?H* 2 8??@??H??Xb8gradient_tape/model/conv2d_15/Conv2D/Conv2DBackpropInputhu  ?B
?
?void cudnn::winograd_nonfused::winogradForwardFilter4x4<float, float>(cudnn::winograd_nonfused::WinogradFilterParams<float, float>)(?H* 2 8??@??H??bmodel/conv2d_7/Reluhu  ?B
?
?void tensorflow::functor::RowReduceKernel<float const*, float*, cub::Sum>(float const*, float*, int, int, cub::Sum, std::iterator_traits<float const*>::value_type)*?2?28??@??H??b8gradient_tape/model/conv2d_transpose/BiasAdd/BiasAddGradhu  ?B
?
?void tensorflow::functor::FillPhiloxRandomKernelLaunch<tensorflow::random::UniformDistribution<tensorflow::random::PhiloxRandom, float> >(unsigned long const*, unsigned long const*, tensorflow::random::PhiloxRandom, tensorflow::random::UniformDistribution<tensorflow::random::PhiloxRandom, float>::ResultElementType*, long, tensorflow::random::UniformDistribution<tensorflow::random::PhiloxRandom, float>)*?298??@??H??b4model/dropout_4/dropout/random_uniform/RandomUniformhuZU?B
?
?void tensorflow::functor::RowReduceKernel<float const*, float*, cub::Sum>(float const*, float*, int, int, cub::Sum, std::iterator_traits<float const*>::value_type)*?2?28??@??H??b1gradient_tape/model/conv2d_13/BiasAdd/BiasAddGradhu  ?B
?
?void tensorflow::functor::RowReduceKernel<float const*, float*, cub::Sum>(float const*, float*, int, int, cub::Sum, std::iterator_traits<float const*>::value_type)*?2?28??@??H??b0gradient_tape/model/conv2d_9/BiasAdd/BiasAddGradhu  ?B
?
?void tensorflow::functor::RowReduceKernel<float const*, float*, cub::Sum>(float const*, float*, int, int, cub::Sum, std::iterator_traits<float const*>::value_type)*?2?28??@??H??b0gradient_tape/model/conv2d_8/BiasAdd/BiasAddGradhu  ?B
?
?void cudnn::ops::nchwToNhwcKernel<float, float, float, false, true, (cudnnKernelDataType_t)2>(cudnn::ops::nchw2nhwc_params_t<float>, float const*, float*)'?!*?2 28??@??H??b'model/conv2d_transpose/conv2d_transposehu  ?B
?
?void tensorflow::functor::RowReduceKernel<float const*, float*, cub::Sum>(float const*, float*, int, int, cub::Sum, std::iterator_traits<float const*>::value_type)*?2?28??@??H??b1gradient_tape/model/conv2d_12/BiasAdd/BiasAddGradhu  ?B
?
?void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseUnaryOp<Eigen::internal::scalar_inverse_op<float>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseUnaryOp<Eigen::internal::scalar_inverse_op<float>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int)*?298??@??H??bCgradient_tape/weighted_binary_crossentropy/logistic_loss/ReciprocalhuZU?B
s
 Neg_GPU_DT_FLOAT_DT_FLOAT_kernel*?2?	8??@??H??b.weighted_binary_crossentropy/logistic_loss/Neghu  ?B
?
?void cudnn::ops::nchwToNhwcKernel<float, float, float, false, true, (cudnnKernelDataType_t)2>(cudnn::ops::nchw2nhwc_params_t<float>, float const*, float*)'?!*?2 28??@??H??XbJgradient_tape/model/conv2d_transpose/conv2d_transpose/Conv2DBackpropFilterhu  ?B
?
?void cudnn::ops::nchwToNhwcKernel<float, float, float, false, true, (cudnnKernelDataType_t)2>(cudnn::ops::nchw2nhwc_params_t<float>, float const*, float*)'?!*?2 28??@??H??Xb9gradient_tape/model/conv2d_10/Conv2D/Conv2DBackpropFilterhu  ?B
w
"Log1p_GPU_DT_FLOAT_DT_FLOAT_kernel*?2?	8??@??H??b0weighted_binary_crossentropy/logistic_loss/Log1phu  ?B
?
?void tensorflow::functor::SwapDimension1And2InTensor3UsingTiles<unsigned int, 256, 32, 32, false>(unsigned int const*, tensorflow::functor::Dimension<3>, unsigned int*) ?!*?2?8??@??H??bCmodel/dropout_2/dropout/Mul_1-0-TransposeNHWCToNCHW-LayoutOptimizerhu  ?B
?
"AddV2_GPU_DT_FLOAT_DT_FLOAT_kernel*?2?8??@??H??b<gradient_tape/weighted_binary_crossentropy/logistic_loss/addhuZU?B
?
 Neg_GPU_DT_FLOAT_DT_FLOAT_kernel*?2?	8??@??H??b<gradient_tape/weighted_binary_crossentropy/logistic_loss/Neghu  ?B
`
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*?2?8??@??H??bmodel/dropout_2/dropout/MulhuZU?B
?
Yvoid tensorflow::BiasNCHWKernel<float>(int, float const*, float const*, float*, int, int)*?2&8??@??H??bmodel/conv2d_22/BiasAddhuZU?B
s
 Exp_GPU_DT_FLOAT_DT_FLOAT_kernel*?2?	8??@??H??b.weighted_binary_crossentropy/logistic_loss/Exphu  ?B
?
?void tensorflow::functor::SwapDimension1And2InTensor3UsingTiles<unsigned char, 256, 32, 32, false>(unsigned char const*, tensorflow::functor::Dimension<3>, unsigned char*) ?*?2?8??@??H??b?gradient_tape/model/dropout_4/dropout/ArithmeticOptimizer/ReorderCastLikeAndValuePreserving_bool_Mul_2-1-TransposeNHWCToNCHW-LayoutOptimizerhu  ?B
n
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*?2?8??@??H??b)gradient_tape/model/dropout_2/dropout/MulhuZU?B
?
?void nchwToFoldedNhwcKernel<float, float, float, true, (cudnnKernelDataType_t)2>(int, int, int, int, float const*, float*, int, int, int, int, int, int, int, int, int, int, float, float, cudnn::reduced_divisor, cudnn::reduced_divisor, cudnn::reduced_divisor)&?!* 2?8??@??H??b)model/conv2d_transpose_1/conv2d_transposehu  ?B
?
?void tensorflow::functor::ApplyAdamKernel<float>(int, float*, float*, float*, float const*, float const*, float const*, float const*, float const*, float const*, float const*, bool) *?2&8??@??H??b%Adam/Adam/update_18/ResourceApplyAdamhuZU?B
?
?void tensorflow::functor::ApplyAdamKernel<float>(int, float*, float*, float*, float const*, float const*, float const*, float const*, float const*, float const*, float const*, bool) *?2&8??@??H??b%Adam/Adam/update_52/ResourceApplyAdamhuZU?B
?
?void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<float, Eigen::TensorMap<Eigen::Tensor<bool const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<float, Eigen::TensorMap<Eigen::Tensor<bool const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*?298??@??H??bmodel/dropout_2/dropout/CasthuZU?B
?
?void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<float, Eigen::TensorMap<Eigen::Tensor<bool const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<float, Eigen::TensorMap<Eigen::Tensor<bool const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*?298??@??H??bXmodel/dropout_2/dropout/ArithmeticOptimizer/ReorderCastLikeAndValuePreserving_float_CasthuZU?B
?
?void cudnn::ops::nhwcToNchwKernel<float, float, float, true, false, (cudnnKernelDataType_t)0>(cudnn::ops::nhwc2nchw_params_t<float>, float const*, float*)&?!*?2?8??@??H??XbLgradient_tape/model/conv2d_transpose_1/conv2d_transpose/Conv2DBackpropFilterhu  ?B
?
?void tensorflow::functor::SwapDimension1And2InTensor3UsingTiles<unsigned char, 256, 32, 32, false>(unsigned char const*, tensorflow::functor::Dimension<3>, unsigned char*) ?*?2?8??@??H??b?gradient_tape/model/dropout_1/dropout/ArithmeticOptimizer/ReorderCastLikeAndValuePreserving_bool_Mul_2-1-TransposeNHWCToNCHW-LayoutOptimizerhu  ?B
?
?void cudnn::ops::nchwToNhwcKernel<float, float, float, false, true, (cudnnKernelDataType_t)2>(cudnn::ops::nchw2nhwc_params_t<float>, float const*, float*)'?!*?2?8??@??H??Xb>gradient_tape/model/conv2d_transpose_1/conv2d_transpose/Conv2Dhu  ?B
?
?void tensorflow::functor::ShuffleInTensor3Simple<float, 2, 1, 0, false>(int, float const*, tensorflow::functor::Dimension<3>, float*)*?2&8??@??H??Xb9gradient_tape/model/conv2d_15/Conv2D/Conv2DBackpropFilterhuZU?B
?
?void tensorflow::functor::ShuffleInTensor3Simple<float, 2, 1, 0, false>(int, float const*, tensorflow::functor::Dimension<3>, float*)*?2&8??@??H??Xb8gradient_tape/model/conv2d_7/Conv2D/Conv2DBackpropFilterhuZU?B
?
?void tensorflow::functor::ShuffleInTensor3Simple<float, 2, 1, 0, false>(int, float const*, tensorflow::functor::Dimension<3>, float*)*?2&8??@??H??bmodel/conv2d_15/ReluhuZU?B
?
?void tensorflow::functor::RowReduceKernel<float const*, float*, cub::Sum>(float const*, float*, int, int, cub::Sum, std::iterator_traits<float const*>::value_type)*?2?d8??@??H??b1gradient_tape/model/conv2d_10/BiasAdd/BiasAddGradhu  ?B
?
?void tensorflow::functor::RowReduceKernel<float const*, float*, cub::Sum>(float const*, float*, int, int, cub::Sum, std::iterator_traits<float const*>::value_type)*?2?d8??@??H??b1gradient_tape/model/conv2d_11/BiasAdd/BiasAddGradhu  ?B
?
?void tensorflow::functor::ShuffleInTensor3Simple<float, 2, 1, 0, false>(int, float const*, tensorflow::functor::Dimension<3>, float*)*?2&8??@??H??Xb8gradient_tape/model/conv2d_15/Conv2D/Conv2DBackpropInputhuZU?B
?
?void tensorflow::functor::ShuffleInTensor3Simple<float, 2, 1, 0, false>(int, float const*, tensorflow::functor::Dimension<3>, float*)*?2&8??@??H??Xb7gradient_tape/model/conv2d_7/Conv2D/Conv2DBackpropInputhuZU?B
?
?void tensorflow::functor::ShuffleInTensor3Simple<float, 2, 1, 0, false>(int, float const*, tensorflow::functor::Dimension<3>, float*)*?2&8??@??H??bmodel/conv2d_7/ReluhuZU?B
p
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*?2?8??@??H??b+gradient_tape/model/dropout_3/dropout/Mul_2huZU?B
b
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*?2?8??@??H??bmodel/dropout_3/dropout/Mul_1huZU?B
?
?void tensorflow::functor::ShuffleInTensor3Simple<float, 2, 1, 0, false>(int, float const*, tensorflow::functor::Dimension<3>, float*)*?2&8??@??H??Xb>gradient_tape/model/conv2d_transpose_1/conv2d_transpose/Conv2DhuZU?B
?
?void tensorflow::functor::ShuffleInTensor3Simple<float, 2, 1, 0, false>(int, float const*, tensorflow::functor::Dimension<3>, float*)*?2&8??@??H??b)model/conv2d_transpose_1/conv2d_transposehuZU?B
?
?void cub::DeviceReduceKernel<cub::DeviceReducePolicy<float, float, int, tensorflow::functor::Sum<float> >::Policy600, float*, float*, int, tensorflow::functor::Sum<float> >(float*, float*, int, cub::GridEvenShare<int>, tensorflow::functor::Sum<float>)0*?2?8??@??H??b!weighted_binary_crossentropy/Meanhu  ?B
q
(GreaterEqual_GPU_DT_FLOAT_DT_BOOL_kernel*?2?8??@??H??b$model/dropout_2/dropout/GreaterEqualhuZU?B
?
?void cudnn::winograd_nonfused::winogradForwardFilter4x4<float, float>(cudnn::winograd_nonfused::WinogradFilterParams<float, float>)(?H* 28??@??H??Xb8gradient_tape/model/conv2d_16/Conv2D/Conv2DBackpropInputhu  ?B
?
?void cudnn::winograd_nonfused::winogradWgradOutput4x4<float, float>(cudnn::winograd_nonfused::WinogradWgradOutputParams<float, float>)=?H* 28??@??H??Xb9gradient_tape/model/conv2d_17/Conv2D/Conv2DBackpropFilterhuZU?B
?
?void cudnn::winograd_nonfused::winogradForwardFilter4x4<float, float>(cudnn::winograd_nonfused::WinogradFilterParams<float, float>)(?H* 2 8??@??H??Xb7gradient_tape/model/conv2d_6/Conv2D/Conv2DBackpropInputhu  ?B
?
?void cudnn::winograd_nonfused::winogradWgradOutput4x4<float, float>(cudnn::winograd_nonfused::WinogradWgradOutputParams<float, float>)=?H* 28??@??H??Xb8gradient_tape/model/conv2d_5/Conv2D/Conv2DBackpropFilterhuZU?B
?
?void tensorflow::functor::FillPhiloxRandomKernelLaunch<tensorflow::random::UniformDistribution<tensorflow::random::PhiloxRandom, float> >(unsigned long const*, unsigned long const*, tensorflow::random::PhiloxRandom, tensorflow::random::UniformDistribution<tensorflow::random::PhiloxRandom, float>::ResultElementType*, long, tensorflow::random::UniformDistribution<tensorflow::random::PhiloxRandom, float>)*?298??@??H??b4model/dropout_2/dropout/random_uniform/RandomUniformhuZU?B
?
?void cudnn::winograd_nonfused::winogradForwardFilter4x4<float, float>(cudnn::winograd_nonfused::WinogradFilterParams<float, float>)(?H* 2 8?|@?|H?|bmodel/conv2d_6/Reluhu  ?B
?
?void cudnn::ops::nchwToNhwcKernel<float, float, float, false, true, (cudnnKernelDataType_t)2>(cudnn::ops::nchw2nhwc_params_t<float>, float const*, float*)'?!*?228?z@?zH?zXb9gradient_tape/model/conv2d_10/Conv2D/Conv2DBackpropFilterhu  ?B
?
?void tensorflow::functor::SwapDimension1And2InTensor3UsingTiles<unsigned int, 256, 32, 32, false>(unsigned int const*, tensorflow::functor::Dimension<3>, unsigned int*) ?!*?2?8?x@?xH?xbCmodel/dropout_3/dropout/Mul_1-0-TransposeNHWCToNCHW-LayoutOptimizerhu  ?B
]
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*?2?8?u@?uH?ubmodel/dropout_3/dropout/MulhuZU?B
k
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*?2?8?u@?uH?ub)gradient_tape/model/dropout_3/dropout/MulhuZU?B
?
?void cub::DeviceSegmentedReduceKernel<cub::DeviceReducePolicy<float, float, int, cub::Sum>::Policy600, float const*, float*, cub::TransformInputIterator<int, tensorflow::functor::RowOffset, cub::CountingInputIterator<int, long>, long>, int, cub::Sum, float>(float const*, float*, cub::TransformInputIterator<int, tensorflow::functor::RowOffset, cub::CountingInputIterator<int, long>, long>, cub::TransformInputIterator<int, tensorflow::functor::RowOffset, cub::CountingInputIterator<int, long>, long>, int, cub::Sum, float) 0*?228?t@?tH?tb1gradient_tape/model/conv2d_22/BiasAdd/BiasAddGradhu  ?B
?
?void tensorflow::functor::ShuffleInTensor3Simple<float, 2, 1, 0, false>(int, float const*, tensorflow::functor::Dimension<3>, float*)*?2&8?p@?pH?pXbLgradient_tape/model/conv2d_transpose_1/conv2d_transpose/Conv2DBackpropFilterhuZU?B
?
?void tensorflow::functor::SwapDimension1And2InTensor3UsingTiles<unsigned char, 256, 32, 32, false>(unsigned char const*, tensorflow::functor::Dimension<3>, unsigned char*) ?*?2?8?o@?oH?ob?gradient_tape/model/dropout_2/dropout/ArithmeticOptimizer/ReorderCastLikeAndValuePreserving_bool_Mul_2-1-TransposeNHWCToNCHW-LayoutOptimizerhu  ?B
?
?void tensorflow::functor::ApplyAdamKernel<float>(int, float*, float*, float*, float const*, float const*, float const*, float const*, float const*, float const*, float const*, bool) *?2&8?m@?mH?mb%Adam/Adam/update_14/ResourceApplyAdamhuZU?B
?
?void tensorflow::functor::ApplyAdamKernel<float>(int, float*, float*, float*, float const*, float const*, float const*, float const*, float const*, float const*, float const*, bool) *?2&8?j@?jH?jb%Adam/Adam/update_54/ResourceApplyAdamhuZU?B
?
?void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<float, Eigen::TensorMap<Eigen::Tensor<bool const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<float, Eigen::TensorMap<Eigen::Tensor<bool const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*?298?e@?eH?ebmodel/dropout_3/dropout/CasthuZU?B
?
?void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<float, Eigen::TensorMap<Eigen::Tensor<bool const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<float, Eigen::TensorMap<Eigen::Tensor<bool const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*?298?e@?eH?ebXmodel/dropout_3/dropout/ArithmeticOptimizer/ReorderCastLikeAndValuePreserving_float_CasthuZU?B
?
?void tensorflow::functor::ApplyAdamKernel<float>(int, float*, float*, float*, float const*, float const*, float const*, float const*, float const*, float const*, float const*, bool) *?2&8?c@?cH?cb%Adam/Adam/update_48/ResourceApplyAdamhuZU?B
?
?void tensorflow::functor::ShuffleInTensor3Simple<float, 2, 1, 0, false>(int, float const*, tensorflow::functor::Dimension<3>, float*)*?2&8?`@?`H?`Xb9gradient_tape/model/conv2d_16/Conv2D/Conv2DBackpropFilterhuZU?B
?
?void tensorflow::functor::ShuffleInTensor3Simple<float, 2, 1, 0, false>(int, float const*, tensorflow::functor::Dimension<3>, float*)*?2&8?`@?`H?`Xb8gradient_tape/model/conv2d_6/Conv2D/Conv2DBackpropFilterhuZU?B
?
?void tensorflow::functor::ShuffleInTensor3Simple<float, 2, 1, 0, false>(int, float const*, tensorflow::functor::Dimension<3>, float*)*?2&8?_@?_H?_bmodel/conv2d_16/ReluhuZU?B
?
?void tensorflow::functor::ShuffleInTensor3Simple<float, 2, 1, 0, false>(int, float const*, tensorflow::functor::Dimension<3>, float*)*?2&8?^@?^H?^bmodel/conv2d_6/ReluhuZU?B
?
?void tensorflow::functor::ShuffleInTensor3Simple<float, 2, 1, 0, false>(int, float const*, tensorflow::functor::Dimension<3>, float*)*?2&8?]@?]H?]Xb7gradient_tape/model/conv2d_6/Conv2D/Conv2DBackpropInputhuZU?B
?
?void tensorflow::functor::ShuffleInTensor3Simple<float, 2, 1, 0, false>(int, float const*, tensorflow::functor::Dimension<3>, float*)*?2&8?]@?]H?]Xb8gradient_tape/model/conv2d_16/Conv2D/Conv2DBackpropInputhuZU?B
?
?void tensorflow::functor::FillPhiloxRandomKernelLaunch<tensorflow::random::UniformDistribution<tensorflow::random::PhiloxRandom, float> >(unsigned long const*, unsigned long const*, tensorflow::random::PhiloxRandom, tensorflow::random::UniformDistribution<tensorflow::random::PhiloxRandom, float>::ResultElementType*, long, tensorflow::random::UniformDistribution<tensorflow::random::PhiloxRandom, float>)*?298?O@?OH?Ob4model/dropout_3/dropout/random_uniform/RandomUniformhuZU?B
n
(GreaterEqual_GPU_DT_FLOAT_DT_BOOL_kernel*?2?8?O@?OH?Ob$model/dropout_3/dropout/GreaterEqualhuZU?B
?
?void cudnn::winograd_nonfused::winogradWgradOutput4x4<float, float>(cudnn::winograd_nonfused::WinogradWgradOutputParams<float, float>)=?H* 28?M@?MH?MXb9gradient_tape/model/conv2d_18/Conv2D/Conv2DBackpropFilterhuZU?B
?
?void cudnn::winograd_nonfused::winogradForwardFilter4x4<float, float>(cudnn::winograd_nonfused::WinogradFilterParams<float, float>)(?H* 28?K@?KH?KXb8gradient_tape/model/conv2d_17/Conv2D/Conv2DBackpropInputhu  ?B
?
?void cudnn::winograd_nonfused::winogradForwardFilter4x4<float, float>(cudnn::winograd_nonfused::WinogradFilterParams<float, float>)(?H* 28?K@?KH?Kbmodel/conv2d_5/Reluhu  ?B
?
?void cudnn::winograd_nonfused::winogradWgradOutput4x4<float, float>(cudnn::winograd_nonfused::WinogradWgradOutputParams<float, float>)=?H* 28?J@?JH?JXb8gradient_tape/model/conv2d_4/Conv2D/Conv2DBackpropFilterhuZU?B
?
?void cudnn::winograd_nonfused::winogradForwardFilter4x4<float, float>(cudnn::winograd_nonfused::WinogradFilterParams<float, float>)(?H* 28?H@?HH?HXb7gradient_tape/model/conv2d_5/Conv2D/Conv2DBackpropInputhu  ?B
?
?void cudnn::winograd_nonfused::winogradForwardFilter4x4<float, float>(cudnn::winograd_nonfused::WinogradFilterParams<float, float>)(?H* 28?G@?GH?Gbmodel/conv2d_17/Reluhu  ?B
?
?void tensorflow::functor::SwapDimension1And2InTensor3UsingTiles<unsigned char, 256, 32, 32, false>(unsigned char const*, tensorflow::functor::Dimension<3>, unsigned char*) ?*?2?8?E@?EH?Eb?gradient_tape/model/dropout_3/dropout/ArithmeticOptimizer/ReorderCastLikeAndValuePreserving_bool_Mul_2-1-TransposeNHWCToNCHW-LayoutOptimizerhu  ?B
?
?void tensorflow::functor::ApplyAdamKernel<float>(int, float*, float*, float*, float const*, float const*, float const*, float const*, float const*, float const*, float const*, bool) *?2&8?@@?@H?@b%Adam/Adam/update_60/ResourceApplyAdamhuZU?B
?
?void tensorflow::functor::ShuffleInTensor3Simple<float, 2, 1, 0, false>(int, float const*, tensorflow::functor::Dimension<3>, float*)*?2&8?=@?=H?=Xb9gradient_tape/model/conv2d_17/Conv2D/Conv2DBackpropFilterhuZU?B
?
?void tensorflow::functor::ApplyAdamKernel<float>(int, float*, float*, float*, float const*, float const*, float const*, float const*, float const*, float const*, float const*, bool) *?2&8?<@?<H?<b%Adam/Adam/update_12/ResourceApplyAdamhuZU?B
?
?void tensorflow::functor::ShuffleInTensor3Simple<float, 2, 1, 0, false>(int, float const*, tensorflow::functor::Dimension<3>, float*)*?2&8?<@?<H?<Xb8gradient_tape/model/conv2d_5/Conv2D/Conv2DBackpropFilterhuZU?B
?
?void tensorflow::functor::ShuffleInTensor3Simple<float, 2, 1, 0, false>(int, float const*, tensorflow::functor::Dimension<3>, float*)*?2&8?:@?:H?:bmodel/conv2d_17/ReluhuZU?B
?
?void tensorflow::functor::ShuffleInTensor3Simple<float, 2, 1, 0, false>(int, float const*, tensorflow::functor::Dimension<3>, float*)*?2&8?9@?9H?9Xb8gradient_tape/model/conv2d_17/Conv2D/Conv2DBackpropInputhuZU?B
?
?void tensorflow::functor::ShuffleInTensor3Simple<float, 2, 1, 0, false>(int, float const*, tensorflow::functor::Dimension<3>, float*)*?2&8?8@?8H?8bmodel/conv2d_5/ReluhuZU?B
?
?void tensorflow::functor::ShuffleInTensor3Simple<float, 2, 1, 0, false>(int, float const*, tensorflow::functor::Dimension<3>, float*)*?2&8?7@?7H?7Xb7gradient_tape/model/conv2d_5/Conv2D/Conv2DBackpropInputhuZU?B
?
?void tensorflow::functor::ShuffleInTensor3Simple<float, 2, 1, 0, false>(int, float const*, tensorflow::functor::Dimension<3>, float*)*?2&8?0@?0H?0b)model/conv2d_transpose_2/conv2d_transposehuZU?B
?
?void tensorflow::functor::ApplyAdamKernel<float>(int, float*, float*, float*, float const*, float const*, float const*, float const*, float const*, float const*, float const*, bool) *?28?.@?.H?.b$Adam/Adam/update_9/ResourceApplyAdamhuZU?B
?
?void tensorflow::functor::ApplyAdamKernel<float>(int, float*, float*, float*, float const*, float const*, float const*, float const*, float const*, float const*, float const*, bool) *?2$8?.@?.H?.b$Adam/Adam/update_8/ResourceApplyAdamhuZU?B
?
?void tensorflow::functor::ApplyAdamKernel<float>(int, float*, float*, float*, float const*, float const*, float const*, float const*, float const*, float const*, float const*, bool) *?2$8?.@?.H?.b%Adam/Adam/update_62/ResourceApplyAdamhuZU?B
?
?void tensorflow::functor::ShuffleInTensor3Simple<float, 2, 1, 0, false>(int, float const*, tensorflow::functor::Dimension<3>, float*)*?2&8?-@?-H?-bmodel/conv2d_18/ReluhuZU?B
?
?void tensorflow::functor::ApplyAdamKernel<float>(int, float*, float*, float*, float const*, float const*, float const*, float const*, float const*, float const*, float const*, bool) *?2 8?,@?,H?,b%Adam/Adam/update_56/ResourceApplyAdamhuZU?B
?
?void tensorflow::functor::ApplyAdamKernel<float>(int, float*, float*, float*, float const*, float const*, float const*, float const*, float const*, float const*, float const*, bool) *?28?*@?*H?*b%Adam/Adam/update_31/ResourceApplyAdamhuZU?B
?
?void tensorflow::functor::ShuffleInTensor3Simple<float, 2, 1, 0, false>(int, float const*, tensorflow::functor::Dimension<3>, float*)*?2&8?)@?)H?)Xb>gradient_tape/model/conv2d_transpose_2/conv2d_transpose/Conv2DhuZU?B
?
?void tensorflow::functor::ShuffleInTensor3Simple<float, 2, 1, 0, false>(int, float const*, tensorflow::functor::Dimension<3>, float*)*?2&8?(@?(H?(Xb8gradient_tape/model/conv2d_18/Conv2D/Conv2DBackpropInputhuZU?B
?
?void tensorflow::functor::ShuffleInTensor3Simple<float, 2, 1, 0, false>(int, float const*, tensorflow::functor::Dimension<3>, float*)*?2&8?(@?(H?(XbLgradient_tape/model/conv2d_transpose_2/conv2d_transpose/Conv2DBackpropFilterhuZU?B
?
?void tensorflow::functor::ShuffleInTensor3Simple<float, 2, 1, 0, false>(int, float const*, tensorflow::functor::Dimension<3>, float*)*?2&8?'@?'H?'Xb9gradient_tape/model/conv2d_18/Conv2D/Conv2DBackpropFilterhuZU?B
?
?void tensorflow::functor::ShuffleInTensor3Simple<float, 2, 1, 0, false>(int, float const*, tensorflow::functor::Dimension<3>, float*)*?2&8?'@?'H?'Xb8gradient_tape/model/conv2d_4/Conv2D/Conv2DBackpropFilterhuZU?B
?
Scask_cudnn::computeWgradSplitKOffsetsKernel(cask_cudnn::ComputeSplitKOffsetsParams)*?2P8?&@?&H?&Xb9gradient_tape/model/conv2d_19/Conv2D/Conv2DBackpropFilterhu  ?B
?
?void tensorflow::functor::ApplyAdamKernel<float>(int, float*, float*, float*, float const*, float const*, float const*, float const*, float const*, float const*, float const*, bool) *?28?&@?&H?&b$Adam/Adam/update_6/ResourceApplyAdamhuZU?B
?
?void tensorflow::functor::ApplyAdamKernel<float>(int, float*, float*, float*, float const*, float const*, float const*, float const*, float const*, float const*, float const*, bool) *?2	8?%@?%H?%b$Adam/Adam/update_2/ResourceApplyAdamhuZU?B
?
?void tensorflow::functor::ShuffleInTensor3Simple<float, 2, 1, 0, false>(int, float const*, tensorflow::functor::Dimension<3>, float*)*?2&8?%@?%H?%bmodel/conv2d_4/ReluhuZU?B
?
?void tensorflow::functor::ApplyAdamKernel<float>(int, float*, float*, float*, float const*, float const*, float const*, float const*, float const*, float const*, float const*, bool) *?28?#@?#H?#b%Adam/Adam/update_64/ResourceApplyAdamhuZU?B
?
?void tensorflow::functor::ShuffleInTensor3Simple<float, 2, 1, 0, false>(int, float const*, tensorflow::functor::Dimension<3>, float*)*?2&8?#@?#H?#Xb7gradient_tape/model/conv2d_4/Conv2D/Conv2DBackpropInputhuZU?B
?
?void cudnn::ops::nchwToNhwcKernel<float, float, float, false, true, (cudnnKernelDataType_t)2>(cudnn::ops::nchw2nhwc_params_t<float>, float const*, float*)'?!*?2?8?#@?#H?#bmodel/conv2d_4/Reluhu  ?B
?
?void tensorflow::functor::ApplyAdamKernel<float>(int, float*, float*, float*, float const*, float const*, float const*, float const*, float const*, float const*, float const*, bool) *?28?"@?"H?"b%Adam/Adam/update_25/ResourceApplyAdamhuZU?B
?
?void tensorflow::functor::ApplyAdamKernel<float>(int, float*, float*, float*, float const*, float const*, float const*, float const*, float const*, float const*, float const*, bool) *?28?"@?"H?"b%Adam/Adam/update_66/ResourceApplyAdamhuZU?B
?
?void tensorflow::functor::ApplyAdamKernel<float>(int, float*, float*, float*, float const*, float const*, float const*, float const*, float const*, float const*, float const*, bool) *?28?!@?!H?!b%Adam/Adam/update_34/ResourceApplyAdamhuZU?B
?
?void tensorflow::functor::ApplyAdamKernel<float>(int, float*, float*, float*, float const*, float const*, float const*, float const*, float const*, float const*, float const*, bool) *?28?!@?!H?!b"Adam/Adam/update/ResourceApplyAdamhuZU?B
?
?void tensorflow::functor::ApplyAdamKernel<float>(int, float*, float*, float*, float const*, float const*, float const*, float const*, float const*, float const*, float const*, bool) *?28?!@?!H?!b%Adam/Adam/update_10/ResourceApplyAdamhuZU?B
?
?void tensorflow::functor::ApplyAdamKernel<float>(int, float*, float*, float*, float const*, float const*, float const*, float const*, float const*, float const*, float const*, bool) *?28?!@?!H?!b%Adam/Adam/update_29/ResourceApplyAdamhuZU?B
?
?void tensorflow::functor::ApplyAdamKernel<float>(int, float*, float*, float*, float const*, float const*, float const*, float const*, float const*, float const*, float const*, bool) *?28?!@?!H?!b%Adam/Adam/update_41/ResourceApplyAdamhuZU?B
?
?void tensorflow::functor::ApplyAdamKernel<float>(int, float*, float*, float*, float const*, float const*, float const*, float const*, float const*, float const*, float const*, bool) *?28?!@?!H?!b%Adam/Adam/update_42/ResourceApplyAdamhuZU?B
?
?void tensorflow::functor::ApplyAdamKernel<float>(int, float*, float*, float*, float const*, float const*, float const*, float const*, float const*, float const*, float const*, bool) *?28?!@?!H?!b%Adam/Adam/update_37/ResourceApplyAdamhuZU?B
?
?void tensorflow::functor::ApplyAdamKernel<float>(int, float*, float*, float*, float const*, float const*, float const*, float const*, float const*, float const*, float const*, bool) *?28? @? H? b%Adam/Adam/update_22/ResourceApplyAdamhuZU?B
?
?void tensorflow::functor::ApplyAdamKernel<float>(int, float*, float*, float*, float const*, float const*, float const*, float const*, float const*, float const*, float const*, bool) *?28? @? H? b%Adam/Adam/update_67/ResourceApplyAdamhuZU?B
?
?void nchwToFoldedNhwcKernel<float, float, float, true, (cudnnKernelDataType_t)2>(int, int, int, int, float const*, float*, int, int, int, int, int, int, int, int, int, int, float, float, cudnn::reduced_divisor, cudnn::reduced_divisor, cudnn::reduced_divisor)&?!* 2?8? @? H? b)model/conv2d_transpose_3/conv2d_transposehu  ?B
?
?void tensorflow::functor::ApplyAdamKernel<float>(int, float*, float*, float*, float const*, float const*, float const*, float const*, float const*, float const*, float const*, bool) *?28? @? H? b%Adam/Adam/update_33/ResourceApplyAdamhuZU?B
?
?void tensorflow::functor::ApplyAdamKernel<float>(int, float*, float*, float*, float const*, float const*, float const*, float const*, float const*, float const*, float const*, bool) *?28? @? H? b%Adam/Adam/update_47/ResourceApplyAdamhuZU?B
?
?void tensorflow::functor::ApplyAdamKernel<float>(int, float*, float*, float*, float const*, float const*, float const*, float const*, float const*, float const*, float const*, bool) *?28? @? H? b%Adam/Adam/update_69/ResourceApplyAdamhuZU?B
?
?void tensorflow::functor::ApplyAdamKernel<float>(int, float*, float*, float*, float const*, float const*, float const*, float const*, float const*, float const*, float const*, bool) *?28? @? H? b%Adam/Adam/update_71/ResourceApplyAdamhuZU?B
?
?void tensorflow::functor::ApplyAdamKernel<float>(int, float*, float*, float*, float const*, float const*, float const*, float const*, float const*, float const*, float const*, bool) *?28? @? H? b%Adam/Adam/update_21/ResourceApplyAdamhuZU?B
?
?void tensorflow::functor::ApplyAdamKernel<float>(int, float*, float*, float*, float const*, float const*, float const*, float const*, float const*, float const*, float const*, bool) *?28? @? H? b%Adam/Adam/update_27/ResourceApplyAdamhuZU?B
?
?void tensorflow::functor::ApplyAdamKernel<float>(int, float*, float*, float*, float const*, float const*, float const*, float const*, float const*, float const*, float const*, bool) *?28? @? H? b%Adam/Adam/update_39/ResourceApplyAdamhuZU?B
?
?void tensorflow::functor::ApplyAdamKernel<float>(int, float*, float*, float*, float const*, float const*, float const*, float const*, float const*, float const*, float const*, bool) *?28? @? H? b%Adam/Adam/update_50/ResourceApplyAdamhuZU?B
?
?void tensorflow::functor::ShuffleInTensor3Simple<float, 2, 1, 0, false>(int, float const*, tensorflow::functor::Dimension<3>, float*)*?2$8? @? H? Xb8gradient_tape/model/conv2d_3/Conv2D/Conv2DBackpropFilterhuZU?B
?
?void tensorflow::functor::ApplyAdamKernel<float>(int, float*, float*, float*, float const*, float const*, float const*, float const*, float const*, float const*, float const*, bool) *?28?@?H?b%Adam/Adam/update_45/ResourceApplyAdamhuZU?B
?
?void tensorflow::functor::ApplyAdamKernel<float>(int, float*, float*, float*, float const*, float const*, float const*, float const*, float const*, float const*, float const*, bool) *?28?@?H?b%Adam/Adam/update_53/ResourceApplyAdamhuZU?B
?
?void tensorflow::functor::ShuffleInTensor3Simple<float, 2, 1, 0, false>(int, float const*, tensorflow::functor::Dimension<3>, float*)*?2$8?@?H?Xb9gradient_tape/model/conv2d_19/Conv2D/Conv2DBackpropFilterhuZU?B
?
?void tensorflow::functor::ApplyAdamKernel<float>(int, float*, float*, float*, float const*, float const*, float const*, float const*, float const*, float const*, float const*, bool) *?28?@?H?b%Adam/Adam/update_19/ResourceApplyAdamhuZU?B
?
?void tensorflow::functor::ApplyAdamKernel<float>(int, float*, float*, float*, float const*, float const*, float const*, float const*, float const*, float const*, float const*, bool) *?28?@?H?b%Adam/Adam/update_58/ResourceApplyAdamhuZU?B
?
?void tensorflow::functor::ApplyAdamKernel<float>(int, float*, float*, float*, float const*, float const*, float const*, float const*, float const*, float const*, float const*, bool) *?2	8?@?H?b%Adam/Adam/update_68/ResourceApplyAdamhuZU?B
?
?void tensorflow::functor::ShuffleInTensor3Simple<float, 2, 1, 0, false>(int, float const*, tensorflow::functor::Dimension<3>, float*)*?28?@?H?bmodel/conv2d_2/ReluhuZU?B
?
?void tensorflow::functor::ShuffleInTensor3Simple<float, 2, 1, 0, false>(int, float const*, tensorflow::functor::Dimension<3>, float*)*?2$8?@?H?Xb7gradient_tape/model/conv2d_3/Conv2D/Conv2DBackpropInputhuZU?B
?
?void tensorflow::functor::ApplyAdamKernel<float>(int, float*, float*, float*, float const*, float const*, float const*, float const*, float const*, float const*, float const*, bool) *?28?@?H?b$Adam/Adam/update_4/ResourceApplyAdamhuZU?B
?
?void tensorflow::functor::ApplyAdamKernel<float>(int, float*, float*, float*, float const*, float const*, float const*, float const*, float const*, float const*, float const*, bool) *?28?@?H?b%Adam/Adam/update_16/ResourceApplyAdamhuZU?B
?
?void tensorflow::functor::ShuffleInTensor3Simple<float, 2, 1, 0, false>(int, float const*, tensorflow::functor::Dimension<3>, float*)*?2$8?@?H?bmodel/conv2d_19/ReluhuZU?B
?
?void tensorflow::functor::ShuffleInTensor3Simple<float, 2, 1, 0, false>(int, float const*, tensorflow::functor::Dimension<3>, float*)*?2$8?@?H?bmodel/conv2d_3/ReluhuZU?B
?
?void tensorflow::functor::ApplyAdamKernel<float>(int, float*, float*, float*, float const*, float const*, float const*, float const*, float const*, float const*, float const*, bool) *?28?@?H?b%Adam/Adam/update_35/ResourceApplyAdamhuZU?B
?
?void tensorflow::functor::ApplyAdamKernel<float>(int, float*, float*, float*, float const*, float const*, float const*, float const*, float const*, float const*, float const*, bool) *?28?@?H?b%Adam/Adam/update_43/ResourceApplyAdamhuZU?B
?
?void tensorflow::functor::ApplyAdamKernel<float>(int, float*, float*, float*, float const*, float const*, float const*, float const*, float const*, float const*, float const*, bool) *?28?@?H?b%Adam/Adam/update_49/ResourceApplyAdamhuZU?B
?
?void tensorflow::functor::ShuffleInTensor3Simple<float, 2, 1, 0, false>(int, float const*, tensorflow::functor::Dimension<3>, float*)*?28?@?H?bmodel/conv2d_20/ReluhuZU?B
?
?void tensorflow::functor::ShuffleInTensor3Simple<float, 2, 1, 0, false>(int, float const*, tensorflow::functor::Dimension<3>, float*)*?2$8?@?H?Xb8gradient_tape/model/conv2d_19/Conv2D/Conv2DBackpropInputhuZU?B
?
~void cudnn::winograd::generateWinogradTilesKernel<0, float, float>(cudnn::winograd::GenerateWinogradTilesParams<float, float>)(?D* 2 8?@?H?bmodel/conv2d_18/ReluhuZU?B
?
?void tensorflow::functor::ApplyAdamKernel<float>(int, float*, float*, float*, float const*, float const*, float const*, float const*, float const*, float const*, float const*, bool) *?28?@?H?b%Adam/Adam/update_55/ResourceApplyAdamhuZU?B
?
?void tensorflow::functor::ShuffleInTensor3Simple<float, 2, 1, 0, false>(int, float const*, tensorflow::functor::Dimension<3>, float*)*?28?@?H?b)model/conv2d_transpose_4/conv2d_transposehuZU?B
?
?void tensorflow::functor::ShuffleInTensor3Simple<float, 2, 1, 0, false>(int, float const*, tensorflow::functor::Dimension<3>, float*)*?28?@?H?Xb7gradient_tape/model/conv2d_2/Conv2D/Conv2DBackpropInputhuZU?B
?
?void tensorflow::functor::ShuffleInTensor3Simple<float, 2, 1, 0, false>(int, float const*, tensorflow::functor::Dimension<3>, float*)*?2 8?@?H?b)model/conv2d_transpose_3/conv2d_transposehuZU?B
?
?void tensorflow::functor::ShuffleInTensor3Simple<float, 2, 1, 0, false>(int, float const*, tensorflow::functor::Dimension<3>, float*)*?2	8?@?H?Xb7gradient_tape/model/conv2d_1/Conv2D/Conv2DBackpropInputhuZU?B
?
?void tensorflow::functor::ShuffleInTensor3Simple<float, 2, 1, 0, false>(int, float const*, tensorflow::functor::Dimension<3>, float*)*?2	8?@?H?bmodel/conv2d_21/ReluhuZU?B
?
?void tensorflow::functor::ShuffleInTensor3Simple<float, 2, 1, 0, false>(int, float const*, tensorflow::functor::Dimension<3>, float*)*?2	8?@?H?Xb8gradient_tape/model/conv2d_21/Conv2D/Conv2DBackpropInputhuZU?B
?
?void tensorflow::functor::ShuffleInTensor3Simple<float, 2, 1, 0, false>(int, float const*, tensorflow::functor::Dimension<3>, float*)*?28?@?H?Xb8gradient_tape/model/conv2d_20/Conv2D/Conv2DBackpropInputhuZU?B
?
?void tensorflow::functor::ApplyAdamKernel<float>(int, float*, float*, float*, float const*, float const*, float const*, float const*, float const*, float const*, float const*, bool) *?28?@?H?b%Adam/Adam/update_17/ResourceApplyAdamhuZU?B
?
?void tensorflow::functor::ApplyAdamKernel<float>(int, float*, float*, float*, float const*, float const*, float const*, float const*, float const*, float const*, float const*, bool) *?28?@?H?b%Adam/Adam/update_51/ResourceApplyAdamhuZU?B
?
?void tensorflow::functor::ApplyAdamKernel<float>(int, float*, float*, float*, float const*, float const*, float const*, float const*, float const*, float const*, float const*, bool) *?28?@?H?b%Adam/Adam/update_63/ResourceApplyAdamhuZU?B
?
?void tensorflow::functor::ApplyAdamKernel<float>(int, float*, float*, float*, float const*, float const*, float const*, float const*, float const*, float const*, float const*, bool) *?28?@?H?b$Adam/Adam/update_3/ResourceApplyAdamhuZU?B
?
?void tensorflow::functor::ApplyAdamKernel<float>(int, float*, float*, float*, float const*, float const*, float const*, float const*, float const*, float const*, float const*, bool) *?28?@?H?b%Adam/Adam/update_11/ResourceApplyAdamhuZU?B
?
?void tensorflow::functor::ApplyAdamKernel<float>(int, float*, float*, float*, float const*, float const*, float const*, float const*, float const*, float const*, float const*, bool) *?28?@?H?b%Adam/Adam/update_59/ResourceApplyAdamhuZU?B
?
?void tensorflow::functor::ApplyAdamKernel<float>(int, float*, float*, float*, float const*, float const*, float const*, float const*, float const*, float const*, float const*, bool) *?28?@?H?b%Adam/Adam/update_61/ResourceApplyAdamhuZU?B
?
?void tensorflow::functor::ShuffleInTensor3Simple<float, 2, 1, 0, false>(int, float const*, tensorflow::functor::Dimension<3>, float*)*?28?@?H?Xb9gradient_tape/model/conv2d_20/Conv2D/Conv2DBackpropFilterhuZU?B
?
?void tensorflow::functor::ShuffleInTensor3Simple<float, 2, 1, 0, false>(int, float const*, tensorflow::functor::Dimension<3>, float*)*?2 8?@?H?Xb>gradient_tape/model/conv2d_transpose_3/conv2d_transpose/Conv2DhuZU?B
?
?void tensorflow::functor::ShuffleInTensor3Simple<float, 2, 1, 0, false>(int, float const*, tensorflow::functor::Dimension<3>, float*)*?2 8?@?H?XbLgradient_tape/model/conv2d_transpose_3/conv2d_transpose/Conv2DBackpropFilterhuZU?B
?
?void tensorflow::functor::ApplyAdamKernel<float>(int, float*, float*, float*, float const*, float const*, float const*, float const*, float const*, float const*, float const*, bool) *?28?@?H?b%Adam/Adam/update_23/ResourceApplyAdamhuZU?B
?
?void tensorflow::functor::ApplyAdamKernel<float>(int, float*, float*, float*, float const*, float const*, float const*, float const*, float const*, float const*, float const*, bool) *?28?@?H?b%Adam/Adam/update_57/ResourceApplyAdamhuZU?B
?
?void tensorflow::functor::ApplyAdamKernel<float>(int, float*, float*, float*, float const*, float const*, float const*, float const*, float const*, float const*, float const*, bool) *?28?@?H?b%Adam/Adam/update_15/ResourceApplyAdamhuZU?B
?
?void tensorflow::functor::ApplyAdamKernel<float>(int, float*, float*, float*, float const*, float const*, float const*, float const*, float const*, float const*, float const*, bool) *?28?@?H?b%Adam/Adam/update_13/ResourceApplyAdamhuZU?B
?
?void tensorflow::functor::ApplyAdamKernel<float>(int, float*, float*, float*, float const*, float const*, float const*, float const*, float const*, float const*, float const*, bool) *?28?@?H?b%Adam/Adam/update_65/ResourceApplyAdamhuZU?B
?
?void tensorflow::functor::ShuffleInTensor3Simple<float, 2, 1, 0, false>(int, float const*, tensorflow::functor::Dimension<3>, float*)*?28?@?H?Xb8gradient_tape/model/conv2d_2/Conv2D/Conv2DBackpropFilterhuZU?B
?
?void tensorflow::functor::ApplyAdamKernel<float>(int, float*, float*, float*, float const*, float const*, float const*, float const*, float const*, float const*, float const*, bool) *?28?@?H?b$Adam/Adam/update_5/ResourceApplyAdamhuZU?B
?
?void tensorflow::functor::ApplyAdamKernel<float>(int, float*, float*, float*, float const*, float const*, float const*, float const*, float const*, float const*, float const*, bool) *?28?@?H?b$Adam/Adam/update_7/ResourceApplyAdamhuZU?B
?
?void tensorflow::functor::ColumnReduceKernel<float const*, float*, cub::Sum>(float const*, float*, int, int, cub::Sum, std::iterator_traits<float const*>::value_type)?!*  28?@?H?b1gradient_tape/model/conv2d_15/BiasAdd/BiasAddGradhuZU?B
?
?void tensorflow::functor::ShuffleInTensor3Simple<float, 2, 1, 0, false>(int, float const*, tensorflow::functor::Dimension<3>, float*)*?2	8?@?H?Xb9gradient_tape/model/conv2d_21/Conv2D/Conv2DBackpropFilterhuZU?B
?
?void tensorflow::functor::ShuffleInTensor3Simple<float, 2, 1, 0, false>(int, float const*, tensorflow::functor::Dimension<3>, float*)*?28?@?H?XbLgradient_tape/model/conv2d_transpose_4/conv2d_transpose/Conv2DBackpropFilterhuZU?B
?
?void tensorflow::functor::ShuffleInTensor3Simple<float, 2, 1, 0, false>(int, float const*, tensorflow::functor::Dimension<3>, float*)*?2	8?@?H?Xb8gradient_tape/model/conv2d_1/Conv2D/Conv2DBackpropFilterhuZU?B
?
Tvoid cask_cudnn::computeOffsetsKernel<true, false>(cask_cudnn::ComputeOffsetsParams)*?2e8?@?H?Xb8gradient_tape/model/conv2d_22/Conv2D/Conv2DBackpropInputhu  ?B
?
?void tensorflow::functor::ApplyAdamKernel<float>(int, float*, float*, float*, float const*, float const*, float const*, float const*, float const*, float const*, float const*, bool) *?28?@?H?b%Adam/Adam/update_70/ResourceApplyAdamhuZU?B
?
?void tensorflow::functor::ShuffleInTensor3Simple<float, 2, 1, 0, false>(int, float const*, tensorflow::functor::Dimension<3>, float*)*?2	8?@?H?bmodel/conv2d_1/ReluhuZU?B
?
?void tensorflow::functor::ColumnReduceKernel<float const*, float*, cub::Sum>(float const*, float*, int, int, cub::Sum, std::iterator_traits<float const*>::value_type)?!*  2 8?@?H?b1gradient_tape/model/conv2d_10/BiasAdd/BiasAddGradhuZU?B
?
?void tensorflow::functor::ShuffleInTensor3Simple<float, 2, 1, 0, false>(int, float const*, tensorflow::functor::Dimension<3>, float*)*?28?@?H?Xb>gradient_tape/model/conv2d_transpose_4/conv2d_transpose/Conv2DhuZU?B
?
?void tensorflow::functor::ColumnReduceKernel<float const*, float*, cub::Sum>(float const*, float*, int, int, cub::Sum, std::iterator_traits<float const*>::value_type)?!*  28?@?H?b:gradient_tape/model/conv2d_transpose_4/BiasAdd/BiasAddGradhuZU?B
?
?void tensorflow::functor::ApplyAdamKernel<float>(int, float*, float*, float*, float const*, float const*, float const*, float const*, float const*, float const*, float const*, bool) *?28?@?H?b$Adam/Adam/update_1/ResourceApplyAdamhuZU?B
?
?void tensorflow::functor::ColumnReduceKernel<float const*, float*, cub::Sum>(float const*, float*, int, int, cub::Sum, std::iterator_traits<float const*>::value_type)?!*  28?@?H?b0gradient_tape/model/conv2d_7/BiasAdd/BiasAddGradhuZU?B
?
?void tensorflow::functor::ColumnReduceMax16ColumnsKernel<float const*, float*, cub::Sum>(float const*, float*, int, int, cub::Sum, std::iterator_traits<float const*>::value_type)?!* 28?@?H?b1gradient_tape/model/conv2d_22/BiasAdd/BiasAddGradhuZU?B
?
Ncask_cudnn::computeWgradBOffsetsKernel(cask_cudnn::ComputeWgradBOffsetsParams)*?28?@?H?Xb9gradient_tape/model/conv2d_19/Conv2D/Conv2DBackpropFilterhu  ?B
?
~void cudnn::winograd::generateWinogradTilesKernel<0, float, float>(cudnn::winograd::GenerateWinogradTilesParams<float, float>)(?D* 28?@?H?bmodel/conv2d_20/ReluhuZU?B
I
 Pow_GPU_DT_FLOAT_DT_FLOAT_kernel*?28?@?H?bAdam/PowhuZU?B
?
~void cudnn::winograd::generateWinogradTilesKernel<0, float, float>(cudnn::winograd::GenerateWinogradTilesParams<float, float>)(?D* 28?@?H?bmodel/conv2d_19/ReluhuZU?B
?
~void cudnn::winograd::generateWinogradTilesKernel<0, float, float>(cudnn::winograd::GenerateWinogradTilesParams<float, float>)(?D* 28?@?H?bmodel/conv2d_3/ReluhuZU?B
?
?void tensorflow::functor::ColumnReduceKernel<float const*, float*, cub::Sum>(float const*, float*, int, int, cub::Sum, std::iterator_traits<float const*>::value_type)?!*  28?@?H?b0gradient_tape/model/conv2d_8/BiasAdd/BiasAddGradhuZU?B
?
?void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::div_no_nan_op<float>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, int>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::div_no_nan_op<float>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, int>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*?28?@?H?b
div_no_nanhuZU?B
?
~void cudnn::winograd::generateWinogradTilesKernel<0, float, float>(cudnn::winograd::GenerateWinogradTilesParams<float, float>)(?D* 28?@?H?bmodel/conv2d_2/ReluhuZU?B
?
?void tensorflow::functor::ColumnReduceKernel<float const*, float*, cub::Sum>(float const*, float*, int, int, cub::Sum, std::iterator_traits<float const*>::value_type)?!*  28?@?H?b8gradient_tape/model/conv2d_transpose/BiasAdd/BiasAddGradhuZU?B
?
?void tensorflow::functor::ColumnReduceKernel<float const*, float*, cub::Sum>(float const*, float*, int, int, cub::Sum, std::iterator_traits<float const*>::value_type)?!*  2 8?@?H?b1gradient_tape/model/conv2d_11/BiasAdd/BiasAddGradhuZU?B
?
?void tensorflow::functor::ColumnReduceKernel<float const*, float*, cub::Sum>(float const*, float*, int, int, cub::Sum, std::iterator_traits<float const*>::value_type)?!*  28?@?H?b1gradient_tape/model/conv2d_13/BiasAdd/BiasAddGradhuZU?B
P
%LogicalAnd_GPU_DT_BOOL_DT_BOOL_kernel*?28?@?H?b
LogicalAndhuZU?B
?
?void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::div_no_nan_op<float>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, int>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::div_no_nan_op<float>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, int>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*?28?@?H?b0weighted_binary_crossentropy/weighted_loss/valuehuZU?B
?
?void cub::DeviceReduceSingleTileKernel<cub::DeviceReducePolicy<float, float, int, tensorflow::functor::Sum<float> >::Policy600, float*, tensorflow::TransformOutputIterator<float, float, tensorflow::functor::DividesBy<float, float>, long>, int, tensorflow::functor::Sum<float>, float>(float*, tensorflow::TransformOutputIterator<float, float, tensorflow::functor::DividesBy<float, float>, long>, int, tensorflow::functor::Sum<float>, float)0*?28?@?H?b!weighted_binary_crossentropy/Meanhu  ?B
?
~void cudnn::winograd::generateWinogradTilesKernel<0, float, float>(cudnn::winograd::GenerateWinogradTilesParams<float, float>)(?D* 28?@?H?Xb8gradient_tape/model/conv2d_18/Conv2D/Conv2DBackpropInputhuZU?B
?
?void tensorflow::functor::ColumnReduceKernel<float const*, float*, cub::Sum>(float const*, float*, int, int, cub::Sum, std::iterator_traits<float const*>::value_type)?!*  28?@?H?b0gradient_tape/model/conv2d_4/BiasAdd/BiasAddGradhuZU?B
?
?void tensorflow::functor::ColumnReduceKernel<float const*, float*, cub::Sum>(float const*, float*, int, int, cub::Sum, std::iterator_traits<float const*>::value_type)?!*  28?@?H?b0gradient_tape/model/conv2d_6/BiasAdd/BiasAddGradhuZU?B
?
?void tensorflow::functor::ColumnReduceKernel<float const*, float*, cub::Sum>(float const*, float*, int, int, cub::Sum, std::iterator_traits<float const*>::value_type)?!*  28?@?H?b1gradient_tape/model/conv2d_12/BiasAdd/BiasAddGradhuZU?B
?
Uvoid cask_cudnn::computeOffsetsKernel<false, false>(cask_cudnn::ComputeOffsetsParams)*?2e8?@?H?bmodel/conv2d/Reluhu  ?B
?
~void cudnn::winograd::generateWinogradTilesKernel<0, float, float>(cudnn::winograd::GenerateWinogradTilesParams<float, float>)(?D* 28?@?H?Xb8gradient_tape/model/conv2d_21/Conv2D/Conv2DBackpropInputhuZU?B
?
~void cudnn::winograd::generateWinogradTilesKernel<0, float, float>(cudnn::winograd::GenerateWinogradTilesParams<float, float>)(?D* 2 8?@?H?Xb7gradient_tape/model/conv2d_4/Conv2D/Conv2DBackpropInputhuZU?B
?
?void tensorflow::functor::ColumnReduceKernel<float const*, float*, cub::Sum>(float const*, float*, int, int, cub::Sum, std::iterator_traits<float const*>::value_type)?!*  28?@?H?b.gradient_tape/model/conv2d/BiasAdd/BiasAddGradhuZU?B
?
?void tensorflow::functor::ColumnReduceKernel<float const*, float*, cub::Sum>(float const*, float*, int, int, cub::Sum, std::iterator_traits<float const*>::value_type)?!*  28?@?H?b1gradient_tape/model/conv2d_14/BiasAdd/BiasAddGradhuZU?B
?
?void tensorflow::functor::ColumnReduceKernel<float const*, float*, cub::Sum>(float const*, float*, int, int, cub::Sum, std::iterator_traits<float const*>::value_type)?!*  28?@?H?b:gradient_tape/model/conv2d_transpose_1/BiasAdd/BiasAddGradhuZU?B
?
?void tensorflow::functor::ColumnReduceKernel<float const*, float*, cub::Sum>(float const*, float*, int, int, cub::Sum, std::iterator_traits<float const*>::value_type)?!*  28?@?H?b0gradient_tape/model/conv2d_9/BiasAdd/BiasAddGradhuZU?B
?
?void tensorflow::functor::ShuffleInTensor3Simple<float, 2, 1, 0, false>(int, float const*, tensorflow::functor::Dimension<3>, float*)*?28?@?H?bmodel/conv2d/ReluhuZU?B
?
?void tensorflow::functor::ColumnReduceKernel<float const*, float*, cub::Sum>(float const*, float*, int, int, cub::Sum, std::iterator_traits<float const*>::value_type)?!*  28?@?H?b1gradient_tape/model/conv2d_20/BiasAdd/BiasAddGradhuZU?B
?
?void tensorflow::functor::ColumnReduceKernel<float const*, float*, cub::Sum>(float const*, float*, int, int, cub::Sum, std::iterator_traits<float const*>::value_type)?!*  28?@?H?b0gradient_tape/model/conv2d_3/BiasAdd/BiasAddGradhuZU?B
?
~void cudnn::winograd::generateWinogradTilesKernel<0, float, float>(cudnn::winograd::GenerateWinogradTilesParams<float, float>)(?D* 28?@?H?Xb7gradient_tape/model/conv2d_3/Conv2D/Conv2DBackpropInputhuZU?B
?
?void tensorflow::functor::ColumnReduceKernel<float const*, float*, cub::Sum>(float const*, float*, int, int, cub::Sum, std::iterator_traits<float const*>::value_type)?!*  28?@?H?b1gradient_tape/model/conv2d_18/BiasAdd/BiasAddGradhuZU?B
?
?void tensorflow::functor::ColumnReduceKernel<float const*, float*, cub::Sum>(float const*, float*, int, int, cub::Sum, std::iterator_traits<float const*>::value_type)?!*  28?@?H?b1gradient_tape/model/conv2d_16/BiasAdd/BiasAddGradhuZU?B
?
?void tensorflow::functor::ColumnReduceKernel<float const*, float*, cub::Sum>(float const*, float*, int, int, cub::Sum, std::iterator_traits<float const*>::value_type)?!*  28?@?H?b1gradient_tape/model/conv2d_17/BiasAdd/BiasAddGradhuZU?B
?
?void tensorflow::functor::ShuffleInTensor3Simple<float, 2, 1, 0, false>(int, float const*, tensorflow::functor::Dimension<3>, float*)*?28?@?H?Xb6gradient_tape/model/conv2d/Conv2D/Conv2DBackpropFilterhuZU?B
?
?void tensorflow::functor::ColumnReduceKernel<float const*, float*, cub::Sum>(float const*, float*, int, int, cub::Sum, std::iterator_traits<float const*>::value_type)?!*  28?@?H?b0gradient_tape/model/conv2d_1/BiasAdd/BiasAddGradhuZU?B
?
?void tensorflow::functor::ColumnReduceKernel<float const*, float*, cub::Sum>(float const*, float*, int, int, cub::Sum, std::iterator_traits<float const*>::value_type)?!*  28?@?H?b1gradient_tape/model/conv2d_21/BiasAdd/BiasAddGradhuZU?B
?
?void tensorflow::functor::ColumnReduceKernel<float const*, float*, cub::Sum>(float const*, float*, int, int, cub::Sum, std::iterator_traits<float const*>::value_type)?!*  28?@?H?b1gradient_tape/model/conv2d_19/BiasAdd/BiasAddGradhuZU?B
?
?void tensorflow::functor::ColumnReduceKernel<float const*, float*, cub::Sum>(float const*, float*, int, int, cub::Sum, std::iterator_traits<float const*>::value_type)?!*  28?@?H?b0gradient_tape/model/conv2d_2/BiasAdd/BiasAddGradhuZU?B
?
?void tensorflow::functor::ColumnReduceKernel<float const*, float*, cub::Sum>(float const*, float*, int, int, cub::Sum, std::iterator_traits<float const*>::value_type)?!*  28?@?H?b:gradient_tape/model/conv2d_transpose_3/BiasAdd/BiasAddGradhuZU?B
?
?void tensorflow::functor::ColumnReduceKernel<float const*, float*, cub::Sum>(float const*, float*, int, int, cub::Sum, std::iterator_traits<float const*>::value_type)?!*  28?@?H?b0gradient_tape/model/conv2d_5/BiasAdd/BiasAddGradhuZU?B
?
?void tensorflow::functor::ColumnReduceKernel<float const*, float*, cub::Sum>(float const*, float*, int, int, cub::Sum, std::iterator_traits<float const*>::value_type)?!*  28?@?H?b:gradient_tape/model/conv2d_transpose_2/BiasAdd/BiasAddGradhuZU?B
?
~void cudnn::winograd::generateWinogradTilesKernel<0, float, float>(cudnn::winograd::GenerateWinogradTilesParams<float, float>)(?D* 28?@?H?Xb8gradient_tape/model/conv2d_19/Conv2D/Conv2DBackpropInputhuZU?B
?
Uvoid cask_cudnn::computeOffsetsKernel<false, false>(cask_cudnn::ComputeOffsetsParams)*?2e8?@?H?bmodel/conv2d_21/Reluhu  ?B
?
?void tensorflow::functor::CleanupSegments<float*, float*, cub::Sum>(float*, float*, int, int, int, cub::Sum, std::iterator_traits<float*>::value_type)*  28?@?H?b.gradient_tape/model/conv2d/BiasAdd/BiasAddGradhuZU?B
?
?void tensorflow::functor::CleanupSegments<float*, float*, cub::Sum>(float*, float*, int, int, int, cub::Sum, std::iterator_traits<float*>::value_type)*  28?@?H?b:gradient_tape/model/conv2d_transpose_3/BiasAdd/BiasAddGradhuZU?B
?
?void tensorflow::functor::CleanupSegments<float*, float*, cub::Sum>(float*, float*, int, int, int, cub::Sum, std::iterator_traits<float*>::value_type)*  28?@?H?b1gradient_tape/model/conv2d_17/BiasAdd/BiasAddGradhuZU?B
?
?void tensorflow::functor::CleanupSegments<float*, float*, cub::Sum>(float*, float*, int, int, int, cub::Sum, std::iterator_traits<float*>::value_type)*  28?@?H?b1gradient_tape/model/conv2d_14/BiasAdd/BiasAddGradhuZU?B
?
?void tensorflow::functor::CleanupSegments<float*, float*, cub::Sum>(float*, float*, int, int, int, cub::Sum, std::iterator_traits<float*>::value_type)*  28?@?H?b1gradient_tape/model/conv2d_18/BiasAdd/BiasAddGradhuZU?B
?
?void tensorflow::functor::CleanupSegments<float*, float*, cub::Sum>(float*, float*, int, int, int, cub::Sum, std::iterator_traits<float*>::value_type)*  28?@?H?b:gradient_tape/model/conv2d_transpose_2/BiasAdd/BiasAddGradhuZU?B
?
?void tensorflow::functor::CleanupSegments<float*, float*, cub::Sum>(float*, float*, int, int, int, cub::Sum, std::iterator_traits<float*>::value_type)*  28?@?H?b1gradient_tape/model/conv2d_15/BiasAdd/BiasAddGradhuZU?B
?
?void tensorflow::functor::CleanupSegments<float*, float*, cub::Sum>(float*, float*, int, int, int, cub::Sum, std::iterator_traits<float*>::value_type)*  28?@?H?b0gradient_tape/model/conv2d_7/BiasAdd/BiasAddGradhuZU?B
K
"AddV2_GPU_DT_INT64_DT_INT64_kernel*?28?@?H?bAdam/addhuZU?B
?
?void tensorflow::functor::CleanupSegments<float*, float*, cub::Sum>(float*, float*, int, int, int, cub::Sum, std::iterator_traits<float*>::value_type)*  28?@?H?b1gradient_tape/model/conv2d_20/BiasAdd/BiasAddGradhuZU?B
?
?void tensorflow::functor::CleanupSegments<float*, float*, cub::Sum>(float*, float*, int, int, int, cub::Sum, std::iterator_traits<float*>::value_type)*  28?@?H?b1gradient_tape/model/conv2d_21/BiasAdd/BiasAddGradhuZU?B
?
?void tensorflow::functor::CleanupSegments<float*, float*, cub::Sum>(float*, float*, int, int, int, cub::Sum, std::iterator_traits<float*>::value_type)*  28?@?H?b:gradient_tape/model/conv2d_transpose_4/BiasAdd/BiasAddGradhuZU?B
?
?void tensorflow::functor::CleanupSegments<float*, float*, cub::Sum>(float*, float*, int, int, int, cub::Sum, std::iterator_traits<float*>::value_type)*  28?@?H?b1gradient_tape/model/conv2d_19/BiasAdd/BiasAddGradhuZU?B
?
?void tensorflow::functor::CleanupSegments<float*, float*, cub::Sum>(float*, float*, int, int, int, cub::Sum, std::iterator_traits<float*>::value_type)*  28?@?H?b0gradient_tape/model/conv2d_2/BiasAdd/BiasAddGradhuZU?B
?
?void tensorflow::functor::CleanupSegments<float*, float*, cub::Sum>(float*, float*, int, int, int, cub::Sum, std::iterator_traits<float*>::value_type)*  28?@?H?b1gradient_tape/model/conv2d_16/BiasAdd/BiasAddGradhuZU?B
?
?void tensorflow::functor::CleanupSegments<float*, float*, cub::Sum>(float*, float*, int, int, int, cub::Sum, std::iterator_traits<float*>::value_type)*  28?@?H?b0gradient_tape/model/conv2d_5/BiasAdd/BiasAddGradhuZU?B
?
?void tensorflow::functor::CleanupSegments<float*, float*, cub::Sum>(float*, float*, int, int, int, cub::Sum, std::iterator_traits<float*>::value_type)*  28?@?H?b0gradient_tape/model/conv2d_6/BiasAdd/BiasAddGradhuZU?B
?
?void tensorflow::functor::CleanupSegments<float*, float*, cub::Sum>(float*, float*, int, int, int, cub::Sum, std::iterator_traits<float*>::value_type)*  28?@?H?b:gradient_tape/model/conv2d_transpose_1/BiasAdd/BiasAddGradhuZU?B
?
?void tensorflow::functor::CleanupSegments<float*, float*, cub::Sum>(float*, float*, int, int, int, cub::Sum, std::iterator_traits<float*>::value_type)*  28?@?H?b0gradient_tape/model/conv2d_1/BiasAdd/BiasAddGradhuZU?B
?
?void tensorflow::functor::CleanupSegments<float*, float*, cub::Sum>(float*, float*, int, int, int, cub::Sum, std::iterator_traits<float*>::value_type)*  28?@?H?b0gradient_tape/model/conv2d_4/BiasAdd/BiasAddGradhuZU?B
?
~void cudnn::winograd::generateWinogradTilesKernel<0, float, float>(cudnn::winograd::GenerateWinogradTilesParams<float, float>)(?D* 28?@?H?Xb7gradient_tape/model/conv2d_1/Conv2D/Conv2DBackpropInputhuZU?B
?
?void tensorflow::functor::CleanupSegments<float*, float*, cub::Sum>(float*, float*, int, int, int, cub::Sum, std::iterator_traits<float*>::value_type)*  28?@?H?b0gradient_tape/model/conv2d_3/BiasAdd/BiasAddGradhuZU?B
K
 Pow_GPU_DT_FLOAT_DT_FLOAT_kernel*?28?@?H?b
Adam/Pow_1huZU?B
?
?void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*?28?@?H?bAssignAddVariableOp_1huZU?B
?
?void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<long, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<long const, long const>, Eigen::TensorMap<Eigen::Tensor<long, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<long const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<long, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<long const, long const>, Eigen::TensorMap<Eigen::Tensor<long, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<long const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*?28?@?H?bAssignAddVariableOp_2huZU?B
?
Uvoid cask_cudnn::computeOffsetsKernel<false, false>(cask_cudnn::ComputeOffsetsParams)*?28?@?H?Xb>gradient_tape/model/conv2d_transpose_3/conv2d_transpose/Conv2Dhu  ?B
?
Uvoid cask_cudnn::computeOffsetsKernel<false, false>(cask_cudnn::ComputeOffsetsParams)*?2e8?@?H?bmodel/conv2d_1/Reluhu  ?B
?
Uvoid cask_cudnn::computeOffsetsKernel<false, false>(cask_cudnn::ComputeOffsetsParams)*?28?@?H?Xb<gradient_tape/model/conv2d_transpose/conv2d_transpose/Conv2Dhu  ?B
?
Uvoid cask_cudnn::computeOffsetsKernel<false, false>(cask_cudnn::ComputeOffsetsParams)*?28?@?H?Xb>gradient_tape/model/conv2d_transpose_4/conv2d_transpose/Conv2Dhu  ?B
?
?void tensorflow::functor::ShuffleInTensor3Simple<float, 2, 1, 0, false>(int, float const*, tensorflow::functor::Dimension<3>, float*)*?28?@?H?Xbmodel/conv2d_22/Conv2DhuZU?B
D
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*?28?@?H?bMulhuZU?B
?
?void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*?28?@?H?bAssignAddVariableOphuZU?B
?
Uvoid cask_cudnn::computeOffsetsKernel<false, false>(cask_cudnn::ComputeOffsetsParams)*?28?@?H?Xb>gradient_tape/model/conv2d_transpose_2/conv2d_transpose/Conv2Dhu  ?B
?
?void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<float, Eigen::TensorMap<Eigen::Tensor<long const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<float, Eigen::TensorMap<Eigen::Tensor<long const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*?28?@?H?bAdam/Cast_1huZU?B
?
~void cudnn::winograd::generateWinogradTilesKernel<0, float, float>(cudnn::winograd::GenerateWinogradTilesParams<float, float>)(?D* 28?@?H?Xb8gradient_tape/model/conv2d_20/Conv2D/Conv2DBackpropInputhuZU?B
?
?void tensorflow::functor::ShuffleInTensor3Simple<float, 2, 1, 0, false>(int, float const*, tensorflow::functor::Dimension<3>, float*)*?28?@?H?Xb8gradient_tape/model/conv2d_22/Conv2D/Conv2DBackpropInputhuZU?B
?
~void cudnn::winograd::generateWinogradTilesKernel<0, float, float>(cudnn::winograd::GenerateWinogradTilesParams<float, float>)(?D* 28?@?H?Xb7gradient_tape/model/conv2d_2/Conv2D/Conv2DBackpropInputhuZU?B
?
Ncask_cudnn::computeWgradBOffsetsKernel(cask_cudnn::ComputeWgradBOffsetsParams)*?28?@?H?Xb8gradient_tape/model/conv2d_3/Conv2D/Conv2DBackpropFilterhu  ?B
?
?void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<long, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<long const, long const>, Eigen::TensorMap<Eigen::Tensor<long, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<long const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<long, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<long const, long const>, Eigen::TensorMap<Eigen::Tensor<long, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<long const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*?28?@?H?bAdam/Adam/AssignAddVariableOphuZU?B
?
Scask_cudnn::computeWgradSplitKOffsetsKernel(cask_cudnn::ComputeSplitKOffsetsParams)*?2P8?@?H?Xb8gradient_tape/model/conv2d_2/Conv2D/Conv2DBackpropFilterhu  ?B
?
Scask_cudnn::computeWgradSplitKOffsetsKernel(cask_cudnn::ComputeSplitKOffsetsParams)*?2P8?@?H?Xb8gradient_tape/model/conv2d_3/Conv2D/Conv2DBackpropFilterhu  ?B
?
?void tensorflow::functor::ShuffleInTensor3Simple<float, 2, 1, 0, false>(int, float const*, tensorflow::functor::Dimension<3>, float*)*?28?@?H?Xb9gradient_tape/model/conv2d_22/Conv2D/Conv2DBackpropFilterhuZU?B
?
Ncask_cudnn::computeWgradBOffsetsKernel(cask_cudnn::ComputeWgradBOffsetsParams)*?28?@?H?Xb8gradient_tape/model/conv2d_2/Conv2D/Conv2DBackpropFilterhu  ?B
?
Ncask_cudnn::computeWgradBOffsetsKernel(cask_cudnn::ComputeWgradBOffsetsParams)*?28?@?H?XbLgradient_tape/model/conv2d_transpose_3/conv2d_transpose/Conv2DBackpropFilterhu  ?B
?
Scask_cudnn::computeWgradSplitKOffsetsKernel(cask_cudnn::ComputeSplitKOffsetsParams)*?2(8?@?H?XbLgradient_tape/model/conv2d_transpose_3/conv2d_transpose/Conv2DBackpropFilterhu  ?B
?
Dcask_cudnn::computeBOffsetsKernel(cask_cudnn::ComputeBOffsetsParams)*?28?@?H?Xb8gradient_tape/model/conv2d_22/Conv2D/Conv2DBackpropInputhu  ?B