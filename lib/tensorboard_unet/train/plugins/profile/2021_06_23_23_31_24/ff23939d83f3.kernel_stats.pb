
£
½void wgrad_alg0_engine<float, 512, 6, 5, 3, 3, 3, false, 512>(int, int, int, float const*, int, float*, float const*, kernel_grad_params, unsigned long long, int, float, int, int, int, int)_*2	Δ8Ώ ό@Ώ όHΏ όXb9gradient_tape/model/conv2d_20/Conv2D/Conv2DBackpropFilterhu³ͺ&B
’
½void wgrad_alg0_engine<float, 512, 6, 5, 3, 3, 3, false, 512>(int, int, int, float const*, int, float*, float const*, kernel_grad_params, unsigned long long, int, float, int, int, int, int)_*2Δ8ΞΘ@ΞΘHΞΘXb8gradient_tape/model/conv2d_1/Conv2D/Conv2DBackpropFilterhu³ͺ&B
£
½void wgrad_alg0_engine<float, 512, 6, 5, 3, 3, 3, false, 512>(int, int, int, float const*, int, float*, float const*, kernel_grad_params, unsigned long long, int, float, int, int, int, int)_*2Δ8Ο@ΟHΟXb9gradient_tape/model/conv2d_21/Conv2D/Conv2DBackpropFilterhu³ͺ&B
{
<ampere_scudnn_winograd_128x128_ldg1_ldg4_relu_tile148t_nt_v1~*22
8ο@οHοbmodel/conv2d_20/ReluhuMUB
‘
<ampere_scudnn_winograd_128x128_ldg1_ldg4_relu_tile148t_nt_v1~*2d
8ώ@ώHώXb8gradient_tape/model/conv2d_20/Conv2D/Conv2DBackpropInputhuMUB
’
<ampere_scudnn_winograd_128x128_ldg1_ldg4_relu_tile148t_nt_v1~*2Θ
8³¬Β@³¬ΒH³¬ΒXb8gradient_tape/model/conv2d_18/Conv2D/Conv2DBackpropInputhuMUB
{
<ampere_scudnn_winograd_128x128_ldg1_ldg4_relu_tile148t_nt_v1~*2d
8΅ϊ­@΅ϊ­H΅ϊ­bmodel/conv2d_18/ReluhuMUB

6ampere_scudnn_128x64_stridedB_splitK_xregs_large_nn_v1 *2P8υςͺ@υςͺHυςͺXb9gradient_tape/model/conv2d_19/Conv2D/Conv2DBackpropFilterhu  ΘA

6ampere_scudnn_128x64_stridedB_splitK_xregs_large_nn_v1 *2P8£@£H£Xb8gradient_tape/model/conv2d_3/Conv2D/Conv2DBackpropFilterhu  ΘA
‘
<ampere_scudnn_winograd_128x128_ldg1_ldg4_relu_tile148t_nt_v1~*22
8Φ@ΦHΦXb8gradient_tape/model/conv2d_21/Conv2D/Conv2DBackpropInputhuMUB
{
<ampere_scudnn_winograd_128x128_ldg1_ldg4_relu_tile148t_nt_v1~*22
8ΧΒ@ΧΒHΧΒbmodel/conv2d_21/ReluhuMUB
 
<ampere_scudnn_winograd_128x128_ldg1_ldg4_relu_tile148t_nt_v1~*22
8φ@φHφXb7gradient_tape/model/conv2d_1/Conv2D/Conv2DBackpropInputhuMUB
z
<ampere_scudnn_winograd_128x128_ldg1_ldg4_relu_tile148t_nt_v1~*22
8Ζ@ΖHΖbmodel/conv2d_1/ReluhuMUB
ι
?void cudnn::bn_bw_1C11_kernel_new<float, float, float2, 512, true, 1>(float, float, float, float, cudnnTensorStruct, float const*, cudnnTensorStruct, float const*, cudnnTensorStruct, float*, float const*, float*, float*, float const*, float const*, float)(*28ψ@ψHψb>gradient_tape/model/batch_normalization_7/FusedBatchNormGradV3hu  ΘB
}
ampere_sgemm_128x128_ntv*2$8ψ@ψHψXb9gradient_tape/model/conv2d_18/Conv2D/Conv2DBackpropFilterhuMUB
y
ampere_sgemm_128x128_ntv*2$8·@·H·Xb8gradient_tape/model/conv2d_12/Conv2D/Conv2DBackpropInputhuMUB
S
ampere_sgemm_128x128_nnv*2$8y@yHybmodel/conv2d_12/ReluhuMUB

?void cutlass_cudnn::Kernel<cutlass_tensorop_s1688dgrad_optimized_tf32_128x128_16x4_unity_stride>(cutlass_tensorop_s1688dgrad_optimized_tf32_128x128_16x4_unity_stride::Params)χ *2(8Έv@ΈvHΈvXb8gradient_tape/model/conv2d_11/Conv2D/Conv2DBackpropInputh

6ampere_scudnn_128x64_stridedB_splitK_xregs_large_nn_v1 *2P8Ψs@ΨsHΨsXb8gradient_tape/model/conv2d_2/Conv2D/Conv2DBackpropFilterhu  ΘA
ε
void tensorflow::functor::ShuffleInTensor3Simple<float, 2, 1, 0, false>(int, float const*, tensorflow::functor::Dimension<3>, float*)*2&8Ψ»n@Ψ»nHΨ»nXb9gradient_tape/model/conv2d_11/Conv2D/Conv2DBackpropFilterhuZUB
Ί
~sm80_xmma_fprop_implicit_gemm_tf32f32_tf32f32_f32_nhwckrsc_nhwc_tilesize128x128x16_stage4_warpsize2x2x1_g1_tensor16x8x8_kernelς *2
 8Ψn@ΨnHΨnPXbmodel/conv2d_11/Reluh
κ
«void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorSlicingOp<Eigen::array<int, 2ul> const, Eigen::array<int, 2ul> const, Eigen::TensorMap<Eigen::Tensor<float, 2, 1, int>, 16, Eigen::MakePointer> >, Eigen::TensorMap<Eigen::Tensor<float const, 2, 1, int>, 16, Eigen::MakePointer> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorSlicingOp<Eigen::array<int, 2ul> const, Eigen::array<int, 2ul> const, Eigen::TensorMap<Eigen::Tensor<float, 2, 1, int>, 16, Eigen::MakePointer> >, Eigen::TensorMap<Eigen::Tensor<float const, 2, 1, int>, 16, Eigen::MakePointer> const> const, Eigen::GpuDevice>, int)*298έj@ά­5H½―5bmodel/concatenate_4/concathuZUB
ς
void cutlass_cudnn::Kernel<cutlass_tensorop_s1688wgrad_analytic_tf32_128x256_16x3>(cutlass_tensorop_s1688wgrad_analytic_tf32_128x256_16x3::Params)? ΐ*2 	8Ήσf@ΉσfHΉσfXb9gradient_tape/model/conv2d_11/Conv2D/Conv2DBackpropFilterh

<ampere_scudnn_winograd_128x128_ldg1_ldg4_relu_tile148t_nt_v1~*2d
8Ϊνb@ΪνbHΪνbXb8gradient_tape/model/conv2d_19/Conv2D/Conv2DBackpropInputhuMUB

<ampere_scudnn_winograd_128x128_ldg1_ldg4_relu_tile148t_nt_v1~*2d
8Ϊ`@Ϊ`HΪ`Xb7gradient_tape/model/conv2d_3/Conv2D/Conv2DBackpropInputhuMUB
z
ampere_sgemm_128x128_ntv*2$8Ή_@Ή_HΉ_Xb9gradient_tape/model/conv2d_12/Conv2D/Conv2DBackpropFilterhuMUB
w
<ampere_scudnn_winograd_128x128_ldg1_ldg4_relu_tile148t_nt_v1~*2d
8€^@€^H€^bmodel/conv2d_3/ReluhuMUB
x
<ampere_scudnn_winograd_128x128_ldg1_ldg4_relu_tile148t_nt_v1~*2d
8Ϊ^@Ϊ^HΪ^bmodel/conv2d_19/ReluhuMUB
y
ampere_sgemm_128x128_ntv*22$8ϊϊZ@ϊϊZHϊϊZXb8gradient_tape/model/conv2d_16/Conv2D/Conv2DBackpropInputhuMUB
y
ampere_sgemm_128x128_ntv*2$8ϋ₯Y@ϋ₯YHϋ₯YXb8gradient_tape/model/conv2d_14/Conv2D/Conv2DBackpropInputhuMUB
S
ampere_sgemm_128x128_nnv*2$8ϋR@ϋRHϋRbmodel/conv2d_14/ReluhuMUB
α
}void cudnn::winograd_nonfused::winogradWgradData4x4<float, float>(cudnn::winograd_nonfused::WinogradDataParams<float, float>)@Α*2Θ28»Q@»QH»QXb9gradient_tape/model/conv2d_18/Conv2D/Conv2DBackpropFilterhuZUB
S
ampere_sgemm_128x128_nnv*22$8ΊύP@ΊύPHΊύPbmodel/conv2d_16/ReluhuMUB
ͺ
Υvoid cudnn::bn_fw_tr_1C11_kernel_NCHW<float, float, 512, true, 1>(cudnnTensorStruct, float const*, cudnnTensorStruct, float*, float const*, float const*, float, float, float*, float*, float*, float*, float, float)(*28ΫΔO@ΫΔOHΫΔOb,model/batch_normalization_7/FusedBatchNormV3hu  ΘB

½void wgrad_alg0_engine<float, 128, 5, 5, 3, 3, 3, false, 512>(int, int, int, float const*, int, float*, float const*, kernel_grad_params, unsigned long long, int, float, int, int, int, int)P*2Δ8ΫΕL@ΫΕLHΫΕLXb6gradient_tape/model/conv2d/Conv2D/Conv2DBackpropFilterhu  HB
μ
‘void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_product_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorConversionOp<float, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_cmp_op<float const, float const, (Eigen::internal::ComparisonName)5>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorCwiseNullaryOp<Eigen::internal::scalar_constant_op<float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_product_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorConversionOp<float, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_cmp_op<float const, float const, (Eigen::internal::ComparisonName)5>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorCwiseNullaryOp<Eigen::internal::scalar_constant_op<float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const> const> const> const, Eigen::GpuDevice>, long)*298Ϋ?K@Ϋ?KHΫ?Kb&gradient_tape/model/conv2d_21/ReluGradhuZUB
μ
‘void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_product_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorConversionOp<float, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_cmp_op<float const, float const, (Eigen::internal::ComparisonName)5>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorCwiseNullaryOp<Eigen::internal::scalar_constant_op<float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_product_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorConversionOp<float, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_cmp_op<float const, float const, (Eigen::internal::ComparisonName)5>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorCwiseNullaryOp<Eigen::internal::scalar_constant_op<float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const> const> const> const, Eigen::GpuDevice>, long)*298 K@ KH Kb&gradient_tape/model/conv2d_20/ReluGradhuZUB
λ
‘void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_product_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorConversionOp<float, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_cmp_op<float const, float const, (Eigen::internal::ComparisonName)5>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorCwiseNullaryOp<Eigen::internal::scalar_constant_op<float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_product_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorConversionOp<float, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_cmp_op<float const, float const, (Eigen::internal::ComparisonName)5>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorCwiseNullaryOp<Eigen::internal::scalar_constant_op<float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const> const> const> const, Eigen::GpuDevice>, long)*298K@KHKb%gradient_tape/model/conv2d_1/ReluGradhuZUB
ι
‘void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_product_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorConversionOp<float, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_cmp_op<float const, float const, (Eigen::internal::ComparisonName)5>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorCwiseNullaryOp<Eigen::internal::scalar_constant_op<float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_product_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorConversionOp<float, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_cmp_op<float const, float const, (Eigen::internal::ComparisonName)5>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorCwiseNullaryOp<Eigen::internal::scalar_constant_op<float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const> const> const> const, Eigen::GpuDevice>, long)*298»ύJ@»ύJH»ύJb#gradient_tape/model/conv2d/ReluGradhuZUB
©
οvoid Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*298σH@σHHσHbAdam/gradients/AddN_5huZUB
³
½void wgrad_alg0_engine<float, 512, 6, 5, 3, 3, 3, false, 512>(int, int, int, float const*, int, float*, float const*, kernel_grad_params, unsigned long long, int, float, int, int, int, int)_*28ΌΫE@ΌΫEHΌΫEXbLgradient_tape/model/conv2d_transpose_4/conv2d_transpose/Conv2DBackpropFilterhu³ͺ&B
¦
Ιvoid cudnn::ops::pooling_bw_kernel_max<float, float, cudnn::maxpooling_func<float, (cudnnNanPropagation_t)0>, false>(cudnnTensorStruct, float const*, cudnnTensorStruct, float const*, cudnnTensorStruct, float const*, cudnnTensorStruct, float*, cudnnPoolingStruct, float, float, int, cudnn::reduced_divisor, cudnn::reduced_divisor)  *2228ϋD@ϋDHϋDb5gradient_tape/model/max_pooling2d/MaxPool/MaxPoolGradhu  ΘB
 
½void wgrad_alg0_engine<float, 128, 5, 5, 3, 3, 3, false, 512>(int, int, int, float const*, int, float*, float const*, kernel_grad_params, unsigned long long, int, float, int, int, int, int)P*2Δ8¨B@¨BH¨BXb9gradient_tape/model/conv2d_22/Conv2D/Conv2DBackpropFilterhu  HB
z
ampere_sgemm_128x128_ntv*2$8B@BHBXb9gradient_tape/model/conv2d_14/Conv2D/Conv2DBackpropFilterhuMUB
z
ampere_sgemm_128x128_ntv*2$8ΜA@ΜAHΜAXb9gradient_tape/model/conv2d_16/Conv2D/Conv2DBackpropFilterhuMUB
y
ampere_sgemm_128x128_ntv*2$8Ϋε@@Ϋε@HΫε@Xb8gradient_tape/model/conv2d_13/Conv2D/Conv2DBackpropInputhuMUB
x
ampere_sgemm_128x128_ntv*2$8φ>@φ>Hφ>Xb7gradient_tape/model/conv2d_9/Conv2D/Conv2DBackpropInputhuMUB

<ampere_scudnn_winograd_128x128_ldg1_ldg4_relu_tile148t_nt_v1~*22
8Όη>@Όη>HΌη>Xb7gradient_tape/model/conv2d_2/Conv2D/Conv2DBackpropInputhuMUB
S
ampere_sgemm_128x128_nnv*2$8ά >@ά >Hά >bmodel/conv2d_13/ReluhuMUB
ζ
?void cudnn::bn_bw_1C11_kernel_new<float, float, float2, 128, true, 1>(float, float, float, float, cudnnTensorStruct, float const*, cudnnTensorStruct, float const*, cudnnTensorStruct, float*, float const*, float*, float*, float const*, float const*, float)(*28όφ=@όφ=Hόφ=b>gradient_tape/model/batch_normalization_6/FusedBatchNormGradV3hu  ΘB
R
ampere_sgemm_128x128_nnv*2$8η=@η=Hη=bmodel/conv2d_9/ReluhuMUB
w
<ampere_scudnn_winograd_128x128_ldg1_ldg4_relu_tile148t_nt_v1~*2d
8ό?9@ό?9Hό?9bmodel/conv2d_2/ReluhuMUB

ήvoid xmma_cudnn::gemm::kernel<xmma_cudnn::implicit_gemm::fprop_indexed::Kernel_traits<xmma_cudnn::Ampere_hmma_tf32_traits<unsigned int, float>, xmma_cudnn::Cta_tile<xmma_cudnn::Ampere, 128, 128, 16, 2, 2, 1, 1>, xmma_cudnn::implicit_gemm::fprop_indexed::Gmem_tile_a_t<xmma_cudnn::Ampere_hmma_tf32_traits<unsigned int, float>, xmma_cudnn::Cta_tile<xmma_cudnn::Ampere, 128, 128, 16, 2, 2, 1, 1>, xmma_cudnn::implicit_gemm::Input_related<0, 0, 0, false>, 16, false, xmma_cudnn::implicit_gemm::fprop_indexed::Gmem_tile_base_a<xmma_cudnn::Ampere_hmma_tf32_traits<unsigned int, float>, xmma_cudnn::Cta_tile<xmma_cudnn::Ampere, 128, 128, 16, 2, 2, 1, 1>, xmma_cudnn::implicit_gemm::Input_related<0, 0, 0, false>, 16, xmma_cudnn::Row, 16, 128> >, xmma_cudnn::implicit_gemm::fprop_indexed::Gmem_tile_c_t<xmma_cudnn::Ampere_hmma_tf32_traits<unsigned int, float>, xmma_cudnn::Cta_tile<xmma_cudnn::Ampere, 128, 128, 16, 2, 2, 1, 1>, 16, xmma_cudnn::Fragment_c<xmma_cudnn::Ampere_hmma_tf32_traits<unsigned int, float>, xmma_cudnn::Cta_tile<xmma_cudnn::Ampere, 128, 128, 16, 2, 2, 1, 1>, false> >, xmma_cudnn::implicit_gemm::Input_related<0, 0, 0, false>, 4> >(xmma_cudnn::implicit_gemm::fprop_indexed::Kernel_traits<xmma_cudnn::Ampere_hmma_tf32_traits<unsigned int, float>, xmma_cudnn::Cta_tile<xmma_cudnn::Ampere, 128, 128, 16, 2, 2, 1, 1>, xmma_cudnn::implicit_gemm::fprop_indexed::Gmem_tile_a_t<xmma_cudnn::Ampere_hmma_tf32_traits<unsigned int, float>, xmma_cudnn::Cta_tile<xmma_cudnn::Ampere, 128, 128, 16, 2, 2, 1, 1>, xmma_cudnn::implicit_gemm::Input_related<0, 0, 0, false>, 16, false, xmma_cudnn::implicit_gemm::fprop_indexed::Gmem_tile_base_a<xmma_cudnn::Ampere_hmma_tf32_traits<unsigned int, float>, xmma_cudnn::Cta_tile<xmma_cudnn::Ampere, 128, 128, 16, 2, 2, 1, 1>, xmma_cudnn::implicit_gemm::Input_related<0, 0, 0, false>, 16, xmma_cudnn::Row, 16, 128> >, xmma_cudnn::implicit_gemm::fprop_indexed::Gmem_tile_c_t<xmma_cudnn::Ampere_hmma_tf32_traits<unsigned int, float>, xmma_cudnn::Cta_tile<xmma_cudnn::Ampere, 128, 128, 16, 2, 2, 1, 1>, 16, xmma_cudnn::Fragment_c<xmma_cudnn::Ampere_hmma_tf32_traits<unsigned int, float>, xmma_cudnn::Cta_tile<xmma_cudnn::Ampere, 128, 128, 16, 2, 2, 1, 1>, false> >, xmma_cudnn::implicit_gemm::Input_related<0, 0, 0, false>, 4>::Params)ό *2
8ΌΊ9@ΌΊ9HΌΊ9PXbmodel/conv2d_10/Reluh
?
±void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 4, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorSlicingOp<Eigen::DSizes<int, 4> const, Eigen::DSizes<int, 4> const, Eigen::TensorMap<Eigen::Tensor<float const, 4, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 4, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorSlicingOp<Eigen::DSizes<int, 4> const, Eigen::DSizes<int, 4> const, Eigen::TensorMap<Eigen::Tensor<float const, 4, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int)*298½8@½8H½8b)gradient_tape/model/concatenate_4/Slice_1huZUB
ύ
±void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 4, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorSlicingOp<Eigen::DSizes<int, 4> const, Eigen::DSizes<int, 4> const, Eigen::TensorMap<Eigen::Tensor<float const, 4, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 4, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorSlicingOp<Eigen::DSizes<int, 4> const, Eigen::DSizes<int, 4> const, Eigen::TensorMap<Eigen::Tensor<float const, 4, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int)*298Ό8@Ό8HΌ8b'gradient_tape/model/concatenate_4/SlicehuZUB
ς
void cutlass_cudnn::Kernel<cutlass_tensorop_s1688wgrad_analytic_tf32_128x256_16x3>(cutlass_tensorop_s1688wgrad_analytic_tf32_128x256_16x3::Params)? ΐ*2 8½ά7@½ά7H½ά7Xb9gradient_tape/model/conv2d_10/Conv2D/Conv2DBackpropFilterh
κ
«void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorSlicingOp<Eigen::array<int, 2ul> const, Eigen::array<int, 2ul> const, Eigen::TensorMap<Eigen::Tensor<float, 2, 1, int>, 16, Eigen::MakePointer> >, Eigen::TensorMap<Eigen::Tensor<float const, 2, 1, int>, 16, Eigen::MakePointer> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorSlicingOp<Eigen::array<int, 2ul> const, Eigen::array<int, 2ul> const, Eigen::TensorMap<Eigen::Tensor<float, 2, 1, int>, 16, Eigen::MakePointer> >, Eigen::TensorMap<Eigen::Tensor<float const, 2, 1, int>, 16, Eigen::MakePointer> const> const, Eigen::GpuDevice>, int)*298ό°5@ήΝHγbmodel/concatenate_3/concathuZUB
y
ampere_sgemm_128x128_ntv*2$8½ώ4@½ώ4H½ώ4Xb8gradient_tape/model/conv2d_10/Conv2D/Conv2DBackpropInputhuMUB

Yvoid tensorflow::BiasNCHWKernel<float>(int, float const*, float const*, float*, int, int)*2&8ύΩ4@ύΩ4HύΩ4b model/conv2d_transpose_4/BiasAddhuZUB

<ampere_scudnn_winograd_128x128_ldg1_ldg4_relu_tile148t_nt_v1~*2d8¨3@¨3H¨3Xb7gradient_tape/model/conv2d_4/Conv2D/Conv2DBackpropInputhuMUB
ϋ
¬void cutlass_cudnn::Kernel<cutlass_tensorop_s1688dgrad_optimized_tf32_128x64_16x6_unity_stride>(cutlass_tensorop_s1688dgrad_optimized_tf32_128x64_16x6_unity_stride::Params)¨ ΐ*2Δ8½χ1@½χ1H½χ1b)model/conv2d_transpose_4/conv2d_transposeh
β
void foldedNhwcToNchwKernel<float, float, float, true, (cudnnKernelDataType_t)0>(int, int, int, int, int, int, int, float const*, float*, int, int, int, int, int, int, int, int, int, int, float, float, cudnn::reduced_divisor, cudnn::reduced_divisor, cudnn::reduced_divisor)&!* 2 28ύΰ1@ύΰ1Hύΰ1b)model/conv2d_transpose_4/conv2d_transposehu  ΘB
ε
void tensorflow::functor::ShuffleInTensor3Simple<float, 2, 1, 0, false>(int, float const*, tensorflow::functor::Dimension<3>, float*)*2&8½ͺ1@½ͺ1H½ͺ1Xb9gradient_tape/model/conv2d_12/Conv2D/Conv2DBackpropFilterhuZUB
z
ampere_sgemm_128x128_ntv*2$8έ¨1@έ¨1Hέ¨1Xb9gradient_tape/model/conv2d_13/Conv2D/Conv2DBackpropFilterhuMUB
Ό
void cudnn::winograd_nonfused::winogradForwardData4x4<float, float>(cudnn::winograd_nonfused::WinogradDataParams<float, float>)@±*2$8έύ0@έύ0Hέύ0bmodel/conv2d_16/ReluhuZUB
y
ampere_sgemm_128x128_ntv*2$8ν/@ν/Hν/Xb8gradient_tape/model/conv2d_9/Conv2D/Conv2DBackpropFilterhuMUB
y
ampere_sgemm_128x128_ntv*22$8½/@½/H½/Xb8gradient_tape/model/conv2d_17/Conv2D/Conv2DBackpropInputhuMUB
η
void cudnn::winograd_nonfused::winogradForwardOutput4x4<float, float>(cudnn::winograd_nonfused::WinogradOutputParams<float, float>)0*2$8ήω.@ήω.Hήω.Xb8gradient_tape/model/conv2d_16/Conv2D/Conv2DBackpropInputhu¦ͺ¦B
Ί
~sm80_xmma_fprop_implicit_gemm_tf32f32_tf32f32_f32_nhwckrsc_nchw_tilesize128x128x16_stage4_warpsize2x2x1_g1_tensor16x8x8_kernelς *2ρ8³.@³.H³.PXbmodel/conv2d_4/Reluh
ͺ
3ampere_scudnn_128x64_stridedB_splitK_interior_nn_v1*2(8σ-@σ-Hσ-XbLgradient_tape/model/conv2d_transpose_3/conv2d_transpose/Conv2DBackpropFilterhu  ΘA
x
ampere_sgemm_128x128_ntv*22$8½ο-@½ο-H½ο-Xb7gradient_tape/model/conv2d_5/Conv2D/Conv2DBackpropInputhuMUB
ε
void tensorflow::functor::ShuffleInTensor3Simple<float, 2, 1, 0, false>(int, float const*, tensorflow::functor::Dimension<3>, float*)*2&8½Κ-@½Κ-H½Κ-Xb9gradient_tape/model/conv2d_10/Conv2D/Conv2DBackpropFilterhuZUB
R
ampere_sgemm_128x128_nnv*22$8½ω,@½ω,H½ω,bmodel/conv2d_5/ReluhuMUB
S
ampere_sgemm_128x128_nnv*22$8ύρ,@ύρ,Hύρ,bmodel/conv2d_17/ReluhuMUB
y
ampere_sgemm_128x128_ntv*2$8½Ε,@½Ε,H½Ε,Xb8gradient_tape/model/conv2d_15/Conv2D/Conv2DBackpropInputhuMUB
x
ampere_sgemm_128x128_ntv*2$8½§+@½§+H½§+Xb7gradient_tape/model/conv2d_7/Conv2D/Conv2DBackpropInputhuMUB
S
ampere_sgemm_128x128_nnv*2$8ρ*@ρ*Hρ*bmodel/conv2d_15/ReluhuMUB
R
ampere_sgemm_128x128_nnv*2$8½Σ*@½Σ*H½Σ*bmodel/conv2d_7/ReluhuMUB
?
΅void tensorflow::functor::ApplyAdamKernel<float>(int, float*, float*, float*, float const*, float const*, float const*, float const*, float const*, float const*, float const*, bool) *2&8ώ¬(@ώ¬(Hώ¬(b%Adam/Adam/update_30/ResourceApplyAdamhuZUB
ΰ
}void cudnn::winograd_nonfused::winogradWgradData4x4<float, float>(cudnn::winograd_nonfused::WinogradDataParams<float, float>)@Α*2x28½(@½(H½(Xb9gradient_tape/model/conv2d_16/Conv2D/Conv2DBackpropFilterhuZUB
β
void cudnn::winograd_nonfused::winogradWgradDelta4x4<float, float>(cudnn::winograd_nonfused::WinogradDeltaParams<float, float>)@*2d28Ύ(@Ύ(HΎ(Xb9gradient_tape/model/conv2d_18/Conv2D/Conv2DBackpropFilterhuZUB

(ampere_scudnn_128x64_relu_interior_nn_v1*2Δ8Ύ&@Ύ&HΎ&Xb>gradient_tape/model/conv2d_transpose_4/conv2d_transpose/Conv2DhuMUB
μ
‘void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_product_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorConversionOp<float, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_cmp_op<float const, float const, (Eigen::internal::ComparisonName)5>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorCwiseNullaryOp<Eigen::internal::scalar_constant_op<float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_product_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorConversionOp<float, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_cmp_op<float const, float const, (Eigen::internal::ComparisonName)5>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorCwiseNullaryOp<Eigen::internal::scalar_constant_op<float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const> const> const> const, Eigen::GpuDevice>, long)*298Τ%@Τ%HΤ%b&gradient_tape/model/conv2d_19/ReluGradhuZUB
λ
‘void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_product_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorConversionOp<float, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_cmp_op<float const, float const, (Eigen::internal::ComparisonName)5>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorCwiseNullaryOp<Eigen::internal::scalar_constant_op<float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_product_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorConversionOp<float, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_cmp_op<float const, float const, (Eigen::internal::ComparisonName)5>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorCwiseNullaryOp<Eigen::internal::scalar_constant_op<float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const> const> const> const, Eigen::GpuDevice>, long)*298ΎΣ%@ΎΣ%HΎΣ%b%gradient_tape/model/conv2d_3/ReluGradhuZUB
μ
‘void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_product_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorConversionOp<float, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_cmp_op<float const, float const, (Eigen::internal::ComparisonName)5>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorCwiseNullaryOp<Eigen::internal::scalar_constant_op<float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_product_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorConversionOp<float, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_cmp_op<float const, float const, (Eigen::internal::ComparisonName)5>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorCwiseNullaryOp<Eigen::internal::scalar_constant_op<float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const> const> const> const, Eigen::GpuDevice>, long)*298έΚ%@έΚ%HέΚ%b&gradient_tape/model/conv2d_18/ReluGradhuZUB
λ
‘void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_product_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorConversionOp<float, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_cmp_op<float const, float const, (Eigen::internal::ComparisonName)5>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorCwiseNullaryOp<Eigen::internal::scalar_constant_op<float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_product_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorConversionOp<float, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_cmp_op<float const, float const, (Eigen::internal::ComparisonName)5>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorCwiseNullaryOp<Eigen::internal::scalar_constant_op<float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const> const> const> const, Eigen::GpuDevice>, long)*298Ό%@Ό%HΌ%b%gradient_tape/model/conv2d_2/ReluGradhuZUB
ͺ
Υvoid cudnn::bn_fw_tr_1C11_kernel_NCHW<float, float, 128, true, 1>(cudnnTensorStruct, float const*, cudnnTensorStruct, float*, float const*, float const*, float, float, float*, float*, float*, float*, float, float)(*28Ύ%@Ύ%HΎ%b,model/batch_normalization_6/FusedBatchNormV3hu  ΘB
²
½void wgrad_alg0_engine<float, 128, 6, 8, 3, 3, 5, false, 512>(int, int, int, float const*, int, float*, float const*, kernel_grad_params, unsigned long long, int, float, int, int, int, int){R* 228χ$@χ$Hχ$XbLgradient_tape/model/conv2d_transpose_2/conv2d_transpose/Conv2DBackpropFilterhuMUB

(ampere_scudnn_128x64_relu_interior_nn_v1*2
8ΎΛ$@ΎΛ$HΎΛ$Xb<gradient_tape/model/conv2d_transpose/conv2d_transpose/Conv2DhuMUB
©
οvoid Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*298Ε$@Ε$HΕ$bAdam/gradients/AddN_4huZUB
p
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*2'8$@$H$b+gradient_tape/model/dropout_7/dropout/Mul_2huZUB
^
%ampere_scudnn_128x32_relu_small_nn_v1**@2N8$@$H$bmodel/conv2d/ReluhuMUB

,ampere_scudnn_128x64_stridedB_interior_nn_v1*2N8ή$@ή$Hή$Xb8gradient_tape/model/conv2d_22/Conv2D/Conv2DBackpropInputhu  ΘA
b
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*2'8ύ$@ύ$Hύ$bmodel/dropout_7/dropout/Mul_1huZUB
¨
Ιvoid cudnn::ops::pooling_bw_kernel_max<float, float, cudnn::maxpooling_func<float, (cudnnNanPropagation_t)0>, false>(cudnnTensorStruct, float const*, cudnnTensorStruct, float const*, cudnnTensorStruct, float const*, cudnnTensorStruct, float*, cudnnPoolingStruct, float, float, int, cudnn::reduced_divisor, cudnn::reduced_divisor) *ψ228Ύ±#@Ύ±#HΎ±#b7gradient_tape/model/max_pooling2d_1/MaxPool/MaxPoolGradhu ΐΑB
z
ampere_sgemm_128x128_ntv*2$8σ!@σ!Hσ!Xb9gradient_tape/model/conv2d_17/Conv2D/Conv2DBackpropFilterhuMUB

(ampere_scudnn_128x64_relu_interior_nn_v1*2ρ8Ε!@Ε!HΕ!Xb>gradient_tape/model/conv2d_transpose_3/conv2d_transpose/Conv2DhuMUB
z
ampere_sgemm_128x128_ntv*2$8ή¬!@ή¬!Hή¬!Xb9gradient_tape/model/conv2d_15/Conv2D/Conv2DBackpropFilterhuMUB
x
ampere_sgemm_128x128_ntv*2$8ήͺ!@ήͺ!Hήͺ!Xb7gradient_tape/model/conv2d_8/Conv2D/Conv2DBackpropInputhuMUB
γ
?void cudnn::bn_bw_1C11_kernel_new<float, float, float2, 512, true, 1>(float, float, float, float, cudnnTensorStruct, float const*, cudnnTensorStruct, float const*, cudnnTensorStruct, float*, float const*, float*, float*, float const*, float const*, float)(*2 8ή©!@ή©!Hή©!b<gradient_tape/model/batch_normalization/FusedBatchNormGradV3hu  ΘB

(ampere_scudnn_128x64_relu_interior_nn_v1*28Ύ!@Ύ!HΎ!Xb>gradient_tape/model/conv2d_transpose_2/conv2d_transpose/Conv2DhuMUB
y
ampere_sgemm_128x128_ntv*2$8Ύ!@Ύ!HΎ!Xb8gradient_tape/model/conv2d_5/Conv2D/Conv2DBackpropFilterhuMUB
R
ampere_sgemm_128x128_nnv*2$8ώ± @ώ± Hώ± bmodel/conv2d_8/ReluhuMUB
y
ampere_sgemm_128x128_ntv*2$8ώͺ @ώͺ Hώͺ Xb8gradient_tape/model/conv2d_7/Conv2D/Conv2DBackpropFilterhuMUB
y
ampere_sgemm_128x128_ntv*2$8 @ H Xb8gradient_tape/model/conv2d_4/Conv2D/Conv2DBackpropFilterhuMUB
ε
ώvoid cudnn::bn_bw_1C11_kernel_new<float, float, float2, 32, true, 1>(float, float, float, float, cudnnTensorStruct, float const*, cudnnTensorStruct, float const*, cudnnTensorStruct, float*, float const*, float*, float*, float const*, float const*, float)(*28ήΪ@ήΪHήΪb>gradient_tape/model/batch_normalization_5/FusedBatchNormGradV3hu  ΘB
ΰ
void cudnn::ops::pooling_fw_4d_kernel<float, float, cudnn::maxpooling_func<float, (cudnnNanPropagation_t)0>, (cudnnPoolingMode_t)0, false>(cudnnTensorStruct, float const*, cudnnTensorStruct, float*, cudnnPoolingStruct, float, float, int, cudnn::reduced_divisor, cudnn::reduced_divisor)(  *2228ή@ήHήbmodel/max_pooling2d/MaxPoolhu  ΘB
»
ϊvoid implicit_convolve_sgemm<float, float, 1024, 5, 5, 3, 3, 3, 1, false, false, true>(int, int, int, float const*, int, float*, float const*, kernel_conv_params, unsigned long long, int, float, float, int, float const*, float const*, bool, int, int)=*2ΐΈ8ώ§@ώ§Hώ§Xbmodel/conv2d_22/Conv2DhuZUB
ύ
±void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 4, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorSlicingOp<Eigen::DSizes<int, 4> const, Eigen::DSizes<int, 4> const, Eigen::TensorMap<Eigen::Tensor<float const, 4, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 4, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorSlicingOp<Eigen::DSizes<int, 4> const, Eigen::DSizes<int, 4> const, Eigen::TensorMap<Eigen::Tensor<float const, 4, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int)*298ώΊ@ώΊHώΊb'gradient_tape/model/concatenate_3/SlicehuZUB
?
±void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 4, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorSlicingOp<Eigen::DSizes<int, 4> const, Eigen::DSizes<int, 4> const, Eigen::TensorMap<Eigen::Tensor<float const, 4, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 4, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorSlicingOp<Eigen::DSizes<int, 4> const, Eigen::DSizes<int, 4> const, Eigen::TensorMap<Eigen::Tensor<float const, 4, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int)*298ή±@ή±Hή±b)gradient_tape/model/concatenate_3/Slice_1huZUB
κ
«void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorSlicingOp<Eigen::array<int, 2ul> const, Eigen::array<int, 2ul> const, Eigen::TensorMap<Eigen::Tensor<float, 2, 1, int>, 16, Eigen::MakePointer> >, Eigen::TensorMap<Eigen::Tensor<float const, 2, 1, int>, 16, Eigen::MakePointer> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorSlicingOp<Eigen::array<int, 2ul> const, Eigen::array<int, 2ul> const, Eigen::TensorMap<Eigen::Tensor<float, 2, 1, int>, 16, Eigen::MakePointer> >, Eigen::TensorMap<Eigen::Tensor<float const, 2, 1, int>, 16, Eigen::MakePointer> const> const, Eigen::GpuDevice>, int)*298Ύλ@Ώ―H?»bmodel/concatenate_2/concathuZUB
ύ
?void cutlass_cudnn::Kernel<cutlass_tensorop_s1688dgrad_optimized_tf32_128x128_16x4_unity_stride>(cutlass_tensorop_s1688dgrad_optimized_tf32_128x128_16x4_unity_stride::Params)χ *2ρ8Ύ±@Ύ±HΎ±b)model/conv2d_transpose_3/conv2d_transposeh

Yvoid tensorflow::BiasNCHWKernel<float>(int, float const*, float const*, float*, int, int)*2&8ί@ίHίb model/conv2d_transpose_3/BiasAddhuZUB
β
void foldedNhwcToNchwKernel<float, float, float, true, (cudnnKernelDataType_t)0>(int, int, int, int, int, int, int, float const*, float*, int, int, int, int, int, int, int, int, int, int, float, float, cudnn::reduced_divisor, cudnn::reduced_divisor, cudnn::reduced_divisor)&!* 2Θ28Ώ@ΏHΏb)model/conv2d_transpose_3/conv2d_transposehu  ΘB
R
ampere_sgemm_128x128_nnv*2$8ήχ@ήχHήχbmodel/conv2d_6/ReluhuMUB

void cutlass_cudnn::Kernel<cutlass_tensorop_s1688wgrad_analytic_tf32_128x256_16x3>(cutlass_tensorop_s1688wgrad_analytic_tf32_128x256_16x3::Params)? ΐ*2 8ίδ@ίδHίδXbJgradient_tape/model/conv2d_transpose/conv2d_transpose/Conv2DBackpropFilterh
Ό
void cudnn::winograd_nonfused::winogradForwardData4x4<float, float>(cudnn::winograd_nonfused::WinogradDataParams<float, float>)@±*2$8ΎΠ@ΎΠHΎΠbmodel/conv2d_17/ReluhuZUB
»
void cudnn::winograd_nonfused::winogradForwardData4x4<float, float>(cudnn::winograd_nonfused::WinogradDataParams<float, float>)@±*2$8ήΟ@ήΟHήΟbmodel/conv2d_5/ReluhuZUB
δ
~sm80_xmma_fprop_implicit_gemm_tf32f32_tf32f32_f32_nhwckrsc_nchw_tilesize128x128x16_stage4_warpsize2x2x1_g1_tensor16x8x8_kernelς *2(8Λ@ΛHΛPXb>gradient_tape/model/conv2d_transpose_1/conv2d_transpose/Conv2Dh
β
void cudnn::winograd_nonfused::winogradForwardData4x4<float, float>(cudnn::winograd_nonfused::WinogradDataParams<float, float>)@±*2$8Η@ΗHΗXb8gradient_tape/model/conv2d_16/Conv2D/Conv2DBackpropInputhuZUB
α
void cudnn::winograd_nonfused::winogradForwardData4x4<float, float>(cudnn::winograd_nonfused::WinogradDataParams<float, float>)@±*2$8ίΐ@ίΐHίΐXb7gradient_tape/model/conv2d_5/Conv2D/Conv2DBackpropInputhuZUB
β
void cudnn::winograd_nonfused::winogradForwardData4x4<float, float>(cudnn::winograd_nonfused::WinogradDataParams<float, float>)@±*2$8Ό@ΌHΌXb8gradient_tape/model/conv2d_17/Conv2D/Conv2DBackpropInputhuZUB
μ
void cudnn::ops::nchwToNhwcKernel<float, float, float, false, true, (cudnnKernelDataType_t)2>(cudnn::ops::nchw2nhwc_params_t<float>, float const*, float*)'!*2Θ28?³@?³H?³b)model/conv2d_transpose_4/conv2d_transposehu  ΘB
`
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*2'8³@³H³bmodel/dropout_7/dropout/MulhuZUB
n
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*2'8²@²H²b)gradient_tape/model/dropout_7/dropout/MulhuZUB

¨void tensorflow::functor::SwapDimension1And2InTensor3UsingTiles<unsigned int, 256, 32, 32, false>(unsigned int const*, tensorflow::functor::Dimension<3>, unsigned int*) !*2 8Ώ±@Ώ±HΏ±bCmodel/dropout_7/dropout/Mul_1-0-TransposeNHWCToNCHW-LayoutOptimizerhu  ΘB
Ό
void cudnn::winograd_nonfused::winogradForwardData4x4<float, float>(cudnn::winograd_nonfused::WinogradDataParams<float, float>)@±*28ή@ήHήbmodel/conv2d_14/ReluhuZUB
y
ampere_sgemm_128x128_ntv*2$8ί@ίHίXb8gradient_tape/model/conv2d_8/Conv2D/Conv2DBackpropFilterhuMUB
δ
void cub::DeviceSegmentedReduceKernel<cub::DeviceReducePolicy<float, float, int, cub::Sum>::Policy600, float const*, float*, cub::TransformInputIterator<int, tensorflow::functor::RowOffset, cub::CountingInputIterator<int, long>, long>, int, cub::Sum, float>(float const*, float*, cub::TransformInputIterator<int, tensorflow::functor::RowOffset, cub::CountingInputIterator<int, long>, long>, cub::TransformInputIterator<int, tensorflow::functor::RowOffset, cub::CountingInputIterator<int, long>, long>, int, cub::Sum, float) 0*2ΐ8Ώ@ΏHΏb1gradient_tape/model/conv2d_20/BiasAdd/BiasAddGradhu  ΘB
δ
void cub::DeviceSegmentedReduceKernel<cub::DeviceReducePolicy<float, float, int, cub::Sum>::Policy600, float const*, float*, cub::TransformInputIterator<int, tensorflow::functor::RowOffset, cub::CountingInputIterator<int, long>, long>, int, cub::Sum, float>(float const*, float*, cub::TransformInputIterator<int, tensorflow::functor::RowOffset, cub::CountingInputIterator<int, long>, long>, cub::TransformInputIterator<int, tensorflow::functor::RowOffset, cub::CountingInputIterator<int, long>, long>, int, cub::Sum, float) 0*2ΐ8ί@ίHίb1gradient_tape/model/conv2d_21/BiasAdd/BiasAddGradhu  ΘB
η
void cudnn::winograd_nonfused::winogradForwardOutput4x4<float, float>(cudnn::winograd_nonfused::WinogradOutputParams<float, float>)0*28Ύώ@ΎώHΎώXb8gradient_tape/model/conv2d_14/Conv2D/Conv2DBackpropInputhu¦ͺ¦B

void cutlass_cudnn::Kernel<cutlass_tensorop_s1688wgrad_analytic_tf32_128x256_16x3>(cutlass_tensorop_s1688wgrad_analytic_tf32_128x256_16x3::Params)? ΐ*2	8Ώϊ@ΏϊHΏϊXbLgradient_tape/model/conv2d_transpose_1/conv2d_transpose/Conv2DBackpropFilterh
γ
void cub::DeviceSegmentedReduceKernel<cub::DeviceReducePolicy<float, float, int, cub::Sum>::Policy600, float const*, float*, cub::TransformInputIterator<int, tensorflow::functor::RowOffset, cub::CountingInputIterator<int, long>, long>, int, cub::Sum, float>(float const*, float*, cub::TransformInputIterator<int, tensorflow::functor::RowOffset, cub::CountingInputIterator<int, long>, long>, cub::TransformInputIterator<int, tensorflow::functor::RowOffset, cub::CountingInputIterator<int, long>, long>, int, cub::Sum, float) 0*2ΐ8ίλ@ίλHίλb0gradient_tape/model/conv2d_1/BiasAdd/BiasAddGradhu  ΘB
ν
void cub::DeviceSegmentedReduceKernel<cub::DeviceReducePolicy<float, float, int, cub::Sum>::Policy600, float const*, float*, cub::TransformInputIterator<int, tensorflow::functor::RowOffset, cub::CountingInputIterator<int, long>, long>, int, cub::Sum, float>(float const*, float*, cub::TransformInputIterator<int, tensorflow::functor::RowOffset, cub::CountingInputIterator<int, long>, long>, cub::TransformInputIterator<int, tensorflow::functor::RowOffset, cub::CountingInputIterator<int, long>, long>, int, cub::Sum, float) 0*2ΐ8ήλ@ήλHήλb:gradient_tape/model/conv2d_transpose_4/BiasAdd/BiasAddGradhu  ΘB
α
void cub::DeviceSegmentedReduceKernel<cub::DeviceReducePolicy<float, float, int, cub::Sum>::Policy600, float const*, float*, cub::TransformInputIterator<int, tensorflow::functor::RowOffset, cub::CountingInputIterator<int, long>, long>, int, cub::Sum, float>(float const*, float*, cub::TransformInputIterator<int, tensorflow::functor::RowOffset, cub::CountingInputIterator<int, long>, long>, cub::TransformInputIterator<int, tensorflow::functor::RowOffset, cub::CountingInputIterator<int, long>, long>, int, cub::Sum, float) 0*2ΐ8Ύθ@ΎθHΎθb.gradient_tape/model/conv2d/BiasAdd/BiasAddGradhu  ΘB
§
Lvoid cudnn::ops::scalePackedTensor_kernel<float, float>(long, float*, float)*2??8?έ@?έH?έb5gradient_tape/model/max_pooling2d/MaxPool/MaxPoolGradhu  ΘB
η
void cudnn::winograd_nonfused::winogradForwardOutput4x4<float, float>(cudnn::winograd_nonfused::WinogradOutputParams<float, float>)0*2$8ΎΠ@ΎΠHΎΠXb8gradient_tape/model/conv2d_17/Conv2D/Conv2DBackpropInputhu¦ͺ¦B
ζ
void cudnn::winograd_nonfused::winogradForwardOutput4x4<float, float>(cudnn::winograd_nonfused::WinogradOutputParams<float, float>)0*2$8ώΟ@ώΟHώΟXb7gradient_tape/model/conv2d_5/Conv2D/Conv2DBackpropInputhu¦ͺ¦B
ΐ
void cudnn::winograd_nonfused::winogradForwardOutput4x4<float, float>(cudnn::winograd_nonfused::WinogradOutputParams<float, float>)0*2$8ΏΜ@ΏΜHΏΜbmodel/conv2d_5/Reluhu¦ͺ¦B
Α
void cudnn::winograd_nonfused::winogradForwardOutput4x4<float, float>(cudnn::winograd_nonfused::WinogradOutputParams<float, float>)0*2$8Λ@ΛHΛbmodel/conv2d_16/Reluhu¦ͺ¦B
Α
void cudnn::winograd_nonfused::winogradForwardOutput4x4<float, float>(cudnn::winograd_nonfused::WinogradOutputParams<float, float>)0*2$8ίΚ@ίΚHίΚbmodel/conv2d_17/Reluhu¦ͺ¦B
θ
void cudnn::winograd_nonfused::winogradWgradOutput4x4<float, float>(cudnn::winograd_nonfused::WinogradWgradOutputParams<float, float>)=H* 2 @8?Ε@?ΕH?ΕXb9gradient_tape/model/conv2d_12/Conv2D/Conv2DBackpropFilterhuZUB
x
ampere_sgemm_128x128_ntv*2$8ε@εHεXb7gradient_tape/model/conv2d_6/Conv2D/Conv2DBackpropInputhuMUB
ύ
?void cutlass_cudnn::Kernel<cutlass_tensorop_s1688dgrad_optimized_tf32_128x128_16x4_unity_stride>(cutlass_tensorop_s1688dgrad_optimized_tf32_128x128_16x4_unity_stride::Params)χ *2τ8Θ@ΘHΘb)model/conv2d_transpose_2/conv2d_transposeh
δ
void tensorflow::functor::ShuffleInTensor3Simple<float, 2, 1, 0, false>(int, float const*, tensorflow::functor::Dimension<3>, float*)*2&8ί@ίHίXb8gradient_tape/model/conv2d_11/Conv2D/Conv2DBackpropInputhuZUB
ϊ
?void cutlass_cudnn::Kernel<cutlass_tensorop_s1688dgrad_optimized_tf32_128x128_16x4_unity_stride>(cutlass_tensorop_s1688dgrad_optimized_tf32_128x128_16x4_unity_stride::Params)χ *2(8ώ@ώHώb'model/conv2d_transpose/conv2d_transposeh
Ύ
void tensorflow::functor::ShuffleInTensor3Simple<float, 2, 1, 0, false>(int, float const*, tensorflow::functor::Dimension<3>, float*)*2&8ΏΒ@ΏΒHΏΒbmodel/conv2d_11/ReluhuZUB
ύ
?void cutlass_cudnn::Kernel<cutlass_tensorop_s1688dgrad_optimized_tf32_128x128_16x4_unity_stride>(cutlass_tensorop_s1688dgrad_optimized_tf32_128x128_16x4_unity_stride::Params)χ *2 8ί@ίHίb)model/conv2d_transpose_1/conv2d_transposeh
?
Υvoid Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<float, Eigen::TensorMap<Eigen::Tensor<bool const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<float, Eigen::TensorMap<Eigen::Tensor<bool const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*298ίΓ@ίΓHίΓbXmodel/dropout_7/dropout/ArithmeticOptimizer/ReorderCastLikeAndValuePreserving_float_CasthuZUB

Υvoid Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<float, Eigen::TensorMap<Eigen::Tensor<bool const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<float, Eigen::TensorMap<Eigen::Tensor<bool const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*298ΎΎ@ΎΎHΎΎbmodel/dropout_7/dropout/CasthuZUB
β
void cudnn::winograd_nonfused::winogradWgradDelta4x4<float, float>(cudnn::winograd_nonfused::WinogradDeltaParams<float, float>)@*2<28@HXb9gradient_tape/model/conv2d_17/Conv2D/Conv2DBackpropFilterhuZUB
ΰ
}void cudnn::winograd_nonfused::winogradWgradData4x4<float, float>(cudnn::winograd_nonfused::WinogradDataParams<float, float>)@Α*2<28ή@ήHήXb9gradient_tape/model/conv2d_17/Conv2D/Conv2DBackpropFilterhuZUB
α
void cudnn::winograd_nonfused::winogradWgradDelta4x4<float, float>(cudnn::winograd_nonfused::WinogradDeltaParams<float, float>)@*2<28@HXb8gradient_tape/model/conv2d_4/Conv2D/Conv2DBackpropFilterhuZUB
β
void cudnn::winograd_nonfused::winogradWgradDelta4x4<float, float>(cudnn::winograd_nonfused::WinogradDeltaParams<float, float>)@*2<28@HXb9gradient_tape/model/conv2d_16/Conv2D/Conv2DBackpropFilterhuZUB
ί
}void cudnn::winograd_nonfused::winogradWgradData4x4<float, float>(cudnn::winograd_nonfused::WinogradDataParams<float, float>)@Α*2<28ώ@ώHώXb8gradient_tape/model/conv2d_5/Conv2D/Conv2DBackpropFilterhuZUB
α
void cudnn::winograd_nonfused::winogradWgradDelta4x4<float, float>(cudnn::winograd_nonfused::WinogradDeltaParams<float, float>)@*2<28?@?H?Xb8gradient_tape/model/conv2d_5/Conv2D/Conv2DBackpropFilterhuZUB
ΰ
}void cudnn::winograd_nonfused::winogradWgradData4x4<float, float>(cudnn::winograd_nonfused::WinogradDataParams<float, float>)@Α*2`28Ώ@ΏHΏXb9gradient_tape/model/conv2d_14/Conv2D/Conv2DBackpropFilterhuZUB
?
΅void tensorflow::functor::ApplyAdamKernel<float>(int, float*, float*, float*, float const*, float const*, float const*, float const*, float const*, float const*, float const*, bool) *2&8?@?H?b%Adam/Adam/update_36/ResourceApplyAdamhuZUB
?
΅void tensorflow::functor::ApplyAdamKernel<float>(int, float*, float*, float*, float const*, float const*, float const*, float const*, float const*, float const*, float const*, bool) *2&8Ύ@ΎHΎb%Adam/Adam/update_28/ResourceApplyAdamhuZUB
§
Υvoid cudnn::bn_fw_tr_1C11_kernel_NCHW<float, float, 512, true, 1>(cudnnTensorStruct, float const*, cudnnTensorStruct, float*, float const*, float const*, float, float, float*, float*, float*, float*, float, float)(*2 8Δ@ΔHΔb*model/batch_normalization/FusedBatchNormV3hu  ΘB
μ
‘void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_product_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorConversionOp<float, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_cmp_op<float const, float const, (Eigen::internal::ComparisonName)5>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorCwiseNullaryOp<Eigen::internal::scalar_constant_op<float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_product_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorConversionOp<float, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_cmp_op<float const, float const, (Eigen::internal::ComparisonName)5>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorCwiseNullaryOp<Eigen::internal::scalar_constant_op<float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const> const> const> const, Eigen::GpuDevice>, long)*298ίό@ίόHίόb&gradient_tape/model/conv2d_16/ReluGradhuZUB
μ
‘void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_product_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorConversionOp<float, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_cmp_op<float const, float const, (Eigen::internal::ComparisonName)5>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorCwiseNullaryOp<Eigen::internal::scalar_constant_op<float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_product_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorConversionOp<float, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_cmp_op<float const, float const, (Eigen::internal::ComparisonName)5>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorCwiseNullaryOp<Eigen::internal::scalar_constant_op<float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const> const> const> const, Eigen::GpuDevice>, long)*298ψ@ψHψb&gradient_tape/model/conv2d_17/ReluGradhuZUB
λ
‘void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_product_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorConversionOp<float, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_cmp_op<float const, float const, (Eigen::internal::ComparisonName)5>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorCwiseNullaryOp<Eigen::internal::scalar_constant_op<float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_product_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorConversionOp<float, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_cmp_op<float const, float const, (Eigen::internal::ComparisonName)5>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorCwiseNullaryOp<Eigen::internal::scalar_constant_op<float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const> const> const> const, Eigen::GpuDevice>, long)*298Ώυ@ΏυHΏυb%gradient_tape/model/conv2d_5/ReluGradhuZUB
λ
‘void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_product_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorConversionOp<float, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_cmp_op<float const, float const, (Eigen::internal::ComparisonName)5>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorCwiseNullaryOp<Eigen::internal::scalar_constant_op<float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_product_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorConversionOp<float, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_cmp_op<float const, float const, (Eigen::internal::ComparisonName)5>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorCwiseNullaryOp<Eigen::internal::scalar_constant_op<float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const> const> const> const, Eigen::GpuDevice>, long)*298ίτ@ίτHίτb%gradient_tape/model/conv2d_4/ReluGradhuZUB
©
οvoid Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*298ί²@ί²Hί²bAdam/gradients/AddN_3huZUB
©
Τvoid cudnn::bn_fw_tr_1C11_kernel_NCHW<float, float, 32, true, 1>(cudnnTensorStruct, float const*, cudnnTensorStruct, float*, float const*, float const*, float, float, float*, float*, float*, float*, float, float)(*28ίͺ@ίͺHίͺb,model/batch_normalization_5/FusedBatchNormV3hu  ΘB
ε
void tensorflow::functor::ShuffleInTensor3Simple<float, 2, 1, 0, false>(int, float const*, tensorflow::functor::Dimension<3>, float*)*2&8ί©@ί©Hί©Xb9gradient_tape/model/conv2d_13/Conv2D/Conv2DBackpropFilterhuZUB
p
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*2Δ8@Hb+gradient_tape/model/dropout_6/dropout/Mul_2huZUB
b
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*2Δ8Ώ@ΏHΏbmodel/dropout_6/dropout/Mul_1huZUB
δ
void tensorflow::functor::ShuffleInTensor3Simple<float, 2, 1, 0, false>(int, float const*, tensorflow::functor::Dimension<3>, float*)*2&8?Γ@?ΓH?ΓXb8gradient_tape/model/conv2d_9/Conv2D/Conv2DBackpropFilterhuZUB
y
ampere_sgemm_128x128_ntv*2$8Α@ΑHΑXb8gradient_tape/model/conv2d_6/Conv2D/Conv2DBackpropFilterhuMUB
Ό
void cudnn::winograd_nonfused::winogradForwardData4x4<float, float>(cudnn::winograd_nonfused::WinogradDataParams<float, float>)@±*28ί@ίHίbmodel/conv2d_12/ReluhuZUB
η
void cudnn::winograd_nonfused::winogradForwardOutput4x4<float, float>(cudnn::winograd_nonfused::WinogradOutputParams<float, float>)0*28ζ@ζHζXb8gradient_tape/model/conv2d_12/Conv2D/Conv2DBackpropInputhu¦ͺ¦B
ε
?void cudnn::bn_bw_1C11_kernel_new<float, float, float2, 128, true, 1>(float, float, float, float, cudnnTensorStruct, float const*, cudnnTensorStruct, float const*, cudnnTensorStruct, float*, float const*, float*, float*, float const*, float const*, float)(*2@8ίΓ@ίΓHίΓb>gradient_tape/model/batch_normalization_1/FusedBatchNormGradV3hu  ΘB
β
void cudnn::ops::pooling_fw_4d_kernel<float, float, cudnn::maxpooling_func<float, (cudnnNanPropagation_t)0>, (cudnnPoolingMode_t)0, false>(cudnnTensorStruct, float const*, cudnnTensorStruct, float*, cudnnPoolingStruct, float, float, int, cudnn::reduced_divisor, cudnn::reduced_divisor)( *ψ228@Hbmodel/max_pooling2d_1/MaxPoolhu ΐΑB
q
(GreaterEqual_GPU_DT_FLOAT_DT_BOOL_kernel*2'8@Hb$model/dropout_7/dropout/GreaterEqualhuZUB
ε
void cudnn::winograd_nonfused::winogradForwardFilter4x4<float, float>(cudnn::winograd_nonfused::WinogradFilterParams<float, float>)(H* 28­@­H­Xb8gradient_tape/model/conv2d_10/Conv2D/Conv2DBackpropInputhu  ΘB

΄void cudnn::pooling_bw_kernel_max_nchw_fully_packed_small<float, float, 2, (cudnnNanPropagation_t)0>(cudnnTensorStruct, float const*, cudnnTensorStruct, float const*, cudnnTensorStruct, float const*, cudnnTensorStruct, float*, cudnnPoolingStruct, float, float, cudnn::reduced_divisor, cudnn::reduced_divisor) d*228?«@?«H?«b7gradient_tape/model/max_pooling2d_2/MaxPool/MaxPoolGradhu  ΘB
Ύ
void cudnn::winograd_nonfused::winogradForwardFilter4x4<float, float>(cudnn::winograd_nonfused::WinogradFilterParams<float, float>)(H* 2 @8?«@?«H?«bmodel/conv2d_12/Reluhu  ΘB
δ
void cudnn::winograd_nonfused::winogradForwardFilter4x4<float, float>(cudnn::winograd_nonfused::WinogradFilterParams<float, float>)(H* 2 @8ΐͺ@ΐͺHΐͺXb8gradient_tape/model/conv2d_12/Conv2D/Conv2DBackpropInputhu  ΘB
ύ
±void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 4, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorSlicingOp<Eigen::DSizes<int, 4> const, Eigen::DSizes<int, 4> const, Eigen::TensorMap<Eigen::Tensor<float const, 4, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 4, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorSlicingOp<Eigen::DSizes<int, 4> const, Eigen::DSizes<int, 4> const, Eigen::TensorMap<Eigen::Tensor<float const, 4, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int)*298¦@¦H¦b'gradient_tape/model/concatenate_2/SlicehuZUB
?
±void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 4, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorSlicingOp<Eigen::DSizes<int, 4> const, Eigen::DSizes<int, 4> const, Eigen::TensorMap<Eigen::Tensor<float const, 4, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 4, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorSlicingOp<Eigen::DSizes<int, 4> const, Eigen::DSizes<int, 4> const, Eigen::TensorMap<Eigen::Tensor<float const, 4, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int)*298@Hb)gradient_tape/model/concatenate_2/Slice_1huZUB
κ
«void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorSlicingOp<Eigen::array<int, 2ul> const, Eigen::array<int, 2ul> const, Eigen::TensorMap<Eigen::Tensor<float, 2, 1, int>, 16, Eigen::MakePointer> >, Eigen::TensorMap<Eigen::Tensor<float const, 2, 1, int>, 16, Eigen::MakePointer> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorSlicingOp<Eigen::array<int, 2ul> const, Eigen::array<int, 2ul> const, Eigen::TensorMap<Eigen::Tensor<float, 2, 1, int>, 16, Eigen::MakePointer> >, Eigen::TensorMap<Eigen::Tensor<float const, 2, 1, int>, 16, Eigen::MakePointer> const> const, Eigen::GpuDevice>, int)*298ίΞ@ ηHΏηbmodel/concatenate_1/concathuZUB
ΰ
}void cudnn::winograd_nonfused::winogradWgradData4x4<float, float>(cudnn::winograd_nonfused::WinogradDataParams<float, float>)@Α*2@28Ώ€@Ώ€HΏ€Xb9gradient_tape/model/conv2d_12/Conv2D/Conv2DBackpropFilterhuZUB

Yvoid tensorflow::BiasNCHWKernel<float>(int, float const*, float const*, float*, int, int)*2&8Ώ@ΏHΏb model/conv2d_transpose_2/BiasAddhuZUB
α
void foldedNhwcToNchwKernel<float, float, float, true, (cudnnKernelDataType_t)0>(int, int, int, int, int, int, int, float const*, float*, int, int, int, int, int, int, int, int, int, int, float, float, cudnn::reduced_divisor, cudnn::reduced_divisor, cudnn::reduced_divisor)&!* 2228ΏΚ@ΏΚHΏΚb)model/conv2d_transpose_2/conv2d_transposehu  ΘB
ξ
void tensorflow::functor::FillPhiloxRandomKernelLaunch<tensorflow::random::UniformDistribution<tensorflow::random::PhiloxRandom, float> >(unsigned long const*, unsigned long const*, tensorflow::random::PhiloxRandom, tensorflow::random::UniformDistribution<tensorflow::random::PhiloxRandom, float>::ResultElementType*, long, tensorflow::random::UniformDistribution<tensorflow::random::PhiloxRandom, float>)*298 Η@ ΗH Ηb4model/dropout_7/dropout/random_uniform/RandomUniformhuZUB
λ
void cudnn::ops::nchwToNhwcKernel<float, float, float, false, true, (cudnnKernelDataType_t)2>(cudnn::ops::nchw2nhwc_params_t<float>, float const*, float*)'!*2228₯@₯H₯b)model/conv2d_transpose_3/conv2d_transposehu  ΘB
n
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*2Δ8?€@?€H?€b)gradient_tape/model/dropout_6/dropout/MulhuZUB
`
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*2Δ8ί€@ί€Hί€bmodel/dropout_6/dropout/MulhuZUB

¨void tensorflow::functor::SwapDimension1And2InTensor3UsingTiles<unsigned int, 256, 32, 32, false>(unsigned int const*, tensorflow::functor::Dimension<3>, unsigned int*) !*2N8£@£H£bCmodel/dropout_6/dropout/Mul_1-0-TransposeNHWCToNCHW-LayoutOptimizerhu  ΘB
γ
void cub::DeviceSegmentedReduceKernel<cub::DeviceReducePolicy<float, float, int, cub::Sum>::Policy600, float const*, float*, cub::TransformInputIterator<int, tensorflow::functor::RowOffset, cub::CountingInputIterator<int, long>, long>, int, cub::Sum, float>(float const*, float*, cub::TransformInputIterator<int, tensorflow::functor::RowOffset, cub::CountingInputIterator<int, long>, long>, cub::TransformInputIterator<int, tensorflow::functor::RowOffset, cub::CountingInputIterator<int, long>, long>, int, cub::Sum, float) 0*28? @? H? b0gradient_tape/model/conv2d_3/BiasAdd/BiasAddGradhu  ΘB
α
void cudnn::winograd_nonfused::winogradForwardData4x4<float, float>(cudnn::winograd_nonfused::WinogradDataParams<float, float>)@±*28? @? H? Xb7gradient_tape/model/conv2d_7/Conv2D/Conv2DBackpropInputhuZUB
β
void cudnn::winograd_nonfused::winogradForwardData4x4<float, float>(cudnn::winograd_nonfused::WinogradDataParams<float, float>)@±*28 @ H Xb8gradient_tape/model/conv2d_15/Conv2D/Conv2DBackpropInputhuZUB
δ
void cub::DeviceSegmentedReduceKernel<cub::DeviceReducePolicy<float, float, int, cub::Sum>::Policy600, float const*, float*, cub::TransformInputIterator<int, tensorflow::functor::RowOffset, cub::CountingInputIterator<int, long>, long>, int, cub::Sum, float>(float const*, float*, cub::TransformInputIterator<int, tensorflow::functor::RowOffset, cub::CountingInputIterator<int, long>, long>, cub::TransformInputIterator<int, tensorflow::functor::RowOffset, cub::CountingInputIterator<int, long>, long>, int, cub::Sum, float) 0*28@Hb1gradient_tape/model/conv2d_18/BiasAdd/BiasAddGradhu  ΘB
α
void cudnn::winograd_nonfused::winogradForwardData4x4<float, float>(cudnn::winograd_nonfused::WinogradDataParams<float, float>)@±*28ί@ίHίXb7gradient_tape/model/conv2d_6/Conv2D/Conv2DBackpropInputhuZUB
»
void cudnn::winograd_nonfused::winogradForwardData4x4<float, float>(cudnn::winograd_nonfused::WinogradDataParams<float, float>)@±*28ΰ@ΰHΰbmodel/conv2d_7/ReluhuZUB
β
void cudnn::winograd_nonfused::winogradForwardData4x4<float, float>(cudnn::winograd_nonfused::WinogradDataParams<float, float>)@±*28Ώ@ΏHΏXb8gradient_tape/model/conv2d_14/Conv2D/Conv2DBackpropInputhuZUB
ώ
void cudnn::ops::nhwcToNchwKernel<float, float, float, true, false, (cudnnKernelDataType_t)0>(cudnn::ops::nhwc2nchw_params_t<float>, float const*, float*)&!*2 8 @ H Xb9gradient_tape/model/conv2d_11/Conv2D/Conv2DBackpropFilterhu  ΘB
Α
void cudnn::winograd_nonfused::winogradForwardOutput4x4<float, float>(cudnn::winograd_nonfused::WinogradOutputParams<float, float>)0*28@Hbmodel/conv2d_15/Reluhu¦ͺ¦B
ΐ
void cudnn::winograd_nonfused::winogradForwardOutput4x4<float, float>(cudnn::winograd_nonfused::WinogradOutputParams<float, float>)0*28@Hbmodel/conv2d_6/Reluhu¦ͺ¦B
θ
void cudnn::winograd_nonfused::winogradWgradOutput4x4<float, float>(cudnn::winograd_nonfused::WinogradWgradOutputParams<float, float>)=H* 2@8@HXb9gradient_tape/model/conv2d_13/Conv2D/Conv2DBackpropFilterhuZUB
Ό
void cudnn::winograd_nonfused::winogradForwardData4x4<float, float>(cudnn::winograd_nonfused::WinogradDataParams<float, float>)@±*28@Hbmodel/conv2d_15/ReluhuZUB
η
void cudnn::winograd_nonfused::winogradForwardOutput4x4<float, float>(cudnn::winograd_nonfused::WinogradOutputParams<float, float>)0*28?@?H?Xb8gradient_tape/model/conv2d_15/Conv2D/Conv2DBackpropInputhu¦ͺ¦B
ζ
void cudnn::winograd_nonfused::winogradForwardOutput4x4<float, float>(cudnn::winograd_nonfused::WinogradOutputParams<float, float>)0*28@HXb7gradient_tape/model/conv2d_7/Conv2D/Conv2DBackpropInputhu¦ͺ¦B
ΐ
void cudnn::winograd_nonfused::winogradForwardOutput4x4<float, float>(cudnn::winograd_nonfused::WinogradOutputParams<float, float>)0*28@Hbmodel/conv2d_7/Reluhu¦ͺ¦B
Α
void cudnn::winograd_nonfused::winogradForwardOutput4x4<float, float>(cudnn::winograd_nonfused::WinogradOutputParams<float, float>)0*28@Hbmodel/conv2d_14/Reluhu¦ͺ¦B
δ
void cub::DeviceSegmentedReduceKernel<cub::DeviceReducePolicy<float, float, int, cub::Sum>::Policy600, float const*, float*, cub::TransformInputIterator<int, tensorflow::functor::RowOffset, cub::CountingInputIterator<int, long>, long>, int, cub::Sum, float>(float const*, float*, cub::TransformInputIterator<int, tensorflow::functor::RowOffset, cub::CountingInputIterator<int, long>, long>, cub::TransformInputIterator<int, tensorflow::functor::RowOffset, cub::CountingInputIterator<int, long>, long>, int, cub::Sum, float) 0*28ί@ίHίb1gradient_tape/model/conv2d_19/BiasAdd/BiasAddGradhu  ΘB
ν
void cub::DeviceSegmentedReduceKernel<cub::DeviceReducePolicy<float, float, int, cub::Sum>::Policy600, float const*, float*, cub::TransformInputIterator<int, tensorflow::functor::RowOffset, cub::CountingInputIterator<int, long>, long>, int, cub::Sum, float>(float const*, float*, cub::TransformInputIterator<int, tensorflow::functor::RowOffset, cub::CountingInputIterator<int, long>, long>, cub::TransformInputIterator<int, tensorflow::functor::RowOffset, cub::CountingInputIterator<int, long>, long>, int, cub::Sum, float) 0*28ί@ίHίb:gradient_tape/model/conv2d_transpose_3/BiasAdd/BiasAddGradhu  ΘB
γ
void cub::DeviceSegmentedReduceKernel<cub::DeviceReducePolicy<float, float, int, cub::Sum>::Policy600, float const*, float*, cub::TransformInputIterator<int, tensorflow::functor::RowOffset, cub::CountingInputIterator<int, long>, long>, int, cub::Sum, float>(float const*, float*, cub::TransformInputIterator<int, tensorflow::functor::RowOffset, cub::CountingInputIterator<int, long>, long>, cub::TransformInputIterator<int, tensorflow::functor::RowOffset, cub::CountingInputIterator<int, long>, long>, int, cub::Sum, float) 0*28ί@ίHίb0gradient_tape/model/conv2d_2/BiasAdd/BiasAddGradhu  ΘB
©
Lvoid cudnn::ops::scalePackedTensor_kernel<float, float>(long, float*, float)*2??8?ω@?ωH?ωb7gradient_tape/model/max_pooling2d_1/MaxPool/MaxPoolGradhu  ΘB
ύ
void cudnn::ops::nchwToNhwcKernel<float, float, float, false, true, (cudnnKernelDataType_t)2>(cudnn::ops::nchw2nhwc_params_t<float>, float const*, float*)'!*2 8Ώω@ΏωHΏωXb8gradient_tape/model/conv2d_11/Conv2D/Conv2DBackpropInputhu  ΘB
Χ
void cudnn::ops::nchwToNhwcKernel<float, float, float, false, true, (cudnnKernelDataType_t)2>(cudnn::ops::nchw2nhwc_params_t<float>, float const*, float*)'!*2 8ψ@ψHψbmodel/conv2d_11/Reluhu  ΘB
η
void cudnn::winograd_nonfused::winogradWgradOutput4x4<float, float>(cudnn::winograd_nonfused::WinogradWgradOutputParams<float, float>)=H* 2@8ίε@ίεHίεXb8gradient_tape/model/conv2d_9/Conv2D/Conv2DBackpropFilterhuZUB
α
void cudnn::winograd_nonfused::winogradWgradDelta4x4<float, float>(cudnn::winograd_nonfused::WinogradDeltaParams<float, float>)@*2028ΰ
@ΰ
Hΰ
Xb8gradient_tape/model/conv2d_7/Conv2D/Conv2DBackpropFilterhuZUB

Υvoid Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<float, Eigen::TensorMap<Eigen::Tensor<bool const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<float, Eigen::TensorMap<Eigen::Tensor<bool const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*298Ώ
@Ώ
HΏ
bmodel/dropout_6/dropout/CasthuZUB
?
Υvoid Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<float, Eigen::TensorMap<Eigen::Tensor<bool const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<float, Eigen::TensorMap<Eigen::Tensor<bool const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*298Ώ
@Ώ
HΏ
bXmodel/dropout_6/dropout/ArithmeticOptimizer/ReorderCastLikeAndValuePreserving_float_CasthuZUB
β
void cudnn::winograd_nonfused::winogradWgradDelta4x4<float, float>(cudnn::winograd_nonfused::WinogradDeltaParams<float, float>)@*2028Ώ
@Ώ
HΏ
Xb9gradient_tape/model/conv2d_14/Conv2D/Conv2DBackpropFilterhuZUB
β
void cudnn::winograd_nonfused::winogradWgradDelta4x4<float, float>(cudnn::winograd_nonfused::WinogradDeltaParams<float, float>)@*2028ί
@ί
Hί
Xb9gradient_tape/model/conv2d_15/Conv2D/Conv2DBackpropFilterhuZUB
α
void cudnn::winograd_nonfused::winogradWgradDelta4x4<float, float>(cudnn::winograd_nonfused::WinogradDeltaParams<float, float>)@*2028Ώ
@Ώ
HΏ
Xb8gradient_tape/model/conv2d_6/Conv2D/Conv2DBackpropFilterhuZUB
?
΅void tensorflow::functor::ApplyAdamKernel<float>(int, float*, float*, float*, float const*, float const*, float const*, float const*, float const*, float const*, float const*, bool) *2&8
@
H
b%Adam/Adam/update_26/ResourceApplyAdamhuZUB
ί
}void cudnn::winograd_nonfused::winogradWgradData4x4<float, float>(cudnn::winograd_nonfused::WinogradDataParams<float, float>)@Α*228ί
@ί
Hί
Xb8gradient_tape/model/conv2d_4/Conv2D/Conv2DBackpropFilterhuZUB
?
΅void tensorflow::functor::ApplyAdamKernel<float>(int, float*, float*, float*, float const*, float const*, float const*, float const*, float const*, float const*, float const*, bool) *2&8ί
@ί
Hί
b%Adam/Adam/update_38/ResourceApplyAdamhuZUB
ί
}void cudnn::winograd_nonfused::winogradWgradData4x4<float, float>(cudnn::winograd_nonfused::WinogradDataParams<float, float>)@Α*2028Ώ
@Ώ
HΏ
Xb8gradient_tape/model/conv2d_7/Conv2D/Conv2DBackpropFilterhuZUB
ΰ
}void cudnn::winograd_nonfused::winogradWgradData4x4<float, float>(cudnn::winograd_nonfused::WinogradDataParams<float, float>)@Α*2028
@
H
Xb9gradient_tape/model/conv2d_15/Conv2D/Conv2DBackpropFilterhuZUB
Ύ
void tensorflow::functor::ShuffleInTensor3Simple<float, 2, 1, 0, false>(int, float const*, tensorflow::functor::Dimension<3>, float*)*2&8ψ	@ψ	Hψ	bmodel/conv2d_12/ReluhuZUB
Ύ
void tensorflow::functor::ShuffleInTensor3Simple<float, 2, 1, 0, false>(int, float const*, tensorflow::functor::Dimension<3>, float*)*2&8?φ	@?φ	H?φ	bmodel/conv2d_10/ReluhuZUB
δ
void tensorflow::functor::ShuffleInTensor3Simple<float, 2, 1, 0, false>(int, float const*, tensorflow::functor::Dimension<3>, float*)*2&8ΰς	@ΰς	Hΰς	Xb8gradient_tape/model/conv2d_12/Conv2D/Conv2DBackpropInputhuZUB
δ
void tensorflow::functor::ShuffleInTensor3Simple<float, 2, 1, 0, false>(int, float const*, tensorflow::functor::Dimension<3>, float*)*2&8ΐγ	@ΐγ	Hΐγ	Xb8gradient_tape/model/conv2d_10/Conv2D/Conv2DBackpropInputhuZUB
ο
void cudnn::bn_bw_1C11_singleread<float, 512, true, 1, 2, 0>(float, float, float, float, cudnnTensorStruct, float const*, cudnnTensorStruct, float const*, cudnnTensorStruct, float*, float const*, float*, float*, float const*, float const*, float, cudnn::reduced_divisor, int, cudnn::reduced_divisor, cudnn::bnBwPersistentState*, int, float, float, float, int, float, cudnnStatus_t*, bool)( ΐ*28ΏΝ	@ΏΝ	HΏΝ	b>gradient_tape/model/batch_normalization_4/FusedBatchNormGradV3huZUB
λ
‘void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_product_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorConversionOp<float, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_cmp_op<float const, float const, (Eigen::internal::ComparisonName)5>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorCwiseNullaryOp<Eigen::internal::scalar_constant_op<float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_product_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorConversionOp<float, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_cmp_op<float const, float const, (Eigen::internal::ComparisonName)5>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorCwiseNullaryOp<Eigen::internal::scalar_constant_op<float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const> const> const> const, Eigen::GpuDevice>, long)*298ΰΗ	@ΰΗ	HΰΗ	b%gradient_tape/model/conv2d_7/ReluGradhuZUB
μ
‘void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_product_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorConversionOp<float, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_cmp_op<float const, float const, (Eigen::internal::ComparisonName)5>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorCwiseNullaryOp<Eigen::internal::scalar_constant_op<float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_product_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorConversionOp<float, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_cmp_op<float const, float const, (Eigen::internal::ComparisonName)5>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorCwiseNullaryOp<Eigen::internal::scalar_constant_op<float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const> const> const> const, Eigen::GpuDevice>, long)*298Η	@Η	HΗ	b&gradient_tape/model/conv2d_14/ReluGradhuZUB
λ
‘void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_product_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorConversionOp<float, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_cmp_op<float const, float const, (Eigen::internal::ComparisonName)5>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorCwiseNullaryOp<Eigen::internal::scalar_constant_op<float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_product_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorConversionOp<float, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_cmp_op<float const, float const, (Eigen::internal::ComparisonName)5>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorCwiseNullaryOp<Eigen::internal::scalar_constant_op<float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const> const> const> const, Eigen::GpuDevice>, long)*298ΐΖ	@ΐΖ	HΐΖ	b%gradient_tape/model/conv2d_6/ReluGradhuZUB
μ
‘void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_product_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorConversionOp<float, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_cmp_op<float const, float const, (Eigen::internal::ComparisonName)5>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorCwiseNullaryOp<Eigen::internal::scalar_constant_op<float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_product_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorConversionOp<float, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_cmp_op<float const, float const, (Eigen::internal::ComparisonName)5>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorCwiseNullaryOp<Eigen::internal::scalar_constant_op<float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const> const> const> const, Eigen::GpuDevice>, long)*298ΐΓ	@ΐΓ	HΐΓ	b&gradient_tape/model/conv2d_15/ReluGradhuZUB
©
Υvoid cudnn::bn_fw_tr_1C11_kernel_NCHW<float, float, 128, true, 1>(cudnnTensorStruct, float const*, cudnnTensorStruct, float*, float const*, float const*, float, float, float*, float*, float*, float*, float, float)(*2@8ίΏ	@ίΏ	HίΏ	b,model/batch_normalization_1/FusedBatchNormV3hu  ΘB
β
«void tensorflow::functor::SwapDimension1And2InTensor3UsingTiles<unsigned char, 256, 32, 32, false>(unsigned char const*, tensorflow::functor::Dimension<3>, unsigned char*)  *2 8  	@  	H  	bgradient_tape/model/dropout_7/dropout/ArithmeticOptimizer/ReorderCastLikeAndValuePreserving_bool_Mul_2-1-TransposeNHWCToNCHW-LayoutOptimizerhu  ΘB
©
οvoid Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*298ΰ	@ΰ	Hΰ	bAdam/gradients/AddN_2huZUB
p
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*2β	8	@	H	b+gradient_tape/model/dropout_5/dropout/Mul_2huZUB
n
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*2β	8ΰ	@ΰ	Hΰ	b)gradient_tape/model/dropout/dropout/Mul_2huZUB
b
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*2β	8ΰ	@ΰ	Hΰ	bmodel/dropout_5/dropout/Mul_1huZUB
`
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*2β	8ΐ	@ΐ	Hΐ	bmodel/dropout/dropout/Mul_1huZUB
?
΅void tensorflow::functor::ApplyAdamKernel<float>(int, float*, float*, float*, float const*, float const*, float const*, float const*, float const*, float const*, float const*, bool) *2&8ϋ@ϋHϋb%Adam/Adam/update_32/ResourceApplyAdamhuZUB

΄void cudnn::pooling_bw_kernel_max_nchw_fully_packed_small<float, float, 2, (cudnnNanPropagation_t)0>(cudnnTensorStruct, float const*, cudnnTensorStruct, float const*, cudnnTensorStruct, float const*, cudnnTensorStruct, float*, cudnnPoolingStruct, float, float, cudnn::reduced_divisor, cudnn::reduced_divisor)  *228 ε@ εH εb7gradient_tape/model/max_pooling2d_4/MaxPool/MaxPoolGradhu  ΘB
ε
ώvoid cudnn::bn_bw_1C11_kernel_new<float, float, float2, 32, true, 1>(float, float, float, float, cudnnTensorStruct, float const*, cudnnTensorStruct, float const*, cudnnTensorStruct, float*, float const*, float*, float*, float const*, float const*, float)(*28?Σ@?ΣH?Σb>gradient_tape/model/batch_normalization_2/FusedBatchNormGradV3hu  ΘB
α
void cudnn::winograd_nonfused::winogradForwardData4x4<float, float>(cudnn::winograd_nonfused::WinogradDataParams<float, float>)@±*28Ώ@ΏHΏXb7gradient_tape/model/conv2d_8/Conv2D/Conv2DBackpropInputhuZUB
β
void cudnn::winograd_nonfused::winogradForwardData4x4<float, float>(cudnn::winograd_nonfused::WinogradDataParams<float, float>)@±*28?@?H?Xb8gradient_tape/model/conv2d_13/Conv2D/Conv2DBackpropInputhuZUB
β
void cudnn::winograd_nonfused::winogradForwardData4x4<float, float>(cudnn::winograd_nonfused::WinogradDataParams<float, float>)@±*28Ώ@ΏHΏXb8gradient_tape/model/conv2d_12/Conv2D/Conv2DBackpropInputhuZUB
»
void cudnn::winograd_nonfused::winogradForwardData4x4<float, float>(cudnn::winograd_nonfused::WinogradDataParams<float, float>)@±*28?@?H?bmodel/conv2d_9/ReluhuZUB
α
void cudnn::winograd_nonfused::winogradForwardData4x4<float, float>(cudnn::winograd_nonfused::WinogradDataParams<float, float>)@±*28Ώ@ΏHΏXb7gradient_tape/model/conv2d_9/Conv2D/Conv2DBackpropInputhuZUB
Ό
void cudnn::winograd_nonfused::winogradForwardData4x4<float, float>(cudnn::winograd_nonfused::WinogradDataParams<float, float>)@±*28@Hbmodel/conv2d_13/ReluhuZUB

΄void cudnn::pooling_bw_kernel_max_nchw_fully_packed_small<float, float, 2, (cudnnNanPropagation_t)0>(cudnnTensorStruct, float const*, cudnnTensorStruct, float const*, cudnnTensorStruct, float const*, cudnnTensorStruct, float*, cudnnPoolingStruct, float, float, cudnn::reduced_divisor, cudnn::reduced_divisor) *228?@?H?b7gradient_tape/model/max_pooling2d_3/MaxPool/MaxPoolGradhu  ΘB
η
void cudnn::winograd_nonfused::winogradForwardOutput4x4<float, float>(cudnn::winograd_nonfused::WinogradOutputParams<float, float>)0*28 @ H Xb8gradient_tape/model/conv2d_13/Conv2D/Conv2DBackpropInputhu¦ͺ¦B
ζ
void cudnn::winograd_nonfused::winogradForwardOutput4x4<float, float>(cudnn::winograd_nonfused::WinogradOutputParams<float, float>)0*28?@?H?Xb7gradient_tape/model/conv2d_9/Conv2D/Conv2DBackpropInputhu¦ͺ¦B
ΐ
void cudnn::winograd_nonfused::winogradForwardOutput4x4<float, float>(cudnn::winograd_nonfused::WinogradOutputParams<float, float>)0*28?@?H?bmodel/conv2d_8/Reluhu¦ͺ¦B
Α
void cudnn::winograd_nonfused::winogradForwardOutput4x4<float, float>(cudnn::winograd_nonfused::WinogradOutputParams<float, float>)0*28@Hbmodel/conv2d_13/Reluhu¦ͺ¦B
Α
void cudnn::winograd_nonfused::winogradForwardOutput4x4<float, float>(cudnn::winograd_nonfused::WinogradOutputParams<float, float>)0*28 @ H bmodel/conv2d_12/Reluhu¦ͺ¦B
ΐ
void cudnn::winograd_nonfused::winogradForwardOutput4x4<float, float>(cudnn::winograd_nonfused::WinogradOutputParams<float, float>)0*28ί@ίHίbmodel/conv2d_9/Reluhu¦ͺ¦B
φ
void tensorflow::functor::ShuffleInTensor3Simple<float, 2, 1, 0, false>(int, float const*, tensorflow::functor::Dimension<3>, float*)*2&8 @ H XbJgradient_tape/model/conv2d_transpose/conv2d_transpose/Conv2DBackpropFilterhuZUB
β
void cudnn::ops::pooling_fw_4d_kernel<float, float, cudnn::maxpooling_func<float, (cudnnNanPropagation_t)0>, (cudnnPoolingMode_t)0, false>(cudnnTensorStruct, float const*, cudnnTensorStruct, float*, cudnnPoolingStruct, float, float, int, cudnn::reduced_divisor, cudnn::reduced_divisor)( *Θ2 28ά@άHάbmodel/max_pooling2d_2/MaxPoolhu @B
―
ͺvoid tensorflow::functor::SwapDimension1And2InTensor3UsingTiles<unsigned int, 1024, 1024, 2, false>(unsigned int const*, tensorflow::functor::Dimension<3>, unsigned int*) `*2Δ8ΐ?@ΐ?Hΐ?b\gradient_tape/model/conv2d/Conv2D/Conv2DBackpropFilter-0-TransposeNHWCToNCHW-LayoutOptimizerhuZUB
q
(GreaterEqual_GPU_DT_FLOAT_DT_BOOL_kernel*2Δ8ΐΡ@ΐΡHΐΡb$model/dropout_6/dropout/GreaterEqualhuZUB
ύ
±void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 4, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorSlicingOp<Eigen::DSizes<int, 4> const, Eigen::DSizes<int, 4> const, Eigen::TensorMap<Eigen::Tensor<float const, 4, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 4, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorSlicingOp<Eigen::DSizes<int, 4> const, Eigen::DSizes<int, 4> const, Eigen::TensorMap<Eigen::Tensor<float const, 4, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int)*298 £@ £H £b'gradient_tape/model/concatenate_1/SlicehuZUB
γ
void cudnn::winograd_nonfused::winogradForwardFilter4x4<float, float>(cudnn::winograd_nonfused::WinogradFilterParams<float, float>)(H* 2@8ΐ@ΐHΐXb7gradient_tape/model/conv2d_9/Conv2D/Conv2DBackpropInputhu  ΘB
?
±void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 4, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorSlicingOp<Eigen::DSizes<int, 4> const, Eigen::DSizes<int, 4> const, Eigen::TensorMap<Eigen::Tensor<float const, 4, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 4, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorSlicingOp<Eigen::DSizes<int, 4> const, Eigen::DSizes<int, 4> const, Eigen::TensorMap<Eigen::Tensor<float const, 4, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int)*298?@?H?b)gradient_tape/model/concatenate_1/Slice_1huZUB
δ
void cudnn::winograd_nonfused::winogradForwardFilter4x4<float, float>(cudnn::winograd_nonfused::WinogradFilterParams<float, float>)(H* 2@8Ώ@ΏHΏXb8gradient_tape/model/conv2d_13/Conv2D/Conv2DBackpropInputhu  ΘB
Ύ
void cudnn::winograd_nonfused::winogradForwardFilter4x4<float, float>(cudnn::winograd_nonfused::WinogradFilterParams<float, float>)(H* 2@8ί@ίHίbmodel/conv2d_13/Reluhu  ΘB
½
void cudnn::winograd_nonfused::winogradForwardFilter4x4<float, float>(cudnn::winograd_nonfused::WinogradFilterParams<float, float>)(H* 2@8ΰ@ΰHΰbmodel/conv2d_9/Reluhu  ΘB
θ
«void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorSlicingOp<Eigen::array<int, 2ul> const, Eigen::array<int, 2ul> const, Eigen::TensorMap<Eigen::Tensor<float, 2, 1, int>, 16, Eigen::MakePointer> >, Eigen::TensorMap<Eigen::Tensor<float const, 2, 1, int>, 16, Eigen::MakePointer> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorSlicingOp<Eigen::array<int, 2ul> const, Eigen::array<int, 2ul> const, Eigen::TensorMap<Eigen::Tensor<float, 2, 1, int>, 16, Eigen::MakePointer> >, Eigen::TensorMap<Eigen::Tensor<float const, 2, 1, int>, 16, Eigen::MakePointer> const> const, Eigen::GpuDevice>, int)*298ΐύ@ΐ½Hΐbmodel/concatenate/concathuZUB
β
void cudnn::winograd_nonfused::winogradWgradDelta4x4<float, float>(cudnn::winograd_nonfused::WinogradDeltaParams<float, float>)@*2 28Ώά@ΏάHΏάXb9gradient_tape/model/conv2d_13/Conv2D/Conv2DBackpropFilterhuZUB

Yvoid tensorflow::BiasNCHWKernel<float>(int, float const*, float const*, float*, int, int)*2&8ΏΫ@ΏΫHΏΫb model/conv2d_transpose_1/BiasAddhuZUB
β
void cudnn::winograd_nonfused::winogradForwardData4x4<float, float>(cudnn::winograd_nonfused::WinogradDataParams<float, float>)@±*28?Ψ@?ΨH?ΨXb8gradient_tape/model/conv2d_10/Conv2D/Conv2DBackpropInputhuZUB
α
void cudnn::winograd_nonfused::winogradWgradDelta4x4<float, float>(cudnn::winograd_nonfused::WinogradDeltaParams<float, float>)@*2 28?Ψ@?ΨH?ΨXb8gradient_tape/model/conv2d_8/Conv2D/Conv2DBackpropFilterhuZUB
ί
}void cudnn::winograd_nonfused::winogradWgradData4x4<float, float>(cudnn::winograd_nonfused::WinogradDataParams<float, float>)@Α*2 28Ψ@ΨHΨXb8gradient_tape/model/conv2d_9/Conv2D/Conv2DBackpropFilterhuZUB
β
void cudnn::winograd_nonfused::winogradWgradDelta4x4<float, float>(cudnn::winograd_nonfused::WinogradDeltaParams<float, float>)@*2 28 Υ@ ΥH ΥXb9gradient_tape/model/conv2d_12/Conv2D/Conv2DBackpropFilterhuZUB
ΰ
}void cudnn::winograd_nonfused::winogradWgradData4x4<float, float>(cudnn::winograd_nonfused::WinogradDataParams<float, float>)@Α*2 28 Τ@ ΤH ΤXb9gradient_tape/model/conv2d_13/Conv2D/Conv2DBackpropFilterhuZUB
α
void cudnn::winograd_nonfused::winogradWgradDelta4x4<float, float>(cudnn::winograd_nonfused::WinogradDeltaParams<float, float>)@*2 28Σ@ΣHΣXb8gradient_tape/model/conv2d_9/Conv2D/Conv2DBackpropFilterhuZUB
Α
θvoid cudnn::bn_fw_tr_1C11_singleread<float, 512, true, 1, 2, 0>(cudnnTensorStruct, float const*, cudnnTensorStruct, float*, float const*, float const*, float, float, float*, float*, float*, float*, float, float, cudnn::reduced_divisor, int, cudnn::reduced_divisor, cudnn::bnFwPersistentState*, int, float, float, float, int, float, float, cudnnStatus_t*, bool)'  *28ΰΞ@ΰΞHΰΞb,model/batch_normalization_4/FusedBatchNormV3hu  ΘB
θ
void cudnn::winograd_nonfused::winogradWgradOutput4x4<float, float>(cudnn::winograd_nonfused::WinogradWgradOutputParams<float, float>)=H* 2 8ΏΊ@ΏΊHΏΊXb9gradient_tape/model/conv2d_14/Conv2D/Conv2DBackpropFilterhuZUB
α
void foldedNhwcToNchwKernel<float, float, float, true, (cudnnKernelDataType_t)0>(int, int, int, int, int, int, int, float const*, float*, int, int, int, int, int, int, int, int, int, int, float, float, cudnn::reduced_divisor, cudnn::reduced_divisor, cudnn::reduced_divisor)&!* 228 °@ °H °b)model/conv2d_transpose_1/conv2d_transposehu  ΘB
δ
void cub::DeviceSegmentedReduceKernel<cub::DeviceReducePolicy<float, float, int, cub::Sum>::Policy600, float const*, float*, cub::TransformInputIterator<int, tensorflow::functor::RowOffset, cub::CountingInputIterator<int, long>, long>, int, cub::Sum, float>(float const*, float*, cub::TransformInputIterator<int, tensorflow::functor::RowOffset, cub::CountingInputIterator<int, long>, long>, cub::TransformInputIterator<int, tensorflow::functor::RowOffset, cub::CountingInputIterator<int, long>, long>, int, cub::Sum, float) 0*228Ώ―@Ώ―HΏ―b1gradient_tape/model/conv2d_17/BiasAdd/BiasAddGradhu  ΘB
δ
void cub::DeviceSegmentedReduceKernel<cub::DeviceReducePolicy<float, float, int, cub::Sum>::Policy600, float const*, float*, cub::TransformInputIterator<int, tensorflow::functor::RowOffset, cub::CountingInputIterator<int, long>, long>, int, cub::Sum, float>(float const*, float*, cub::TransformInputIterator<int, tensorflow::functor::RowOffset, cub::CountingInputIterator<int, long>, long>, cub::TransformInputIterator<int, tensorflow::functor::RowOffset, cub::CountingInputIterator<int, long>, long>, int, cub::Sum, float) 0*228―@―H―b1gradient_tape/model/conv2d_16/BiasAdd/BiasAddGradhu  ΘB
γ
void cub::DeviceSegmentedReduceKernel<cub::DeviceReducePolicy<float, float, int, cub::Sum>::Policy600, float const*, float*, cub::TransformInputIterator<int, tensorflow::functor::RowOffset, cub::CountingInputIterator<int, long>, long>, int, cub::Sum, float>(float const*, float*, cub::TransformInputIterator<int, tensorflow::functor::RowOffset, cub::CountingInputIterator<int, long>, long>, cub::TransformInputIterator<int, tensorflow::functor::RowOffset, cub::CountingInputIterator<int, long>, long>, int, cub::Sum, float) 0*228ί­@ί­Hί­b0gradient_tape/model/conv2d_5/BiasAdd/BiasAddGradhu  ΘB
γ
void cub::DeviceSegmentedReduceKernel<cub::DeviceReducePolicy<float, float, int, cub::Sum>::Policy600, float const*, float*, cub::TransformInputIterator<int, tensorflow::functor::RowOffset, cub::CountingInputIterator<int, long>, long>, int, cub::Sum, float>(float const*, float*, cub::TransformInputIterator<int, tensorflow::functor::RowOffset, cub::CountingInputIterator<int, long>, long>, cub::TransformInputIterator<int, tensorflow::functor::RowOffset, cub::CountingInputIterator<int, long>, long>, int, cub::Sum, float) 0*228«@«H«b0gradient_tape/model/conv2d_4/BiasAdd/BiasAddGradhu  ΘB
ν
void cub::DeviceSegmentedReduceKernel<cub::DeviceReducePolicy<float, float, int, cub::Sum>::Policy600, float const*, float*, cub::TransformInputIterator<int, tensorflow::functor::RowOffset, cub::CountingInputIterator<int, long>, long>, int, cub::Sum, float>(float const*, float*, cub::TransformInputIterator<int, tensorflow::functor::RowOffset, cub::CountingInputIterator<int, long>, long>, cub::TransformInputIterator<int, tensorflow::functor::RowOffset, cub::CountingInputIterator<int, long>, long>, int, cub::Sum, float) 0*228ΰ₯@ΰ₯Hΰ₯b:gradient_tape/model/conv2d_transpose_2/BiasAdd/BiasAddGradhu  ΘB
ξ
void tensorflow::functor::FillPhiloxRandomKernelLaunch<tensorflow::random::UniformDistribution<tensorflow::random::PhiloxRandom, float> >(unsigned long const*, unsigned long const*, tensorflow::random::PhiloxRandom, tensorflow::random::UniformDistribution<tensorflow::random::PhiloxRandom, float>::ResultElementType*, long, tensorflow::random::UniformDistribution<tensorflow::random::PhiloxRandom, float>)*298?’@?’H?’b4model/dropout_6/dropout/random_uniform/RandomUniformhuZUB
η
void cudnn::winograd_nonfused::winogradWgradOutput4x4<float, float>(cudnn::winograd_nonfused::WinogradWgradOutputParams<float, float>)=H* 2@8?‘@?‘H?‘Xb8gradient_tape/model/conv2d_8/Conv2D/Conv2DBackpropFilterhuZUB
λ
void cudnn::ops::nchwToNhwcKernel<float, float, float, false, true, (cudnnKernelDataType_t)2>(cudnn::ops::nchw2nhwc_params_t<float>, float const*, float*)'!*228ί @ί Hί b)model/conv2d_transpose_2/conv2d_transposehu  ΘB

¨void tensorflow::functor::SwapDimension1And2InTensor3UsingTiles<unsigned int, 256, 32, 32, false>(unsigned int const*, tensorflow::functor::Dimension<3>, unsigned int*) !*2Π(8ΐ @ΐ Hΐ bCmodel/dropout_5/dropout/Mul_1-0-TransposeNHWCToNCHW-LayoutOptimizerhu  ΘB
Υ
void cudnn::ops::nchwToNhwcKernel<float, float, float, false, true, (cudnnKernelDataType_t)2>(cudnn::ops::nchw2nhwc_params_t<float>, float const*, float*)'!*2228ΐ@ΐHΐbmodel/conv2d_4/Reluhu  ΘB

void cudnn::ops::nchwToNhwcKernel<float, float, float, false, true, (cudnnKernelDataType_t)2>(cudnn::ops::nchw2nhwc_params_t<float>, float const*, float*)'!*228 @ H Xb>gradient_tape/model/conv2d_transpose_1/conv2d_transpose/Conv2Dhu  ΘB

void cudnn::ops::nchwToNhwcKernel<float, float, float, false, true, (cudnnKernelDataType_t)2>(cudnn::ops::nchw2nhwc_params_t<float>, float const*, float*)'!*228?@?H?XbLgradient_tape/model/conv2d_transpose_1/conv2d_transpose/Conv2DBackpropFilterhu  ΘB
ώ
void cudnn::ops::nhwcToNchwKernel<float, float, float, true, false, (cudnnKernelDataType_t)0>(cudnn::ops::nhwc2nchw_params_t<float>, float const*, float*)&!*28Ώ@ΏHΏXb9gradient_tape/model/conv2d_10/Conv2D/Conv2DBackpropFilterhu  ΘB
`
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*2β	8ΐ@ΐHΐbmodel/dropout_5/dropout/MulhuZUB

¨void tensorflow::functor::SwapDimension1And2InTensor3UsingTiles<unsigned int, 256, 32, 32, false>(unsigned int const*, tensorflow::functor::Dimension<3>, unsigned int*) !*2'8Ώ@ΏHΏbAmodel/dropout/dropout/Mul_1-0-TransposeNHWCToNCHW-LayoutOptimizerhu  ΘB
n
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*2β	8@Hb)gradient_tape/model/dropout_5/dropout/MulhuZUB
l
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*2β	8?@?H?b'gradient_tape/model/dropout/dropout/MulhuZUB
ε
void tensorflow::functor::ShuffleInTensor3Simple<float, 2, 1, 0, false>(int, float const*, tensorflow::functor::Dimension<3>, float*)*2&8 @ H Xb9gradient_tape/model/conv2d_14/Conv2D/Conv2DBackpropFilterhuZUB
ζ
void cudnn::winograd_nonfused::winogradForwardOutput4x4<float, float>(cudnn::winograd_nonfused::WinogradOutputParams<float, float>)0*28@HXb7gradient_tape/model/conv2d_6/Conv2D/Conv2DBackpropInputhu¦ͺ¦B
»
void cudnn::winograd_nonfused::winogradForwardData4x4<float, float>(cudnn::winograd_nonfused::WinogradDataParams<float, float>)@±*28@Hbmodel/conv2d_6/ReluhuZUB
^
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*2β	8ΐ@ΐHΐbmodel/dropout/dropout/MulhuZUB
Χ
void cudnn::ops::nchwToNhwcKernel<float, float, float, false, true, (cudnnKernelDataType_t)2>(cudnn::ops::nchw2nhwc_params_t<float>, float const*, float*)'!*28ΰ@ΰHΰbmodel/conv2d_10/Reluhu  ΘB
δ
void tensorflow::functor::ShuffleInTensor3Simple<float, 2, 1, 0, false>(int, float const*, tensorflow::functor::Dimension<3>, float*)*2&8ΐϋ@ΐϋHΐϋXb8gradient_tape/model/conv2d_8/Conv2D/Conv2DBackpropFilterhuZUB

void cudnn::ops::nhwcToNchwKernel<float, float, float, true, false, (cudnnKernelDataType_t)0>(cudnn::ops::nhwc2nchw_params_t<float>, float const*, float*)&!*28Ώ«@Ώ«HΏ«XbJgradient_tape/model/conv2d_transpose/conv2d_transpose/Conv2DBackpropFilterhu  ΘB
?
΅void tensorflow::functor::ApplyAdamKernel<float>(int, float*, float*, float*, float const*, float const*, float const*, float const*, float const*, float const*, float const*, bool) *2&8@Hb%Adam/Adam/update_44/ResourceApplyAdamhuZUB
Ρ
void nchwToFoldedNhwcKernel<float, float, float, true, (cudnnKernelDataType_t)2>(int, int, int, int, float const*, float*, int, int, int, int, int, int, int, int, int, int, float, float, cudnn::reduced_divisor, cudnn::reduced_divisor, cudnn::reduced_divisor)&!* 28ΰ@ΰHΰb'model/conv2d_transpose/conv2d_transposehu  ΘB
?
Υvoid Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<float, Eigen::TensorMap<Eigen::Tensor<bool const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<float, Eigen::TensorMap<Eigen::Tensor<bool const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*298ί@ίHίbXmodel/dropout_5/dropout/ArithmeticOptimizer/ReorderCastLikeAndValuePreserving_float_CasthuZUB
©
Τvoid cudnn::bn_fw_tr_1C11_kernel_NCHW<float, float, 32, true, 1>(cudnnTensorStruct, float const*, cudnnTensorStruct, float*, float const*, float const*, float, float, float*, float*, float*, float*, float, float)(*28ί@ίHίb,model/batch_normalization_2/FusedBatchNormV3hu  ΘB

Υvoid Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<float, Eigen::TensorMap<Eigen::Tensor<bool const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<float, Eigen::TensorMap<Eigen::Tensor<bool const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*298ΐ@ΐHΐbmodel/dropout_5/dropout/CasthuZUB
Π
Υvoid Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<float, Eigen::TensorMap<Eigen::Tensor<bool const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<float, Eigen::TensorMap<Eigen::Tensor<bool const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*298 @ H bVmodel/dropout/dropout/ArithmeticOptimizer/ReorderCastLikeAndValuePreserving_float_CasthuZUB
?
΅void tensorflow::functor::ApplyAdamKernel<float>(int, float*, float*, float*, float const*, float const*, float const*, float const*, float const*, float const*, float const*, bool) *2&8@Hb%Adam/Adam/update_24/ResourceApplyAdamhuZUB

Υvoid Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<float, Eigen::TensorMap<Eigen::Tensor<bool const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<float, Eigen::TensorMap<Eigen::Tensor<bool const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*298?@?H?bmodel/dropout/dropout/CasthuZUB
ί
}void cudnn::winograd_nonfused::winogradWgradData4x4<float, float>(cudnn::winograd_nonfused::WinogradDataParams<float, float>)@Α*228 @ H Xb8gradient_tape/model/conv2d_6/Conv2D/Conv2DBackpropFilterhuZUB
δ
void tensorflow::functor::ShuffleInTensor3Simple<float, 2, 1, 0, false>(int, float const*, tensorflow::functor::Dimension<3>, float*)*2&8@HXb8gradient_tape/model/conv2d_13/Conv2D/Conv2DBackpropInputhuZUB
Ύ
void tensorflow::functor::ShuffleInTensor3Simple<float, 2, 1, 0, false>(int, float const*, tensorflow::functor::Dimension<3>, float*)*2&8ΐϊ@ΐϊHΐϊbmodel/conv2d_13/ReluhuZUB
½
void tensorflow::functor::ShuffleInTensor3Simple<float, 2, 1, 0, false>(int, float const*, tensorflow::functor::Dimension<3>, float*)*2&8 ψ@ ψH ψbmodel/conv2d_9/ReluhuZUB
γ
void tensorflow::functor::ShuffleInTensor3Simple<float, 2, 1, 0, false>(int, float const*, tensorflow::functor::Dimension<3>, float*)*2&8ΐσ@ΐσHΐσXb7gradient_tape/model/conv2d_9/Conv2D/Conv2DBackpropInputhuZUB
μ
‘void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_product_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorConversionOp<float, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_cmp_op<float const, float const, (Eigen::internal::ComparisonName)5>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorCwiseNullaryOp<Eigen::internal::scalar_constant_op<float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_product_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorConversionOp<float, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_cmp_op<float const, float const, (Eigen::internal::ComparisonName)5>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorCwiseNullaryOp<Eigen::internal::scalar_constant_op<float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const> const> const> const, Eigen::GpuDevice>, long)*298ΰν@ΰνHΰνb&gradient_tape/model/conv2d_12/ReluGradhuZUB
λ
‘void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_product_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorConversionOp<float, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_cmp_op<float const, float const, (Eigen::internal::ComparisonName)5>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorCwiseNullaryOp<Eigen::internal::scalar_constant_op<float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_product_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorConversionOp<float, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_cmp_op<float const, float const, (Eigen::internal::ComparisonName)5>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorCwiseNullaryOp<Eigen::internal::scalar_constant_op<float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const> const> const> const, Eigen::GpuDevice>, long)*298 μ@ μH μb%gradient_tape/model/conv2d_8/ReluGradhuZUB
λ
‘void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_product_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorConversionOp<float, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_cmp_op<float const, float const, (Eigen::internal::ComparisonName)5>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorCwiseNullaryOp<Eigen::internal::scalar_constant_op<float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_product_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorConversionOp<float, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_cmp_op<float const, float const, (Eigen::internal::ComparisonName)5>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorCwiseNullaryOp<Eigen::internal::scalar_constant_op<float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const> const> const> const, Eigen::GpuDevice>, long)*298ΐλ@ΐλHΐλb%gradient_tape/model/conv2d_9/ReluGradhuZUB
μ
‘void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_product_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorConversionOp<float, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_cmp_op<float const, float const, (Eigen::internal::ComparisonName)5>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorCwiseNullaryOp<Eigen::internal::scalar_constant_op<float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_product_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorConversionOp<float, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_cmp_op<float const, float const, (Eigen::internal::ComparisonName)5>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorCwiseNullaryOp<Eigen::internal::scalar_constant_op<float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const> const> const> const, Eigen::GpuDevice>, long)*298ΐι@ΐιHΐιb&gradient_tape/model/conv2d_13/ReluGradhuZUB
α
«void tensorflow::functor::SwapDimension1And2InTensor3UsingTiles<unsigned char, 256, 32, 32, false>(unsigned char const*, tensorflow::functor::Dimension<3>, unsigned char*)  *2N8 α@ αH αbgradient_tape/model/dropout_6/dropout/ArithmeticOptimizer/ReorderCastLikeAndValuePreserving_bool_Mul_2-1-TransposeNHWCToNCHW-LayoutOptimizerhu  ΘB
p
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*2ρ8ά@άHάb+gradient_tape/model/dropout_1/dropout/Mul_2huZUB
p
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*2ρ8ίΧ@ίΧHίΧb+gradient_tape/model/dropout_4/dropout/Mul_2huZUB
©
οvoid Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*298ίΣ@ίΣHίΣbAdam/gradients/AddN_1huZUB
b
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*2ρ8 ?@ ?H ?bmodel/dropout_1/dropout/Mul_1huZUB
b
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*2ρ8Ν@ΝHΝbmodel/dropout_4/dropout/Mul_1huZUB
θ
void tensorflow::functor::ShuffleInTensor3Simple<float, 2, 1, 0, false>(int, float const*, tensorflow::functor::Dimension<3>, float*)*2&8ΐ₯@ΐ₯Hΐ₯Xb<gradient_tape/model/conv2d_transpose/conv2d_transpose/Conv2DhuZUB
Ρ
void tensorflow::functor::ShuffleInTensor3Simple<float, 2, 1, 0, false>(int, float const*, tensorflow::functor::Dimension<3>, float*)*2&8ΐ@ΐHΐb'model/conv2d_transpose/conv2d_transposehuZUB
»
void cudnn::winograd_nonfused::winogradForwardData4x4<float, float>(cudnn::winograd_nonfused::WinogradDataParams<float, float>)@±*28ΰ@ΰHΰbmodel/conv2d_8/ReluhuZUB
ζ
void cudnn::winograd_nonfused::winogradForwardOutput4x4<float, float>(cudnn::winograd_nonfused::WinogradOutputParams<float, float>)0*28ΰ@ΰHΰXb7gradient_tape/model/conv2d_8/Conv2D/Conv2DBackpropInputhu¦ͺ¦B
Η
void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float, float>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float, float>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float, float>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float, float>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*298 ώ@ ώH ώbAdam/gradients/AddNhuZUB
β
void cudnn::ops::pooling_fw_4d_kernel<float, float, cudnn::maxpooling_func<float, (cudnnNanPropagation_t)0>, (cudnnPoolingMode_t)0, false>(cudnnTensorStruct, float const*, cudnnTensorStruct, float*, cudnnPoolingStruct, float, float, int, cudnn::reduced_divisor, cudnn::reduced_divisor)( *Θ2@28 ω@ ωH ωbmodel/max_pooling2d_3/MaxPoolhu @B
o
(GreaterEqual_GPU_DT_FLOAT_DT_BOOL_kernel*2β	8ΐφ@ΐφHΐφb"model/dropout/dropout/GreaterEqualhuZUB
q
(GreaterEqual_GPU_DT_FLOAT_DT_BOOL_kernel*2β	8ς@ςHςb$model/dropout_5/dropout/GreaterEqualhuZUB
γ
void cudnn::winograd_nonfused::winogradForwardFilter4x4<float, float>(cudnn::winograd_nonfused::WinogradFilterParams<float, float>)(H* 2@8 έ@ έH έXb7gradient_tape/model/conv2d_8/Conv2D/Conv2DBackpropInputhu  ΘB
ϋ
±void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 4, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorSlicingOp<Eigen::DSizes<int, 4> const, Eigen::DSizes<int, 4> const, Eigen::TensorMap<Eigen::Tensor<float const, 4, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 4, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorSlicingOp<Eigen::DSizes<int, 4> const, Eigen::DSizes<int, 4> const, Eigen::TensorMap<Eigen::Tensor<float const, 4, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int)*298 ά@ άH άb%gradient_tape/model/concatenate/SlicehuZUB
ύ
±void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 4, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorSlicingOp<Eigen::DSizes<int, 4> const, Eigen::DSizes<int, 4> const, Eigen::TensorMap<Eigen::Tensor<float const, 4, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 4, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorSlicingOp<Eigen::DSizes<int, 4> const, Eigen::DSizes<int, 4> const, Eigen::TensorMap<Eigen::Tensor<float const, 4, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int)*298ΰΫ@ΰΫHΰΫb'gradient_tape/model/concatenate/Slice_1huZUB
δ
void cudnn::winograd_nonfused::winogradForwardFilter4x4<float, float>(cudnn::winograd_nonfused::WinogradFilterParams<float, float>)(H* 2 8 Υ@ ΥH ΥXb8gradient_tape/model/conv2d_14/Conv2D/Conv2DBackpropInputhu  ΘB
½
void cudnn::winograd_nonfused::winogradForwardFilter4x4<float, float>(cudnn::winograd_nonfused::WinogradFilterParams<float, float>)(H* 2@8 Σ@ ΣH Σbmodel/conv2d_8/Reluhu  ΘB
Ύ
void cudnn::winograd_nonfused::winogradForwardFilter4x4<float, float>(cudnn::winograd_nonfused::WinogradFilterParams<float, float>)(H* 2 8ΐΡ@ΐΡHΐΡbmodel/conv2d_14/Reluhu  ΘB

Yvoid tensorflow::BiasNCHWKernel<float>(int, float const*, float const*, float*, int, int)*2&8Ί@ΊHΊbmodel/conv2d_transpose/BiasAddhuZUB
η
void cudnn::winograd_nonfused::winogradForwardOutput4x4<float, float>(cudnn::winograd_nonfused::WinogradOutputParams<float, float>)0*28ΐΉ@ΐΉHΐΉXb8gradient_tape/model/conv2d_10/Conv2D/Conv2DBackpropInputhu¦ͺ¦B
θ
void cudnn::winograd_nonfused::winogradWgradOutput4x4<float, float>(cudnn::winograd_nonfused::WinogradWgradOutputParams<float, float>)=H* 2 8ΐΈ@ΐΈHΐΈXb9gradient_tape/model/conv2d_15/Conv2D/Conv2DBackpropFilterhuZUB

£void tensorflow::functor::RowReduceKernel<float const*, float*, cub::Sum>(float const*, float*, int, int, cub::Sum, std::iterator_traits<float const*>::value_type)*28 ³@ ³H ³b:gradient_tape/model/conv2d_transpose_1/BiasAdd/BiasAddGradhu  ΘB
ί
}void cudnn::winograd_nonfused::winogradWgradData4x4<float, float>(cudnn::winograd_nonfused::WinogradDataParams<float, float>)@Α*228ΰ²@ΰ²Hΰ²Xb8gradient_tape/model/conv2d_8/Conv2D/Conv2DBackpropFilterhuZUB
ϊ
£void tensorflow::functor::RowReduceKernel<float const*, float*, cub::Sum>(float const*, float*, int, int, cub::Sum, std::iterator_traits<float const*>::value_type)*28ΰ²@ΰ²Hΰ²b1gradient_tape/model/conv2d_15/BiasAdd/BiasAddGradhu  ΘB
ω
£void tensorflow::functor::RowReduceKernel<float const*, float*, cub::Sum>(float const*, float*, int, int, cub::Sum, std::iterator_traits<float const*>::value_type)*28 ²@ ²H ²b0gradient_tape/model/conv2d_7/BiasAdd/BiasAddGradhu  ΘB
ω
£void tensorflow::functor::RowReduceKernel<float const*, float*, cub::Sum>(float const*, float*, int, int, cub::Sum, std::iterator_traits<float const*>::value_type)*28²@²H²b0gradient_tape/model/conv2d_6/BiasAdd/BiasAddGradhu  ΘB
ύ
void cudnn::ops::nchwToNhwcKernel<float, float, float, false, true, (cudnnKernelDataType_t)2>(cudnn::ops::nchw2nhwc_params_t<float>, float const*, float*)'!*2 28ΐ¬@ΥHΐΧXb9gradient_tape/model/conv2d_11/Conv2D/Conv2DBackpropFilterhu  ΘB
μ
void tensorflow::functor::FillPhiloxRandomKernelLaunch<tensorflow::random::UniformDistribution<tensorflow::random::PhiloxRandom, float> >(unsigned long const*, unsigned long const*, tensorflow::random::PhiloxRandom, tensorflow::random::UniformDistribution<tensorflow::random::PhiloxRandom, float>::ResultElementType*, long, tensorflow::random::UniformDistribution<tensorflow::random::PhiloxRandom, float>)*298ΰ©@ΰ©Hΰ©b2model/dropout/dropout/random_uniform/RandomUniformhuZUB
η
void cudnn::winograd_nonfused::winogradWgradOutput4x4<float, float>(cudnn::winograd_nonfused::WinogradWgradOutputParams<float, float>)=H* 2 8ΐ¨@ΐ¨Hΐ¨Xb8gradient_tape/model/conv2d_7/Conv2D/Conv2DBackpropFilterhuZUB
ξ
void tensorflow::functor::FillPhiloxRandomKernelLaunch<tensorflow::random::UniformDistribution<tensorflow::random::PhiloxRandom, float> >(unsigned long const*, unsigned long const*, tensorflow::random::PhiloxRandom, tensorflow::random::UniformDistribution<tensorflow::random::PhiloxRandom, float>::ResultElementType*, long, tensorflow::random::UniformDistribution<tensorflow::random::PhiloxRandom, float>)*298Ώ¦@Ώ¦HΏ¦b4model/dropout_5/dropout/random_uniform/RandomUniformhuZUB
ϊ
£void tensorflow::functor::RowReduceKernel<float const*, float*, cub::Sum>(float const*, float*, int, int, cub::Sum, std::iterator_traits<float const*>::value_type)*28@Hb1gradient_tape/model/conv2d_14/BiasAdd/BiasAddGradhu  ΘB
λ
void cudnn::ops::nchwToNhwcKernel<float, float, float, false, true, (cudnnKernelDataType_t)2>(cudnn::ops::nchw2nhwc_params_t<float>, float const*, float*)'!*228Ώ@ΏHΏb)model/conv2d_transpose_1/conv2d_transposehu  ΘB
ί
void foldedNhwcToNchwKernel<float, float, float, true, (cudnnKernelDataType_t)0>(int, int, int, int, int, int, int, float const*, float*, int, int, int, int, int, int, int, int, int, int, float, float, cudnn::reduced_divisor, cudnn::reduced_divisor, cudnn::reduced_divisor)&!* 228ΐ@ΐHΐb'model/conv2d_transpose/conv2d_transposehu  ΘB

¨void tensorflow::functor::SwapDimension1And2InTensor3UsingTiles<unsigned int, 256, 32, 32, false>(unsigned int const*, tensorflow::functor::Dimension<3>, unsigned int*) !*2¨8ί@ίHίbCmodel/dropout_1/dropout/Mul_1-0-TransposeNHWCToNCHW-LayoutOptimizerhu  ΘB

¨void tensorflow::functor::SwapDimension1And2InTensor3UsingTiles<unsigned int, 256, 32, 32, false>(unsigned int const*, tensorflow::functor::Dimension<3>, unsigned int*) !*28Ώ@ΏHΏbCmodel/dropout_4/dropout/Mul_1-0-TransposeNHWCToNCHW-LayoutOptimizerhu  ΘB
`
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*2ρ8 @ H bmodel/dropout_1/dropout/MulhuZUB
n
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*2ρ8 @ H b)gradient_tape/model/dropout_4/dropout/MulhuZUB

void cudnn::ops::nchwToNhwcKernel<float, float, float, false, true, (cudnnKernelDataType_t)2>(cudnn::ops::nchw2nhwc_params_t<float>, float const*, float*)'!*228ί@ίHίXbLgradient_tape/model/conv2d_transpose_1/conv2d_transpose/Conv2DBackpropFilterhu  ΘB

void cudnn::ops::nchwToNhwcKernel<float, float, float, false, true, (cudnnKernelDataType_t)2>(cudnn::ops::nchw2nhwc_params_t<float>, float const*, float*)'!*228 @ H XbJgradient_tape/model/conv2d_transpose/conv2d_transpose/Conv2DBackpropFilterhu  ΘB
`
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*2ρ8ΐ@ΐHΐbmodel/dropout_4/dropout/MulhuZUB
n
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*2ρ8@Hb)gradient_tape/model/dropout_1/dropout/MulhuZUB
?
΅void tensorflow::functor::ApplyAdamKernel<float>(int, float*, float*, float*, float const*, float const*, float const*, float const*, float const*, float const*, float const*, bool) *2&8ΐή@ΐήHΐήb%Adam/Adam/update_46/ResourceApplyAdamhuZUB
?
΅void tensorflow::functor::ApplyAdamKernel<float>(int, float*, float*, float*, float const*, float const*, float const*, float const*, float const*, float const*, float const*, bool) *2&8Ϊ@ΪHΪb%Adam/Adam/update_20/ResourceApplyAdamhuZUB
ο
void cudnn::bn_bw_1C11_singleread<float, 512, true, 1, 2, 0>(float, float, float, float, cudnnTensorStruct, float const*, cudnnTensorStruct, float const*, cudnnTensorStruct, float*, float const*, float*, float*, float const*, float const*, float, cudnn::reduced_divisor, int, cudnn::reduced_divisor, cudnn::bnBwPersistentState*, int, float, float, float, int, float, cudnnStatus_t*, bool)( ΐ*28 Ω@ ΩH Ωb>gradient_tape/model/batch_normalization_3/FusedBatchNormGradV3huZUB
?
Υvoid Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<float, Eigen::TensorMap<Eigen::Tensor<bool const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<float, Eigen::TensorMap<Eigen::Tensor<bool const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*298ΐΧ@ΐΧHΐΧbXmodel/dropout_4/dropout/ArithmeticOptimizer/ReorderCastLikeAndValuePreserving_float_CasthuZUB
?
Υvoid Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<float, Eigen::TensorMap<Eigen::Tensor<bool const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<float, Eigen::TensorMap<Eigen::Tensor<bool const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*298Φ@ΦHΦbXmodel/dropout_1/dropout/ArithmeticOptimizer/ReorderCastLikeAndValuePreserving_float_CasthuZUB

Υvoid Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<float, Eigen::TensorMap<Eigen::Tensor<bool const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<float, Eigen::TensorMap<Eigen::Tensor<bool const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*298ΐΥ@ΐΥHΐΥbmodel/dropout_4/dropout/CasthuZUB

Υvoid Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<float, Eigen::TensorMap<Eigen::Tensor<bool const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<float, Eigen::TensorMap<Eigen::Tensor<bool const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*298ΐΣ@ΐΣHΐΣbmodel/dropout_1/dropout/CasthuZUB
α
«void tensorflow::functor::SwapDimension1And2InTensor3UsingTiles<unsigned char, 256, 32, 32, false>(unsigned char const*, tensorflow::functor::Dimension<3>, unsigned char*)  *2Π(8ΐΚ@ΐΚHΐΚbgradient_tape/model/dropout_5/dropout/ArithmeticOptimizer/ReorderCastLikeAndValuePreserving_bool_Mul_2-1-TransposeNHWCToNCHW-LayoutOptimizerhu  ΘB
μ
‘void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_product_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorConversionOp<float, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_cmp_op<float const, float const, (Eigen::internal::ComparisonName)5>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorCwiseNullaryOp<Eigen::internal::scalar_constant_op<float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_product_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorConversionOp<float, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_cmp_op<float const, float const, (Eigen::internal::ComparisonName)5>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorCwiseNullaryOp<Eigen::internal::scalar_constant_op<float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const> const> const> const, Eigen::GpuDevice>, long)*298ΰΒ@ΰΒHΰΒb&gradient_tape/model/conv2d_11/ReluGradhuZUB
μ
‘void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_product_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorConversionOp<float, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_cmp_op<float const, float const, (Eigen::internal::ComparisonName)5>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorCwiseNullaryOp<Eigen::internal::scalar_constant_op<float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_product_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorConversionOp<float, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_cmp_op<float const, float const, (Eigen::internal::ComparisonName)5>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorCwiseNullaryOp<Eigen::internal::scalar_constant_op<float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const> const> const> const, Eigen::GpuDevice>, long)*298ΐΑ@ΐΑHΐΑb&gradient_tape/model/conv2d_10/ReluGradhuZUB
δ
void tensorflow::functor::ShuffleInTensor3Simple<float, 2, 1, 0, false>(int, float const*, tensorflow::functor::Dimension<3>, float*)*2&8ΰΐ@ΰΐHΰΐXb8gradient_tape/model/conv2d_14/Conv2D/Conv2DBackpropInputhuZUB
p
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*2Ή8ΰΊ@ΰΊHΰΊb+gradient_tape/model/dropout_2/dropout/Mul_2huZUB
ϋ
void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorSelectOp<Eigen::TensorMap<Eigen::Tensor<bool const, 1, 1, int>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, int>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorSelectOp<Eigen::TensorMap<Eigen::Tensor<bool const, 1, 1, int>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, int>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int)*298ΰΊ@ΰΊHΰΊbAgradient_tape/weighted_binary_crossentropy/logistic_loss/Select_2huZUB
ί
«void tensorflow::functor::SwapDimension1And2InTensor3UsingTiles<unsigned char, 256, 32, 32, false>(unsigned char const*, tensorflow::functor::Dimension<3>, unsigned char*)  *2'8ΰΊ@ΰΊHΰΊbgradient_tape/model/dropout/dropout/ArithmeticOptimizer/ReorderCastLikeAndValuePreserving_bool_Mul_2-1-TransposeNHWCToNCHW-LayoutOptimizerhu  ΘB
½
void tensorflow::functor::ShuffleInTensor3Simple<float, 2, 1, 0, false>(int, float const*, tensorflow::functor::Dimension<3>, float*)*2&8Ί@ΊHΊbmodel/conv2d_8/ReluhuZUB
ν
void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorSelectOp<Eigen::TensorMap<Eigen::Tensor<bool const, 1, 1, int>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, int>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorSelectOp<Eigen::TensorMap<Eigen::Tensor<bool const, 1, 1, int>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, int>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int)*298Ί@ΊHΊb3weighted_binary_crossentropy/logistic_loss/Select_1huZUB
ω
void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorSelectOp<Eigen::TensorMap<Eigen::Tensor<bool const, 1, 1, int>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, int>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorSelectOp<Eigen::TensorMap<Eigen::Tensor<bool const, 1, 1, int>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, int>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int)*298ΰΉ@ΰΉHΰΉb?gradient_tape/weighted_binary_crossentropy/logistic_loss/SelecthuZUB
γ
void tensorflow::functor::ShuffleInTensor3Simple<float, 2, 1, 0, false>(int, float const*, tensorflow::functor::Dimension<3>, float*)*2&8ΰΉ@ΰΉHΰΉXb7gradient_tape/model/conv2d_8/Conv2D/Conv2DBackpropInputhuZUB
ϋ
void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorSelectOp<Eigen::TensorMap<Eigen::Tensor<bool const, 1, 1, int>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, int>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorSelectOp<Eigen::TensorMap<Eigen::Tensor<bool const, 1, 1, int>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, int>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int)*298ΐΉ@ΐΉHΐΉbAgradient_tape/weighted_binary_crossentropy/logistic_loss/Select_3huZUB
λ
void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorSelectOp<Eigen::TensorMap<Eigen::Tensor<bool const, 1, 1, int>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, int>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorSelectOp<Eigen::TensorMap<Eigen::Tensor<bool const, 1, 1, int>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, int>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int)*298 Ή@ ΉH Ήb1weighted_binary_crossentropy/logistic_loss/SelecthuZUB
q
"AddV2_GPU_DT_FLOAT_DT_FLOAT_kernel*2Ή8ΰΈ@ΰΈHΰΈb*weighted_binary_crossentropy/logistic_losshuZUB

 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*2Ή8 ·@ ·H ·b@gradient_tape/weighted_binary_crossentropy/logistic_loss/mul/MulhuZUB
s
 Sub_GPU_DT_FLOAT_DT_FLOAT_kernel*2Ή8·@·H·b.weighted_binary_crossentropy/logistic_loss/subhuZUB

 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*2Ή8ΰ³@ΰ³Hΰ³b<gradient_tape/weighted_binary_crossentropy/logistic_loss/mulhuZUB
Ύ
void tensorflow::functor::ShuffleInTensor3Simple<float, 2, 1, 0, false>(int, float const*, tensorflow::functor::Dimension<3>, float*)*2&8ΰ―@ΰ―Hΰ―bmodel/conv2d_14/ReluhuZUB
b
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*2Ή8Ώ―@Ώ―HΏ―bmodel/dropout_2/dropout/Mul_1huZUB
s
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*2Ή8ΰ¬@ΰ¬Hΰ¬b.weighted_binary_crossentropy/logistic_loss/mulhuZUB
?
΅void tensorflow::functor::ApplyAdamKernel<float>(int, float*, float*, float*, float const*, float const*, float const*, float const*, float const*, float const*, float const*, bool) *2&8ΰ§@ΰ§Hΰ§b%Adam/Adam/update_40/ResourceApplyAdamhuZUB

 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*2Ή8ί @ί Hί b>gradient_tape/weighted_binary_crossentropy/logistic_loss/mul_1huZUB
Ω
void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 4, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorBroadcastingOp<Eigen::array<int, 4ul> const, Eigen::TensorMap<Eigen::Tensor<float const, 4, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 4, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorBroadcastingOp<Eigen::array<int, 4ul> const, Eigen::TensorMap<Eigen::Tensor<float const, 4, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int) *298ΐ@ΐHΐb1gradient_tape/weighted_binary_crossentropy/Tile_1huZUB
θ
void cudnn::winograd_nonfused::winogradWgradOutput4x4<float, float>(cudnn::winograd_nonfused::WinogradWgradOutputParams<float, float>)=H* 28 @ H Xb9gradient_tape/model/conv2d_16/Conv2D/Conv2DBackpropFilterhuZUB
α
void cudnn::ops::pooling_fw_4d_kernel<float, float, cudnn::maxpooling_func<float, (cudnnNanPropagation_t)0>, (cudnnPoolingMode_t)0, false>(cudnnTensorStruct, float const*, cudnnTensorStruct, float*, cudnnPoolingStruct, float, float, int, cudnn::reduced_divisor, cudnn::reduced_divisor)(  *2228ΰ@ΰHΰbmodel/max_pooling2d_4/MaxPoolhuωOΓB
q
(GreaterEqual_GPU_DT_FLOAT_DT_BOOL_kernel*2ρ8ΐ@ΐHΐb$model/dropout_4/dropout/GreaterEqualhuZUB
q
(GreaterEqual_GPU_DT_FLOAT_DT_BOOL_kernel*2ρ8ΰ@ΰHΰb$model/dropout_1/dropout/GreaterEqualhuZUB
η
void cudnn::winograd_nonfused::winogradWgradOutput4x4<float, float>(cudnn::winograd_nonfused::WinogradWgradOutputParams<float, float>)=H* 2 8ΰ?@ΰ?Hΰ?Xb8gradient_tape/model/conv2d_6/Conv2D/Conv2DBackpropFilterhuZUB

(GreaterEqual_GPU_DT_FLOAT_DT_BOOL_kernel*2Ή8ΐϊ@ΐϊHΐϊb7weighted_binary_crossentropy/logistic_loss/GreaterEqualhuZUB
½
void cudnn::winograd_nonfused::winogradForwardFilter4x4<float, float>(cudnn::winograd_nonfused::WinogradFilterParams<float, float>)(H* 2 8ς@ςHςbmodel/conv2d_7/Reluhu  ΘB
δ
void cudnn::winograd_nonfused::winogradForwardFilter4x4<float, float>(cudnn::winograd_nonfused::WinogradFilterParams<float, float>)(H* 2 8?ρ@?ρH?ρXb8gradient_tape/model/conv2d_15/Conv2D/Conv2DBackpropInputhu  ΘB
Α
θvoid cudnn::bn_fw_tr_1C11_singleread<float, 512, true, 1, 2, 0>(cudnnTensorStruct, float const*, cudnnTensorStruct, float*, float const*, float const*, float, float, float*, float*, float*, float*, float, float, cudnn::reduced_divisor, int, cudnn::reduced_divisor, cudnn::bnFwPersistentState*, int, float, float, float, int, float, float, cudnnStatus_t*, bool)'  *28ίπ@ίπHίπb,model/batch_normalization_3/FusedBatchNormV3hu  ΘB
γ
void cudnn::winograd_nonfused::winogradForwardFilter4x4<float, float>(cudnn::winograd_nonfused::WinogradFilterParams<float, float>)(H* 2 8π@πHπXb7gradient_tape/model/conv2d_7/Conv2D/Conv2DBackpropInputhu  ΘB
Ύ
void cudnn::winograd_nonfused::winogradForwardFilter4x4<float, float>(cudnn::winograd_nonfused::WinogradFilterParams<float, float>)(H* 2 8ξ@ξHξbmodel/conv2d_15/Reluhu  ΘB
ξ
void tensorflow::functor::FillPhiloxRandomKernelLaunch<tensorflow::random::UniformDistribution<tensorflow::random::PhiloxRandom, float> >(unsigned long const*, unsigned long const*, tensorflow::random::PhiloxRandom, tensorflow::random::UniformDistribution<tensorflow::random::PhiloxRandom, float>::ResultElementType*, long, tensorflow::random::UniformDistribution<tensorflow::random::PhiloxRandom, float>)*298ν@νHνb4model/dropout_1/dropout/random_uniform/RandomUniformhuZUB
ξ
void tensorflow::functor::FillPhiloxRandomKernelLaunch<tensorflow::random::UniformDistribution<tensorflow::random::PhiloxRandom, float> >(unsigned long const*, unsigned long const*, tensorflow::random::PhiloxRandom, tensorflow::random::UniformDistribution<tensorflow::random::PhiloxRandom, float>::ResultElementType*, long, tensorflow::random::UniformDistribution<tensorflow::random::PhiloxRandom, float>)*298ΐζ@ΐζHΐζb4model/dropout_4/dropout/random_uniform/RandomUniformhuZUB

£void tensorflow::functor::RowReduceKernel<float const*, float*, cub::Sum>(float const*, float*, int, int, cub::Sum, std::iterator_traits<float const*>::value_type)*228ΐδ@ΐδHΐδb8gradient_tape/model/conv2d_transpose/BiasAdd/BiasAddGradhu  ΘB
ϊ
£void tensorflow::functor::RowReduceKernel<float const*, float*, cub::Sum>(float const*, float*, int, int, cub::Sum, std::iterator_traits<float const*>::value_type)*228 ί@ ίH ίb1gradient_tape/model/conv2d_13/BiasAdd/BiasAddGradhu  ΘB
ϊ
£void tensorflow::functor::RowReduceKernel<float const*, float*, cub::Sum>(float const*, float*, int, int, cub::Sum, std::iterator_traits<float const*>::value_type)*228ΐέ@ΐέHΐέb1gradient_tape/model/conv2d_12/BiasAdd/BiasAddGradhu  ΘB
ω
£void tensorflow::functor::RowReduceKernel<float const*, float*, cub::Sum>(float const*, float*, int, int, cub::Sum, std::iterator_traits<float const*>::value_type)*228 έ@ έH έb0gradient_tape/model/conv2d_8/BiasAdd/BiasAddGradhu  ΘB
ό
void cudnn::ops::nchwToNhwcKernel<float, float, float, false, true, (cudnnKernelDataType_t)2>(cudnn::ops::nchw2nhwc_params_t<float>, float const*, float*)'!*2 28ΰά@ΰάHΰάXb8gradient_tape/model/conv2d_11/Conv2D/Conv2DBackpropInputhu  ΘB
ω
£void tensorflow::functor::RowReduceKernel<float const*, float*, cub::Sum>(float const*, float*, int, int, cub::Sum, std::iterator_traits<float const*>::value_type)*228ΰά@ΰάHΰάb0gradient_tape/model/conv2d_9/BiasAdd/BiasAddGradhu  ΘB

void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseUnaryOp<Eigen::internal::scalar_inverse_op<float>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseUnaryOp<Eigen::internal::scalar_inverse_op<float>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int)*298ΐά@ΐάHΐάbCgradient_tape/weighted_binary_crossentropy/logistic_loss/ReciprocalhuZUB

¨void tensorflow::functor::SwapDimension1And2InTensor3UsingTiles<unsigned int, 256, 32, 32, false>(unsigned int const*, tensorflow::functor::Dimension<3>, unsigned int*) !*2ΐ8ΰΪ@ΰΪHΰΪbCmodel/dropout_2/dropout/Mul_1-0-TransposeNHWCToNCHW-LayoutOptimizerhu  ΘB
w
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*2Ή8Ϊ@ΪHΪb2gradient_tape/weighted_binary_crossentropy/truedivhuZUB
s
 Neg_GPU_DT_FLOAT_DT_FLOAT_kernel*2β	8Ω@ΩHΩb.weighted_binary_crossentropy/logistic_loss/Neghu  ΘB
`
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*2Ή8ΐΧ@ΐΧHΐΧbmodel/dropout_2/dropout/MulhuZUB

void cudnn::ops::nchwToNhwcKernel<float, float, float, false, true, (cudnnKernelDataType_t)2>(cudnn::ops::nchw2nhwc_params_t<float>, float const*, float*)'!*2 28Χ@ΧHΧXbJgradient_tape/model/conv2d_transpose/conv2d_transpose/Conv2DBackpropFilterhu  ΘB
ι
void cudnn::ops::nchwToNhwcKernel<float, float, float, false, true, (cudnnKernelDataType_t)2>(cudnn::ops::nchw2nhwc_params_t<float>, float const*, float*)'!*2 28ΰΦ@ΰΦHΰΦb'model/conv2d_transpose/conv2d_transposehu  ΘB
w
"Log1p_GPU_DT_FLOAT_DT_FLOAT_kernel*2β	8Φ@ΦHΦb0weighted_binary_crossentropy/logistic_loss/Log1phu  ΘB
Φ
void cudnn::ops::nchwToNhwcKernel<float, float, float, false, true, (cudnnKernelDataType_t)0>(cudnn::ops::nchw2nhwc_params_t<float>, float const*, float*)(!*2 28ΰΥ@ΰΥHΰΥbmodel/conv2d_10/Reluhu  ΘB
Φ
void cudnn::ops::nchwToNhwcKernel<float, float, float, false, true, (cudnnKernelDataType_t)0>(cudnn::ops::nchw2nhwc_params_t<float>, float const*, float*)(!*2 28ΐΥ@ΐΥHΐΥbmodel/conv2d_11/Reluhu  ΘB
ύ
void cudnn::ops::nchwToNhwcKernel<float, float, float, false, true, (cudnnKernelDataType_t)2>(cudnn::ops::nchw2nhwc_params_t<float>, float const*, float*)'!*2 28ΐΥ@ΐΥHΐΥXb9gradient_tape/model/conv2d_10/Conv2D/Conv2DBackpropFilterhu  ΘB
Φ
void cudnn::ops::nhwcToNchwKernel<float, float, float, true, false, (cudnnKernelDataType_t)0>(cudnn::ops::nhwc2nchw_params_t<float>, float const*, float*)&!*2 28Υ@ΥHΥbmodel/conv2d_11/Reluhu  ΘB

"AddV2_GPU_DT_FLOAT_DT_FLOAT_kernel*2Ή8ΐΤ@ΐΤHΐΤb<gradient_tape/weighted_binary_crossentropy/logistic_loss/addhuZUB

 Neg_GPU_DT_FLOAT_DT_FLOAT_kernel*2β	8ΐΤ@ΐΤHΐΤb<gradient_tape/weighted_binary_crossentropy/logistic_loss/Neghu  ΘB

Yvoid tensorflow::BiasNCHWKernel<float>(int, float const*, float const*, float*, int, int)*2&8ΐΡ@ΐΡHΐΡbmodel/conv2d_22/BiasAddhuZUB
s
 Exp_GPU_DT_FLOAT_DT_FLOAT_kernel*2β	8ΰΠ@ΰΠHΰΠb.weighted_binary_crossentropy/logistic_loss/Exphu  ΘB
Φ
void cudnn::ops::nhwcToNchwKernel<float, float, float, true, false, (cudnnKernelDataType_t)0>(cudnn::ops::nhwc2nchw_params_t<float>, float const*, float*)&!*2 28ΰΛ@ΰΛHΰΛbmodel/conv2d_10/Reluhu  ΘB
ό
void cudnn::ops::nhwcToNchwKernel<float, float, float, true, false, (cudnnKernelDataType_t)0>(cudnn::ops::nhwc2nchw_params_t<float>, float const*, float*)&!*2 28ΐΚ@ΐΚHΐΚXb8gradient_tape/model/conv2d_11/Conv2D/Conv2DBackpropInputhu  ΘB
α
«void tensorflow::functor::SwapDimension1And2InTensor3UsingTiles<unsigned char, 256, 32, 32, false>(unsigned char const*, tensorflow::functor::Dimension<3>, unsigned char*)  *28ΐΚ@ΐΚHΐΚbgradient_tape/model/dropout_4/dropout/ArithmeticOptimizer/ReorderCastLikeAndValuePreserving_bool_Mul_2-1-TransposeNHWCToNCHW-LayoutOptimizerhu  ΘB
Φ
void cudnn::ops::nchwToNhwcKernel<float, float, float, false, true, (cudnnKernelDataType_t)2>(cudnn::ops::nchw2nhwc_params_t<float>, float const*, float*)'!*2 28ΏΑ@ΏΑHΏΑbmodel/conv2d_11/Reluhu  ΘB
n
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*2Ή8ΐ@ΐHΐb)gradient_tape/model/dropout_2/dropout/MulhuZUB
Σ
void nchwToFoldedNhwcKernel<float, float, float, true, (cudnnKernelDataType_t)2>(int, int, int, int, float const*, float*, int, int, int, int, int, int, int, int, int, int, float, float, cudnn::reduced_divisor, cudnn::reduced_divisor, cudnn::reduced_divisor)&!* 28Ύ@ΎHΎb)model/conv2d_transpose_1/conv2d_transposehu  ΘB
?
΅void tensorflow::functor::ApplyAdamKernel<float>(int, float*, float*, float*, float const*, float const*, float const*, float const*, float const*, float const*, float const*, bool) *2&8 »@ »H »b%Adam/Adam/update_52/ResourceApplyAdamhuZUB
?
΅void tensorflow::functor::ApplyAdamKernel<float>(int, float*, float*, float*, float const*, float const*, float const*, float const*, float const*, float const*, float const*, bool) *2&8ΰΊ@ΰΊHΰΊb%Adam/Adam/update_18/ResourceApplyAdamhuZUB

void cudnn::ops::nhwcToNchwKernel<float, float, float, true, false, (cudnnKernelDataType_t)0>(cudnn::ops::nhwc2nchw_params_t<float>, float const*, float*)&!*28ΰ·@ΰ·Hΰ·XbLgradient_tape/model/conv2d_transpose_1/conv2d_transpose/Conv2DBackpropFilterhu  ΘB
?
Υvoid Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<float, Eigen::TensorMap<Eigen::Tensor<bool const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<float, Eigen::TensorMap<Eigen::Tensor<bool const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*298ΐΆ@ΐΆHΐΆbXmodel/dropout_2/dropout/ArithmeticOptimizer/ReorderCastLikeAndValuePreserving_float_CasthuZUB

void cudnn::ops::nchwToNhwcKernel<float, float, float, false, true, (cudnnKernelDataType_t)2>(cudnn::ops::nchw2nhwc_params_t<float>, float const*, float*)'!*28 Ά@ ΆH ΆXb>gradient_tape/model/conv2d_transpose_1/conv2d_transpose/Conv2Dhu  ΘB

Υvoid Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<float, Eigen::TensorMap<Eigen::Tensor<bool const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<float, Eigen::TensorMap<Eigen::Tensor<bool const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*298Ά@ΆHΆbmodel/dropout_2/dropout/CasthuZUB
α
«void tensorflow::functor::SwapDimension1And2InTensor3UsingTiles<unsigned char, 256, 32, 32, false>(unsigned char const*, tensorflow::functor::Dimension<3>, unsigned char*)  *2¨8±@±H±bgradient_tape/model/dropout_1/dropout/ArithmeticOptimizer/ReorderCastLikeAndValuePreserving_bool_Mul_2-1-TransposeNHWCToNCHW-LayoutOptimizerhu  ΘB
ε
void tensorflow::functor::ShuffleInTensor3Simple<float, 2, 1, 0, false>(int, float const*, tensorflow::functor::Dimension<3>, float*)*2&8 °@ °H °Xb9gradient_tape/model/conv2d_15/Conv2D/Conv2DBackpropFilterhuZUB
ϊ
£void tensorflow::functor::RowReduceKernel<float const*, float*, cub::Sum>(float const*, float*, int, int, cub::Sum, std::iterator_traits<float const*>::value_type)*2d8­@­H­b1gradient_tape/model/conv2d_10/BiasAdd/BiasAddGradhu  ΘB
δ
void tensorflow::functor::ShuffleInTensor3Simple<float, 2, 1, 0, false>(int, float const*, tensorflow::functor::Dimension<3>, float*)*2&8 ¬@ ¬H ¬Xb8gradient_tape/model/conv2d_15/Conv2D/Conv2DBackpropInputhuZUB
δ
void tensorflow::functor::ShuffleInTensor3Simple<float, 2, 1, 0, false>(int, float const*, tensorflow::functor::Dimension<3>, float*)*2&8ΰ«@ΰ«Hΰ«Xb8gradient_tape/model/conv2d_7/Conv2D/Conv2DBackpropFilterhuZUB
ϊ
£void tensorflow::functor::RowReduceKernel<float const*, float*, cub::Sum>(float const*, float*, int, int, cub::Sum, std::iterator_traits<float const*>::value_type)*2d8?ͺ@?ͺH?ͺb1gradient_tape/model/conv2d_11/BiasAdd/BiasAddGradhu  ΘB
γ
void tensorflow::functor::ShuffleInTensor3Simple<float, 2, 1, 0, false>(int, float const*, tensorflow::functor::Dimension<3>, float*)*2&8ΐ¦@ΐ¦Hΐ¦Xb7gradient_tape/model/conv2d_7/Conv2D/Conv2DBackpropInputhuZUB
Ύ
void tensorflow::functor::ShuffleInTensor3Simple<float, 2, 1, 0, false>(int, float const*, tensorflow::functor::Dimension<3>, float*)*2&8 ¦@ ¦H ¦bmodel/conv2d_15/ReluhuZUB
½
void tensorflow::functor::ShuffleInTensor3Simple<float, 2, 1, 0, false>(int, float const*, tensorflow::functor::Dimension<3>, float*)*2&8Ώ’@Ώ’HΏ’bmodel/conv2d_7/ReluhuZUB
b
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*28 @ H bmodel/dropout_3/dropout/Mul_1huZUB
κ
void tensorflow::functor::ShuffleInTensor3Simple<float, 2, 1, 0, false>(int, float const*, tensorflow::functor::Dimension<3>, float*)*2&8ΰ@ΰHΰXb>gradient_tape/model/conv2d_transpose_1/conv2d_transpose/Conv2DhuZUB
p
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*28@Hb+gradient_tape/model/dropout_3/dropout/Mul_2huZUB
Σ
void tensorflow::functor::ShuffleInTensor3Simple<float, 2, 1, 0, false>(int, float const*, tensorflow::functor::Dimension<3>, float*)*2&8 @ H b)model/conv2d_transpose_1/conv2d_transposehuZUB
Δ
ϋvoid cub::DeviceReduceKernel<cub::DeviceReducePolicy<float, float, int, tensorflow::functor::Sum<float> >::Policy600, float*, float*, int, tensorflow::functor::Sum<float> >(float*, float*, int, cub::GridEvenShare<int>, tensorflow::functor::Sum<float>)0*2Ή8ΐ@ΐHΐb!weighted_binary_crossentropy/Meanhu  ΘB
q
(GreaterEqual_GPU_DT_FLOAT_DT_BOOL_kernel*2Ή8ΐ@ΐHΐb$model/dropout_2/dropout/GreaterEqualhuZUB
θ
void cudnn::winograd_nonfused::winogradWgradOutput4x4<float, float>(cudnn::winograd_nonfused::WinogradWgradOutputParams<float, float>)=H* 28ΐ@ΐHΐXb9gradient_tape/model/conv2d_17/Conv2D/Conv2DBackpropFilterhuZUB
η
void cudnn::winograd_nonfused::winogradWgradOutput4x4<float, float>(cudnn::winograd_nonfused::WinogradWgradOutputParams<float, float>)=H* 28ΐ@ΐHΐXb8gradient_tape/model/conv2d_5/Conv2D/Conv2DBackpropFilterhuZUB
ξ
void tensorflow::functor::FillPhiloxRandomKernelLaunch<tensorflow::random::UniformDistribution<tensorflow::random::PhiloxRandom, float> >(unsigned long const*, unsigned long const*, tensorflow::random::PhiloxRandom, tensorflow::random::UniformDistribution<tensorflow::random::PhiloxRandom, float>::ResultElementType*, long, tensorflow::random::UniformDistribution<tensorflow::random::PhiloxRandom, float>)*298@Hb4model/dropout_2/dropout/random_uniform/RandomUniformhuZUB
α
void cudnn::winograd_nonfused::winogradForwardFilter4x4<float, float>(cudnn::winograd_nonfused::WinogradFilterParams<float, float>)(H* 28~@~H~Xb8gradient_tape/model/conv2d_16/Conv2D/Conv2DBackpropInputhu  ΘB
»
void cudnn::winograd_nonfused::winogradForwardFilter4x4<float, float>(cudnn::winograd_nonfused::WinogradFilterParams<float, float>)(H* 28ΰ}@ΰ}Hΰ}bmodel/conv2d_16/Reluhu  ΘB
ΰ
void cudnn::winograd_nonfused::winogradForwardFilter4x4<float, float>(cudnn::winograd_nonfused::WinogradFilterParams<float, float>)(H* 2 8 z@ zH zXb7gradient_tape/model/conv2d_6/Conv2D/Conv2DBackpropInputhu  ΘB
Ί
void cudnn::winograd_nonfused::winogradForwardFilter4x4<float, float>(cudnn::winograd_nonfused::WinogradFilterParams<float, float>)(H* 2 8z@zHzbmodel/conv2d_6/Reluhu  ΘB
ΰ
void cub::DeviceSegmentedReduceKernel<cub::DeviceReducePolicy<float, float, int, cub::Sum>::Policy600, float const*, float*, cub::TransformInputIterator<int, tensorflow::functor::RowOffset, cub::CountingInputIterator<int, long>, long>, int, cub::Sum, float>(float const*, float*, cub::TransformInputIterator<int, tensorflow::functor::RowOffset, cub::CountingInputIterator<int, long>, long>, cub::TransformInputIterator<int, tensorflow::functor::RowOffset, cub::CountingInputIterator<int, long>, long>, int, cub::Sum, float) 0*228 y@ yH yb1gradient_tape/model/conv2d_22/BiasAdd/BiasAddGradhu  ΘB
k
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*28 x@ xH xb)gradient_tape/model/dropout_3/dropout/MulhuZUB

¨void tensorflow::functor::SwapDimension1And2InTensor3UsingTiles<unsigned int, 256, 32, 32, false>(unsigned int const*, tensorflow::functor::Dimension<3>, unsigned int*) !*2 8ΰv@ΰvHΰvbCmodel/dropout_3/dropout/Mul_1-0-TransposeNHWCToNCHW-LayoutOptimizerhu  ΘB
ϊ
void cudnn::ops::nchwToNhwcKernel<float, float, float, false, true, (cudnnKernelDataType_t)2>(cudnn::ops::nchw2nhwc_params_t<float>, float const*, float*)'!*228?u@?uH?uXb9gradient_tape/model/conv2d_10/Conv2D/Conv2DBackpropFilterhu  ΘB
]
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*28ΰt@ΰtHΰtbmodel/dropout_3/dropout/MulhuZUB
Σ
void cudnn::ops::nchwToNhwcKernel<float, float, float, false, true, (cudnnKernelDataType_t)2>(cudnn::ops::nchw2nhwc_params_t<float>, float const*, float*)'!*228t@tHtbmodel/conv2d_10/Reluhu  ΘB
υ
void tensorflow::functor::ShuffleInTensor3Simple<float, 2, 1, 0, false>(int, float const*, tensorflow::functor::Dimension<3>, float*)*2&8Ώq@ΏqHΏqXbLgradient_tape/model/conv2d_transpose_1/conv2d_transpose/Conv2DBackpropFilterhuZUB
ή
«void tensorflow::functor::SwapDimension1And2InTensor3UsingTiles<unsigned char, 256, 32, 32, false>(unsigned char const*, tensorflow::functor::Dimension<3>, unsigned char*)  *2ΐ8ΰn@ΰnHΰnbgradient_tape/model/dropout_2/dropout/ArithmeticOptimizer/ReorderCastLikeAndValuePreserving_bool_Mul_2-1-TransposeNHWCToNCHW-LayoutOptimizerhu  ΘB
ό
΅void tensorflow::functor::ApplyAdamKernel<float>(int, float*, float*, float*, float const*, float const*, float const*, float const*, float const*, float const*, float const*, bool) *2&8ΐh@ΐhHΐhb%Adam/Adam/update_54/ResourceApplyAdamhuZUB
ό
΅void tensorflow::functor::ApplyAdamKernel<float>(int, float*, float*, float*, float const*, float const*, float const*, float const*, float const*, float const*, float const*, bool) *2&8h@hHhb%Adam/Adam/update_14/ResourceApplyAdamhuZUB

Υvoid Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<float, Eigen::TensorMap<Eigen::Tensor<bool const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<float, Eigen::TensorMap<Eigen::Tensor<bool const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*298ΰd@ΰdHΰdbmodel/dropout_3/dropout/CasthuZUB
β
void tensorflow::functor::ShuffleInTensor3Simple<float, 2, 1, 0, false>(int, float const*, tensorflow::functor::Dimension<3>, float*)*2&8?c@?cH?cXb9gradient_tape/model/conv2d_16/Conv2D/Conv2DBackpropFilterhuZUB
Ο
Υvoid Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<float, Eigen::TensorMap<Eigen::Tensor<bool const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<float, Eigen::TensorMap<Eigen::Tensor<bool const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*298ΐc@ΐcHΐcbXmodel/dropout_3/dropout/ArithmeticOptimizer/ReorderCastLikeAndValuePreserving_float_CasthuZUB
α
void tensorflow::functor::ShuffleInTensor3Simple<float, 2, 1, 0, false>(int, float const*, tensorflow::functor::Dimension<3>, float*)*2&8b@bHbXb8gradient_tape/model/conv2d_6/Conv2D/Conv2DBackpropFilterhuZUB
α
void tensorflow::functor::ShuffleInTensor3Simple<float, 2, 1, 0, false>(int, float const*, tensorflow::functor::Dimension<3>, float*)*2&8ΰa@ΰaHΰaXb8gradient_tape/model/conv2d_16/Conv2D/Conv2DBackpropInputhuZUB
»
void tensorflow::functor::ShuffleInTensor3Simple<float, 2, 1, 0, false>(int, float const*, tensorflow::functor::Dimension<3>, float*)*2&8ΐ`@ΐ`Hΐ`bmodel/conv2d_16/ReluhuZUB
ό
΅void tensorflow::functor::ApplyAdamKernel<float>(int, float*, float*, float*, float const*, float const*, float const*, float const*, float const*, float const*, float const*, bool) *2&8 `@ `H `b%Adam/Adam/update_48/ResourceApplyAdamhuZUB
ΰ
void tensorflow::functor::ShuffleInTensor3Simple<float, 2, 1, 0, false>(int, float const*, tensorflow::functor::Dimension<3>, float*)*2&8 _@ _H _Xb7gradient_tape/model/conv2d_6/Conv2D/Conv2DBackpropInputhuZUB
Ί
void tensorflow::functor::ShuffleInTensor3Simple<float, 2, 1, 0, false>(int, float const*, tensorflow::functor::Dimension<3>, float*)*2&8ΰ^@ΰ^Hΰ^bmodel/conv2d_6/ReluhuZUB
δ
void cudnn::winograd_nonfused::winogradWgradOutput4x4<float, float>(cudnn::winograd_nonfused::WinogradWgradOutputParams<float, float>)=H* 28P@PHPXb8gradient_tape/model/conv2d_4/Conv2D/Conv2DBackpropFilterhuZUB
ε
void cudnn::winograd_nonfused::winogradWgradOutput4x4<float, float>(cudnn::winograd_nonfused::WinogradWgradOutputParams<float, float>)=H* 28ΰN@ΰNHΰNXb9gradient_tape/model/conv2d_18/Conv2D/Conv2DBackpropFilterhuZUB
n
(GreaterEqual_GPU_DT_FLOAT_DT_BOOL_kernel*28ΐN@ΐNHΐNb$model/dropout_3/dropout/GreaterEqualhuZUB
λ
void tensorflow::functor::FillPhiloxRandomKernelLaunch<tensorflow::random::UniformDistribution<tensorflow::random::PhiloxRandom, float> >(unsigned long const*, unsigned long const*, tensorflow::random::PhiloxRandom, tensorflow::random::UniformDistribution<tensorflow::random::PhiloxRandom, float>::ResultElementType*, long, tensorflow::random::UniformDistribution<tensorflow::random::PhiloxRandom, float>)*298ΐN@ΐNHΐNb4model/dropout_3/dropout/random_uniform/RandomUniformhuZUB
ΰ
void cudnn::winograd_nonfused::winogradForwardFilter4x4<float, float>(cudnn::winograd_nonfused::WinogradFilterParams<float, float>)(H* 28ΐI@ΐIHΐIXb7gradient_tape/model/conv2d_5/Conv2D/Conv2DBackpropInputhu  ΘB
Ί
void cudnn::winograd_nonfused::winogradForwardFilter4x4<float, float>(cudnn::winograd_nonfused::WinogradFilterParams<float, float>)(H* 28H@HHHbmodel/conv2d_5/Reluhu  ΘB
α
void cudnn::winograd_nonfused::winogradForwardFilter4x4<float, float>(cudnn::winograd_nonfused::WinogradFilterParams<float, float>)(H* 28ΐG@ΐGHΐGXb8gradient_tape/model/conv2d_17/Conv2D/Conv2DBackpropInputhu  ΘB
»
void cudnn::winograd_nonfused::winogradForwardFilter4x4<float, float>(cudnn::winograd_nonfused::WinogradFilterParams<float, float>)(H* 28 F@ FH Fbmodel/conv2d_17/Reluhu  ΘB
ή
«void tensorflow::functor::SwapDimension1And2InTensor3UsingTiles<unsigned char, 256, 32, 32, false>(unsigned char const*, tensorflow::functor::Dimension<3>, unsigned char*)  *2 8ΐE@ΐEHΐEbgradient_tape/model/dropout_3/dropout/ArithmeticOptimizer/ReorderCastLikeAndValuePreserving_bool_Mul_2-1-TransposeNHWCToNCHW-LayoutOptimizerhu  ΘB
ό
΅void tensorflow::functor::ApplyAdamKernel<float>(int, float*, float*, float*, float const*, float const*, float const*, float const*, float const*, float const*, float const*, bool) *2&8Α?@Α?HΑ?b%Adam/Adam/update_60/ResourceApplyAdamhuZUB
α
void tensorflow::functor::ShuffleInTensor3Simple<float, 2, 1, 0, false>(int, float const*, tensorflow::functor::Dimension<3>, float*)*2&8?=@?=H?=Xb8gradient_tape/model/conv2d_17/Conv2D/Conv2DBackpropInputhuZUB
β
void tensorflow::functor::ShuffleInTensor3Simple<float, 2, 1, 0, false>(int, float const*, tensorflow::functor::Dimension<3>, float*)*2&8ΐ<@ΐ<Hΐ<Xb9gradient_tape/model/conv2d_17/Conv2D/Conv2DBackpropFilterhuZUB
Π
void nchwToFoldedNhwcKernel<float, float, float, true, (cudnnKernelDataType_t)2>(int, int, int, int, float const*, float*, int, int, int, int, int, int, int, int, int, int, float, float, cudnn::reduced_divisor, cudnn::reduced_divisor, cudnn::reduced_divisor)&!* 28<@<H<b)model/conv2d_transpose_2/conv2d_transposehu  ΘB
α
void tensorflow::functor::ShuffleInTensor3Simple<float, 2, 1, 0, false>(int, float const*, tensorflow::functor::Dimension<3>, float*)*2&8 ;@ ;H ;Xb8gradient_tape/model/conv2d_5/Conv2D/Conv2DBackpropFilterhuZUB
ό
΅void tensorflow::functor::ApplyAdamKernel<float>(int, float*, float*, float*, float const*, float const*, float const*, float const*, float const*, float const*, float const*, bool) *2&8ΰ:@ΰ:Hΰ:b%Adam/Adam/update_12/ResourceApplyAdamhuZUB
Ί
void tensorflow::functor::ShuffleInTensor3Simple<float, 2, 1, 0, false>(int, float const*, tensorflow::functor::Dimension<3>, float*)*2&8ΐ9@ΐ9Hΐ9bmodel/conv2d_5/ReluhuZUB
»
void tensorflow::functor::ShuffleInTensor3Simple<float, 2, 1, 0, false>(int, float const*, tensorflow::functor::Dimension<3>, float*)*2&89@9H9bmodel/conv2d_17/ReluhuZUB
ΰ
void tensorflow::functor::ShuffleInTensor3Simple<float, 2, 1, 0, false>(int, float const*, tensorflow::functor::Dimension<3>, float*)*2&8ΐ8@ΐ8Hΐ8Xb7gradient_tape/model/conv2d_5/Conv2D/Conv2DBackpropInputhuZUB
Π
void tensorflow::functor::ShuffleInTensor3Simple<float, 2, 1, 0, false>(int, float const*, tensorflow::functor::Dimension<3>, float*)*2&8ΰ0@ΰ0Hΰ0b)model/conv2d_transpose_2/conv2d_transposehuZUB
ό
΅void tensorflow::functor::ApplyAdamKernel<float>(int, float*, float*, float*, float const*, float const*, float const*, float const*, float const*, float const*, float const*, bool) *2 8 .@ .H .b%Adam/Adam/update_56/ResourceApplyAdamhuZUB
ό
΅void tensorflow::functor::ApplyAdamKernel<float>(int, float*, float*, float*, float const*, float const*, float const*, float const*, float const*, float const*, float const*, bool) *2$8ΐ,@ΐ,Hΐ,b%Adam/Adam/update_62/ResourceApplyAdamhuZUB
ϋ
΅void tensorflow::functor::ApplyAdamKernel<float>(int, float*, float*, float*, float const*, float const*, float const*, float const*, float const*, float const*, float const*, bool) *2$8ΰ+@ΰ+Hΰ+b$Adam/Adam/update_8/ResourceApplyAdamhuZUB
η
void tensorflow::functor::ShuffleInTensor3Simple<float, 2, 1, 0, false>(int, float const*, tensorflow::functor::Dimension<3>, float*)*2&8ΰ+@ΰ+Hΰ+Xb>gradient_tape/model/conv2d_transpose_2/conv2d_transpose/Conv2DhuZUB
»
void tensorflow::functor::ShuffleInTensor3Simple<float, 2, 1, 0, false>(int, float const*, tensorflow::functor::Dimension<3>, float*)*2&8+@+H+bmodel/conv2d_18/ReluhuZUB
β
void tensorflow::functor::ShuffleInTensor3Simple<float, 2, 1, 0, false>(int, float const*, tensorflow::functor::Dimension<3>, float*)*2&8ΰ*@ΰ*Hΰ*Xb9gradient_tape/model/conv2d_18/Conv2D/Conv2DBackpropFilterhuZUB
υ
void tensorflow::functor::ShuffleInTensor3Simple<float, 2, 1, 0, false>(int, float const*, tensorflow::functor::Dimension<3>, float*)*2&8*@*H*XbLgradient_tape/model/conv2d_transpose_2/conv2d_transpose/Conv2DBackpropFilterhuZUB
α
void tensorflow::functor::ShuffleInTensor3Simple<float, 2, 1, 0, false>(int, float const*, tensorflow::functor::Dimension<3>, float*)*2&8ΰ'@ΰ'Hΰ'Xb8gradient_tape/model/conv2d_18/Conv2D/Conv2DBackpropInputhuZUB
Ί
void tensorflow::functor::ShuffleInTensor3Simple<float, 2, 1, 0, false>(int, float const*, tensorflow::functor::Dimension<3>, float*)*2&8ΰ&@ΰ&Hΰ&bmodel/conv2d_4/ReluhuZUB
α
void tensorflow::functor::ShuffleInTensor3Simple<float, 2, 1, 0, false>(int, float const*, tensorflow::functor::Dimension<3>, float*)*2&8ΐ&@ΐ&Hΐ&Xb8gradient_tape/model/conv2d_4/Conv2D/Conv2DBackpropFilterhuZUB
ό
΅void tensorflow::functor::ApplyAdamKernel<float>(int, float*, float*, float*, float const*, float const*, float const*, float const*, float const*, float const*, float const*, bool) *28ΐ%@ΐ%Hΐ%b%Adam/Adam/update_25/ResourceApplyAdamhuZUB
ϋ
΅void tensorflow::functor::ApplyAdamKernel<float>(int, float*, float*, float*, float const*, float const*, float const*, float const*, float const*, float const*, float const*, bool) *28ΐ%@ΐ%Hΐ%b$Adam/Adam/update_6/ResourceApplyAdamhuZUB
ό
΅void tensorflow::functor::ApplyAdamKernel<float>(int, float*, float*, float*, float const*, float const*, float const*, float const*, float const*, float const*, float const*, bool) *28α#@α#Hα#b%Adam/Adam/update_64/ResourceApplyAdamhuZUB
Σ
void cudnn::ops::nchwToNhwcKernel<float, float, float, false, true, (cudnnKernelDataType_t)2>(cudnn::ops::nchw2nhwc_params_t<float>, float const*, float*)'!*28ΐ#@ΐ#Hΐ#bmodel/conv2d_4/Reluhu  ΘB
ϋ
΅void tensorflow::functor::ApplyAdamKernel<float>(int, float*, float*, float*, float const*, float const*, float const*, float const*, float const*, float const*, float const*, bool) *2	8 #@ #H #b$Adam/Adam/update_2/ResourceApplyAdamhuZUB
ΰ
void tensorflow::functor::ShuffleInTensor3Simple<float, 2, 1, 0, false>(int, float const*, tensorflow::functor::Dimension<3>, float*)*2&8ΐ"@ΐ"Hΐ"Xb7gradient_tape/model/conv2d_4/Conv2D/Conv2DBackpropInputhuZUB
ό
΅void tensorflow::functor::ApplyAdamKernel<float>(int, float*, float*, float*, float const*, float const*, float const*, float const*, float const*, float const*, float const*, bool) *28 "@ "H "b%Adam/Adam/update_66/ResourceApplyAdamhuZUB
ό
΅void tensorflow::functor::ApplyAdamKernel<float>(int, float*, float*, float*, float const*, float const*, float const*, float const*, float const*, float const*, float const*, bool) *28 !@ !H !b%Adam/Adam/update_34/ResourceApplyAdamhuZUB
ό
΅void tensorflow::functor::ApplyAdamKernel<float>(int, float*, float*, float*, float const*, float const*, float const*, float const*, float const*, float const*, float const*, bool) *28!@!H!b%Adam/Adam/update_31/ResourceApplyAdamhuZUB
ό
΅void tensorflow::functor::ApplyAdamKernel<float>(int, float*, float*, float*, float const*, float const*, float const*, float const*, float const*, float const*, float const*, bool) *28!@!H!b%Adam/Adam/update_41/ResourceApplyAdamhuZUB
ό
΅void tensorflow::functor::ApplyAdamKernel<float>(int, float*, float*, float*, float const*, float const*, float const*, float const*, float const*, float const*, float const*, bool) *28ΰ @ΰ Hΰ b%Adam/Adam/update_29/ResourceApplyAdamhuZUB
ϋ
΅void tensorflow::functor::ApplyAdamKernel<float>(int, float*, float*, float*, float const*, float const*, float const*, float const*, float const*, float const*, float const*, bool) *28ΰ @ΰ Hΰ b$Adam/Adam/update_9/ResourceApplyAdamhuZUB
ό
΅void tensorflow::functor::ApplyAdamKernel<float>(int, float*, float*, float*, float const*, float const*, float const*, float const*, float const*, float const*, float const*, bool) *28ΐ @ΐ Hΐ b%Adam/Adam/update_19/ResourceApplyAdamhuZUB
ό
΅void tensorflow::functor::ApplyAdamKernel<float>(int, float*, float*, float*, float const*, float const*, float const*, float const*, float const*, float const*, float const*, bool) *28ΐ @ΐ Hΐ b%Adam/Adam/update_69/ResourceApplyAdamhuZUB
ό
΅void tensorflow::functor::ApplyAdamKernel<float>(int, float*, float*, float*, float const*, float const*, float const*, float const*, float const*, float const*, float const*, bool) *28  @  H  b%Adam/Adam/update_22/ResourceApplyAdamhuZUB
ό
΅void tensorflow::functor::ApplyAdamKernel<float>(int, float*, float*, float*, float const*, float const*, float const*, float const*, float const*, float const*, float const*, bool) *28  @  H  b%Adam/Adam/update_37/ResourceApplyAdamhuZUB
ό
΅void tensorflow::functor::ApplyAdamKernel<float>(int, float*, float*, float*, float const*, float const*, float const*, float const*, float const*, float const*, float const*, bool) *28  @  H  b%Adam/Adam/update_39/ResourceApplyAdamhuZUB
ό
΅void tensorflow::functor::ApplyAdamKernel<float>(int, float*, float*, float*, float const*, float const*, float const*, float const*, float const*, float const*, float const*, bool) *28  @  H  b%Adam/Adam/update_42/ResourceApplyAdamhuZUB
ό
΅void tensorflow::functor::ApplyAdamKernel<float>(int, float*, float*, float*, float const*, float const*, float const*, float const*, float const*, float const*, float const*, bool) *28 @ H b%Adam/Adam/update_50/ResourceApplyAdamhuZUB
ό
΅void tensorflow::functor::ApplyAdamKernel<float>(int, float*, float*, float*, float const*, float const*, float const*, float const*, float const*, float const*, float const*, bool) *28?@?H?b%Adam/Adam/update_33/ResourceApplyAdamhuZUB
ό
΅void tensorflow::functor::ApplyAdamKernel<float>(int, float*, float*, float*, float const*, float const*, float const*, float const*, float const*, float const*, float const*, bool) *28ΰ@ΰHΰb%Adam/Adam/update_27/ResourceApplyAdamhuZUB
ό
΅void tensorflow::functor::ApplyAdamKernel<float>(int, float*, float*, float*, float const*, float const*, float const*, float const*, float const*, float const*, float const*, bool) *28ΰ@ΰHΰb%Adam/Adam/update_45/ResourceApplyAdamhuZUB
ό
΅void tensorflow::functor::ApplyAdamKernel<float>(int, float*, float*, float*, float const*, float const*, float const*, float const*, float const*, float const*, float const*, bool) *28ΰ@ΰHΰb%Adam/Adam/update_58/ResourceApplyAdamhuZUB
ΰ
void tensorflow::functor::ShuffleInTensor3Simple<float, 2, 1, 0, false>(int, float const*, tensorflow::functor::Dimension<3>, float*)*28ΰ@ΰHΰXb7gradient_tape/model/conv2d_2/Conv2D/Conv2DBackpropInputhuZUB
Π
void nchwToFoldedNhwcKernel<float, float, float, true, (cudnnKernelDataType_t)2>(int, int, int, int, float const*, float*, int, int, int, int, int, int, int, int, int, int, float, float, cudnn::reduced_divisor, cudnn::reduced_divisor, cudnn::reduced_divisor)&!* 28ΐ@ΐHΐb)model/conv2d_transpose_3/conv2d_transposehu  ΘB
ω
΅void tensorflow::functor::ApplyAdamKernel<float>(int, float*, float*, float*, float const*, float const*, float const*, float const*, float const*, float const*, float const*, bool) *28ΐ@ΐHΐb"Adam/Adam/update/ResourceApplyAdamhuZUB
ό
΅void tensorflow::functor::ApplyAdamKernel<float>(int, float*, float*, float*, float const*, float const*, float const*, float const*, float const*, float const*, float const*, bool) *28ΐ@ΐHΐb%Adam/Adam/update_47/ResourceApplyAdamhuZUB
ό
΅void tensorflow::functor::ApplyAdamKernel<float>(int, float*, float*, float*, float const*, float const*, float const*, float const*, float const*, float const*, float const*, bool) *28ΐ@ΐHΐb%Adam/Adam/update_53/ResourceApplyAdamhuZUB
ό
΅void tensorflow::functor::ApplyAdamKernel<float>(int, float*, float*, float*, float const*, float const*, float const*, float const*, float const*, float const*, float const*, bool) *28ΐ@ΐHΐb%Adam/Adam/update_71/ResourceApplyAdamhuZUB
ό
΅void tensorflow::functor::ApplyAdamKernel<float>(int, float*, float*, float*, float const*, float const*, float const*, float const*, float const*, float const*, float const*, bool) *2	8ΐ@ΐHΐb%Adam/Adam/update_68/ResourceApplyAdamhuZUB
ό
΅void tensorflow::functor::ApplyAdamKernel<float>(int, float*, float*, float*, float const*, float const*, float const*, float const*, float const*, float const*, float const*, bool) *28 @ H b%Adam/Adam/update_16/ResourceApplyAdamhuZUB
ό
΅void tensorflow::functor::ApplyAdamKernel<float>(int, float*, float*, float*, float const*, float const*, float const*, float const*, float const*, float const*, float const*, bool) *28 @ H b%Adam/Adam/update_21/ResourceApplyAdamhuZUB
Π
void tensorflow::functor::ShuffleInTensor3Simple<float, 2, 1, 0, false>(int, float const*, tensorflow::functor::Dimension<3>, float*)*28@Hb)model/conv2d_transpose_4/conv2d_transposehuZUB
ό
΅void tensorflow::functor::ApplyAdamKernel<float>(int, float*, float*, float*, float const*, float const*, float const*, float const*, float const*, float const*, float const*, bool) *28ΰ@ΰHΰb%Adam/Adam/update_10/ResourceApplyAdamhuZUB
ϋ
΅void tensorflow::functor::ApplyAdamKernel<float>(int, float*, float*, float*, float const*, float const*, float const*, float const*, float const*, float const*, float const*, bool) *28ΰ@ΰHΰb$Adam/Adam/update_4/ResourceApplyAdamhuZUB
ό
΅void tensorflow::functor::ApplyAdamKernel<float>(int, float*, float*, float*, float const*, float const*, float const*, float const*, float const*, float const*, float const*, bool) *28ΐ@ΐHΐb%Adam/Adam/update_67/ResourceApplyAdamhuZUB
ό
΅void tensorflow::functor::ApplyAdamKernel<float>(int, float*, float*, float*, float const*, float const*, float const*, float const*, float const*, float const*, float const*, bool) *28@Hb%Adam/Adam/update_49/ResourceApplyAdamhuZUB
Ί
void tensorflow::functor::ShuffleInTensor3Simple<float, 2, 1, 0, false>(int, float const*, tensorflow::functor::Dimension<3>, float*)*28@Hbmodel/conv2d_2/ReluhuZUB
β
void tensorflow::functor::ShuffleInTensor3Simple<float, 2, 1, 0, false>(int, float const*, tensorflow::functor::Dimension<3>, float*)*2$8@HXb9gradient_tape/model/conv2d_19/Conv2D/Conv2DBackpropFilterhuZUB
ό
΅void tensorflow::functor::ApplyAdamKernel<float>(int, float*, float*, float*, float const*, float const*, float const*, float const*, float const*, float const*, float const*, bool) *28ΰ@ΰHΰb%Adam/Adam/update_35/ResourceApplyAdamhuZUB
»
void tensorflow::functor::ShuffleInTensor3Simple<float, 2, 1, 0, false>(int, float const*, tensorflow::functor::Dimension<3>, float*)*2$8ΰ@ΰHΰbmodel/conv2d_19/ReluhuZUB
ό
΅void tensorflow::functor::ApplyAdamKernel<float>(int, float*, float*, float*, float const*, float const*, float const*, float const*, float const*, float const*, float const*, bool) *28ΐ@ΐHΐb%Adam/Adam/update_15/ResourceApplyAdamhuZUB
ό
΅void tensorflow::functor::ApplyAdamKernel<float>(int, float*, float*, float*, float const*, float const*, float const*, float const*, float const*, float const*, float const*, bool) *28ΐ@ΐHΐb%Adam/Adam/update_43/ResourceApplyAdamhuZUB
α
void tensorflow::functor::ShuffleInTensor3Simple<float, 2, 1, 0, false>(int, float const*, tensorflow::functor::Dimension<3>, float*)*2$8ΐ@ΐHΐXb8gradient_tape/model/conv2d_19/Conv2D/Conv2DBackpropInputhuZUB
α
void tensorflow::functor::ShuffleInTensor3Simple<float, 2, 1, 0, false>(int, float const*, tensorflow::functor::Dimension<3>, float*)*2$8ΐ@ΐHΐXb8gradient_tape/model/conv2d_3/Conv2D/Conv2DBackpropFilterhuZUB
ΰ
void tensorflow::functor::ShuffleInTensor3Simple<float, 2, 1, 0, false>(int, float const*, tensorflow::functor::Dimension<3>, float*)*2$8ΐ@ΐHΐXb7gradient_tape/model/conv2d_3/Conv2D/Conv2DBackpropInputhuZUB
Ί
void tensorflow::functor::ShuffleInTensor3Simple<float, 2, 1, 0, false>(int, float const*, tensorflow::functor::Dimension<3>, float*)*2$8 @ H bmodel/conv2d_3/ReluhuZUB
»
void tensorflow::functor::ShuffleInTensor3Simple<float, 2, 1, 0, false>(int, float const*, tensorflow::functor::Dimension<3>, float*)*28@Hbmodel/conv2d_20/ReluhuZUB
Π
void tensorflow::functor::ShuffleInTensor3Simple<float, 2, 1, 0, false>(int, float const*, tensorflow::functor::Dimension<3>, float*)*2 8@Hb)model/conv2d_transpose_3/conv2d_transposehuZUB
β
void tensorflow::functor::ShuffleInTensor3Simple<float, 2, 1, 0, false>(int, float const*, tensorflow::functor::Dimension<3>, float*)*2	8ΰ@ΰHΰXb9gradient_tape/model/conv2d_21/Conv2D/Conv2DBackpropFilterhuZUB
α
void tensorflow::functor::ShuffleInTensor3Simple<float, 2, 1, 0, false>(int, float const*, tensorflow::functor::Dimension<3>, float*)*28ΰ@ΰHΰXb8gradient_tape/model/conv2d_2/Conv2D/Conv2DBackpropFilterhuZUB
ό
΅void tensorflow::functor::ApplyAdamKernel<float>(int, float*, float*, float*, float const*, float const*, float const*, float const*, float const*, float const*, float const*, bool) *28ΐ@ΐHΐb%Adam/Adam/update_51/ResourceApplyAdamhuZUB
ό
΅void tensorflow::functor::ApplyAdamKernel<float>(int, float*, float*, float*, float const*, float const*, float const*, float const*, float const*, float const*, float const*, bool) *28 @ H b%Adam/Adam/update_55/ResourceApplyAdamhuZUB
ό
΅void tensorflow::functor::ApplyAdamKernel<float>(int, float*, float*, float*, float const*, float const*, float const*, float const*, float const*, float const*, float const*, bool) *28@Hb%Adam/Adam/update_13/ResourceApplyAdamhuZUB
ό
΅void tensorflow::functor::ApplyAdamKernel<float>(int, float*, float*, float*, float const*, float const*, float const*, float const*, float const*, float const*, float const*, bool) *28@Hb%Adam/Adam/update_23/ResourceApplyAdamhuZUB
η
void tensorflow::functor::ShuffleInTensor3Simple<float, 2, 1, 0, false>(int, float const*, tensorflow::functor::Dimension<3>, float*)*2 8@HXb>gradient_tape/model/conv2d_transpose_3/conv2d_transpose/Conv2DhuZUB
ό
΅void tensorflow::functor::ApplyAdamKernel<float>(int, float*, float*, float*, float const*, float const*, float const*, float const*, float const*, float const*, float const*, bool) *28ΐ@ΐHΐb%Adam/Adam/update_57/ResourceApplyAdamhuZUB
ό
΅void tensorflow::functor::ApplyAdamKernel<float>(int, float*, float*, float*, float const*, float const*, float const*, float const*, float const*, float const*, float const*, bool) *28ΐ@ΐHΐb%Adam/Adam/update_59/ResourceApplyAdamhuZUB
ό
΅void tensorflow::functor::ApplyAdamKernel<float>(int, float*, float*, float*, float const*, float const*, float const*, float const*, float const*, float const*, float const*, bool) *28ΐ@ΐHΐb%Adam/Adam/update_65/ResourceApplyAdamhuZUB
»
void tensorflow::functor::ShuffleInTensor3Simple<float, 2, 1, 0, false>(int, float const*, tensorflow::functor::Dimension<3>, float*)*2	8ΐ@ΐHΐbmodel/conv2d_21/ReluhuZUB
α
void tensorflow::functor::ShuffleInTensor3Simple<float, 2, 1, 0, false>(int, float const*, tensorflow::functor::Dimension<3>, float*)*28ΐ@ΐHΐXb8gradient_tape/model/conv2d_20/Conv2D/Conv2DBackpropInputhuZUB
ϋ
΅void tensorflow::functor::ApplyAdamKernel<float>(int, float*, float*, float*, float const*, float const*, float const*, float const*, float const*, float const*, float const*, bool) *28 @ H b$Adam/Adam/update_5/ResourceApplyAdamhuZUB
ό
΅void tensorflow::functor::ApplyAdamKernel<float>(int, float*, float*, float*, float const*, float const*, float const*, float const*, float const*, float const*, float const*, bool) *28 @ H b%Adam/Adam/update_63/ResourceApplyAdamhuZUB
ΰ
void tensorflow::functor::ShuffleInTensor3Simple<float, 2, 1, 0, false>(int, float const*, tensorflow::functor::Dimension<3>, float*)*2	8 @ H Xb7gradient_tape/model/conv2d_1/Conv2D/Conv2DBackpropInputhuZUB
ό
΅void tensorflow::functor::ApplyAdamKernel<float>(int, float*, float*, float*, float const*, float const*, float const*, float const*, float const*, float const*, float const*, bool) *28@Hb%Adam/Adam/update_61/ResourceApplyAdamhuZUB
ό
΅void tensorflow::functor::ApplyAdamKernel<float>(int, float*, float*, float*, float const*, float const*, float const*, float const*, float const*, float const*, float const*, bool) *28@Hb%Adam/Adam/update_11/ResourceApplyAdamhuZUB
ό
΅void tensorflow::functor::ApplyAdamKernel<float>(int, float*, float*, float*, float const*, float const*, float const*, float const*, float const*, float const*, float const*, bool) *28@Hb%Adam/Adam/update_17/ResourceApplyAdamhuZUB
α
void tensorflow::functor::ShuffleInTensor3Simple<float, 2, 1, 0, false>(int, float const*, tensorflow::functor::Dimension<3>, float*)*2	8@HXb8gradient_tape/model/conv2d_21/Conv2D/Conv2DBackpropInputhuZUB
β
void tensorflow::functor::ShuffleInTensor3Simple<float, 2, 1, 0, false>(int, float const*, tensorflow::functor::Dimension<3>, float*)*28@HXb9gradient_tape/model/conv2d_20/Conv2D/Conv2DBackpropFilterhuZUB
Έ
void tensorflow::functor::ShuffleInTensor3Simple<float, 2, 1, 0, false>(int, float const*, tensorflow::functor::Dimension<3>, float*)*28?@?H?bmodel/conv2d/ReluhuZUB
ϋ
΅void tensorflow::functor::ApplyAdamKernel<float>(int, float*, float*, float*, float const*, float const*, float const*, float const*, float const*, float const*, float const*, bool) *28ΰ@ΰHΰb$Adam/Adam/update_3/ResourceApplyAdamhuZUB
υ
void tensorflow::functor::ShuffleInTensor3Simple<float, 2, 1, 0, false>(int, float const*, tensorflow::functor::Dimension<3>, float*)*2 8ΰ@ΰHΰXbLgradient_tape/model/conv2d_transpose_3/conv2d_transpose/Conv2DBackpropFilterhuZUB
ϋ
΅void tensorflow::functor::ApplyAdamKernel<float>(int, float*, float*, float*, float const*, float const*, float const*, float const*, float const*, float const*, float const*, bool) *28ΐ@ΐHΐb$Adam/Adam/update_7/ResourceApplyAdamhuZUB
η
void tensorflow::functor::ShuffleInTensor3Simple<float, 2, 1, 0, false>(int, float const*, tensorflow::functor::Dimension<3>, float*)*28ΐ@ΐHΐXb>gradient_tape/model/conv2d_transpose_4/conv2d_transpose/Conv2DhuZUB
΅
~void cudnn::winograd::generateWinogradTilesKernel<0, float, float>(cudnn::winograd::GenerateWinogradTilesParams<float, float>)(D* 2 8 @ H bmodel/conv2d_18/ReluhuZU·B
Ί
void tensorflow::functor::ShuffleInTensor3Simple<float, 2, 1, 0, false>(int, float const*, tensorflow::functor::Dimension<3>, float*)*2	8@Hbmodel/conv2d_1/ReluhuZUB
―
Tvoid cask_cudnn::computeOffsetsKernel<true, false>(cask_cudnn::ComputeOffsetsParams)*2e8ί@ίHίXb8gradient_tape/model/conv2d_22/Conv2D/Conv2DBackpropInputhu  ΘB
ό
΅void tensorflow::functor::ApplyAdamKernel<float>(int, float*, float*, float*, float const*, float const*, float const*, float const*, float const*, float const*, float const*, bool) *28ΐ@ΐHΐb%Adam/Adam/update_70/ResourceApplyAdamhuZUB
υ
void tensorflow::functor::ShuffleInTensor3Simple<float, 2, 1, 0, false>(int, float const*, tensorflow::functor::Dimension<3>, float*)*28ΐ@ΐHΐXbLgradient_tape/model/conv2d_transpose_4/conv2d_transpose/Conv2DBackpropFilterhuZUB
α
void tensorflow::functor::ShuffleInTensor3Simple<float, 2, 1, 0, false>(int, float const*, tensorflow::functor::Dimension<3>, float*)*2	8ΐ@ΐHΐXb8gradient_tape/model/conv2d_1/Conv2D/Conv2DBackpropFilterhuZUB
Ϋ
~void cudnn::winograd::generateWinogradTilesKernel<0, float, float>(cudnn::winograd::GenerateWinogradTilesParams<float, float>)(D* 28@HXb8gradient_tape/model/conv2d_18/Conv2D/Conv2DBackpropInputhuZU·B
ϋ
΅void tensorflow::functor::ApplyAdamKernel<float>(int, float*, float*, float*, float const*, float const*, float const*, float const*, float const*, float const*, float const*, bool) *28@Hb$Adam/Adam/update_1/ResourceApplyAdamhuZUB
ϋ
¦void tensorflow::functor::ColumnReduceKernel<float const*, float*, cub::Sum>(float const*, float*, int, int, cub::Sum, std::iterator_traits<float const*>::value_type)!*  2 8@Hb1gradient_tape/model/conv2d_11/BiasAdd/BiasAddGradhuZUB

²void tensorflow::functor::ColumnReduceMax16ColumnsKernel<float const*, float*, cub::Sum>(float const*, float*, int, int, cub::Sum, std::iterator_traits<float const*>::value_type)!* 28@Hb1gradient_tape/model/conv2d_22/BiasAdd/BiasAddGradhuZUB
΄
~void cudnn::winograd::generateWinogradTilesKernel<0, float, float>(cudnn::winograd::GenerateWinogradTilesParams<float, float>)(D* 28ΰ@ΰHΰbmodel/conv2d_1/ReluhuZU·B
Ο
void nchwToFoldedNhwcKernel<float, float, float, true, (cudnnKernelDataType_t)2>(int, int, int, int, float const*, float*, int, int, int, int, int, int, int, int, int, int, float, float, cudnn::reduced_divisor, cudnn::reduced_divisor, cudnn::reduced_divisor)&!* 2@8ΰ@ΰHΰb)model/conv2d_transpose_4/conv2d_transposehu  ΘB
ϋ
¦void tensorflow::functor::ColumnReduceKernel<float const*, float*, cub::Sum>(float const*, float*, int, int, cub::Sum, std::iterator_traits<float const*>::value_type)!*  28ΰ@ΰHΰb1gradient_tape/model/conv2d_15/BiasAdd/BiasAddGradhuZUB
ϋ
¦void tensorflow::functor::ColumnReduceKernel<float const*, float*, cub::Sum>(float const*, float*, int, int, cub::Sum, std::iterator_traits<float const*>::value_type)!*  28ΰ@ΰHΰb1gradient_tape/model/conv2d_12/BiasAdd/BiasAddGradhuZUB
η
void tensorflow::functor::CleanupSegments<float*, float*, cub::Sum>(float*, float*, int, int, int, cub::Sum, std::iterator_traits<float*>::value_type)*  28ΐ@ΐHΐb0gradient_tape/model/conv2d_7/BiasAdd/BiasAddGradhuZUB
ϋ
¦void tensorflow::functor::ColumnReduceKernel<float const*, float*, cub::Sum>(float const*, float*, int, int, cub::Sum, std::iterator_traits<float const*>::value_type)!*  28 @ H b1gradient_tape/model/conv2d_13/BiasAdd/BiasAddGradhuZUB
ϋ
¦void tensorflow::functor::ColumnReduceKernel<float const*, float*, cub::Sum>(float const*, float*, int, int, cub::Sum, std::iterator_traits<float const*>::value_type)!*  2 8 @ H b1gradient_tape/model/conv2d_10/BiasAdd/BiasAddGradhuZUB
ό
·void cub::DeviceReduceSingleTileKernel<cub::DeviceReducePolicy<float, float, int, tensorflow::functor::Sum<float> >::Policy600, float*, tensorflow::TransformOutputIterator<float, float, tensorflow::functor::DividesBy<float, float>, long>, int, tensorflow::functor::Sum<float>, float>(float*, tensorflow::TransformOutputIterator<float, float, tensorflow::functor::DividesBy<float, float>, long>, int, tensorflow::functor::Sum<float>, float)0*28@Hb!weighted_binary_crossentropy/Meanhu  ΘB
΄
~void cudnn::winograd::generateWinogradTilesKernel<0, float, float>(cudnn::winograd::GenerateWinogradTilesParams<float, float>)(D* 28@Hbmodel/conv2d_2/ReluhuZU·B
Ϋ
~void cudnn::winograd::generateWinogradTilesKernel<0, float, float>(cudnn::winograd::GenerateWinogradTilesParams<float, float>)(D* 28@HXb8gradient_tape/model/conv2d_19/Conv2D/Conv2DBackpropInputhuZU·B

¦void tensorflow::functor::ColumnReduceKernel<float const*, float*, cub::Sum>(float const*, float*, int, int, cub::Sum, std::iterator_traits<float const*>::value_type)!*  28@Hb8gradient_tape/model/conv2d_transpose/BiasAdd/BiasAddGradhuZUB
Ϊ
~void cudnn::winograd::generateWinogradTilesKernel<0, float, float>(cudnn::winograd::GenerateWinogradTilesParams<float, float>)(D* 28ΰ@ΰHΰXb7gradient_tape/model/conv2d_3/Conv2D/Conv2DBackpropInputhuZU·B
ρ
Εvoid Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::div_no_nan_op<float>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, int>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::div_no_nan_op<float>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, int>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*28ΐ@ΐHΐb
div_no_nanhuZUB

Εvoid Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::div_no_nan_op<float>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, int>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::div_no_nan_op<float>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, int>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*28ΐ@ΐHΐb0weighted_binary_crossentropy/weighted_loss/valuehuZUB
Ϊ
~void cudnn::winograd::generateWinogradTilesKernel<0, float, float>(cudnn::winograd::GenerateWinogradTilesParams<float, float>)(D* 2 8ΐ@ΐHΐXb7gradient_tape/model/conv2d_4/Conv2D/Conv2DBackpropInputhuZU·B

¦void tensorflow::functor::ColumnReduceKernel<float const*, float*, cub::Sum>(float const*, float*, int, int, cub::Sum, std::iterator_traits<float const*>::value_type)!*  28ΐ@ΐHΐb:gradient_tape/model/conv2d_transpose_2/BiasAdd/BiasAddGradhuZUB
ϋ
¦void tensorflow::functor::ColumnReduceKernel<float const*, float*, cub::Sum>(float const*, float*, int, int, cub::Sum, std::iterator_traits<float const*>::value_type)!*  28ΐ@ΐHΐb1gradient_tape/model/conv2d_14/BiasAdd/BiasAddGradhuZUB

¦void tensorflow::functor::ColumnReduceKernel<float const*, float*, cub::Sum>(float const*, float*, int, int, cub::Sum, std::iterator_traits<float const*>::value_type)!*  28ΐ@ΐHΐb:gradient_tape/model/conv2d_transpose_1/BiasAdd/BiasAddGradhuZUB
ϋ
¦void tensorflow::functor::ColumnReduceKernel<float const*, float*, cub::Sum>(float const*, float*, int, int, cub::Sum, std::iterator_traits<float const*>::value_type)!*  28 @ H b1gradient_tape/model/conv2d_16/BiasAdd/BiasAddGradhuZUB
ϊ
¦void tensorflow::functor::ColumnReduceKernel<float const*, float*, cub::Sum>(float const*, float*, int, int, cub::Sum, std::iterator_traits<float const*>::value_type)!*  28 @ H b0gradient_tape/model/conv2d_9/BiasAdd/BiasAddGradhuZUB
΅
~void cudnn::winograd::generateWinogradTilesKernel<0, float, float>(cudnn::winograd::GenerateWinogradTilesParams<float, float>)(D* 28@Hbmodel/conv2d_19/ReluhuZU·B

¦void tensorflow::functor::ColumnReduceKernel<float const*, float*, cub::Sum>(float const*, float*, int, int, cub::Sum, std::iterator_traits<float const*>::value_type)!*  28@Hb:gradient_tape/model/conv2d_transpose_4/BiasAdd/BiasAddGradhuZUB
ϋ
¦void tensorflow::functor::ColumnReduceKernel<float const*, float*, cub::Sum>(float const*, float*, int, int, cub::Sum, std::iterator_traits<float const*>::value_type)!*  28@Hb1gradient_tape/model/conv2d_19/BiasAdd/BiasAddGradhuZUB
ϊ
¦void tensorflow::functor::ColumnReduceKernel<float const*, float*, cub::Sum>(float const*, float*, int, int, cub::Sum, std::iterator_traits<float const*>::value_type)!*  28@Hb0gradient_tape/model/conv2d_8/BiasAdd/BiasAddGradhuZUB
΄
~void cudnn::winograd::generateWinogradTilesKernel<0, float, float>(cudnn::winograd::GenerateWinogradTilesParams<float, float>)(D* 28α@αHαbmodel/conv2d_3/ReluhuZU·B
ϋ
¦void tensorflow::functor::ColumnReduceKernel<float const*, float*, cub::Sum>(float const*, float*, int, int, cub::Sum, std::iterator_traits<float const*>::value_type)!*  28α@αHαb1gradient_tape/model/conv2d_17/BiasAdd/BiasAddGradhuZUB
ϋ
¦void tensorflow::functor::ColumnReduceKernel<float const*, float*, cub::Sum>(float const*, float*, int, int, cub::Sum, std::iterator_traits<float const*>::value_type)!*  28ΰ@ΰHΰb1gradient_tape/model/conv2d_18/BiasAdd/BiasAddGradhuZUB

¦void tensorflow::functor::ColumnReduceKernel<float const*, float*, cub::Sum>(float const*, float*, int, int, cub::Sum, std::iterator_traits<float const*>::value_type)!*  28ΰ@ΰHΰb:gradient_tape/model/conv2d_transpose_3/BiasAdd/BiasAddGradhuZUB
ϊ
¦void tensorflow::functor::ColumnReduceKernel<float const*, float*, cub::Sum>(float const*, float*, int, int, cub::Sum, std::iterator_traits<float const*>::value_type)!*  28ΰ@ΰHΰb0gradient_tape/model/conv2d_4/BiasAdd/BiasAddGradhuZUB
ϊ
¦void tensorflow::functor::ColumnReduceKernel<float const*, float*, cub::Sum>(float const*, float*, int, int, cub::Sum, std::iterator_traits<float const*>::value_type)!*  28ΰ@ΰHΰb0gradient_tape/model/conv2d_5/BiasAdd/BiasAddGradhuZUB
ϊ
¦void tensorflow::functor::ColumnReduceKernel<float const*, float*, cub::Sum>(float const*, float*, int, int, cub::Sum, std::iterator_traits<float const*>::value_type)!*  28ΰ@ΰHΰb0gradient_tape/model/conv2d_7/BiasAdd/BiasAddGradhuZUB
I
 Pow_GPU_DT_FLOAT_DT_FLOAT_kernel*28ΐ@ΐHΐbAdam/PowhuZUB
ϊ
¦void tensorflow::functor::ColumnReduceKernel<float const*, float*, cub::Sum>(float const*, float*, int, int, cub::Sum, std::iterator_traits<float const*>::value_type)!*  28ΐ@ΐHΐb0gradient_tape/model/conv2d_6/BiasAdd/BiasAddGradhuZUB

Uvoid cask_cudnn::computeOffsetsKernel<false, false>(cask_cudnn::ComputeOffsetsParams)*2e8 @ H bmodel/conv2d/Reluhu  ΘB
η
void tensorflow::functor::CleanupSegments<float*, float*, cub::Sum>(float*, float*, int, int, int, cub::Sum, std::iterator_traits<float*>::value_type)*  28 @ H b0gradient_tape/model/conv2d_4/BiasAdd/BiasAddGradhuZUB
ψ
¦void tensorflow::functor::ColumnReduceKernel<float const*, float*, cub::Sum>(float const*, float*, int, int, cub::Sum, std::iterator_traits<float const*>::value_type)!*  28 @ H b.gradient_tape/model/conv2d/BiasAdd/BiasAddGradhuZUB
ϋ
¦void tensorflow::functor::ColumnReduceKernel<float const*, float*, cub::Sum>(float const*, float*, int, int, cub::Sum, std::iterator_traits<float const*>::value_type)!*  28 @ H b1gradient_tape/model/conv2d_20/BiasAdd/BiasAddGradhuZUB
ϊ
¦void tensorflow::functor::ColumnReduceKernel<float const*, float*, cub::Sum>(float const*, float*, int, int, cub::Sum, std::iterator_traits<float const*>::value_type)!*  28 @ H b0gradient_tape/model/conv2d_2/BiasAdd/BiasAddGradhuZUB
ϊ
¦void tensorflow::functor::ColumnReduceKernel<float const*, float*, cub::Sum>(float const*, float*, int, int, cub::Sum, std::iterator_traits<float const*>::value_type)!*  28 @ H b0gradient_tape/model/conv2d_3/BiasAdd/BiasAddGradhuZUB
ϋ
¦void tensorflow::functor::ColumnReduceKernel<float const*, float*, cub::Sum>(float const*, float*, int, int, cub::Sum, std::iterator_traits<float const*>::value_type)!*  28@Hb1gradient_tape/model/conv2d_21/BiasAdd/BiasAddGradhuZUB
θ
void tensorflow::functor::CleanupSegments<float*, float*, cub::Sum>(float*, float*, int, int, int, cub::Sum, std::iterator_traits<float*>::value_type)*  28@Hb1gradient_tape/model/conv2d_15/BiasAdd/BiasAddGradhuZUB
Ϋ
~void cudnn::winograd::generateWinogradTilesKernel<0, float, float>(cudnn::winograd::GenerateWinogradTilesParams<float, float>)(D* 28@HXb8gradient_tape/model/conv2d_20/Conv2D/Conv2DBackpropInputhuZU·B
ρ
void tensorflow::functor::CleanupSegments<float*, float*, cub::Sum>(float*, float*, int, int, int, cub::Sum, std::iterator_traits<float*>::value_type)*  28@Hb:gradient_tape/model/conv2d_transpose_4/BiasAdd/BiasAddGradhuZUB
ρ
void tensorflow::functor::CleanupSegments<float*, float*, cub::Sum>(float*, float*, int, int, int, cub::Sum, std::iterator_traits<float*>::value_type)*  28@Hb:gradient_tape/model/conv2d_transpose_1/BiasAdd/BiasAddGradhuZUB
θ
void tensorflow::functor::CleanupSegments<float*, float*, cub::Sum>(float*, float*, int, int, int, cub::Sum, std::iterator_traits<float*>::value_type)*  28ΰ@ΰHΰb1gradient_tape/model/conv2d_18/BiasAdd/BiasAddGradhuZUB
θ
void tensorflow::functor::CleanupSegments<float*, float*, cub::Sum>(float*, float*, int, int, int, cub::Sum, std::iterator_traits<float*>::value_type)*  28ΰ@ΰHΰb1gradient_tape/model/conv2d_19/BiasAdd/BiasAddGradhuZUB
ρ
void tensorflow::functor::CleanupSegments<float*, float*, cub::Sum>(float*, float*, int, int, int, cub::Sum, std::iterator_traits<float*>::value_type)*  28ΰ@ΰHΰb:gradient_tape/model/conv2d_transpose_3/BiasAdd/BiasAddGradhuZUB
θ
void tensorflow::functor::CleanupSegments<float*, float*, cub::Sum>(float*, float*, int, int, int, cub::Sum, std::iterator_traits<float*>::value_type)*  28ΰ@ΰHΰb1gradient_tape/model/conv2d_16/BiasAdd/BiasAddGradhuZUB
ϊ
¦void tensorflow::functor::ColumnReduceKernel<float const*, float*, cub::Sum>(float const*, float*, int, int, cub::Sum, std::iterator_traits<float const*>::value_type)!*  28ΰ@ΰHΰb0gradient_tape/model/conv2d_1/BiasAdd/BiasAddGradhuZUB
η
void tensorflow::functor::CleanupSegments<float*, float*, cub::Sum>(float*, float*, int, int, int, cub::Sum, std::iterator_traits<float*>::value_type)*  28ΐ@ΐHΐb0gradient_tape/model/conv2d_3/BiasAdd/BiasAddGradhuZUB
θ
void tensorflow::functor::CleanupSegments<float*, float*, cub::Sum>(float*, float*, int, int, int, cub::Sum, std::iterator_traits<float*>::value_type)*  28ΐ@ΐHΐb1gradient_tape/model/conv2d_17/BiasAdd/BiasAddGradhuZUB
ί
void tensorflow::functor::ShuffleInTensor3Simple<float, 2, 1, 0, false>(int, float const*, tensorflow::functor::Dimension<3>, float*)*28ΐ@ΐHΐXb6gradient_tape/model/conv2d/Conv2D/Conv2DBackpropFilterhuZUB
K
"AddV2_GPU_DT_INT64_DT_INT64_kernel*28Ώ@ΏHΏbAdam/addhuZUB
ρ
void tensorflow::functor::CleanupSegments<float*, float*, cub::Sum>(float*, float*, int, int, int, cub::Sum, std::iterator_traits<float*>::value_type)*  28Ώ@ΏHΏb:gradient_tape/model/conv2d_transpose_2/BiasAdd/BiasAddGradhuZUB
η
void tensorflow::functor::CleanupSegments<float*, float*, cub::Sum>(float*, float*, int, int, int, cub::Sum, std::iterator_traits<float*>::value_type)*  28 @ H b0gradient_tape/model/conv2d_1/BiasAdd/BiasAddGradhuZUB
θ
void tensorflow::functor::CleanupSegments<float*, float*, cub::Sum>(float*, float*, int, int, int, cub::Sum, std::iterator_traits<float*>::value_type)*  28 @ H b1gradient_tape/model/conv2d_21/BiasAdd/BiasAddGradhuZUB
η
void tensorflow::functor::CleanupSegments<float*, float*, cub::Sum>(float*, float*, int, int, int, cub::Sum, std::iterator_traits<float*>::value_type)*  28 @ H b0gradient_tape/model/conv2d_5/BiasAdd/BiasAddGradhuZUB
θ
void tensorflow::functor::CleanupSegments<float*, float*, cub::Sum>(float*, float*, int, int, int, cub::Sum, std::iterator_traits<float*>::value_type)*  28 @ H b1gradient_tape/model/conv2d_14/BiasAdd/BiasAddGradhuZUB
Ϊ
~void cudnn::winograd::generateWinogradTilesKernel<0, float, float>(cudnn::winograd::GenerateWinogradTilesParams<float, float>)(D* 28@HXb7gradient_tape/model/conv2d_1/Conv2D/Conv2DBackpropInputhuZU·B
ε
void tensorflow::functor::CleanupSegments<float*, float*, cub::Sum>(float*, float*, int, int, int, cub::Sum, std::iterator_traits<float*>::value_type)*  28@Hb.gradient_tape/model/conv2d/BiasAdd/BiasAddGradhuZUB
θ
void tensorflow::functor::CleanupSegments<float*, float*, cub::Sum>(float*, float*, int, int, int, cub::Sum, std::iterator_traits<float*>::value_type)*  28@Hb1gradient_tape/model/conv2d_20/BiasAdd/BiasAddGradhuZUB
η
void tensorflow::functor::CleanupSegments<float*, float*, cub::Sum>(float*, float*, int, int, int, cub::Sum, std::iterator_traits<float*>::value_type)*  28@Hb0gradient_tape/model/conv2d_2/BiasAdd/BiasAddGradhuZUB
η
void tensorflow::functor::CleanupSegments<float*, float*, cub::Sum>(float*, float*, int, int, int, cub::Sum, std::iterator_traits<float*>::value_type)*  28@Hb0gradient_tape/model/conv2d_6/BiasAdd/BiasAddGradhuZUB

Ωvoid Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<long, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<long const, long const>, Eigen::TensorMap<Eigen::Tensor<long, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<long const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<long, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<long const, long const>, Eigen::TensorMap<Eigen::Tensor<long, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<long const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*28α@αHαbAssignAddVariableOp_2huZUB
΅
~void cudnn::winograd::generateWinogradTilesKernel<0, float, float>(cudnn::winograd::GenerateWinogradTilesParams<float, float>)(D* 28ΰ@ΰHΰbmodel/conv2d_20/ReluhuZU·B
΅
~void cudnn::winograd::generateWinogradTilesKernel<0, float, float>(cudnn::winograd::GenerateWinogradTilesParams<float, float>)(D* 28ΐ@ΐHΐbmodel/conv2d_21/ReluhuZU·B
Ϋ
~void cudnn::winograd::generateWinogradTilesKernel<0, float, float>(cudnn::winograd::GenerateWinogradTilesParams<float, float>)(D* 28ΐ@ΐHΐXb8gradient_tape/model/conv2d_21/Conv2D/Conv2DBackpropInputhuZU·B
Ά
Uvoid cask_cudnn::computeOffsetsKernel<false, false>(cask_cudnn::ComputeOffsetsParams)*28 @ H Xb>gradient_tape/model/conv2d_transpose_3/conv2d_transpose/Conv2Dhu  ΘB
―
Scask_cudnn::computeWgradSplitKOffsetsKernel(cask_cudnn::ComputeSplitKOffsetsParams)*2P8@HXb9gradient_tape/model/conv2d_19/Conv2D/Conv2DBackpropFilterhu  ΘB
Ά
Uvoid cask_cudnn::computeOffsetsKernel<false, false>(cask_cudnn::ComputeOffsetsParams)*28@HXb>gradient_tape/model/conv2d_transpose_2/conv2d_transpose/Conv2Dhu  ΘB
Ώ
void tensorflow::functor::ShuffleInTensor3Simple<float, 2, 1, 0, false>(int, float const*, tensorflow::functor::Dimension<3>, float*)*28@HXbmodel/conv2d_22/Conv2DhuZUB
D
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*28ΰ@ΰHΰbMulhuZUB

γvoid Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*28ΰ@ΰHΰbAssignAddVariableOphuZUB

Ωvoid Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<long, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<long const, long const>, Eigen::TensorMap<Eigen::Tensor<long, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<long const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<long, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<long const, long const>, Eigen::TensorMap<Eigen::Tensor<long, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<long const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*28ΰ@ΰHΰbAdam/Adam/AssignAddVariableOphuZUB
΄
Uvoid cask_cudnn::computeOffsetsKernel<false, false>(cask_cudnn::ComputeOffsetsParams)*28ΰ@ΰHΰXb<gradient_tape/model/conv2d_transpose/conv2d_transpose/Conv2Dhu  ΘB
Ά
Uvoid cask_cudnn::computeOffsetsKernel<false, false>(cask_cudnn::ComputeOffsetsParams)*28ΰ@ΰHΰXb>gradient_tape/model/conv2d_transpose_4/conv2d_transpose/Conv2Dhu  ΘB
α
void tensorflow::functor::ShuffleInTensor3Simple<float, 2, 1, 0, false>(int, float const*, tensorflow::functor::Dimension<3>, float*)*28ΰ@ΰHΰXb8gradient_tape/model/conv2d_22/Conv2D/Conv2DBackpropInputhuZUB
ͺ
Ncask_cudnn::computeWgradBOffsetsKernel(cask_cudnn::ComputeWgradBOffsetsParams)*28ΐ@ΐHΐXb9gradient_tape/model/conv2d_19/Conv2D/Conv2DBackpropFilterhu  ΘB

Υvoid Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<float, Eigen::TensorMap<Eigen::Tensor<long const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<float, Eigen::TensorMap<Eigen::Tensor<long const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*28ΐ@ΐHΐbAdam/Cast_1huZUB
Ϊ
~void cudnn::winograd::generateWinogradTilesKernel<0, float, float>(cudnn::winograd::GenerateWinogradTilesParams<float, float>)(D* 28ΐ@ΐHΐXb7gradient_tape/model/conv2d_2/Conv2D/Conv2DBackpropInputhuZU·B
½
Ncask_cudnn::computeWgradBOffsetsKernel(cask_cudnn::ComputeWgradBOffsetsParams)*28 @ H XbLgradient_tape/model/conv2d_transpose_3/conv2d_transpose/Conv2DBackpropFilterhu  ΘB

γvoid Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*28 @ H bAssignAddVariableOp_1huZUB
β
void tensorflow::functor::ShuffleInTensor3Simple<float, 2, 1, 0, false>(int, float const*, tensorflow::functor::Dimension<3>, float*)*28 @ H Xb9gradient_tape/model/conv2d_22/Conv2D/Conv2DBackpropFilterhuZUB
K
 Pow_GPU_DT_FLOAT_DT_FLOAT_kernel*28@Hb
Adam/Pow_1huZUB
©
Ncask_cudnn::computeWgradBOffsetsKernel(cask_cudnn::ComputeWgradBOffsetsParams)*28@HXb8gradient_tape/model/conv2d_3/Conv2D/Conv2DBackpropFilterhu  ΘB
?
Scask_cudnn::computeWgradSplitKOffsetsKernel(cask_cudnn::ComputeSplitKOffsetsParams)*2P8ΰ@ΰHΰXb8gradient_tape/model/conv2d_3/Conv2D/Conv2DBackpropFilterhu  ΘB
P
%LogicalAnd_GPU_DT_BOOL_DT_BOOL_kernel*28ΐ@ΐHΐb
LogicalAndhuZUB
?
Scask_cudnn::computeWgradSplitKOffsetsKernel(cask_cudnn::ComputeSplitKOffsetsParams)*2P8ΐ@ΐHΐXb8gradient_tape/model/conv2d_2/Conv2D/Conv2DBackpropFilterhu  ΘB
©
Ncask_cudnn::computeWgradBOffsetsKernel(cask_cudnn::ComputeWgradBOffsetsParams)*28 @ H Xb8gradient_tape/model/conv2d_2/Conv2D/Conv2DBackpropFilterhu  ΘB
Β
Scask_cudnn::computeWgradSplitKOffsetsKernel(cask_cudnn::ComputeSplitKOffsetsParams)*2(8 @ H XbLgradient_tape/model/conv2d_transpose_3/conv2d_transpose/Conv2DBackpropFilterhu  ΘB

Dcask_cudnn::computeBOffsetsKernel(cask_cudnn::ComputeBOffsetsParams)*28@HXb8gradient_tape/model/conv2d_22/Conv2D/Conv2DBackpropInputhu  ΘB