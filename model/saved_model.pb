??
??
D
AddV2
x"T
y"T
z"T"
Ttype:
2	??
B
AssignVariableOp
resource
value"dtype"
dtypetype?
~
BiasAdd

value"T	
bias"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
8
Const
output"dtype"
valuetensor"
dtypetype
?
Conv2D

input"T
filter"T
output"T"
Ttype:	
2"
strides	list(int)"
use_cudnn_on_gpubool(",
paddingstring:
SAMEVALIDEXPLICIT""
explicit_paddings	list(int)
 "-
data_formatstringNHWC:
NHWCNCHW" 
	dilations	list(int)

?
FusedBatchNormV3
x"T

scale"U
offset"U	
mean"U
variance"U
y"T

batch_mean"U
batch_variance"U
reserve_space_1"U
reserve_space_2"U
reserve_space_3"U"
Ttype:
2"
Utype:
2"
epsilonfloat%??8"&
exponential_avg_factorfloat%  ??";
data_formatstringNHWC:
NHWCNCHWNDHWCNCDHW"
is_trainingbool(
.
Identity

input"T
output"T"	
Ttype
q
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:

2	
e
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool(?

NoOp
M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
@
ReadVariableOp
resource
value"dtype"
dtypetype?
E
Relu
features"T
activations"T"
Ttype:
2	
[
Reshape
tensor"T
shape"Tshape
output"T"	
Ttype"
Tshapetype0:
2	
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0?
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0?
?
Select
	condition

t"T
e"T
output"T"	
Ttype
H
ShardedFilename
basename	
shard

num_shards
filename
9
Softmax
logits"T
softmax"T"
Ttype:
2
?
StatefulPartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring ?
@
StaticRegexFullMatch	
input

output
"
patternstring
N

StringJoin
inputs*N

output"
Nint(0"
	separatorstring 
-
Tanh
x"T
y"T"
Ttype:

2
?
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 ?"serve*2.6.12v2.6.0-101-g3aa40c3ce9d8??
?
conv2d_11/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:P*!
shared_nameconv2d_11/kernel
}
$conv2d_11/kernel/Read/ReadVariableOpReadVariableOpconv2d_11/kernel*&
_output_shapes
:P*
dtype0
t
conv2d_11/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameconv2d_11/bias
m
"conv2d_11/bias/Read/ReadVariableOpReadVariableOpconv2d_11/bias*
_output_shapes
:*
dtype0
x
dense_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@*
shared_namedense_1/kernel
q
"dense_1/kernel/Read/ReadVariableOpReadVariableOpdense_1/kernel*
_output_shapes

:@*
dtype0
p
dense_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_1/bias
i
 dense_1/bias/Read/ReadVariableOpReadVariableOpdense_1/bias*
_output_shapes
:*
dtype0
?
conv2d_10/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:PI*!
shared_nameconv2d_10/kernel
}
$conv2d_10/kernel/Read/ReadVariableOpReadVariableOpconv2d_10/kernel*&
_output_shapes
:PI*
dtype0
t
conv2d_10/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:I*
shared_nameconv2d_10/bias
m
"conv2d_10/bias/Read/ReadVariableOpReadVariableOpconv2d_10/bias*
_output_shapes
:I*
dtype0
~
conv2d/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:P*
shared_nameconv2d/kernel
w
!conv2d/kernel/Read/ReadVariableOpReadVariableOpconv2d/kernel*&
_output_shapes
:P*
dtype0
n
conv2d/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:P*
shared_nameconv2d/bias
g
conv2d/bias/Read/ReadVariableOpReadVariableOpconv2d/bias*
_output_shapes
:P*
dtype0
?
batch_normalization/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:P**
shared_namebatch_normalization/gamma
?
-batch_normalization/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization/gamma*
_output_shapes
:P*
dtype0
?
batch_normalization/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:P*)
shared_namebatch_normalization/beta
?
,batch_normalization/beta/Read/ReadVariableOpReadVariableOpbatch_normalization/beta*
_output_shapes
:P*
dtype0
?
batch_normalization/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:P*0
shared_name!batch_normalization/moving_mean
?
3batch_normalization/moving_mean/Read/ReadVariableOpReadVariableOpbatch_normalization/moving_mean*
_output_shapes
:P*
dtype0
?
#batch_normalization/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:P*4
shared_name%#batch_normalization/moving_variance
?
7batch_normalization/moving_variance/Read/ReadVariableOpReadVariableOp#batch_normalization/moving_variance*
_output_shapes
:P*
dtype0
?
conv2d_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:PP* 
shared_nameconv2d_1/kernel
{
#conv2d_1/kernel/Read/ReadVariableOpReadVariableOpconv2d_1/kernel*&
_output_shapes
:PP*
dtype0
r
conv2d_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:P*
shared_nameconv2d_1/bias
k
!conv2d_1/bias/Read/ReadVariableOpReadVariableOpconv2d_1/bias*
_output_shapes
:P*
dtype0
?
batch_normalization_1/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:P*,
shared_namebatch_normalization_1/gamma
?
/batch_normalization_1/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_1/gamma*
_output_shapes
:P*
dtype0
?
batch_normalization_1/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:P*+
shared_namebatch_normalization_1/beta
?
.batch_normalization_1/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_1/beta*
_output_shapes
:P*
dtype0
?
!batch_normalization_1/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:P*2
shared_name#!batch_normalization_1/moving_mean
?
5batch_normalization_1/moving_mean/Read/ReadVariableOpReadVariableOp!batch_normalization_1/moving_mean*
_output_shapes
:P*
dtype0
?
%batch_normalization_1/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:P*6
shared_name'%batch_normalization_1/moving_variance
?
9batch_normalization_1/moving_variance/Read/ReadVariableOpReadVariableOp%batch_normalization_1/moving_variance*
_output_shapes
:P*
dtype0
?
conv2d_2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:PP* 
shared_nameconv2d_2/kernel
{
#conv2d_2/kernel/Read/ReadVariableOpReadVariableOpconv2d_2/kernel*&
_output_shapes
:PP*
dtype0
r
conv2d_2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:P*
shared_nameconv2d_2/bias
k
!conv2d_2/bias/Read/ReadVariableOpReadVariableOpconv2d_2/bias*
_output_shapes
:P*
dtype0
?
batch_normalization_2/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:P*,
shared_namebatch_normalization_2/gamma
?
/batch_normalization_2/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_2/gamma*
_output_shapes
:P*
dtype0
?
batch_normalization_2/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:P*+
shared_namebatch_normalization_2/beta
?
.batch_normalization_2/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_2/beta*
_output_shapes
:P*
dtype0
?
!batch_normalization_2/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:P*2
shared_name#!batch_normalization_2/moving_mean
?
5batch_normalization_2/moving_mean/Read/ReadVariableOpReadVariableOp!batch_normalization_2/moving_mean*
_output_shapes
:P*
dtype0
?
%batch_normalization_2/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:P*6
shared_name'%batch_normalization_2/moving_variance
?
9batch_normalization_2/moving_variance/Read/ReadVariableOpReadVariableOp%batch_normalization_2/moving_variance*
_output_shapes
:P*
dtype0

NoOpNoOp
?Q
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*?Q
value?QB?Q B?Q
?
layer-0
layer_with_weights-0
layer-1
layer_with_weights-1
layer-2
layer-3
layer-4
layer_with_weights-2
layer-5
layer_with_weights-3
layer-6

signatures
#	_self_saveable_object_factories

	variables
regularization_losses
trainable_variables
	keras_api
%
#_self_saveable_object_factories
?
layer-0
layer_with_weights-0
layer-1
layer_with_weights-1
layer-2
layer-3
layer-4
layer_with_weights-2
layer-5
layer_with_weights-3
layer-6
layer-7
layer-8
layer_with_weights-4
layer-9
layer_with_weights-5
layer-10
layer-11
layer-12
layer-13
#_self_saveable_object_factories
	variables
regularization_losses
 trainable_variables
!	keras_api
?

"kernel
#bias
#$_self_saveable_object_factories
%regularization_losses
&	variables
'trainable_variables
(	keras_api
w
#)_self_saveable_object_factories
*regularization_losses
+	variables
,trainable_variables
-	keras_api
w
#._self_saveable_object_factories
/regularization_losses
0	variables
1trainable_variables
2	keras_api
?

3kernel
4bias
#5_self_saveable_object_factories
6regularization_losses
7	variables
8trainable_variables
9	keras_api
?

:kernel
;bias
#<_self_saveable_object_factories
=regularization_losses
>	variables
?trainable_variables
@	keras_api
 
 
?
A0
B1
C2
D3
E4
F5
G6
H7
I8
J9
K10
L11
M12
N13
O14
P15
Q16
R17
"18
#19
320
421
:22
;23
 
?
A0
B1
C2
D3
G4
H5
I6
J7
M8
N9
O10
P11
"12
#13
314
415
:16
;17
?
Slayer_metrics
Tlayer_regularization_losses
Umetrics

	variables
regularization_losses

Vlayers
trainable_variables
Wnon_trainable_variables
 
%
#X_self_saveable_object_factories
?

Akernel
Bbias
#Y_self_saveable_object_factories
Zregularization_losses
[	variables
\trainable_variables
]	keras_api
?
^axis
	Cgamma
Dbeta
Emoving_mean
Fmoving_variance
#__self_saveable_object_factories
`regularization_losses
a	variables
btrainable_variables
c	keras_api
w
#d_self_saveable_object_factories
eregularization_losses
f	variables
gtrainable_variables
h	keras_api
w
#i_self_saveable_object_factories
jregularization_losses
k	variables
ltrainable_variables
m	keras_api
?

Gkernel
Hbias
#n_self_saveable_object_factories
oregularization_losses
p	variables
qtrainable_variables
r	keras_api
?
saxis
	Igamma
Jbeta
Kmoving_mean
Lmoving_variance
#t_self_saveable_object_factories
uregularization_losses
v	variables
wtrainable_variables
x	keras_api
w
#y_self_saveable_object_factories
zregularization_losses
{	variables
|trainable_variables
}	keras_api
z
#~_self_saveable_object_factories
regularization_losses
?	variables
?trainable_variables
?	keras_api
?

Mkernel
Nbias
$?_self_saveable_object_factories
?regularization_losses
?	variables
?trainable_variables
?	keras_api
?
	?axis
	Ogamma
Pbeta
Qmoving_mean
Rmoving_variance
$?_self_saveable_object_factories
?regularization_losses
?	variables
?trainable_variables
?	keras_api
6
$?_self_saveable_object_factories
?	keras_api
|
$?_self_saveable_object_factories
?regularization_losses
?	variables
?trainable_variables
?	keras_api
|
$?_self_saveable_object_factories
?regularization_losses
?	variables
?trainable_variables
?	keras_api
 
?
A0
B1
C2
D3
E4
F5
G6
H7
I8
J9
K10
L11
M12
N13
O14
P15
Q16
R17
 
V
A0
B1
C2
D3
G4
H5
I6
J7
M8
N9
O10
P11
?
?layer_metrics
 ?layer_regularization_losses
?metrics
	variables
regularization_losses
?layers
 trainable_variables
?non_trainable_variables
\Z
VARIABLE_VALUEconv2d_11/kernel6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE
XV
VARIABLE_VALUEconv2d_11/bias4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE
 
 

"0
#1

"0
#1
?
?layer_metrics
 ?layer_regularization_losses
?metrics
%regularization_losses
&	variables
?layers
'trainable_variables
?non_trainable_variables
 
 
 
 
?
?layer_metrics
 ?layer_regularization_losses
?metrics
*regularization_losses
+	variables
?layers
,trainable_variables
?non_trainable_variables
 
 
 
 
?
?layer_metrics
 ?layer_regularization_losses
?metrics
/regularization_losses
0	variables
?layers
1trainable_variables
?non_trainable_variables
ZX
VARIABLE_VALUEdense_1/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE
VT
VARIABLE_VALUEdense_1/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE
 
 

30
41

30
41
?
?layer_metrics
 ?layer_regularization_losses
?metrics
6regularization_losses
7	variables
?layers
8trainable_variables
?non_trainable_variables
\Z
VARIABLE_VALUEconv2d_10/kernel6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUE
XV
VARIABLE_VALUEconv2d_10/bias4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUE
 
 

:0
;1

:0
;1
?
?layer_metrics
 ?layer_regularization_losses
?metrics
=regularization_losses
>	variables
?layers
?trainable_variables
?non_trainable_variables
IG
VARIABLE_VALUEconv2d/kernel&variables/0/.ATTRIBUTES/VARIABLE_VALUE
GE
VARIABLE_VALUEconv2d/bias&variables/1/.ATTRIBUTES/VARIABLE_VALUE
US
VARIABLE_VALUEbatch_normalization/gamma&variables/2/.ATTRIBUTES/VARIABLE_VALUE
TR
VARIABLE_VALUEbatch_normalization/beta&variables/3/.ATTRIBUTES/VARIABLE_VALUE
[Y
VARIABLE_VALUEbatch_normalization/moving_mean&variables/4/.ATTRIBUTES/VARIABLE_VALUE
_]
VARIABLE_VALUE#batch_normalization/moving_variance&variables/5/.ATTRIBUTES/VARIABLE_VALUE
KI
VARIABLE_VALUEconv2d_1/kernel&variables/6/.ATTRIBUTES/VARIABLE_VALUE
IG
VARIABLE_VALUEconv2d_1/bias&variables/7/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEbatch_normalization_1/gamma&variables/8/.ATTRIBUTES/VARIABLE_VALUE
VT
VARIABLE_VALUEbatch_normalization_1/beta&variables/9/.ATTRIBUTES/VARIABLE_VALUE
^\
VARIABLE_VALUE!batch_normalization_1/moving_mean'variables/10/.ATTRIBUTES/VARIABLE_VALUE
b`
VARIABLE_VALUE%batch_normalization_1/moving_variance'variables/11/.ATTRIBUTES/VARIABLE_VALUE
LJ
VARIABLE_VALUEconv2d_2/kernel'variables/12/.ATTRIBUTES/VARIABLE_VALUE
JH
VARIABLE_VALUEconv2d_2/bias'variables/13/.ATTRIBUTES/VARIABLE_VALUE
XV
VARIABLE_VALUEbatch_normalization_2/gamma'variables/14/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEbatch_normalization_2/beta'variables/15/.ATTRIBUTES/VARIABLE_VALUE
^\
VARIABLE_VALUE!batch_normalization_2/moving_mean'variables/16/.ATTRIBUTES/VARIABLE_VALUE
b`
VARIABLE_VALUE%batch_normalization_2/moving_variance'variables/17/.ATTRIBUTES/VARIABLE_VALUE
 
 
 
1
0
1
2
3
4
5
6
*
E0
F1
K2
L3
Q4
R5
 
 
 

A0
B1

A0
B1
?
?layer_metrics
 ?layer_regularization_losses
?metrics
Zregularization_losses
[	variables
?layers
\trainable_variables
?non_trainable_variables
 
 
 

C0
D1
E2
F3

C0
D1
?
?layer_metrics
 ?layer_regularization_losses
?metrics
`regularization_losses
a	variables
?layers
btrainable_variables
?non_trainable_variables
 
 
 
 
?
?layer_metrics
 ?layer_regularization_losses
?metrics
eregularization_losses
f	variables
?layers
gtrainable_variables
?non_trainable_variables
 
 
 
 
?
?layer_metrics
 ?layer_regularization_losses
?metrics
jregularization_losses
k	variables
?layers
ltrainable_variables
?non_trainable_variables
 
 

G0
H1

G0
H1
?
?layer_metrics
 ?layer_regularization_losses
?metrics
oregularization_losses
p	variables
?layers
qtrainable_variables
?non_trainable_variables
 
 
 

I0
J1
K2
L3

I0
J1
?
?layer_metrics
 ?layer_regularization_losses
?metrics
uregularization_losses
v	variables
?layers
wtrainable_variables
?non_trainable_variables
 
 
 
 
?
?layer_metrics
 ?layer_regularization_losses
?metrics
zregularization_losses
{	variables
?layers
|trainable_variables
?non_trainable_variables
 
 
 
 
?
?layer_metrics
 ?layer_regularization_losses
?metrics
regularization_losses
?	variables
?layers
?trainable_variables
?non_trainable_variables
 
 

M0
N1

M0
N1
?
?layer_metrics
 ?layer_regularization_losses
?metrics
?regularization_losses
?	variables
?layers
?trainable_variables
?non_trainable_variables
 
 
 

O0
P1
Q2
R3

O0
P1
?
?layer_metrics
 ?layer_regularization_losses
?metrics
?regularization_losses
?	variables
?layers
?trainable_variables
?non_trainable_variables
 
 
 
 
 
 
?
?layer_metrics
 ?layer_regularization_losses
?metrics
?regularization_losses
?	variables
?layers
?trainable_variables
?non_trainable_variables
 
 
 
 
?
?layer_metrics
 ?layer_regularization_losses
?metrics
?regularization_losses
?	variables
?layers
?trainable_variables
?non_trainable_variables
 
 
 
f
0
1
2
3
4
5
6
7
8
9
10
11
12
13
*
E0
F1
K2
L3
Q4
R5
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 

E0
F1
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 

K0
L1
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 

Q0
R1
 
 
 
 
 
 
 
 
 
 
?
serving_default_input_4Placeholder*/
_output_shapes
:?????????*
dtype0*$
shape:?????????
?
StatefulPartitionedCallStatefulPartitionedCallserving_default_input_4conv2d/kernelconv2d/biasbatch_normalization/gammabatch_normalization/betabatch_normalization/moving_mean#batch_normalization/moving_varianceconv2d_1/kernelconv2d_1/biasbatch_normalization_1/gammabatch_normalization_1/beta!batch_normalization_1/moving_mean%batch_normalization_1/moving_varianceconv2d_2/kernelconv2d_2/biasbatch_normalization_2/gammabatch_normalization_2/beta!batch_normalization_2/moving_mean%batch_normalization_2/moving_varianceconv2d_11/kernelconv2d_11/biasconv2d_10/kernelconv2d_10/biasdense_1/kerneldense_1/bias*$
Tin
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:?????????I:?????????*:
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8? *.
f)R'
%__inference_signature_wrapper_3595602
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
?

StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename$conv2d_11/kernel/Read/ReadVariableOp"conv2d_11/bias/Read/ReadVariableOp"dense_1/kernel/Read/ReadVariableOp dense_1/bias/Read/ReadVariableOp$conv2d_10/kernel/Read/ReadVariableOp"conv2d_10/bias/Read/ReadVariableOp!conv2d/kernel/Read/ReadVariableOpconv2d/bias/Read/ReadVariableOp-batch_normalization/gamma/Read/ReadVariableOp,batch_normalization/beta/Read/ReadVariableOp3batch_normalization/moving_mean/Read/ReadVariableOp7batch_normalization/moving_variance/Read/ReadVariableOp#conv2d_1/kernel/Read/ReadVariableOp!conv2d_1/bias/Read/ReadVariableOp/batch_normalization_1/gamma/Read/ReadVariableOp.batch_normalization_1/beta/Read/ReadVariableOp5batch_normalization_1/moving_mean/Read/ReadVariableOp9batch_normalization_1/moving_variance/Read/ReadVariableOp#conv2d_2/kernel/Read/ReadVariableOp!conv2d_2/bias/Read/ReadVariableOp/batch_normalization_2/gamma/Read/ReadVariableOp.batch_normalization_2/beta/Read/ReadVariableOp5batch_normalization_2/moving_mean/Read/ReadVariableOp9batch_normalization_2/moving_variance/Read/ReadVariableOpConst*%
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *)
f$R"
 __inference__traced_save_3596911
?
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenameconv2d_11/kernelconv2d_11/biasdense_1/kerneldense_1/biasconv2d_10/kernelconv2d_10/biasconv2d/kernelconv2d/biasbatch_normalization/gammabatch_normalization/betabatch_normalization/moving_mean#batch_normalization/moving_varianceconv2d_1/kernelconv2d_1/biasbatch_normalization_1/gammabatch_normalization_1/beta!batch_normalization_1/moving_mean%batch_normalization_1/moving_varianceconv2d_2/kernelconv2d_2/biasbatch_normalization_2/gammabatch_normalization_2/beta!batch_normalization_2/moving_mean%batch_normalization_2/moving_variance*$
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *,
f'R%
#__inference__traced_restore_3596993??
??
?
B__inference_model_layer_call_and_return_conditional_losses_3596177

inputs?
%conv2d_conv2d_readvariableop_resource:P4
&conv2d_biasadd_readvariableop_resource:P9
+batch_normalization_readvariableop_resource:P;
-batch_normalization_readvariableop_1_resource:PJ
<batch_normalization_fusedbatchnormv3_readvariableop_resource:PL
>batch_normalization_fusedbatchnormv3_readvariableop_1_resource:PA
'conv2d_1_conv2d_readvariableop_resource:PP6
(conv2d_1_biasadd_readvariableop_resource:P;
-batch_normalization_1_readvariableop_resource:P=
/batch_normalization_1_readvariableop_1_resource:PL
>batch_normalization_1_fusedbatchnormv3_readvariableop_resource:PN
@batch_normalization_1_fusedbatchnormv3_readvariableop_1_resource:PA
'conv2d_2_conv2d_readvariableop_resource:PP6
(conv2d_2_biasadd_readvariableop_resource:P;
-batch_normalization_2_readvariableop_resource:P=
/batch_normalization_2_readvariableop_1_resource:PL
>batch_normalization_2_fusedbatchnormv3_readvariableop_resource:PN
@batch_normalization_2_fusedbatchnormv3_readvariableop_1_resource:P
identity??"batch_normalization/AssignNewValue?$batch_normalization/AssignNewValue_1?3batch_normalization/FusedBatchNormV3/ReadVariableOp?5batch_normalization/FusedBatchNormV3/ReadVariableOp_1?"batch_normalization/ReadVariableOp?$batch_normalization/ReadVariableOp_1?$batch_normalization_1/AssignNewValue?&batch_normalization_1/AssignNewValue_1?5batch_normalization_1/FusedBatchNormV3/ReadVariableOp?7batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1?$batch_normalization_1/ReadVariableOp?&batch_normalization_1/ReadVariableOp_1?$batch_normalization_2/AssignNewValue?&batch_normalization_2/AssignNewValue_1?5batch_normalization_2/FusedBatchNormV3/ReadVariableOp?7batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1?$batch_normalization_2/ReadVariableOp?&batch_normalization_2/ReadVariableOp_1?conv2d/BiasAdd/ReadVariableOp?conv2d/Conv2D/ReadVariableOp?conv2d_1/BiasAdd/ReadVariableOp?conv2d_1/Conv2D/ReadVariableOp?conv2d_2/BiasAdd/ReadVariableOp?conv2d_2/Conv2D/ReadVariableOp?
conv2d/Conv2D/ReadVariableOpReadVariableOp%conv2d_conv2d_readvariableop_resource*&
_output_shapes
:P*
dtype02
conv2d/Conv2D/ReadVariableOp?
conv2d/Conv2DConv2Dinputs$conv2d/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????P*
paddingSAME*
strides
2
conv2d/Conv2D?
conv2d/BiasAdd/ReadVariableOpReadVariableOp&conv2d_biasadd_readvariableop_resource*
_output_shapes
:P*
dtype02
conv2d/BiasAdd/ReadVariableOp?
conv2d/BiasAddBiasAddconv2d/Conv2D:output:0%conv2d/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????P2
conv2d/BiasAdd?
"batch_normalization/ReadVariableOpReadVariableOp+batch_normalization_readvariableop_resource*
_output_shapes
:P*
dtype02$
"batch_normalization/ReadVariableOp?
$batch_normalization/ReadVariableOp_1ReadVariableOp-batch_normalization_readvariableop_1_resource*
_output_shapes
:P*
dtype02&
$batch_normalization/ReadVariableOp_1?
3batch_normalization/FusedBatchNormV3/ReadVariableOpReadVariableOp<batch_normalization_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:P*
dtype025
3batch_normalization/FusedBatchNormV3/ReadVariableOp?
5batch_normalization/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp>batch_normalization_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:P*
dtype027
5batch_normalization/FusedBatchNormV3/ReadVariableOp_1?
$batch_normalization/FusedBatchNormV3FusedBatchNormV3conv2d/BiasAdd:output:0*batch_normalization/ReadVariableOp:value:0,batch_normalization/ReadVariableOp_1:value:0;batch_normalization/FusedBatchNormV3/ReadVariableOp:value:0=batch_normalization/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????P:P:P:P:P:*
epsilon%o?:*
exponential_avg_factor%
?#<2&
$batch_normalization/FusedBatchNormV3?
"batch_normalization/AssignNewValueAssignVariableOp<batch_normalization_fusedbatchnormv3_readvariableop_resource1batch_normalization/FusedBatchNormV3:batch_mean:04^batch_normalization/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype02$
"batch_normalization/AssignNewValue?
$batch_normalization/AssignNewValue_1AssignVariableOp>batch_normalization_fusedbatchnormv3_readvariableop_1_resource5batch_normalization/FusedBatchNormV3:batch_variance:06^batch_normalization/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype02&
$batch_normalization/AssignNewValue_1?
activation/ReluRelu(batch_normalization/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:?????????P2
activation/Relus
dropout/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
dropout/dropout/Const?
dropout/dropout/MulMulactivation/Relu:activations:0dropout/dropout/Const:output:0*
T0*/
_output_shapes
:?????????P2
dropout/dropout/Mul{
dropout/dropout/ShapeShapeactivation/Relu:activations:0*
T0*
_output_shapes
:2
dropout/dropout/Shape?
,dropout/dropout/random_uniform/RandomUniformRandomUniformdropout/dropout/Shape:output:0*
T0*/
_output_shapes
:?????????P*
dtype02.
,dropout/dropout/random_uniform/RandomUniform?
dropout/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L>2 
dropout/dropout/GreaterEqual/y?
dropout/dropout/GreaterEqualGreaterEqual5dropout/dropout/random_uniform/RandomUniform:output:0'dropout/dropout/GreaterEqual/y:output:0*
T0*/
_output_shapes
:?????????P2
dropout/dropout/GreaterEqual?
dropout/dropout/CastCast dropout/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*/
_output_shapes
:?????????P2
dropout/dropout/Cast?
dropout/dropout/Mul_1Muldropout/dropout/Mul:z:0dropout/dropout/Cast:y:0*
T0*/
_output_shapes
:?????????P2
dropout/dropout/Mul_1?
conv2d_1/Conv2D/ReadVariableOpReadVariableOp'conv2d_1_conv2d_readvariableop_resource*&
_output_shapes
:PP*
dtype02 
conv2d_1/Conv2D/ReadVariableOp?
conv2d_1/Conv2DConv2Ddropout/dropout/Mul_1:z:0&conv2d_1/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????P*
paddingSAME*
strides
2
conv2d_1/Conv2D?
conv2d_1/BiasAdd/ReadVariableOpReadVariableOp(conv2d_1_biasadd_readvariableop_resource*
_output_shapes
:P*
dtype02!
conv2d_1/BiasAdd/ReadVariableOp?
conv2d_1/BiasAddBiasAddconv2d_1/Conv2D:output:0'conv2d_1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????P2
conv2d_1/BiasAdd?
$batch_normalization_1/ReadVariableOpReadVariableOp-batch_normalization_1_readvariableop_resource*
_output_shapes
:P*
dtype02&
$batch_normalization_1/ReadVariableOp?
&batch_normalization_1/ReadVariableOp_1ReadVariableOp/batch_normalization_1_readvariableop_1_resource*
_output_shapes
:P*
dtype02(
&batch_normalization_1/ReadVariableOp_1?
5batch_normalization_1/FusedBatchNormV3/ReadVariableOpReadVariableOp>batch_normalization_1_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:P*
dtype027
5batch_normalization_1/FusedBatchNormV3/ReadVariableOp?
7batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp@batch_normalization_1_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:P*
dtype029
7batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1?
&batch_normalization_1/FusedBatchNormV3FusedBatchNormV3conv2d_1/BiasAdd:output:0,batch_normalization_1/ReadVariableOp:value:0.batch_normalization_1/ReadVariableOp_1:value:0=batch_normalization_1/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????P:P:P:P:P:*
epsilon%o?:*
exponential_avg_factor%
?#<2(
&batch_normalization_1/FusedBatchNormV3?
$batch_normalization_1/AssignNewValueAssignVariableOp>batch_normalization_1_fusedbatchnormv3_readvariableop_resource3batch_normalization_1/FusedBatchNormV3:batch_mean:06^batch_normalization_1/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype02&
$batch_normalization_1/AssignNewValue?
&batch_normalization_1/AssignNewValue_1AssignVariableOp@batch_normalization_1_fusedbatchnormv3_readvariableop_1_resource7batch_normalization_1/FusedBatchNormV3:batch_variance:08^batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype02(
&batch_normalization_1/AssignNewValue_1?
activation_1/ReluRelu*batch_normalization_1/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:?????????P2
activation_1/Reluw
dropout_1/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
dropout_1/dropout/Const?
dropout_1/dropout/MulMulactivation_1/Relu:activations:0 dropout_1/dropout/Const:output:0*
T0*/
_output_shapes
:?????????P2
dropout_1/dropout/Mul?
dropout_1/dropout/ShapeShapeactivation_1/Relu:activations:0*
T0*
_output_shapes
:2
dropout_1/dropout/Shape?
.dropout_1/dropout/random_uniform/RandomUniformRandomUniform dropout_1/dropout/Shape:output:0*
T0*/
_output_shapes
:?????????P*
dtype020
.dropout_1/dropout/random_uniform/RandomUniform?
 dropout_1/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L>2"
 dropout_1/dropout/GreaterEqual/y?
dropout_1/dropout/GreaterEqualGreaterEqual7dropout_1/dropout/random_uniform/RandomUniform:output:0)dropout_1/dropout/GreaterEqual/y:output:0*
T0*/
_output_shapes
:?????????P2 
dropout_1/dropout/GreaterEqual?
dropout_1/dropout/CastCast"dropout_1/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*/
_output_shapes
:?????????P2
dropout_1/dropout/Cast?
dropout_1/dropout/Mul_1Muldropout_1/dropout/Mul:z:0dropout_1/dropout/Cast:y:0*
T0*/
_output_shapes
:?????????P2
dropout_1/dropout/Mul_1?
conv2d_2/Conv2D/ReadVariableOpReadVariableOp'conv2d_2_conv2d_readvariableop_resource*&
_output_shapes
:PP*
dtype02 
conv2d_2/Conv2D/ReadVariableOp?
conv2d_2/Conv2DConv2Ddropout_1/dropout/Mul_1:z:0&conv2d_2/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????P*
paddingSAME*
strides
2
conv2d_2/Conv2D?
conv2d_2/BiasAdd/ReadVariableOpReadVariableOp(conv2d_2_biasadd_readvariableop_resource*
_output_shapes
:P*
dtype02!
conv2d_2/BiasAdd/ReadVariableOp?
conv2d_2/BiasAddBiasAddconv2d_2/Conv2D:output:0'conv2d_2/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????P2
conv2d_2/BiasAdd?
$batch_normalization_2/ReadVariableOpReadVariableOp-batch_normalization_2_readvariableop_resource*
_output_shapes
:P*
dtype02&
$batch_normalization_2/ReadVariableOp?
&batch_normalization_2/ReadVariableOp_1ReadVariableOp/batch_normalization_2_readvariableop_1_resource*
_output_shapes
:P*
dtype02(
&batch_normalization_2/ReadVariableOp_1?
5batch_normalization_2/FusedBatchNormV3/ReadVariableOpReadVariableOp>batch_normalization_2_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:P*
dtype027
5batch_normalization_2/FusedBatchNormV3/ReadVariableOp?
7batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp@batch_normalization_2_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:P*
dtype029
7batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1?
&batch_normalization_2/FusedBatchNormV3FusedBatchNormV3conv2d_2/BiasAdd:output:0,batch_normalization_2/ReadVariableOp:value:0.batch_normalization_2/ReadVariableOp_1:value:0=batch_normalization_2/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????P:P:P:P:P:*
epsilon%o?:*
exponential_avg_factor%
?#<2(
&batch_normalization_2/FusedBatchNormV3?
$batch_normalization_2/AssignNewValueAssignVariableOp>batch_normalization_2_fusedbatchnormv3_readvariableop_resource3batch_normalization_2/FusedBatchNormV3:batch_mean:06^batch_normalization_2/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype02&
$batch_normalization_2/AssignNewValue?
&batch_normalization_2/AssignNewValue_1AssignVariableOp@batch_normalization_2_fusedbatchnormv3_readvariableop_1_resource7batch_normalization_2/FusedBatchNormV3:batch_variance:08^batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype02(
&batch_normalization_2/AssignNewValue_1?
tf.__operators__.add/AddV2AddV2*batch_normalization_2/FusedBatchNormV3:y:0dropout/dropout/Mul_1:z:0*
T0*/
_output_shapes
:?????????P2
tf.__operators__.add/AddV2?
activation_2/ReluRelutf.__operators__.add/AddV2:z:0*
T0*/
_output_shapes
:?????????P2
activation_2/Reluw
dropout_2/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
dropout_2/dropout/Const?
dropout_2/dropout/MulMulactivation_2/Relu:activations:0 dropout_2/dropout/Const:output:0*
T0*/
_output_shapes
:?????????P2
dropout_2/dropout/Mul?
dropout_2/dropout/ShapeShapeactivation_2/Relu:activations:0*
T0*
_output_shapes
:2
dropout_2/dropout/Shape?
.dropout_2/dropout/random_uniform/RandomUniformRandomUniform dropout_2/dropout/Shape:output:0*
T0*/
_output_shapes
:?????????P*
dtype020
.dropout_2/dropout/random_uniform/RandomUniform?
 dropout_2/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L>2"
 dropout_2/dropout/GreaterEqual/y?
dropout_2/dropout/GreaterEqualGreaterEqual7dropout_2/dropout/random_uniform/RandomUniform:output:0)dropout_2/dropout/GreaterEqual/y:output:0*
T0*/
_output_shapes
:?????????P2 
dropout_2/dropout/GreaterEqual?
dropout_2/dropout/CastCast"dropout_2/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*/
_output_shapes
:?????????P2
dropout_2/dropout/Cast?
dropout_2/dropout/Mul_1Muldropout_2/dropout/Mul:z:0dropout_2/dropout/Cast:y:0*
T0*/
_output_shapes
:?????????P2
dropout_2/dropout/Mul_1~
IdentityIdentitydropout_2/dropout/Mul_1:z:0^NoOp*
T0*/
_output_shapes
:?????????P2

Identity?
NoOpNoOp#^batch_normalization/AssignNewValue%^batch_normalization/AssignNewValue_14^batch_normalization/FusedBatchNormV3/ReadVariableOp6^batch_normalization/FusedBatchNormV3/ReadVariableOp_1#^batch_normalization/ReadVariableOp%^batch_normalization/ReadVariableOp_1%^batch_normalization_1/AssignNewValue'^batch_normalization_1/AssignNewValue_16^batch_normalization_1/FusedBatchNormV3/ReadVariableOp8^batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1%^batch_normalization_1/ReadVariableOp'^batch_normalization_1/ReadVariableOp_1%^batch_normalization_2/AssignNewValue'^batch_normalization_2/AssignNewValue_16^batch_normalization_2/FusedBatchNormV3/ReadVariableOp8^batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1%^batch_normalization_2/ReadVariableOp'^batch_normalization_2/ReadVariableOp_1^conv2d/BiasAdd/ReadVariableOp^conv2d/Conv2D/ReadVariableOp ^conv2d_1/BiasAdd/ReadVariableOp^conv2d_1/Conv2D/ReadVariableOp ^conv2d_2/BiasAdd/ReadVariableOp^conv2d_2/Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*R
_input_shapesA
?:?????????: : : : : : : : : : : : : : : : : : 2H
"batch_normalization/AssignNewValue"batch_normalization/AssignNewValue2L
$batch_normalization/AssignNewValue_1$batch_normalization/AssignNewValue_12j
3batch_normalization/FusedBatchNormV3/ReadVariableOp3batch_normalization/FusedBatchNormV3/ReadVariableOp2n
5batch_normalization/FusedBatchNormV3/ReadVariableOp_15batch_normalization/FusedBatchNormV3/ReadVariableOp_12H
"batch_normalization/ReadVariableOp"batch_normalization/ReadVariableOp2L
$batch_normalization/ReadVariableOp_1$batch_normalization/ReadVariableOp_12L
$batch_normalization_1/AssignNewValue$batch_normalization_1/AssignNewValue2P
&batch_normalization_1/AssignNewValue_1&batch_normalization_1/AssignNewValue_12n
5batch_normalization_1/FusedBatchNormV3/ReadVariableOp5batch_normalization_1/FusedBatchNormV3/ReadVariableOp2r
7batch_normalization_1/FusedBatchNormV3/ReadVariableOp_17batch_normalization_1/FusedBatchNormV3/ReadVariableOp_12L
$batch_normalization_1/ReadVariableOp$batch_normalization_1/ReadVariableOp2P
&batch_normalization_1/ReadVariableOp_1&batch_normalization_1/ReadVariableOp_12L
$batch_normalization_2/AssignNewValue$batch_normalization_2/AssignNewValue2P
&batch_normalization_2/AssignNewValue_1&batch_normalization_2/AssignNewValue_12n
5batch_normalization_2/FusedBatchNormV3/ReadVariableOp5batch_normalization_2/FusedBatchNormV3/ReadVariableOp2r
7batch_normalization_2/FusedBatchNormV3/ReadVariableOp_17batch_normalization_2/FusedBatchNormV3/ReadVariableOp_12L
$batch_normalization_2/ReadVariableOp$batch_normalization_2/ReadVariableOp2P
&batch_normalization_2/ReadVariableOp_1&batch_normalization_2/ReadVariableOp_12>
conv2d/BiasAdd/ReadVariableOpconv2d/BiasAdd/ReadVariableOp2<
conv2d/Conv2D/ReadVariableOpconv2d/Conv2D/ReadVariableOp2B
conv2d_1/BiasAdd/ReadVariableOpconv2d_1/BiasAdd/ReadVariableOp2@
conv2d_1/Conv2D/ReadVariableOpconv2d_1/Conv2D/ReadVariableOp2B
conv2d_2/BiasAdd/ReadVariableOpconv2d_2/BiasAdd/ReadVariableOp2@
conv2d_2/Conv2D/ReadVariableOpconv2d_2/Conv2D/ReadVariableOp:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?
d
F__inference_dropout_9_layer_call_and_return_conditional_losses_3595045

inputs

identity_1b
IdentityIdentityinputs*
T0*/
_output_shapes
:?????????2

Identityq

Identity_1IdentityIdentity:output:0*
T0*/
_output_shapes
:?????????2

Identity_1"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
%__inference_signature_wrapper_3595602
input_4!
unknown:P
	unknown_0:P
	unknown_1:P
	unknown_2:P
	unknown_3:P
	unknown_4:P#
	unknown_5:PP
	unknown_6:P
	unknown_7:P
	unknown_8:P
	unknown_9:P

unknown_10:P$

unknown_11:PP

unknown_12:P

unknown_13:P

unknown_14:P

unknown_15:P

unknown_16:P$

unknown_17:P

unknown_18:$

unknown_19:PI

unknown_20:I

unknown_21:@

unknown_22:
identity

identity_1??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_4unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22*$
Tin
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:?????????I:?????????*:
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8? *+
f&R$
"__inference__wrapped_model_35938512
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:?????????I2

Identity

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*'
_output_shapes
:?????????2

Identity_1h
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*^
_input_shapesM
K:?????????: : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
/
_output_shapes
:?????????
!
_user_specified_name	input_4
?
?
E__inference_conv2d_1_layer_call_and_return_conditional_losses_3596474

inputs8
conv2d_readvariableop_resource:PP-
biasadd_readvariableop_resource:P
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:PP*
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????P*
paddingSAME*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:P*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????P2	
BiasAdds
IdentityIdentityBiasAdd:output:0^NoOp*
T0*/
_output_shapes
:?????????P2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????P: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:?????????P
 
_user_specified_nameinputs
?
b
)__inference_dropout_layer_call_fn_3596438

inputs
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????P* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_dropout_layer_call_and_return_conditional_losses_35946342
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:?????????P2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????P22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????P
 
_user_specified_nameinputs
?
?
R__inference_batch_normalization_2_layer_call_and_return_conditional_losses_3596724

inputs%
readvariableop_resource:P'
readvariableop_1_resource:P6
(fusedbatchnormv3_readvariableop_resource:P8
*fusedbatchnormv3_readvariableop_1_resource:P
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:P*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:P*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:P*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:P*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+???????????????????????????P:P:P:P:P:*
epsilon%o?:*
is_training( 2
FusedBatchNormV3?
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+???????????????????????????P2

Identity?
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+???????????????????????????P: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+???????????????????????????P
 
_user_specified_nameinputs
?
?
C__inference_conv2d_layer_call_and_return_conditional_losses_3596294

inputs8
conv2d_readvariableop_resource:P-
biasadd_readvariableop_resource:P
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:P*
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????P*
paddingSAME*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:P*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????P2	
BiasAdds
IdentityIdentityBiasAdd:output:0^NoOp*
T0*/
_output_shapes
:?????????P2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?	
?
7__inference_batch_normalization_2_layer_call_fn_3596680

inputs
unknown:P
	unknown_0:P
	unknown_1:P
	unknown_2:P
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????P*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *[
fVRT
R__inference_batch_normalization_2_layer_call_and_return_conditional_losses_35941692
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+???????????????????????????P2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+???????????????????????????P: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+???????????????????????????P
 
_user_specified_nameinputs
?
?
R__inference_batch_normalization_2_layer_call_and_return_conditional_losses_3594510

inputs%
readvariableop_resource:P'
readvariableop_1_resource:P6
(fusedbatchnormv3_readvariableop_resource:P8
*fusedbatchnormv3_readvariableop_1_resource:P
identity??AssignNewValue?AssignNewValue_1?FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:P*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:P*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:P*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:P*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????P:P:P:P:P:*
epsilon%o?:*
exponential_avg_factor%
?#<2
FusedBatchNormV3?
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype02
AssignNewValue?
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype02
AssignNewValue_1w
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*/
_output_shapes
:?????????P2

Identity?
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:?????????P: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:W S
/
_output_shapes
:?????????P
 
_user_specified_nameinputs
?
?
*__inference_conv2d_1_layer_call_fn_3596464

inputs!
unknown:PP
	unknown_0:P
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????P*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_conv2d_1_layer_call_and_return_conditional_losses_35943032
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:?????????P2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????P: : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????P
 
_user_specified_nameinputs
?
d
F__inference_dropout_2_layer_call_and_return_conditional_losses_3596803

inputs

identity_1b
IdentityIdentityinputs*
T0*/
_output_shapes
:?????????P2

Identityq

Identity_1IdentityIdentity:output:0*
T0*/
_output_shapes
:?????????P2

Identity_1"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????P:W S
/
_output_shapes
:?????????P
 
_user_specified_nameinputs
?
G
+__inference_flatten_1_layer_call_fn_3596229

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_flatten_1_layer_call_and_return_conditional_losses_35950532
PartitionedCalll
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:?????????@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
)__inference_model_3_layer_call_fn_3595657

inputs!
unknown:P
	unknown_0:P
	unknown_1:P
	unknown_2:P
	unknown_3:P
	unknown_4:P#
	unknown_5:PP
	unknown_6:P
	unknown_7:P
	unknown_8:P
	unknown_9:P

unknown_10:P$

unknown_11:PP

unknown_12:P

unknown_13:P

unknown_14:P

unknown_15:P

unknown_16:P$

unknown_17:P

unknown_18:$

unknown_19:PI

unknown_20:I

unknown_21:@

unknown_22:
identity

identity_1??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22*$
Tin
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:?????????:?????????I*:
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_model_3_layer_call_and_return_conditional_losses_35950912
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????2

Identity?

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*/
_output_shapes
:?????????I2

Identity_1h
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*^
_input_shapesM
K:?????????: : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?=
?
B__inference_model_layer_call_and_return_conditional_losses_3594926
input_1(
conv2d_3594876:P
conv2d_3594878:P)
batch_normalization_3594881:P)
batch_normalization_3594883:P)
batch_normalization_3594885:P)
batch_normalization_3594887:P*
conv2d_1_3594892:PP
conv2d_1_3594894:P+
batch_normalization_1_3594897:P+
batch_normalization_1_3594899:P+
batch_normalization_1_3594901:P+
batch_normalization_1_3594903:P*
conv2d_2_3594908:PP
conv2d_2_3594910:P+
batch_normalization_2_3594913:P+
batch_normalization_2_3594915:P+
batch_normalization_2_3594917:P+
batch_normalization_2_3594919:P
identity??+batch_normalization/StatefulPartitionedCall?-batch_normalization_1/StatefulPartitionedCall?-batch_normalization_2/StatefulPartitionedCall?conv2d/StatefulPartitionedCall? conv2d_1/StatefulPartitionedCall? conv2d_2/StatefulPartitionedCall?
conv2d/StatefulPartitionedCallStatefulPartitionedCallinput_1conv2d_3594876conv2d_3594878*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????P*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_conv2d_layer_call_and_return_conditional_losses_35942462 
conv2d/StatefulPartitionedCall?
+batch_normalization/StatefulPartitionedCallStatefulPartitionedCall'conv2d/StatefulPartitionedCall:output:0batch_normalization_3594881batch_normalization_3594883batch_normalization_3594885batch_normalization_3594887*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????P*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *Y
fTRR
P__inference_batch_normalization_layer_call_and_return_conditional_losses_35942692-
+batch_normalization/StatefulPartitionedCall?
activation/PartitionedCallPartitionedCall4batch_normalization/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????P* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *P
fKRI
G__inference_activation_layer_call_and_return_conditional_losses_35942842
activation/PartitionedCall?
dropout/PartitionedCallPartitionedCall#activation/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????P* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_dropout_layer_call_and_return_conditional_losses_35942912
dropout/PartitionedCall?
 conv2d_1/StatefulPartitionedCallStatefulPartitionedCall dropout/PartitionedCall:output:0conv2d_1_3594892conv2d_1_3594894*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????P*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_conv2d_1_layer_call_and_return_conditional_losses_35943032"
 conv2d_1/StatefulPartitionedCall?
-batch_normalization_1/StatefulPartitionedCallStatefulPartitionedCall)conv2d_1/StatefulPartitionedCall:output:0batch_normalization_1_3594897batch_normalization_1_3594899batch_normalization_1_3594901batch_normalization_1_3594903*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????P*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *[
fVRT
R__inference_batch_normalization_1_layer_call_and_return_conditional_losses_35943262/
-batch_normalization_1/StatefulPartitionedCall?
activation_1/PartitionedCallPartitionedCall6batch_normalization_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????P* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *R
fMRK
I__inference_activation_1_layer_call_and_return_conditional_losses_35943412
activation_1/PartitionedCall?
dropout_1/PartitionedCallPartitionedCall%activation_1/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????P* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_dropout_1_layer_call_and_return_conditional_losses_35943482
dropout_1/PartitionedCall?
 conv2d_2/StatefulPartitionedCallStatefulPartitionedCall"dropout_1/PartitionedCall:output:0conv2d_2_3594908conv2d_2_3594910*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????P*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_conv2d_2_layer_call_and_return_conditional_losses_35943602"
 conv2d_2/StatefulPartitionedCall?
-batch_normalization_2/StatefulPartitionedCallStatefulPartitionedCall)conv2d_2/StatefulPartitionedCall:output:0batch_normalization_2_3594913batch_normalization_2_3594915batch_normalization_2_3594917batch_normalization_2_3594919*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????P*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *[
fVRT
R__inference_batch_normalization_2_layer_call_and_return_conditional_losses_35943832/
-batch_normalization_2/StatefulPartitionedCall?
tf.__operators__.add/AddV2AddV26batch_normalization_2/StatefulPartitionedCall:output:0 dropout/PartitionedCall:output:0*
T0*/
_output_shapes
:?????????P2
tf.__operators__.add/AddV2?
activation_2/PartitionedCallPartitionedCalltf.__operators__.add/AddV2:z:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????P* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *R
fMRK
I__inference_activation_2_layer_call_and_return_conditional_losses_35943992
activation_2/PartitionedCall?
dropout_2/PartitionedCallPartitionedCall%activation_2/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????P* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_dropout_2_layer_call_and_return_conditional_losses_35944062
dropout_2/PartitionedCall?
IdentityIdentity"dropout_2/PartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:?????????P2

Identity?
NoOpNoOp,^batch_normalization/StatefulPartitionedCall.^batch_normalization_1/StatefulPartitionedCall.^batch_normalization_2/StatefulPartitionedCall^conv2d/StatefulPartitionedCall!^conv2d_1/StatefulPartitionedCall!^conv2d_2/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*R
_input_shapesA
?:?????????: : : : : : : : : : : : : : : : : : 2Z
+batch_normalization/StatefulPartitionedCall+batch_normalization/StatefulPartitionedCall2^
-batch_normalization_1/StatefulPartitionedCall-batch_normalization_1/StatefulPartitionedCall2^
-batch_normalization_2/StatefulPartitionedCall-batch_normalization_2/StatefulPartitionedCall2@
conv2d/StatefulPartitionedCallconv2d/StatefulPartitionedCall2D
 conv2d_1/StatefulPartitionedCall conv2d_1/StatefulPartitionedCall2D
 conv2d_2/StatefulPartitionedCall conv2d_2/StatefulPartitionedCall:X T
/
_output_shapes
:?????????
!
_user_specified_name	input_1
?
b
F__inference_flatten_1_layer_call_and_return_conditional_losses_3595053

inputs
identity_
ConstConst*
_output_shapes
:*
dtype0*
valueB"????@   2
Constg
ReshapeReshapeinputsConst:output:0*
T0*'
_output_shapes
:?????????@2	
Reshaped
IdentityIdentityReshape:output:0*
T0*'
_output_shapes
:?????????@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?g
?
#__inference__traced_restore_3596993
file_prefix;
!assignvariableop_conv2d_11_kernel:P/
!assignvariableop_1_conv2d_11_bias:3
!assignvariableop_2_dense_1_kernel:@-
assignvariableop_3_dense_1_bias:=
#assignvariableop_4_conv2d_10_kernel:PI/
!assignvariableop_5_conv2d_10_bias:I:
 assignvariableop_6_conv2d_kernel:P,
assignvariableop_7_conv2d_bias:P:
,assignvariableop_8_batch_normalization_gamma:P9
+assignvariableop_9_batch_normalization_beta:PA
3assignvariableop_10_batch_normalization_moving_mean:PE
7assignvariableop_11_batch_normalization_moving_variance:P=
#assignvariableop_12_conv2d_1_kernel:PP/
!assignvariableop_13_conv2d_1_bias:P=
/assignvariableop_14_batch_normalization_1_gamma:P<
.assignvariableop_15_batch_normalization_1_beta:PC
5assignvariableop_16_batch_normalization_1_moving_mean:PG
9assignvariableop_17_batch_normalization_1_moving_variance:P=
#assignvariableop_18_conv2d_2_kernel:PP/
!assignvariableop_19_conv2d_2_bias:P=
/assignvariableop_20_batch_normalization_2_gamma:P<
.assignvariableop_21_batch_normalization_2_beta:PC
5assignvariableop_22_batch_normalization_2_moving_mean:PG
9assignvariableop_23_batch_normalization_2_moving_variance:P
identity_25??AssignVariableOp?AssignVariableOp_1?AssignVariableOp_10?AssignVariableOp_11?AssignVariableOp_12?AssignVariableOp_13?AssignVariableOp_14?AssignVariableOp_15?AssignVariableOp_16?AssignVariableOp_17?AssignVariableOp_18?AssignVariableOp_19?AssignVariableOp_2?AssignVariableOp_20?AssignVariableOp_21?AssignVariableOp_22?AssignVariableOp_23?AssignVariableOp_3?AssignVariableOp_4?AssignVariableOp_5?AssignVariableOp_6?AssignVariableOp_7?AssignVariableOp_8?AssignVariableOp_9?	
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*?
value?B?B6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB&variables/0/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB&variables/3/.ATTRIBUTES/VARIABLE_VALUEB&variables/4/.ATTRIBUTES/VARIABLE_VALUEB&variables/5/.ATTRIBUTES/VARIABLE_VALUEB&variables/6/.ATTRIBUTES/VARIABLE_VALUEB&variables/7/.ATTRIBUTES/VARIABLE_VALUEB&variables/8/.ATTRIBUTES/VARIABLE_VALUEB&variables/9/.ATTRIBUTES/VARIABLE_VALUEB'variables/10/.ATTRIBUTES/VARIABLE_VALUEB'variables/11/.ATTRIBUTES/VARIABLE_VALUEB'variables/12/.ATTRIBUTES/VARIABLE_VALUEB'variables/13/.ATTRIBUTES/VARIABLE_VALUEB'variables/14/.ATTRIBUTES/VARIABLE_VALUEB'variables/15/.ATTRIBUTES/VARIABLE_VALUEB'variables/16/.ATTRIBUTES/VARIABLE_VALUEB'variables/17/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2/tensor_names?
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*E
value<B:B B B B B B B B B B B B B B B B B B B B B B B B B 2
RestoreV2/shape_and_slices?
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*x
_output_shapesf
d:::::::::::::::::::::::::*'
dtypes
22
	RestoreV2g
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:2

Identity?
AssignVariableOpAssignVariableOp!assignvariableop_conv2d_11_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOpk

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:2

Identity_1?
AssignVariableOp_1AssignVariableOp!assignvariableop_1_conv2d_11_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_1k

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:2

Identity_2?
AssignVariableOp_2AssignVariableOp!assignvariableop_2_dense_1_kernelIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_2k

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:2

Identity_3?
AssignVariableOp_3AssignVariableOpassignvariableop_3_dense_1_biasIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_3k

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:2

Identity_4?
AssignVariableOp_4AssignVariableOp#assignvariableop_4_conv2d_10_kernelIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_4k

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:2

Identity_5?
AssignVariableOp_5AssignVariableOp!assignvariableop_5_conv2d_10_biasIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_5k

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:2

Identity_6?
AssignVariableOp_6AssignVariableOp assignvariableop_6_conv2d_kernelIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_6k

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:2

Identity_7?
AssignVariableOp_7AssignVariableOpassignvariableop_7_conv2d_biasIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_7k

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:2

Identity_8?
AssignVariableOp_8AssignVariableOp,assignvariableop_8_batch_normalization_gammaIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_8k

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:2

Identity_9?
AssignVariableOp_9AssignVariableOp+assignvariableop_9_batch_normalization_betaIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_9n
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:2
Identity_10?
AssignVariableOp_10AssignVariableOp3assignvariableop_10_batch_normalization_moving_meanIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_10n
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:2
Identity_11?
AssignVariableOp_11AssignVariableOp7assignvariableop_11_batch_normalization_moving_varianceIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_11n
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:2
Identity_12?
AssignVariableOp_12AssignVariableOp#assignvariableop_12_conv2d_1_kernelIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_12n
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:2
Identity_13?
AssignVariableOp_13AssignVariableOp!assignvariableop_13_conv2d_1_biasIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_13n
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:2
Identity_14?
AssignVariableOp_14AssignVariableOp/assignvariableop_14_batch_normalization_1_gammaIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_14n
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:2
Identity_15?
AssignVariableOp_15AssignVariableOp.assignvariableop_15_batch_normalization_1_betaIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_15n
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:2
Identity_16?
AssignVariableOp_16AssignVariableOp5assignvariableop_16_batch_normalization_1_moving_meanIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_16n
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:2
Identity_17?
AssignVariableOp_17AssignVariableOp9assignvariableop_17_batch_normalization_1_moving_varianceIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_17n
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:2
Identity_18?
AssignVariableOp_18AssignVariableOp#assignvariableop_18_conv2d_2_kernelIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_18n
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:2
Identity_19?
AssignVariableOp_19AssignVariableOp!assignvariableop_19_conv2d_2_biasIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_19n
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:2
Identity_20?
AssignVariableOp_20AssignVariableOp/assignvariableop_20_batch_normalization_2_gammaIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_20n
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:2
Identity_21?
AssignVariableOp_21AssignVariableOp.assignvariableop_21_batch_normalization_2_betaIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_21n
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:2
Identity_22?
AssignVariableOp_22AssignVariableOp5assignvariableop_22_batch_normalization_2_moving_meanIdentity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_22n
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:2
Identity_23?
AssignVariableOp_23AssignVariableOp9assignvariableop_23_batch_normalization_2_moving_varianceIdentity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_239
NoOpNoOp"/device:CPU:0*
_output_shapes
 2
NoOp?
Identity_24Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2
Identity_24f
Identity_25IdentityIdentity_24:output:0^NoOp_1*
T0*
_output_shapes
: 2
Identity_25?
NoOp_1NoOp^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*"
_acd_function_control_output(*
_output_shapes
 2
NoOp_1"#
identity_25Identity_25:output:0*E
_input_shapes4
2: : : : : : : : : : : : : : : : : : : : : : : : : 2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_10AssignVariableOp_102*
AssignVariableOp_11AssignVariableOp_112*
AssignVariableOp_12AssignVariableOp_122*
AssignVariableOp_13AssignVariableOp_132*
AssignVariableOp_14AssignVariableOp_142*
AssignVariableOp_15AssignVariableOp_152*
AssignVariableOp_16AssignVariableOp_162*
AssignVariableOp_17AssignVariableOp_172*
AssignVariableOp_18AssignVariableOp_182*
AssignVariableOp_19AssignVariableOp_192(
AssignVariableOp_2AssignVariableOp_22*
AssignVariableOp_20AssignVariableOp_202*
AssignVariableOp_21AssignVariableOp_212*
AssignVariableOp_22AssignVariableOp_222*
AssignVariableOp_23AssignVariableOp_232(
AssignVariableOp_3AssignVariableOp_32(
AssignVariableOp_4AssignVariableOp_42(
AssignVariableOp_5AssignVariableOp_52(
AssignVariableOp_6AssignVariableOp_62(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_9:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
?	
?
5__inference_batch_normalization_layer_call_fn_3596307

inputs
unknown:P
	unknown_0:P
	unknown_1:P
	unknown_2:P
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????P*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *Y
fTRR
P__inference_batch_normalization_layer_call_and_return_conditional_losses_35938732
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+???????????????????????????P2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+???????????????????????????P: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+???????????????????????????P
 
_user_specified_nameinputs
?
b
D__inference_dropout_layer_call_and_return_conditional_losses_3596443

inputs

identity_1b
IdentityIdentityinputs*
T0*/
_output_shapes
:?????????P2

Identityq

Identity_1IdentityIdentity:output:0*
T0*/
_output_shapes
:?????????P2

Identity_1"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????P:W S
/
_output_shapes
:?????????P
 
_user_specified_nameinputs
?
d
+__inference_dropout_1_layer_call_fn_3596618

inputs
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????P* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_dropout_1_layer_call_and_return_conditional_losses_35945512
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:?????????P2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????P22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????P
 
_user_specified_nameinputs
?
?
P__inference_batch_normalization_layer_call_and_return_conditional_losses_3596418

inputs%
readvariableop_resource:P'
readvariableop_1_resource:P6
(fusedbatchnormv3_readvariableop_resource:P8
*fusedbatchnormv3_readvariableop_1_resource:P
identity??AssignNewValue?AssignNewValue_1?FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:P*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:P*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:P*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:P*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????P:P:P:P:P:*
epsilon%o?:*
exponential_avg_factor%
?#<2
FusedBatchNormV3?
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype02
AssignNewValue?
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype02
AssignNewValue_1w
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*/
_output_shapes
:?????????P2

Identity?
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:?????????P: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:W S
/
_output_shapes
:?????????P
 
_user_specified_nameinputs
?
?
7__inference_batch_normalization_1_layer_call_fn_3596513

inputs
unknown:P
	unknown_0:P
	unknown_1:P
	unknown_2:P
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????P*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *[
fVRT
R__inference_batch_normalization_1_layer_call_and_return_conditional_losses_35943262
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:?????????P2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:?????????P: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????P
 
_user_specified_nameinputs
?
?
R__inference_batch_normalization_1_layer_call_and_return_conditional_losses_3593999

inputs%
readvariableop_resource:P'
readvariableop_1_resource:P6
(fusedbatchnormv3_readvariableop_resource:P8
*fusedbatchnormv3_readvariableop_1_resource:P
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:P*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:P*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:P*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:P*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+???????????????????????????P:P:P:P:P:*
epsilon%o?:*
is_training( 2
FusedBatchNormV3?
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+???????????????????????????P2

Identity?
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+???????????????????????????P: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+???????????????????????????P
 
_user_specified_nameinputs
?
?
R__inference_batch_normalization_1_layer_call_and_return_conditional_losses_3596562

inputs%
readvariableop_resource:P'
readvariableop_1_resource:P6
(fusedbatchnormv3_readvariableop_resource:P8
*fusedbatchnormv3_readvariableop_1_resource:P
identity??AssignNewValue?AssignNewValue_1?FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:P*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:P*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:P*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:P*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+???????????????????????????P:P:P:P:P:*
epsilon%o?:*
exponential_avg_factor%
?#<2
FusedBatchNormV3?
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype02
AssignNewValue?
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype02
AssignNewValue_1?
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+???????????????????????????P2

Identity?
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+???????????????????????????P: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+???????????????????????????P
 
_user_specified_nameinputs
?
?
+__inference_conv2d_10_layer_call_fn_3596264

inputs!
unknown:PI
	unknown_0:I
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????I*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_conv2d_10_layer_call_and_return_conditional_losses_35950662
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:?????????I2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????P: : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????P
 
_user_specified_nameinputs
?
d
+__inference_dropout_9_layer_call_fn_3596207

inputs
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_dropout_9_layer_call_and_return_conditional_losses_35951902
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:?????????2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
7__inference_batch_normalization_2_layer_call_fn_3596693

inputs
unknown:P
	unknown_0:P
	unknown_1:P
	unknown_2:P
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????P*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *[
fVRT
R__inference_batch_normalization_2_layer_call_and_return_conditional_losses_35943832
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:?????????P2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:?????????P: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????P
 
_user_specified_nameinputs
?
d
F__inference_dropout_1_layer_call_and_return_conditional_losses_3594348

inputs

identity_1b
IdentityIdentityinputs*
T0*/
_output_shapes
:?????????P2

Identityq

Identity_1IdentityIdentity:output:0*
T0*/
_output_shapes
:?????????P2

Identity_1"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????P:W S
/
_output_shapes
:?????????P
 
_user_specified_nameinputs
??
?
D__inference_model_3_layer_call_and_return_conditional_losses_3595808

inputsE
+model_conv2d_conv2d_readvariableop_resource:P:
,model_conv2d_biasadd_readvariableop_resource:P?
1model_batch_normalization_readvariableop_resource:PA
3model_batch_normalization_readvariableop_1_resource:PP
Bmodel_batch_normalization_fusedbatchnormv3_readvariableop_resource:PR
Dmodel_batch_normalization_fusedbatchnormv3_readvariableop_1_resource:PG
-model_conv2d_1_conv2d_readvariableop_resource:PP<
.model_conv2d_1_biasadd_readvariableop_resource:PA
3model_batch_normalization_1_readvariableop_resource:PC
5model_batch_normalization_1_readvariableop_1_resource:PR
Dmodel_batch_normalization_1_fusedbatchnormv3_readvariableop_resource:PT
Fmodel_batch_normalization_1_fusedbatchnormv3_readvariableop_1_resource:PG
-model_conv2d_2_conv2d_readvariableop_resource:PP<
.model_conv2d_2_biasadd_readvariableop_resource:PA
3model_batch_normalization_2_readvariableop_resource:PC
5model_batch_normalization_2_readvariableop_1_resource:PR
Dmodel_batch_normalization_2_fusedbatchnormv3_readvariableop_resource:PT
Fmodel_batch_normalization_2_fusedbatchnormv3_readvariableop_1_resource:PB
(conv2d_11_conv2d_readvariableop_resource:P7
)conv2d_11_biasadd_readvariableop_resource:B
(conv2d_10_conv2d_readvariableop_resource:PI7
)conv2d_10_biasadd_readvariableop_resource:I8
&dense_1_matmul_readvariableop_resource:@5
'dense_1_biasadd_readvariableop_resource:
identity

identity_1?? conv2d_10/BiasAdd/ReadVariableOp?conv2d_10/Conv2D/ReadVariableOp? conv2d_11/BiasAdd/ReadVariableOp?conv2d_11/Conv2D/ReadVariableOp?dense_1/BiasAdd/ReadVariableOp?dense_1/MatMul/ReadVariableOp?9model/batch_normalization/FusedBatchNormV3/ReadVariableOp?;model/batch_normalization/FusedBatchNormV3/ReadVariableOp_1?(model/batch_normalization/ReadVariableOp?*model/batch_normalization/ReadVariableOp_1?;model/batch_normalization_1/FusedBatchNormV3/ReadVariableOp?=model/batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1?*model/batch_normalization_1/ReadVariableOp?,model/batch_normalization_1/ReadVariableOp_1?;model/batch_normalization_2/FusedBatchNormV3/ReadVariableOp?=model/batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1?*model/batch_normalization_2/ReadVariableOp?,model/batch_normalization_2/ReadVariableOp_1?#model/conv2d/BiasAdd/ReadVariableOp?"model/conv2d/Conv2D/ReadVariableOp?%model/conv2d_1/BiasAdd/ReadVariableOp?$model/conv2d_1/Conv2D/ReadVariableOp?%model/conv2d_2/BiasAdd/ReadVariableOp?$model/conv2d_2/Conv2D/ReadVariableOp?
"model/conv2d/Conv2D/ReadVariableOpReadVariableOp+model_conv2d_conv2d_readvariableop_resource*&
_output_shapes
:P*
dtype02$
"model/conv2d/Conv2D/ReadVariableOp?
model/conv2d/Conv2DConv2Dinputs*model/conv2d/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????P*
paddingSAME*
strides
2
model/conv2d/Conv2D?
#model/conv2d/BiasAdd/ReadVariableOpReadVariableOp,model_conv2d_biasadd_readvariableop_resource*
_output_shapes
:P*
dtype02%
#model/conv2d/BiasAdd/ReadVariableOp?
model/conv2d/BiasAddBiasAddmodel/conv2d/Conv2D:output:0+model/conv2d/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????P2
model/conv2d/BiasAdd?
(model/batch_normalization/ReadVariableOpReadVariableOp1model_batch_normalization_readvariableop_resource*
_output_shapes
:P*
dtype02*
(model/batch_normalization/ReadVariableOp?
*model/batch_normalization/ReadVariableOp_1ReadVariableOp3model_batch_normalization_readvariableop_1_resource*
_output_shapes
:P*
dtype02,
*model/batch_normalization/ReadVariableOp_1?
9model/batch_normalization/FusedBatchNormV3/ReadVariableOpReadVariableOpBmodel_batch_normalization_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:P*
dtype02;
9model/batch_normalization/FusedBatchNormV3/ReadVariableOp?
;model/batch_normalization/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpDmodel_batch_normalization_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:P*
dtype02=
;model/batch_normalization/FusedBatchNormV3/ReadVariableOp_1?
*model/batch_normalization/FusedBatchNormV3FusedBatchNormV3model/conv2d/BiasAdd:output:00model/batch_normalization/ReadVariableOp:value:02model/batch_normalization/ReadVariableOp_1:value:0Amodel/batch_normalization/FusedBatchNormV3/ReadVariableOp:value:0Cmodel/batch_normalization/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????P:P:P:P:P:*
epsilon%o?:*
is_training( 2,
*model/batch_normalization/FusedBatchNormV3?
model/activation/ReluRelu.model/batch_normalization/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:?????????P2
model/activation/Relu?
model/dropout/IdentityIdentity#model/activation/Relu:activations:0*
T0*/
_output_shapes
:?????????P2
model/dropout/Identity?
$model/conv2d_1/Conv2D/ReadVariableOpReadVariableOp-model_conv2d_1_conv2d_readvariableop_resource*&
_output_shapes
:PP*
dtype02&
$model/conv2d_1/Conv2D/ReadVariableOp?
model/conv2d_1/Conv2DConv2Dmodel/dropout/Identity:output:0,model/conv2d_1/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????P*
paddingSAME*
strides
2
model/conv2d_1/Conv2D?
%model/conv2d_1/BiasAdd/ReadVariableOpReadVariableOp.model_conv2d_1_biasadd_readvariableop_resource*
_output_shapes
:P*
dtype02'
%model/conv2d_1/BiasAdd/ReadVariableOp?
model/conv2d_1/BiasAddBiasAddmodel/conv2d_1/Conv2D:output:0-model/conv2d_1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????P2
model/conv2d_1/BiasAdd?
*model/batch_normalization_1/ReadVariableOpReadVariableOp3model_batch_normalization_1_readvariableop_resource*
_output_shapes
:P*
dtype02,
*model/batch_normalization_1/ReadVariableOp?
,model/batch_normalization_1/ReadVariableOp_1ReadVariableOp5model_batch_normalization_1_readvariableop_1_resource*
_output_shapes
:P*
dtype02.
,model/batch_normalization_1/ReadVariableOp_1?
;model/batch_normalization_1/FusedBatchNormV3/ReadVariableOpReadVariableOpDmodel_batch_normalization_1_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:P*
dtype02=
;model/batch_normalization_1/FusedBatchNormV3/ReadVariableOp?
=model/batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpFmodel_batch_normalization_1_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:P*
dtype02?
=model/batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1?
,model/batch_normalization_1/FusedBatchNormV3FusedBatchNormV3model/conv2d_1/BiasAdd:output:02model/batch_normalization_1/ReadVariableOp:value:04model/batch_normalization_1/ReadVariableOp_1:value:0Cmodel/batch_normalization_1/FusedBatchNormV3/ReadVariableOp:value:0Emodel/batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????P:P:P:P:P:*
epsilon%o?:*
is_training( 2.
,model/batch_normalization_1/FusedBatchNormV3?
model/activation_1/ReluRelu0model/batch_normalization_1/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:?????????P2
model/activation_1/Relu?
model/dropout_1/IdentityIdentity%model/activation_1/Relu:activations:0*
T0*/
_output_shapes
:?????????P2
model/dropout_1/Identity?
$model/conv2d_2/Conv2D/ReadVariableOpReadVariableOp-model_conv2d_2_conv2d_readvariableop_resource*&
_output_shapes
:PP*
dtype02&
$model/conv2d_2/Conv2D/ReadVariableOp?
model/conv2d_2/Conv2DConv2D!model/dropout_1/Identity:output:0,model/conv2d_2/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????P*
paddingSAME*
strides
2
model/conv2d_2/Conv2D?
%model/conv2d_2/BiasAdd/ReadVariableOpReadVariableOp.model_conv2d_2_biasadd_readvariableop_resource*
_output_shapes
:P*
dtype02'
%model/conv2d_2/BiasAdd/ReadVariableOp?
model/conv2d_2/BiasAddBiasAddmodel/conv2d_2/Conv2D:output:0-model/conv2d_2/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????P2
model/conv2d_2/BiasAdd?
*model/batch_normalization_2/ReadVariableOpReadVariableOp3model_batch_normalization_2_readvariableop_resource*
_output_shapes
:P*
dtype02,
*model/batch_normalization_2/ReadVariableOp?
,model/batch_normalization_2/ReadVariableOp_1ReadVariableOp5model_batch_normalization_2_readvariableop_1_resource*
_output_shapes
:P*
dtype02.
,model/batch_normalization_2/ReadVariableOp_1?
;model/batch_normalization_2/FusedBatchNormV3/ReadVariableOpReadVariableOpDmodel_batch_normalization_2_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:P*
dtype02=
;model/batch_normalization_2/FusedBatchNormV3/ReadVariableOp?
=model/batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpFmodel_batch_normalization_2_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:P*
dtype02?
=model/batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1?
,model/batch_normalization_2/FusedBatchNormV3FusedBatchNormV3model/conv2d_2/BiasAdd:output:02model/batch_normalization_2/ReadVariableOp:value:04model/batch_normalization_2/ReadVariableOp_1:value:0Cmodel/batch_normalization_2/FusedBatchNormV3/ReadVariableOp:value:0Emodel/batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????P:P:P:P:P:*
epsilon%o?:*
is_training( 2.
,model/batch_normalization_2/FusedBatchNormV3?
 model/tf.__operators__.add/AddV2AddV20model/batch_normalization_2/FusedBatchNormV3:y:0model/dropout/Identity:output:0*
T0*/
_output_shapes
:?????????P2"
 model/tf.__operators__.add/AddV2?
model/activation_2/ReluRelu$model/tf.__operators__.add/AddV2:z:0*
T0*/
_output_shapes
:?????????P2
model/activation_2/Relu?
model/dropout_2/IdentityIdentity%model/activation_2/Relu:activations:0*
T0*/
_output_shapes
:?????????P2
model/dropout_2/Identity?
conv2d_11/Conv2D/ReadVariableOpReadVariableOp(conv2d_11_conv2d_readvariableop_resource*&
_output_shapes
:P*
dtype02!
conv2d_11/Conv2D/ReadVariableOp?
conv2d_11/Conv2DConv2D!model/dropout_2/Identity:output:0'conv2d_11/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????*
paddingVALID*
strides
2
conv2d_11/Conv2D?
 conv2d_11/BiasAdd/ReadVariableOpReadVariableOp)conv2d_11_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02"
 conv2d_11/BiasAdd/ReadVariableOp?
conv2d_11/BiasAddBiasAddconv2d_11/Conv2D:output:0(conv2d_11/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????2
conv2d_11/BiasAdd~
conv2d_11/ReluReluconv2d_11/BiasAdd:output:0*
T0*/
_output_shapes
:?????????2
conv2d_11/Relu?
dropout_9/IdentityIdentityconv2d_11/Relu:activations:0*
T0*/
_output_shapes
:?????????2
dropout_9/Identitys
flatten_1/ConstConst*
_output_shapes
:*
dtype0*
valueB"????@   2
flatten_1/Const?
flatten_1/ReshapeReshapedropout_9/Identity:output:0flatten_1/Const:output:0*
T0*'
_output_shapes
:?????????@2
flatten_1/Reshape?
conv2d_10/Conv2D/ReadVariableOpReadVariableOp(conv2d_10_conv2d_readvariableop_resource*&
_output_shapes
:PI*
dtype02!
conv2d_10/Conv2D/ReadVariableOp?
conv2d_10/Conv2DConv2D!model/dropout_2/Identity:output:0'conv2d_10/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????I*
paddingSAME*
strides
2
conv2d_10/Conv2D?
 conv2d_10/BiasAdd/ReadVariableOpReadVariableOp)conv2d_10_biasadd_readvariableop_resource*
_output_shapes
:I*
dtype02"
 conv2d_10/BiasAdd/ReadVariableOp?
conv2d_10/BiasAddBiasAddconv2d_10/Conv2D:output:0(conv2d_10/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????I2
conv2d_10/BiasAdd?
conv2d_10/SoftmaxSoftmaxconv2d_10/BiasAdd:output:0*
T0*/
_output_shapes
:?????????I2
conv2d_10/Softmax?
dense_1/MatMul/ReadVariableOpReadVariableOp&dense_1_matmul_readvariableop_resource*
_output_shapes

:@*
dtype02
dense_1/MatMul/ReadVariableOp?
dense_1/MatMulMatMulflatten_1/Reshape:output:0%dense_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_1/MatMul?
dense_1/BiasAdd/ReadVariableOpReadVariableOp'dense_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02 
dense_1/BiasAdd/ReadVariableOp?
dense_1/BiasAddBiasAdddense_1/MatMul:product:0&dense_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_1/BiasAddp
dense_1/TanhTanhdense_1/BiasAdd:output:0*
T0*'
_output_shapes
:?????????2
dense_1/Tanhk
IdentityIdentitydense_1/Tanh:y:0^NoOp*
T0*'
_output_shapes
:?????????2

Identity?

Identity_1Identityconv2d_10/Softmax:softmax:0^NoOp*
T0*/
_output_shapes
:?????????I2

Identity_1?	
NoOpNoOp!^conv2d_10/BiasAdd/ReadVariableOp ^conv2d_10/Conv2D/ReadVariableOp!^conv2d_11/BiasAdd/ReadVariableOp ^conv2d_11/Conv2D/ReadVariableOp^dense_1/BiasAdd/ReadVariableOp^dense_1/MatMul/ReadVariableOp:^model/batch_normalization/FusedBatchNormV3/ReadVariableOp<^model/batch_normalization/FusedBatchNormV3/ReadVariableOp_1)^model/batch_normalization/ReadVariableOp+^model/batch_normalization/ReadVariableOp_1<^model/batch_normalization_1/FusedBatchNormV3/ReadVariableOp>^model/batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1+^model/batch_normalization_1/ReadVariableOp-^model/batch_normalization_1/ReadVariableOp_1<^model/batch_normalization_2/FusedBatchNormV3/ReadVariableOp>^model/batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1+^model/batch_normalization_2/ReadVariableOp-^model/batch_normalization_2/ReadVariableOp_1$^model/conv2d/BiasAdd/ReadVariableOp#^model/conv2d/Conv2D/ReadVariableOp&^model/conv2d_1/BiasAdd/ReadVariableOp%^model/conv2d_1/Conv2D/ReadVariableOp&^model/conv2d_2/BiasAdd/ReadVariableOp%^model/conv2d_2/Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*^
_input_shapesM
K:?????????: : : : : : : : : : : : : : : : : : : : : : : : 2D
 conv2d_10/BiasAdd/ReadVariableOp conv2d_10/BiasAdd/ReadVariableOp2B
conv2d_10/Conv2D/ReadVariableOpconv2d_10/Conv2D/ReadVariableOp2D
 conv2d_11/BiasAdd/ReadVariableOp conv2d_11/BiasAdd/ReadVariableOp2B
conv2d_11/Conv2D/ReadVariableOpconv2d_11/Conv2D/ReadVariableOp2@
dense_1/BiasAdd/ReadVariableOpdense_1/BiasAdd/ReadVariableOp2>
dense_1/MatMul/ReadVariableOpdense_1/MatMul/ReadVariableOp2v
9model/batch_normalization/FusedBatchNormV3/ReadVariableOp9model/batch_normalization/FusedBatchNormV3/ReadVariableOp2z
;model/batch_normalization/FusedBatchNormV3/ReadVariableOp_1;model/batch_normalization/FusedBatchNormV3/ReadVariableOp_12T
(model/batch_normalization/ReadVariableOp(model/batch_normalization/ReadVariableOp2X
*model/batch_normalization/ReadVariableOp_1*model/batch_normalization/ReadVariableOp_12z
;model/batch_normalization_1/FusedBatchNormV3/ReadVariableOp;model/batch_normalization_1/FusedBatchNormV3/ReadVariableOp2~
=model/batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1=model/batch_normalization_1/FusedBatchNormV3/ReadVariableOp_12X
*model/batch_normalization_1/ReadVariableOp*model/batch_normalization_1/ReadVariableOp2\
,model/batch_normalization_1/ReadVariableOp_1,model/batch_normalization_1/ReadVariableOp_12z
;model/batch_normalization_2/FusedBatchNormV3/ReadVariableOp;model/batch_normalization_2/FusedBatchNormV3/ReadVariableOp2~
=model/batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1=model/batch_normalization_2/FusedBatchNormV3/ReadVariableOp_12X
*model/batch_normalization_2/ReadVariableOp*model/batch_normalization_2/ReadVariableOp2\
,model/batch_normalization_2/ReadVariableOp_1,model/batch_normalization_2/ReadVariableOp_12J
#model/conv2d/BiasAdd/ReadVariableOp#model/conv2d/BiasAdd/ReadVariableOp2H
"model/conv2d/Conv2D/ReadVariableOp"model/conv2d/Conv2D/ReadVariableOp2N
%model/conv2d_1/BiasAdd/ReadVariableOp%model/conv2d_1/BiasAdd/ReadVariableOp2L
$model/conv2d_1/Conv2D/ReadVariableOp$model/conv2d_1/Conv2D/ReadVariableOp2N
%model/conv2d_2/BiasAdd/ReadVariableOp%model/conv2d_2/BiasAdd/ReadVariableOp2L
$model/conv2d_2/Conv2D/ReadVariableOp$model/conv2d_2/Conv2D/ReadVariableOp:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?
e
I__inference_activation_1_layer_call_and_return_conditional_losses_3596608

inputs
identityV
ReluReluinputs*
T0*/
_output_shapes
:?????????P2
Relun
IdentityIdentityRelu:activations:0*
T0*/
_output_shapes
:?????????P2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????P:W S
/
_output_shapes
:?????????P
 
_user_specified_nameinputs
?
?
'__inference_model_layer_call_fn_3594873
input_1!
unknown:P
	unknown_0:P
	unknown_1:P
	unknown_2:P
	unknown_3:P
	unknown_4:P#
	unknown_5:PP
	unknown_6:P
	unknown_7:P
	unknown_8:P
	unknown_9:P

unknown_10:P$

unknown_11:PP

unknown_12:P

unknown_13:P

unknown_14:P

unknown_15:P

unknown_16:P
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????P*.
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_model_layer_call_and_return_conditional_losses_35947932
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:?????????P2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*R
_input_shapesA
?:?????????: : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
/
_output_shapes
:?????????
!
_user_specified_name	input_1
?
?
*__inference_conv2d_2_layer_call_fn_3596644

inputs!
unknown:PP
	unknown_0:P
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????P*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_conv2d_2_layer_call_and_return_conditional_losses_35943602
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:?????????P2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????P: : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????P
 
_user_specified_nameinputs
?
e
I__inference_activation_2_layer_call_and_return_conditional_losses_3594399

inputs
identityV
ReluReluinputs*
T0*/
_output_shapes
:?????????P2
Relun
IdentityIdentityRelu:activations:0*
T0*/
_output_shapes
:?????????P2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????P:W S
/
_output_shapes
:?????????P
 
_user_specified_nameinputs
?
e
F__inference_dropout_1_layer_call_and_return_conditional_losses_3596635

inputs
identity?c
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
dropout/Const{
dropout/MulMulinputsdropout/Const:output:0*
T0*/
_output_shapes
:?????????P2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape?
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*/
_output_shapes
:?????????P*
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L>2
dropout/GreaterEqual/y?
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*/
_output_shapes
:?????????P2
dropout/GreaterEqual?
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*/
_output_shapes
:?????????P2
dropout/Cast?
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*/
_output_shapes
:?????????P2
dropout/Mul_1m
IdentityIdentitydropout/Mul_1:z:0*
T0*/
_output_shapes
:?????????P2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????P:W S
/
_output_shapes
:?????????P
 
_user_specified_nameinputs
?
?
E__inference_conv2d_1_layer_call_and_return_conditional_losses_3594303

inputs8
conv2d_readvariableop_resource:PP-
biasadd_readvariableop_resource:P
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:PP*
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????P*
paddingSAME*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:P*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????P2	
BiasAdds
IdentityIdentityBiasAdd:output:0^NoOp*
T0*/
_output_shapes
:?????????P2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????P: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:?????????P
 
_user_specified_nameinputs
?
?
P__inference_batch_normalization_layer_call_and_return_conditional_losses_3594676

inputs%
readvariableop_resource:P'
readvariableop_1_resource:P6
(fusedbatchnormv3_readvariableop_resource:P8
*fusedbatchnormv3_readvariableop_1_resource:P
identity??AssignNewValue?AssignNewValue_1?FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:P*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:P*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:P*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:P*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????P:P:P:P:P:*
epsilon%o?:*
exponential_avg_factor%
?#<2
FusedBatchNormV3?
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype02
AssignNewValue?
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype02
AssignNewValue_1w
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*/
_output_shapes
:?????????P2

Identity?
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:?????????P: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:W S
/
_output_shapes
:?????????P
 
_user_specified_nameinputs
?
?
)__inference_dense_1_layer_call_fn_3596244

inputs
unknown:@
	unknown_0:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_dense_1_layer_call_and_return_conditional_losses_35950832
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????@: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????@
 
_user_specified_nameinputs
?
?
P__inference_batch_normalization_layer_call_and_return_conditional_losses_3593917

inputs%
readvariableop_resource:P'
readvariableop_1_resource:P6
(fusedbatchnormv3_readvariableop_resource:P8
*fusedbatchnormv3_readvariableop_1_resource:P
identity??AssignNewValue?AssignNewValue_1?FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:P*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:P*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:P*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:P*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+???????????????????????????P:P:P:P:P:*
epsilon%o?:*
exponential_avg_factor%
?#<2
FusedBatchNormV3?
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype02
AssignNewValue?
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype02
AssignNewValue_1?
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+???????????????????????????P2

Identity?
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+???????????????????????????P: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+???????????????????????????P
 
_user_specified_nameinputs
?
?
)__inference_model_3_layer_call_fn_3595144
input_4!
unknown:P
	unknown_0:P
	unknown_1:P
	unknown_2:P
	unknown_3:P
	unknown_4:P#
	unknown_5:PP
	unknown_6:P
	unknown_7:P
	unknown_8:P
	unknown_9:P

unknown_10:P$

unknown_11:PP

unknown_12:P

unknown_13:P

unknown_14:P

unknown_15:P

unknown_16:P$

unknown_17:P

unknown_18:$

unknown_19:PI

unknown_20:I

unknown_21:@

unknown_22:
identity

identity_1??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_4unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22*$
Tin
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:?????????:?????????I*:
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_model_3_layer_call_and_return_conditional_losses_35950912
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????2

Identity?

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*/
_output_shapes
:?????????I2

Identity_1h
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*^
_input_shapesM
K:?????????: : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
/
_output_shapes
:?????????
!
_user_specified_name	input_4
?
e
F__inference_dropout_2_layer_call_and_return_conditional_losses_3594468

inputs
identity?c
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
dropout/Const{
dropout/MulMulinputsdropout/Const:output:0*
T0*/
_output_shapes
:?????????P2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape?
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*/
_output_shapes
:?????????P*
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L>2
dropout/GreaterEqual/y?
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*/
_output_shapes
:?????????P2
dropout/GreaterEqual?
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*/
_output_shapes
:?????????P2
dropout/Cast?
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*/
_output_shapes
:?????????P2
dropout/Mul_1m
IdentityIdentitydropout/Mul_1:z:0*
T0*/
_output_shapes
:?????????P2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????P:W S
/
_output_shapes
:?????????P
 
_user_specified_nameinputs
?
?
R__inference_batch_normalization_2_layer_call_and_return_conditional_losses_3596778

inputs%
readvariableop_resource:P'
readvariableop_1_resource:P6
(fusedbatchnormv3_readvariableop_resource:P8
*fusedbatchnormv3_readvariableop_1_resource:P
identity??AssignNewValue?AssignNewValue_1?FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:P*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:P*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:P*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:P*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????P:P:P:P:P:*
epsilon%o?:*
exponential_avg_factor%
?#<2
FusedBatchNormV3?
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype02
AssignNewValue?
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype02
AssignNewValue_1w
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*/
_output_shapes
:?????????P2

Identity?
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:?????????P: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:W S
/
_output_shapes
:?????????P
 
_user_specified_nameinputs
?
e
I__inference_activation_1_layer_call_and_return_conditional_losses_3594341

inputs
identityV
ReluReluinputs*
T0*/
_output_shapes
:?????????P2
Relun
IdentityIdentityRelu:activations:0*
T0*/
_output_shapes
:?????????P2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????P:W S
/
_output_shapes
:?????????P
 
_user_specified_nameinputs
?
?
+__inference_conv2d_11_layer_call_fn_3596186

inputs!
unknown:P
	unknown_0:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_conv2d_11_layer_call_and_return_conditional_losses_35950342
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:?????????2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????P: : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????P
 
_user_specified_nameinputs
?
e
F__inference_dropout_9_layer_call_and_return_conditional_losses_3595190

inputs
identity?c
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
dropout/Const{
dropout/MulMulinputsdropout/Const:output:0*
T0*/
_output_shapes
:?????????2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape?
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*/
_output_shapes
:?????????*
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L>2
dropout/GreaterEqual/y?
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*/
_output_shapes
:?????????2
dropout/GreaterEqual?
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*/
_output_shapes
:?????????2
dropout/Cast?
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*/
_output_shapes
:?????????2
dropout/Mul_1m
IdentityIdentitydropout/Mul_1:z:0*
T0*/
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
P__inference_batch_normalization_layer_call_and_return_conditional_losses_3593873

inputs%
readvariableop_resource:P'
readvariableop_1_resource:P6
(fusedbatchnormv3_readvariableop_resource:P8
*fusedbatchnormv3_readvariableop_1_resource:P
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:P*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:P*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:P*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:P*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+???????????????????????????P:P:P:P:P:*
epsilon%o?:*
is_training( 2
FusedBatchNormV3?
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+???????????????????????????P2

Identity?
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+???????????????????????????P: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+???????????????????????????P
 
_user_specified_nameinputs
?	
?
7__inference_batch_normalization_2_layer_call_fn_3596667

inputs
unknown:P
	unknown_0:P
	unknown_1:P
	unknown_2:P
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????P*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *[
fVRT
R__inference_batch_normalization_2_layer_call_and_return_conditional_losses_35941252
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+???????????????????????????P2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+???????????????????????????P: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+???????????????????????????P
 
_user_specified_nameinputs
?
?
E__inference_conv2d_2_layer_call_and_return_conditional_losses_3594360

inputs8
conv2d_readvariableop_resource:PP-
biasadd_readvariableop_resource:P
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:PP*
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????P*
paddingSAME*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:P*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????P2	
BiasAdds
IdentityIdentityBiasAdd:output:0^NoOp*
T0*/
_output_shapes
:?????????P2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????P: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:?????????P
 
_user_specified_nameinputs
?
?
R__inference_batch_normalization_1_layer_call_and_return_conditional_losses_3596598

inputs%
readvariableop_resource:P'
readvariableop_1_resource:P6
(fusedbatchnormv3_readvariableop_resource:P8
*fusedbatchnormv3_readvariableop_1_resource:P
identity??AssignNewValue?AssignNewValue_1?FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:P*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:P*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:P*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:P*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????P:P:P:P:P:*
epsilon%o?:*
exponential_avg_factor%
?#<2
FusedBatchNormV3?
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype02
AssignNewValue?
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype02
AssignNewValue_1w
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*/
_output_shapes
:?????????P2

Identity?
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:?????????P: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:W S
/
_output_shapes
:?????????P
 
_user_specified_nameinputs
?	
?
7__inference_batch_normalization_1_layer_call_fn_3596500

inputs
unknown:P
	unknown_0:P
	unknown_1:P
	unknown_2:P
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????P*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *[
fVRT
R__inference_batch_normalization_1_layer_call_and_return_conditional_losses_35940432
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+???????????????????????????P2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+???????????????????????????P: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+???????????????????????????P
 
_user_specified_nameinputs
?
c
D__inference_dropout_layer_call_and_return_conditional_losses_3594634

inputs
identity?c
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
dropout/Const{
dropout/MulMulinputsdropout/Const:output:0*
T0*/
_output_shapes
:?????????P2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape?
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*/
_output_shapes
:?????????P*
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L>2
dropout/GreaterEqual/y?
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*/
_output_shapes
:?????????P2
dropout/GreaterEqual?
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*/
_output_shapes
:?????????P2
dropout/Cast?
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*/
_output_shapes
:?????????P2
dropout/Mul_1m
IdentityIdentitydropout/Mul_1:z:0*
T0*/
_output_shapes
:?????????P2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????P:W S
/
_output_shapes
:?????????P
 
_user_specified_nameinputs
?&
?
D__inference_model_3_layer_call_and_return_conditional_losses_3595486
input_4'
model_3595430:P
model_3595432:P
model_3595434:P
model_3595436:P
model_3595438:P
model_3595440:P'
model_3595442:PP
model_3595444:P
model_3595446:P
model_3595448:P
model_3595450:P
model_3595452:P'
model_3595454:PP
model_3595456:P
model_3595458:P
model_3595460:P
model_3595462:P
model_3595464:P+
conv2d_11_3595467:P
conv2d_11_3595469:+
conv2d_10_3595474:PI
conv2d_10_3595476:I!
dense_1_3595479:@
dense_1_3595481:
identity

identity_1??!conv2d_10/StatefulPartitionedCall?!conv2d_11/StatefulPartitionedCall?dense_1/StatefulPartitionedCall?model/StatefulPartitionedCall?
model/StatefulPartitionedCallStatefulPartitionedCallinput_4model_3595430model_3595432model_3595434model_3595436model_3595438model_3595440model_3595442model_3595444model_3595446model_3595448model_3595450model_3595452model_3595454model_3595456model_3595458model_3595460model_3595462model_3595464*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????P*4
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_model_layer_call_and_return_conditional_losses_35944092
model/StatefulPartitionedCall?
!conv2d_11/StatefulPartitionedCallStatefulPartitionedCall&model/StatefulPartitionedCall:output:0conv2d_11_3595467conv2d_11_3595469*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_conv2d_11_layer_call_and_return_conditional_losses_35950342#
!conv2d_11/StatefulPartitionedCall?
dropout_9/PartitionedCallPartitionedCall*conv2d_11/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_dropout_9_layer_call_and_return_conditional_losses_35950452
dropout_9/PartitionedCall?
flatten_1/PartitionedCallPartitionedCall"dropout_9/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_flatten_1_layer_call_and_return_conditional_losses_35950532
flatten_1/PartitionedCall?
!conv2d_10/StatefulPartitionedCallStatefulPartitionedCall&model/StatefulPartitionedCall:output:0conv2d_10_3595474conv2d_10_3595476*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????I*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_conv2d_10_layer_call_and_return_conditional_losses_35950662#
!conv2d_10/StatefulPartitionedCall?
dense_1/StatefulPartitionedCallStatefulPartitionedCall"flatten_1/PartitionedCall:output:0dense_1_3595479dense_1_3595481*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_dense_1_layer_call_and_return_conditional_losses_35950832!
dense_1/StatefulPartitionedCall?
IdentityIdentity(dense_1/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????2

Identity?

Identity_1Identity*conv2d_10/StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:?????????I2

Identity_1?
NoOpNoOp"^conv2d_10/StatefulPartitionedCall"^conv2d_11/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall^model/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*^
_input_shapesM
K:?????????: : : : : : : : : : : : : : : : : : : : : : : : 2F
!conv2d_10/StatefulPartitionedCall!conv2d_10/StatefulPartitionedCall2F
!conv2d_11/StatefulPartitionedCall!conv2d_11/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall2>
model/StatefulPartitionedCallmodel/StatefulPartitionedCall:X T
/
_output_shapes
:?????????
!
_user_specified_name	input_4
?
?
R__inference_batch_normalization_1_layer_call_and_return_conditional_losses_3596544

inputs%
readvariableop_resource:P'
readvariableop_1_resource:P6
(fusedbatchnormv3_readvariableop_resource:P8
*fusedbatchnormv3_readvariableop_1_resource:P
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:P*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:P*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:P*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:P*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+???????????????????????????P:P:P:P:P:*
epsilon%o?:*
is_training( 2
FusedBatchNormV3?
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+???????????????????????????P2

Identity?
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+???????????????????????????P: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+???????????????????????????P
 
_user_specified_nameinputs
?
c
G__inference_activation_layer_call_and_return_conditional_losses_3596428

inputs
identityV
ReluReluinputs*
T0*/
_output_shapes
:?????????P2
Relun
IdentityIdentityRelu:activations:0*
T0*/
_output_shapes
:?????????P2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????P:W S
/
_output_shapes
:?????????P
 
_user_specified_nameinputs
?
d
+__inference_dropout_2_layer_call_fn_3596798

inputs
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????P* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_dropout_2_layer_call_and_return_conditional_losses_35944682
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:?????????P2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????P22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????P
 
_user_specified_nameinputs
?
b
D__inference_dropout_layer_call_and_return_conditional_losses_3594291

inputs

identity_1b
IdentityIdentityinputs*
T0*/
_output_shapes
:?????????P2

Identityq

Identity_1IdentityIdentity:output:0*
T0*/
_output_shapes
:?????????P2

Identity_1"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????P:W S
/
_output_shapes
:?????????P
 
_user_specified_nameinputs
?
c
D__inference_dropout_layer_call_and_return_conditional_losses_3596455

inputs
identity?c
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
dropout/Const{
dropout/MulMulinputsdropout/Const:output:0*
T0*/
_output_shapes
:?????????P2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape?
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*/
_output_shapes
:?????????P*
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L>2
dropout/GreaterEqual/y?
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*/
_output_shapes
:?????????P2
dropout/GreaterEqual?
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*/
_output_shapes
:?????????P2
dropout/Cast?
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*/
_output_shapes
:?????????P2
dropout/Mul_1m
IdentityIdentitydropout/Mul_1:z:0*
T0*/
_output_shapes
:?????????P2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????P:W S
/
_output_shapes
:?????????P
 
_user_specified_nameinputs
?8
?
 __inference__traced_save_3596911
file_prefix/
+savev2_conv2d_11_kernel_read_readvariableop-
)savev2_conv2d_11_bias_read_readvariableop-
)savev2_dense_1_kernel_read_readvariableop+
'savev2_dense_1_bias_read_readvariableop/
+savev2_conv2d_10_kernel_read_readvariableop-
)savev2_conv2d_10_bias_read_readvariableop,
(savev2_conv2d_kernel_read_readvariableop*
&savev2_conv2d_bias_read_readvariableop8
4savev2_batch_normalization_gamma_read_readvariableop7
3savev2_batch_normalization_beta_read_readvariableop>
:savev2_batch_normalization_moving_mean_read_readvariableopB
>savev2_batch_normalization_moving_variance_read_readvariableop.
*savev2_conv2d_1_kernel_read_readvariableop,
(savev2_conv2d_1_bias_read_readvariableop:
6savev2_batch_normalization_1_gamma_read_readvariableop9
5savev2_batch_normalization_1_beta_read_readvariableop@
<savev2_batch_normalization_1_moving_mean_read_readvariableopD
@savev2_batch_normalization_1_moving_variance_read_readvariableop.
*savev2_conv2d_2_kernel_read_readvariableop,
(savev2_conv2d_2_bias_read_readvariableop:
6savev2_batch_normalization_2_gamma_read_readvariableop9
5savev2_batch_normalization_2_beta_read_readvariableop@
<savev2_batch_normalization_2_moving_mean_read_readvariableopD
@savev2_batch_normalization_2_moving_variance_read_readvariableop
savev2_const

identity_1??MergeV2Checkpoints?
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*2
StaticRegexFullMatchc
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.part2
Constl
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B
_temp/part2	
Const_1?
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: 2
Selectt

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: 2

StringJoinZ

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :2

num_shards
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : 2
ShardedFilename/shard?
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 2
ShardedFilename?	
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*?
value?B?B6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB&variables/0/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB&variables/3/.ATTRIBUTES/VARIABLE_VALUEB&variables/4/.ATTRIBUTES/VARIABLE_VALUEB&variables/5/.ATTRIBUTES/VARIABLE_VALUEB&variables/6/.ATTRIBUTES/VARIABLE_VALUEB&variables/7/.ATTRIBUTES/VARIABLE_VALUEB&variables/8/.ATTRIBUTES/VARIABLE_VALUEB&variables/9/.ATTRIBUTES/VARIABLE_VALUEB'variables/10/.ATTRIBUTES/VARIABLE_VALUEB'variables/11/.ATTRIBUTES/VARIABLE_VALUEB'variables/12/.ATTRIBUTES/VARIABLE_VALUEB'variables/13/.ATTRIBUTES/VARIABLE_VALUEB'variables/14/.ATTRIBUTES/VARIABLE_VALUEB'variables/15/.ATTRIBUTES/VARIABLE_VALUEB'variables/16/.ATTRIBUTES/VARIABLE_VALUEB'variables/17/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2/tensor_names?
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*E
value<B:B B B B B B B B B B B B B B B B B B B B B B B B B 2
SaveV2/shape_and_slices?
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0+savev2_conv2d_11_kernel_read_readvariableop)savev2_conv2d_11_bias_read_readvariableop)savev2_dense_1_kernel_read_readvariableop'savev2_dense_1_bias_read_readvariableop+savev2_conv2d_10_kernel_read_readvariableop)savev2_conv2d_10_bias_read_readvariableop(savev2_conv2d_kernel_read_readvariableop&savev2_conv2d_bias_read_readvariableop4savev2_batch_normalization_gamma_read_readvariableop3savev2_batch_normalization_beta_read_readvariableop:savev2_batch_normalization_moving_mean_read_readvariableop>savev2_batch_normalization_moving_variance_read_readvariableop*savev2_conv2d_1_kernel_read_readvariableop(savev2_conv2d_1_bias_read_readvariableop6savev2_batch_normalization_1_gamma_read_readvariableop5savev2_batch_normalization_1_beta_read_readvariableop<savev2_batch_normalization_1_moving_mean_read_readvariableop@savev2_batch_normalization_1_moving_variance_read_readvariableop*savev2_conv2d_2_kernel_read_readvariableop(savev2_conv2d_2_bias_read_readvariableop6savev2_batch_normalization_2_gamma_read_readvariableop5savev2_batch_normalization_2_beta_read_readvariableop<savev2_batch_normalization_2_moving_mean_read_readvariableop@savev2_batch_normalization_2_moving_variance_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *'
dtypes
22
SaveV2?
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:2(
&MergeV2Checkpoints/checkpoint_prefixes?
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*
_output_shapes
 2
MergeV2Checkpointsr
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: 2

Identity_

Identity_1IdentityIdentity:output:0^NoOp*
T0*
_output_shapes
: 2

Identity_1c
NoOpNoOp^MergeV2Checkpoints*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"!

identity_1Identity_1:output:0*?
_input_shapes?
?: :P::@::PI:I:P:P:P:P:P:P:PP:P:P:P:P:P:PP:P:P:P:P:P: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:,(
&
_output_shapes
:P: 

_output_shapes
::$ 

_output_shapes

:@: 

_output_shapes
::,(
&
_output_shapes
:PI: 

_output_shapes
:I:,(
&
_output_shapes
:P: 

_output_shapes
:P: 	

_output_shapes
:P: 


_output_shapes
:P: 

_output_shapes
:P: 

_output_shapes
:P:,(
&
_output_shapes
:PP: 

_output_shapes
:P: 

_output_shapes
:P: 

_output_shapes
:P: 

_output_shapes
:P: 

_output_shapes
:P:,(
&
_output_shapes
:PP: 

_output_shapes
:P: 

_output_shapes
:P: 

_output_shapes
:P: 

_output_shapes
:P: 

_output_shapes
:P:

_output_shapes
: 
?
?
P__inference_batch_normalization_layer_call_and_return_conditional_losses_3596364

inputs%
readvariableop_resource:P'
readvariableop_1_resource:P6
(fusedbatchnormv3_readvariableop_resource:P8
*fusedbatchnormv3_readvariableop_1_resource:P
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:P*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:P*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:P*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:P*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+???????????????????????????P:P:P:P:P:*
epsilon%o?:*
is_training( 2
FusedBatchNormV3?
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+???????????????????????????P2

Identity?
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+???????????????????????????P: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+???????????????????????????P
 
_user_specified_nameinputs
?=
?
B__inference_model_layer_call_and_return_conditional_losses_3594409

inputs(
conv2d_3594247:P
conv2d_3594249:P)
batch_normalization_3594270:P)
batch_normalization_3594272:P)
batch_normalization_3594274:P)
batch_normalization_3594276:P*
conv2d_1_3594304:PP
conv2d_1_3594306:P+
batch_normalization_1_3594327:P+
batch_normalization_1_3594329:P+
batch_normalization_1_3594331:P+
batch_normalization_1_3594333:P*
conv2d_2_3594361:PP
conv2d_2_3594363:P+
batch_normalization_2_3594384:P+
batch_normalization_2_3594386:P+
batch_normalization_2_3594388:P+
batch_normalization_2_3594390:P
identity??+batch_normalization/StatefulPartitionedCall?-batch_normalization_1/StatefulPartitionedCall?-batch_normalization_2/StatefulPartitionedCall?conv2d/StatefulPartitionedCall? conv2d_1/StatefulPartitionedCall? conv2d_2/StatefulPartitionedCall?
conv2d/StatefulPartitionedCallStatefulPartitionedCallinputsconv2d_3594247conv2d_3594249*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????P*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_conv2d_layer_call_and_return_conditional_losses_35942462 
conv2d/StatefulPartitionedCall?
+batch_normalization/StatefulPartitionedCallStatefulPartitionedCall'conv2d/StatefulPartitionedCall:output:0batch_normalization_3594270batch_normalization_3594272batch_normalization_3594274batch_normalization_3594276*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????P*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *Y
fTRR
P__inference_batch_normalization_layer_call_and_return_conditional_losses_35942692-
+batch_normalization/StatefulPartitionedCall?
activation/PartitionedCallPartitionedCall4batch_normalization/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????P* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *P
fKRI
G__inference_activation_layer_call_and_return_conditional_losses_35942842
activation/PartitionedCall?
dropout/PartitionedCallPartitionedCall#activation/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????P* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_dropout_layer_call_and_return_conditional_losses_35942912
dropout/PartitionedCall?
 conv2d_1/StatefulPartitionedCallStatefulPartitionedCall dropout/PartitionedCall:output:0conv2d_1_3594304conv2d_1_3594306*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????P*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_conv2d_1_layer_call_and_return_conditional_losses_35943032"
 conv2d_1/StatefulPartitionedCall?
-batch_normalization_1/StatefulPartitionedCallStatefulPartitionedCall)conv2d_1/StatefulPartitionedCall:output:0batch_normalization_1_3594327batch_normalization_1_3594329batch_normalization_1_3594331batch_normalization_1_3594333*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????P*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *[
fVRT
R__inference_batch_normalization_1_layer_call_and_return_conditional_losses_35943262/
-batch_normalization_1/StatefulPartitionedCall?
activation_1/PartitionedCallPartitionedCall6batch_normalization_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????P* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *R
fMRK
I__inference_activation_1_layer_call_and_return_conditional_losses_35943412
activation_1/PartitionedCall?
dropout_1/PartitionedCallPartitionedCall%activation_1/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????P* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_dropout_1_layer_call_and_return_conditional_losses_35943482
dropout_1/PartitionedCall?
 conv2d_2/StatefulPartitionedCallStatefulPartitionedCall"dropout_1/PartitionedCall:output:0conv2d_2_3594361conv2d_2_3594363*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????P*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_conv2d_2_layer_call_and_return_conditional_losses_35943602"
 conv2d_2/StatefulPartitionedCall?
-batch_normalization_2/StatefulPartitionedCallStatefulPartitionedCall)conv2d_2/StatefulPartitionedCall:output:0batch_normalization_2_3594384batch_normalization_2_3594386batch_normalization_2_3594388batch_normalization_2_3594390*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????P*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *[
fVRT
R__inference_batch_normalization_2_layer_call_and_return_conditional_losses_35943832/
-batch_normalization_2/StatefulPartitionedCall?
tf.__operators__.add/AddV2AddV26batch_normalization_2/StatefulPartitionedCall:output:0 dropout/PartitionedCall:output:0*
T0*/
_output_shapes
:?????????P2
tf.__operators__.add/AddV2?
activation_2/PartitionedCallPartitionedCalltf.__operators__.add/AddV2:z:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????P* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *R
fMRK
I__inference_activation_2_layer_call_and_return_conditional_losses_35943992
activation_2/PartitionedCall?
dropout_2/PartitionedCallPartitionedCall%activation_2/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????P* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_dropout_2_layer_call_and_return_conditional_losses_35944062
dropout_2/PartitionedCall?
IdentityIdentity"dropout_2/PartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:?????????P2

Identity?
NoOpNoOp,^batch_normalization/StatefulPartitionedCall.^batch_normalization_1/StatefulPartitionedCall.^batch_normalization_2/StatefulPartitionedCall^conv2d/StatefulPartitionedCall!^conv2d_1/StatefulPartitionedCall!^conv2d_2/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*R
_input_shapesA
?:?????????: : : : : : : : : : : : : : : : : : 2Z
+batch_normalization/StatefulPartitionedCall+batch_normalization/StatefulPartitionedCall2^
-batch_normalization_1/StatefulPartitionedCall-batch_normalization_1/StatefulPartitionedCall2^
-batch_normalization_2/StatefulPartitionedCall-batch_normalization_2/StatefulPartitionedCall2@
conv2d/StatefulPartitionedCallconv2d/StatefulPartitionedCall2D
 conv2d_1/StatefulPartitionedCall conv2d_1/StatefulPartitionedCall2D
 conv2d_2/StatefulPartitionedCall conv2d_2/StatefulPartitionedCall:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?&
?
D__inference_model_3_layer_call_and_return_conditional_losses_3595091

inputs'
model_3594986:P
model_3594988:P
model_3594990:P
model_3594992:P
model_3594994:P
model_3594996:P'
model_3594998:PP
model_3595000:P
model_3595002:P
model_3595004:P
model_3595006:P
model_3595008:P'
model_3595010:PP
model_3595012:P
model_3595014:P
model_3595016:P
model_3595018:P
model_3595020:P+
conv2d_11_3595035:P
conv2d_11_3595037:+
conv2d_10_3595067:PI
conv2d_10_3595069:I!
dense_1_3595084:@
dense_1_3595086:
identity

identity_1??!conv2d_10/StatefulPartitionedCall?!conv2d_11/StatefulPartitionedCall?dense_1/StatefulPartitionedCall?model/StatefulPartitionedCall?
model/StatefulPartitionedCallStatefulPartitionedCallinputsmodel_3594986model_3594988model_3594990model_3594992model_3594994model_3594996model_3594998model_3595000model_3595002model_3595004model_3595006model_3595008model_3595010model_3595012model_3595014model_3595016model_3595018model_3595020*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????P*4
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_model_layer_call_and_return_conditional_losses_35944092
model/StatefulPartitionedCall?
!conv2d_11/StatefulPartitionedCallStatefulPartitionedCall&model/StatefulPartitionedCall:output:0conv2d_11_3595035conv2d_11_3595037*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_conv2d_11_layer_call_and_return_conditional_losses_35950342#
!conv2d_11/StatefulPartitionedCall?
dropout_9/PartitionedCallPartitionedCall*conv2d_11/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_dropout_9_layer_call_and_return_conditional_losses_35950452
dropout_9/PartitionedCall?
flatten_1/PartitionedCallPartitionedCall"dropout_9/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_flatten_1_layer_call_and_return_conditional_losses_35950532
flatten_1/PartitionedCall?
!conv2d_10/StatefulPartitionedCallStatefulPartitionedCall&model/StatefulPartitionedCall:output:0conv2d_10_3595067conv2d_10_3595069*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????I*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_conv2d_10_layer_call_and_return_conditional_losses_35950662#
!conv2d_10/StatefulPartitionedCall?
dense_1/StatefulPartitionedCallStatefulPartitionedCall"flatten_1/PartitionedCall:output:0dense_1_3595084dense_1_3595086*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_dense_1_layer_call_and_return_conditional_losses_35950832!
dense_1/StatefulPartitionedCall?
IdentityIdentity(dense_1/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????2

Identity?

Identity_1Identity*conv2d_10/StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:?????????I2

Identity_1?
NoOpNoOp"^conv2d_10/StatefulPartitionedCall"^conv2d_11/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall^model/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*^
_input_shapesM
K:?????????: : : : : : : : : : : : : : : : : : : : : : : : 2F
!conv2d_10/StatefulPartitionedCall!conv2d_10/StatefulPartitionedCall2F
!conv2d_11/StatefulPartitionedCall!conv2d_11/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall2>
model/StatefulPartitionedCallmodel/StatefulPartitionedCall:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
F__inference_conv2d_11_layer_call_and_return_conditional_losses_3595034

inputs8
conv2d_readvariableop_resource:P-
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:P*
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????*
paddingVALID*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????2	
BiasAdd`
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:?????????2
Reluu
IdentityIdentityRelu:activations:0^NoOp*
T0*/
_output_shapes
:?????????2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????P: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:?????????P
 
_user_specified_nameinputs
?
?
)__inference_model_3_layer_call_fn_3595712

inputs!
unknown:P
	unknown_0:P
	unknown_1:P
	unknown_2:P
	unknown_3:P
	unknown_4:P#
	unknown_5:PP
	unknown_6:P
	unknown_7:P
	unknown_8:P
	unknown_9:P

unknown_10:P$

unknown_11:PP

unknown_12:P

unknown_13:P

unknown_14:P

unknown_15:P

unknown_16:P$

unknown_17:P

unknown_18:$

unknown_19:PI

unknown_20:I

unknown_21:@

unknown_22:
identity

identity_1??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22*$
Tin
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:?????????:?????????I*4
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_model_3_layer_call_and_return_conditional_losses_35953192
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????2

Identity?

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*/
_output_shapes
:?????????I2

Identity_1h
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*^
_input_shapesM
K:?????????: : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?
G
+__inference_dropout_1_layer_call_fn_3596613

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????P* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_dropout_1_layer_call_and_return_conditional_losses_35943482
PartitionedCallt
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:?????????P2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????P:W S
/
_output_shapes
:?????????P
 
_user_specified_nameinputs
?
e
F__inference_dropout_2_layer_call_and_return_conditional_losses_3596815

inputs
identity?c
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
dropout/Const{
dropout/MulMulinputsdropout/Const:output:0*
T0*/
_output_shapes
:?????????P2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape?
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*/
_output_shapes
:?????????P*
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L>2
dropout/GreaterEqual/y?
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*/
_output_shapes
:?????????P2
dropout/GreaterEqual?
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*/
_output_shapes
:?????????P2
dropout/Cast?
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*/
_output_shapes
:?????????P2
dropout/Mul_1m
IdentityIdentitydropout/Mul_1:z:0*
T0*/
_output_shapes
:?????????P2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????P:W S
/
_output_shapes
:?????????P
 
_user_specified_nameinputs
?
?
'__inference_model_layer_call_fn_3596014

inputs!
unknown:P
	unknown_0:P
	unknown_1:P
	unknown_2:P
	unknown_3:P
	unknown_4:P#
	unknown_5:PP
	unknown_6:P
	unknown_7:P
	unknown_8:P
	unknown_9:P

unknown_10:P$

unknown_11:PP

unknown_12:P

unknown_13:P

unknown_14:P

unknown_15:P

unknown_16:P
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????P*.
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_model_layer_call_and_return_conditional_losses_35947932
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:?????????P2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*R
_input_shapesA
?:?????????: : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?'
?
D__inference_model_3_layer_call_and_return_conditional_losses_3595545
input_4'
model_3595489:P
model_3595491:P
model_3595493:P
model_3595495:P
model_3595497:P
model_3595499:P'
model_3595501:PP
model_3595503:P
model_3595505:P
model_3595507:P
model_3595509:P
model_3595511:P'
model_3595513:PP
model_3595515:P
model_3595517:P
model_3595519:P
model_3595521:P
model_3595523:P+
conv2d_11_3595526:P
conv2d_11_3595528:+
conv2d_10_3595533:PI
conv2d_10_3595535:I!
dense_1_3595538:@
dense_1_3595540:
identity

identity_1??!conv2d_10/StatefulPartitionedCall?!conv2d_11/StatefulPartitionedCall?dense_1/StatefulPartitionedCall?!dropout_9/StatefulPartitionedCall?model/StatefulPartitionedCall?
model/StatefulPartitionedCallStatefulPartitionedCallinput_4model_3595489model_3595491model_3595493model_3595495model_3595497model_3595499model_3595501model_3595503model_3595505model_3595507model_3595509model_3595511model_3595513model_3595515model_3595517model_3595519model_3595521model_3595523*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????P*.
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_model_layer_call_and_return_conditional_losses_35947932
model/StatefulPartitionedCall?
!conv2d_11/StatefulPartitionedCallStatefulPartitionedCall&model/StatefulPartitionedCall:output:0conv2d_11_3595526conv2d_11_3595528*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_conv2d_11_layer_call_and_return_conditional_losses_35950342#
!conv2d_11/StatefulPartitionedCall?
!dropout_9/StatefulPartitionedCallStatefulPartitionedCall*conv2d_11/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_dropout_9_layer_call_and_return_conditional_losses_35951902#
!dropout_9/StatefulPartitionedCall?
flatten_1/PartitionedCallPartitionedCall*dropout_9/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_flatten_1_layer_call_and_return_conditional_losses_35950532
flatten_1/PartitionedCall?
!conv2d_10/StatefulPartitionedCallStatefulPartitionedCall&model/StatefulPartitionedCall:output:0conv2d_10_3595533conv2d_10_3595535*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????I*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_conv2d_10_layer_call_and_return_conditional_losses_35950662#
!conv2d_10/StatefulPartitionedCall?
dense_1/StatefulPartitionedCallStatefulPartitionedCall"flatten_1/PartitionedCall:output:0dense_1_3595538dense_1_3595540*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_dense_1_layer_call_and_return_conditional_losses_35950832!
dense_1/StatefulPartitionedCall?
IdentityIdentity(dense_1/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????2

Identity?

Identity_1Identity*conv2d_10/StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:?????????I2

Identity_1?
NoOpNoOp"^conv2d_10/StatefulPartitionedCall"^conv2d_11/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall"^dropout_9/StatefulPartitionedCall^model/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*^
_input_shapesM
K:?????????: : : : : : : : : : : : : : : : : : : : : : : : 2F
!conv2d_10/StatefulPartitionedCall!conv2d_10/StatefulPartitionedCall2F
!conv2d_11/StatefulPartitionedCall!conv2d_11/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall2F
!dropout_9/StatefulPartitionedCall!dropout_9/StatefulPartitionedCall2>
model/StatefulPartitionedCallmodel/StatefulPartitionedCall:X T
/
_output_shapes
:?????????
!
_user_specified_name	input_4
?
?
R__inference_batch_normalization_2_layer_call_and_return_conditional_losses_3594169

inputs%
readvariableop_resource:P'
readvariableop_1_resource:P6
(fusedbatchnormv3_readvariableop_resource:P8
*fusedbatchnormv3_readvariableop_1_resource:P
identity??AssignNewValue?AssignNewValue_1?FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:P*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:P*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:P*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:P*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+???????????????????????????P:P:P:P:P:*
epsilon%o?:*
exponential_avg_factor%
?#<2
FusedBatchNormV3?
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype02
AssignNewValue?
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype02
AssignNewValue_1?
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+???????????????????????????P2

Identity?
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+???????????????????????????P: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+???????????????????????????P
 
_user_specified_nameinputs
?
?
R__inference_batch_normalization_2_layer_call_and_return_conditional_losses_3594383

inputs%
readvariableop_resource:P'
readvariableop_1_resource:P6
(fusedbatchnormv3_readvariableop_resource:P8
*fusedbatchnormv3_readvariableop_1_resource:P
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:P*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:P*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:P*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:P*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????P:P:P:P:P:*
epsilon%o?:*
is_training( 2
FusedBatchNormV3w
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*/
_output_shapes
:?????????P2

Identity?
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:?????????P: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:W S
/
_output_shapes
:?????????P
 
_user_specified_nameinputs
?
?
R__inference_batch_normalization_1_layer_call_and_return_conditional_losses_3594593

inputs%
readvariableop_resource:P'
readvariableop_1_resource:P6
(fusedbatchnormv3_readvariableop_resource:P8
*fusedbatchnormv3_readvariableop_1_resource:P
identity??AssignNewValue?AssignNewValue_1?FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:P*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:P*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:P*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:P*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????P:P:P:P:P:*
epsilon%o?:*
exponential_avg_factor%
?#<2
FusedBatchNormV3?
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype02
AssignNewValue?
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype02
AssignNewValue_1w
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*/
_output_shapes
:?????????P2

Identity?
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:?????????P: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:W S
/
_output_shapes
:?????????P
 
_user_specified_nameinputs
?
?
F__inference_conv2d_10_layer_call_and_return_conditional_losses_3596275

inputs8
conv2d_readvariableop_resource:PI-
biasadd_readvariableop_resource:I
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:PI*
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????I*
paddingSAME*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:I*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????I2	
BiasAddi
SoftmaxSoftmaxBiasAdd:output:0*
T0*/
_output_shapes
:?????????I2	
Softmaxt
IdentityIdentitySoftmax:softmax:0^NoOp*
T0*/
_output_shapes
:?????????I2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????P: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:?????????P
 
_user_specified_nameinputs
?
J
.__inference_activation_1_layer_call_fn_3596603

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????P* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *R
fMRK
I__inference_activation_1_layer_call_and_return_conditional_losses_35943412
PartitionedCallt
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:?????????P2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????P:W S
/
_output_shapes
:?????????P
 
_user_specified_nameinputs
?
?
E__inference_conv2d_2_layer_call_and_return_conditional_losses_3596654

inputs8
conv2d_readvariableop_resource:PP-
biasadd_readvariableop_resource:P
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:PP*
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????P*
paddingSAME*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:P*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????P2	
BiasAdds
IdentityIdentityBiasAdd:output:0^NoOp*
T0*/
_output_shapes
:?????????P2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????P: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:?????????P
 
_user_specified_nameinputs
?
?
P__inference_batch_normalization_layer_call_and_return_conditional_losses_3594269

inputs%
readvariableop_resource:P'
readvariableop_1_resource:P6
(fusedbatchnormv3_readvariableop_resource:P8
*fusedbatchnormv3_readvariableop_1_resource:P
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:P*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:P*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:P*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:P*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????P:P:P:P:P:*
epsilon%o?:*
is_training( 2
FusedBatchNormV3w
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*/
_output_shapes
:?????????P2

Identity?
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:?????????P: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:W S
/
_output_shapes
:?????????P
 
_user_specified_nameinputs
?
e
F__inference_dropout_1_layer_call_and_return_conditional_losses_3594551

inputs
identity?c
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
dropout/Const{
dropout/MulMulinputsdropout/Const:output:0*
T0*/
_output_shapes
:?????????P2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape?
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*/
_output_shapes
:?????????P*
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L>2
dropout/GreaterEqual/y?
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*/
_output_shapes
:?????????P2
dropout/GreaterEqual?
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*/
_output_shapes
:?????????P2
dropout/Cast?
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*/
_output_shapes
:?????????P2
dropout/Mul_1m
IdentityIdentitydropout/Mul_1:z:0*
T0*/
_output_shapes
:?????????P2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????P:W S
/
_output_shapes
:?????????P
 
_user_specified_nameinputs
?
?
R__inference_batch_normalization_1_layer_call_and_return_conditional_losses_3596580

inputs%
readvariableop_resource:P'
readvariableop_1_resource:P6
(fusedbatchnormv3_readvariableop_resource:P8
*fusedbatchnormv3_readvariableop_1_resource:P
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:P*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:P*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:P*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:P*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????P:P:P:P:P:*
epsilon%o?:*
is_training( 2
FusedBatchNormV3w
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*/
_output_shapes
:?????????P2

Identity?
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:?????????P: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:W S
/
_output_shapes
:?????????P
 
_user_specified_nameinputs
?
H
,__inference_activation_layer_call_fn_3596423

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????P* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *P
fKRI
G__inference_activation_layer_call_and_return_conditional_losses_35942842
PartitionedCallt
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:?????????P2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????P:W S
/
_output_shapes
:?????????P
 
_user_specified_nameinputs
?
b
F__inference_flatten_1_layer_call_and_return_conditional_losses_3596235

inputs
identity_
ConstConst*
_output_shapes
:*
dtype0*
valueB"????@   2
Constg
ReshapeReshapeinputsConst:output:0*
T0*'
_output_shapes
:?????????@2	
Reshaped
IdentityIdentityReshape:output:0*
T0*'
_output_shapes
:?????????@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?B
?	
B__inference_model_layer_call_and_return_conditional_losses_3594793

inputs(
conv2d_3594743:P
conv2d_3594745:P)
batch_normalization_3594748:P)
batch_normalization_3594750:P)
batch_normalization_3594752:P)
batch_normalization_3594754:P*
conv2d_1_3594759:PP
conv2d_1_3594761:P+
batch_normalization_1_3594764:P+
batch_normalization_1_3594766:P+
batch_normalization_1_3594768:P+
batch_normalization_1_3594770:P*
conv2d_2_3594775:PP
conv2d_2_3594777:P+
batch_normalization_2_3594780:P+
batch_normalization_2_3594782:P+
batch_normalization_2_3594784:P+
batch_normalization_2_3594786:P
identity??+batch_normalization/StatefulPartitionedCall?-batch_normalization_1/StatefulPartitionedCall?-batch_normalization_2/StatefulPartitionedCall?conv2d/StatefulPartitionedCall? conv2d_1/StatefulPartitionedCall? conv2d_2/StatefulPartitionedCall?dropout/StatefulPartitionedCall?!dropout_1/StatefulPartitionedCall?!dropout_2/StatefulPartitionedCall?
conv2d/StatefulPartitionedCallStatefulPartitionedCallinputsconv2d_3594743conv2d_3594745*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????P*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_conv2d_layer_call_and_return_conditional_losses_35942462 
conv2d/StatefulPartitionedCall?
+batch_normalization/StatefulPartitionedCallStatefulPartitionedCall'conv2d/StatefulPartitionedCall:output:0batch_normalization_3594748batch_normalization_3594750batch_normalization_3594752batch_normalization_3594754*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????P*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *Y
fTRR
P__inference_batch_normalization_layer_call_and_return_conditional_losses_35946762-
+batch_normalization/StatefulPartitionedCall?
activation/PartitionedCallPartitionedCall4batch_normalization/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????P* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *P
fKRI
G__inference_activation_layer_call_and_return_conditional_losses_35942842
activation/PartitionedCall?
dropout/StatefulPartitionedCallStatefulPartitionedCall#activation/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????P* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_dropout_layer_call_and_return_conditional_losses_35946342!
dropout/StatefulPartitionedCall?
 conv2d_1/StatefulPartitionedCallStatefulPartitionedCall(dropout/StatefulPartitionedCall:output:0conv2d_1_3594759conv2d_1_3594761*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????P*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_conv2d_1_layer_call_and_return_conditional_losses_35943032"
 conv2d_1/StatefulPartitionedCall?
-batch_normalization_1/StatefulPartitionedCallStatefulPartitionedCall)conv2d_1/StatefulPartitionedCall:output:0batch_normalization_1_3594764batch_normalization_1_3594766batch_normalization_1_3594768batch_normalization_1_3594770*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????P*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *[
fVRT
R__inference_batch_normalization_1_layer_call_and_return_conditional_losses_35945932/
-batch_normalization_1/StatefulPartitionedCall?
activation_1/PartitionedCallPartitionedCall6batch_normalization_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????P* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *R
fMRK
I__inference_activation_1_layer_call_and_return_conditional_losses_35943412
activation_1/PartitionedCall?
!dropout_1/StatefulPartitionedCallStatefulPartitionedCall%activation_1/PartitionedCall:output:0 ^dropout/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????P* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_dropout_1_layer_call_and_return_conditional_losses_35945512#
!dropout_1/StatefulPartitionedCall?
 conv2d_2/StatefulPartitionedCallStatefulPartitionedCall*dropout_1/StatefulPartitionedCall:output:0conv2d_2_3594775conv2d_2_3594777*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????P*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_conv2d_2_layer_call_and_return_conditional_losses_35943602"
 conv2d_2/StatefulPartitionedCall?
-batch_normalization_2/StatefulPartitionedCallStatefulPartitionedCall)conv2d_2/StatefulPartitionedCall:output:0batch_normalization_2_3594780batch_normalization_2_3594782batch_normalization_2_3594784batch_normalization_2_3594786*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????P*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *[
fVRT
R__inference_batch_normalization_2_layer_call_and_return_conditional_losses_35945102/
-batch_normalization_2/StatefulPartitionedCall?
tf.__operators__.add/AddV2AddV26batch_normalization_2/StatefulPartitionedCall:output:0(dropout/StatefulPartitionedCall:output:0*
T0*/
_output_shapes
:?????????P2
tf.__operators__.add/AddV2?
activation_2/PartitionedCallPartitionedCalltf.__operators__.add/AddV2:z:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????P* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *R
fMRK
I__inference_activation_2_layer_call_and_return_conditional_losses_35943992
activation_2/PartitionedCall?
!dropout_2/StatefulPartitionedCallStatefulPartitionedCall%activation_2/PartitionedCall:output:0"^dropout_1/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????P* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_dropout_2_layer_call_and_return_conditional_losses_35944682#
!dropout_2/StatefulPartitionedCall?
IdentityIdentity*dropout_2/StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:?????????P2

Identity?
NoOpNoOp,^batch_normalization/StatefulPartitionedCall.^batch_normalization_1/StatefulPartitionedCall.^batch_normalization_2/StatefulPartitionedCall^conv2d/StatefulPartitionedCall!^conv2d_1/StatefulPartitionedCall!^conv2d_2/StatefulPartitionedCall ^dropout/StatefulPartitionedCall"^dropout_1/StatefulPartitionedCall"^dropout_2/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*R
_input_shapesA
?:?????????: : : : : : : : : : : : : : : : : : 2Z
+batch_normalization/StatefulPartitionedCall+batch_normalization/StatefulPartitionedCall2^
-batch_normalization_1/StatefulPartitionedCall-batch_normalization_1/StatefulPartitionedCall2^
-batch_normalization_2/StatefulPartitionedCall-batch_normalization_2/StatefulPartitionedCall2@
conv2d/StatefulPartitionedCallconv2d/StatefulPartitionedCall2D
 conv2d_1/StatefulPartitionedCall conv2d_1/StatefulPartitionedCall2D
 conv2d_2/StatefulPartitionedCall conv2d_2/StatefulPartitionedCall2B
dropout/StatefulPartitionedCalldropout/StatefulPartitionedCall2F
!dropout_1/StatefulPartitionedCall!dropout_1/StatefulPartitionedCall2F
!dropout_2/StatefulPartitionedCall!dropout_2/StatefulPartitionedCall:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
??
?
D__inference_model_3_layer_call_and_return_conditional_losses_3595932

inputsE
+model_conv2d_conv2d_readvariableop_resource:P:
,model_conv2d_biasadd_readvariableop_resource:P?
1model_batch_normalization_readvariableop_resource:PA
3model_batch_normalization_readvariableop_1_resource:PP
Bmodel_batch_normalization_fusedbatchnormv3_readvariableop_resource:PR
Dmodel_batch_normalization_fusedbatchnormv3_readvariableop_1_resource:PG
-model_conv2d_1_conv2d_readvariableop_resource:PP<
.model_conv2d_1_biasadd_readvariableop_resource:PA
3model_batch_normalization_1_readvariableop_resource:PC
5model_batch_normalization_1_readvariableop_1_resource:PR
Dmodel_batch_normalization_1_fusedbatchnormv3_readvariableop_resource:PT
Fmodel_batch_normalization_1_fusedbatchnormv3_readvariableop_1_resource:PG
-model_conv2d_2_conv2d_readvariableop_resource:PP<
.model_conv2d_2_biasadd_readvariableop_resource:PA
3model_batch_normalization_2_readvariableop_resource:PC
5model_batch_normalization_2_readvariableop_1_resource:PR
Dmodel_batch_normalization_2_fusedbatchnormv3_readvariableop_resource:PT
Fmodel_batch_normalization_2_fusedbatchnormv3_readvariableop_1_resource:PB
(conv2d_11_conv2d_readvariableop_resource:P7
)conv2d_11_biasadd_readvariableop_resource:B
(conv2d_10_conv2d_readvariableop_resource:PI7
)conv2d_10_biasadd_readvariableop_resource:I8
&dense_1_matmul_readvariableop_resource:@5
'dense_1_biasadd_readvariableop_resource:
identity

identity_1?? conv2d_10/BiasAdd/ReadVariableOp?conv2d_10/Conv2D/ReadVariableOp? conv2d_11/BiasAdd/ReadVariableOp?conv2d_11/Conv2D/ReadVariableOp?dense_1/BiasAdd/ReadVariableOp?dense_1/MatMul/ReadVariableOp?(model/batch_normalization/AssignNewValue?*model/batch_normalization/AssignNewValue_1?9model/batch_normalization/FusedBatchNormV3/ReadVariableOp?;model/batch_normalization/FusedBatchNormV3/ReadVariableOp_1?(model/batch_normalization/ReadVariableOp?*model/batch_normalization/ReadVariableOp_1?*model/batch_normalization_1/AssignNewValue?,model/batch_normalization_1/AssignNewValue_1?;model/batch_normalization_1/FusedBatchNormV3/ReadVariableOp?=model/batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1?*model/batch_normalization_1/ReadVariableOp?,model/batch_normalization_1/ReadVariableOp_1?*model/batch_normalization_2/AssignNewValue?,model/batch_normalization_2/AssignNewValue_1?;model/batch_normalization_2/FusedBatchNormV3/ReadVariableOp?=model/batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1?*model/batch_normalization_2/ReadVariableOp?,model/batch_normalization_2/ReadVariableOp_1?#model/conv2d/BiasAdd/ReadVariableOp?"model/conv2d/Conv2D/ReadVariableOp?%model/conv2d_1/BiasAdd/ReadVariableOp?$model/conv2d_1/Conv2D/ReadVariableOp?%model/conv2d_2/BiasAdd/ReadVariableOp?$model/conv2d_2/Conv2D/ReadVariableOp?
"model/conv2d/Conv2D/ReadVariableOpReadVariableOp+model_conv2d_conv2d_readvariableop_resource*&
_output_shapes
:P*
dtype02$
"model/conv2d/Conv2D/ReadVariableOp?
model/conv2d/Conv2DConv2Dinputs*model/conv2d/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????P*
paddingSAME*
strides
2
model/conv2d/Conv2D?
#model/conv2d/BiasAdd/ReadVariableOpReadVariableOp,model_conv2d_biasadd_readvariableop_resource*
_output_shapes
:P*
dtype02%
#model/conv2d/BiasAdd/ReadVariableOp?
model/conv2d/BiasAddBiasAddmodel/conv2d/Conv2D:output:0+model/conv2d/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????P2
model/conv2d/BiasAdd?
(model/batch_normalization/ReadVariableOpReadVariableOp1model_batch_normalization_readvariableop_resource*
_output_shapes
:P*
dtype02*
(model/batch_normalization/ReadVariableOp?
*model/batch_normalization/ReadVariableOp_1ReadVariableOp3model_batch_normalization_readvariableop_1_resource*
_output_shapes
:P*
dtype02,
*model/batch_normalization/ReadVariableOp_1?
9model/batch_normalization/FusedBatchNormV3/ReadVariableOpReadVariableOpBmodel_batch_normalization_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:P*
dtype02;
9model/batch_normalization/FusedBatchNormV3/ReadVariableOp?
;model/batch_normalization/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpDmodel_batch_normalization_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:P*
dtype02=
;model/batch_normalization/FusedBatchNormV3/ReadVariableOp_1?
*model/batch_normalization/FusedBatchNormV3FusedBatchNormV3model/conv2d/BiasAdd:output:00model/batch_normalization/ReadVariableOp:value:02model/batch_normalization/ReadVariableOp_1:value:0Amodel/batch_normalization/FusedBatchNormV3/ReadVariableOp:value:0Cmodel/batch_normalization/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????P:P:P:P:P:*
epsilon%o?:*
exponential_avg_factor%
?#<2,
*model/batch_normalization/FusedBatchNormV3?
(model/batch_normalization/AssignNewValueAssignVariableOpBmodel_batch_normalization_fusedbatchnormv3_readvariableop_resource7model/batch_normalization/FusedBatchNormV3:batch_mean:0:^model/batch_normalization/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype02*
(model/batch_normalization/AssignNewValue?
*model/batch_normalization/AssignNewValue_1AssignVariableOpDmodel_batch_normalization_fusedbatchnormv3_readvariableop_1_resource;model/batch_normalization/FusedBatchNormV3:batch_variance:0<^model/batch_normalization/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype02,
*model/batch_normalization/AssignNewValue_1?
model/activation/ReluRelu.model/batch_normalization/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:?????????P2
model/activation/Relu
model/dropout/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
model/dropout/dropout/Const?
model/dropout/dropout/MulMul#model/activation/Relu:activations:0$model/dropout/dropout/Const:output:0*
T0*/
_output_shapes
:?????????P2
model/dropout/dropout/Mul?
model/dropout/dropout/ShapeShape#model/activation/Relu:activations:0*
T0*
_output_shapes
:2
model/dropout/dropout/Shape?
2model/dropout/dropout/random_uniform/RandomUniformRandomUniform$model/dropout/dropout/Shape:output:0*
T0*/
_output_shapes
:?????????P*
dtype024
2model/dropout/dropout/random_uniform/RandomUniform?
$model/dropout/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L>2&
$model/dropout/dropout/GreaterEqual/y?
"model/dropout/dropout/GreaterEqualGreaterEqual;model/dropout/dropout/random_uniform/RandomUniform:output:0-model/dropout/dropout/GreaterEqual/y:output:0*
T0*/
_output_shapes
:?????????P2$
"model/dropout/dropout/GreaterEqual?
model/dropout/dropout/CastCast&model/dropout/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*/
_output_shapes
:?????????P2
model/dropout/dropout/Cast?
model/dropout/dropout/Mul_1Mulmodel/dropout/dropout/Mul:z:0model/dropout/dropout/Cast:y:0*
T0*/
_output_shapes
:?????????P2
model/dropout/dropout/Mul_1?
$model/conv2d_1/Conv2D/ReadVariableOpReadVariableOp-model_conv2d_1_conv2d_readvariableop_resource*&
_output_shapes
:PP*
dtype02&
$model/conv2d_1/Conv2D/ReadVariableOp?
model/conv2d_1/Conv2DConv2Dmodel/dropout/dropout/Mul_1:z:0,model/conv2d_1/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????P*
paddingSAME*
strides
2
model/conv2d_1/Conv2D?
%model/conv2d_1/BiasAdd/ReadVariableOpReadVariableOp.model_conv2d_1_biasadd_readvariableop_resource*
_output_shapes
:P*
dtype02'
%model/conv2d_1/BiasAdd/ReadVariableOp?
model/conv2d_1/BiasAddBiasAddmodel/conv2d_1/Conv2D:output:0-model/conv2d_1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????P2
model/conv2d_1/BiasAdd?
*model/batch_normalization_1/ReadVariableOpReadVariableOp3model_batch_normalization_1_readvariableop_resource*
_output_shapes
:P*
dtype02,
*model/batch_normalization_1/ReadVariableOp?
,model/batch_normalization_1/ReadVariableOp_1ReadVariableOp5model_batch_normalization_1_readvariableop_1_resource*
_output_shapes
:P*
dtype02.
,model/batch_normalization_1/ReadVariableOp_1?
;model/batch_normalization_1/FusedBatchNormV3/ReadVariableOpReadVariableOpDmodel_batch_normalization_1_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:P*
dtype02=
;model/batch_normalization_1/FusedBatchNormV3/ReadVariableOp?
=model/batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpFmodel_batch_normalization_1_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:P*
dtype02?
=model/batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1?
,model/batch_normalization_1/FusedBatchNormV3FusedBatchNormV3model/conv2d_1/BiasAdd:output:02model/batch_normalization_1/ReadVariableOp:value:04model/batch_normalization_1/ReadVariableOp_1:value:0Cmodel/batch_normalization_1/FusedBatchNormV3/ReadVariableOp:value:0Emodel/batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????P:P:P:P:P:*
epsilon%o?:*
exponential_avg_factor%
?#<2.
,model/batch_normalization_1/FusedBatchNormV3?
*model/batch_normalization_1/AssignNewValueAssignVariableOpDmodel_batch_normalization_1_fusedbatchnormv3_readvariableop_resource9model/batch_normalization_1/FusedBatchNormV3:batch_mean:0<^model/batch_normalization_1/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype02,
*model/batch_normalization_1/AssignNewValue?
,model/batch_normalization_1/AssignNewValue_1AssignVariableOpFmodel_batch_normalization_1_fusedbatchnormv3_readvariableop_1_resource=model/batch_normalization_1/FusedBatchNormV3:batch_variance:0>^model/batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype02.
,model/batch_normalization_1/AssignNewValue_1?
model/activation_1/ReluRelu0model/batch_normalization_1/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:?????????P2
model/activation_1/Relu?
model/dropout_1/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
model/dropout_1/dropout/Const?
model/dropout_1/dropout/MulMul%model/activation_1/Relu:activations:0&model/dropout_1/dropout/Const:output:0*
T0*/
_output_shapes
:?????????P2
model/dropout_1/dropout/Mul?
model/dropout_1/dropout/ShapeShape%model/activation_1/Relu:activations:0*
T0*
_output_shapes
:2
model/dropout_1/dropout/Shape?
4model/dropout_1/dropout/random_uniform/RandomUniformRandomUniform&model/dropout_1/dropout/Shape:output:0*
T0*/
_output_shapes
:?????????P*
dtype026
4model/dropout_1/dropout/random_uniform/RandomUniform?
&model/dropout_1/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L>2(
&model/dropout_1/dropout/GreaterEqual/y?
$model/dropout_1/dropout/GreaterEqualGreaterEqual=model/dropout_1/dropout/random_uniform/RandomUniform:output:0/model/dropout_1/dropout/GreaterEqual/y:output:0*
T0*/
_output_shapes
:?????????P2&
$model/dropout_1/dropout/GreaterEqual?
model/dropout_1/dropout/CastCast(model/dropout_1/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*/
_output_shapes
:?????????P2
model/dropout_1/dropout/Cast?
model/dropout_1/dropout/Mul_1Mulmodel/dropout_1/dropout/Mul:z:0 model/dropout_1/dropout/Cast:y:0*
T0*/
_output_shapes
:?????????P2
model/dropout_1/dropout/Mul_1?
$model/conv2d_2/Conv2D/ReadVariableOpReadVariableOp-model_conv2d_2_conv2d_readvariableop_resource*&
_output_shapes
:PP*
dtype02&
$model/conv2d_2/Conv2D/ReadVariableOp?
model/conv2d_2/Conv2DConv2D!model/dropout_1/dropout/Mul_1:z:0,model/conv2d_2/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????P*
paddingSAME*
strides
2
model/conv2d_2/Conv2D?
%model/conv2d_2/BiasAdd/ReadVariableOpReadVariableOp.model_conv2d_2_biasadd_readvariableop_resource*
_output_shapes
:P*
dtype02'
%model/conv2d_2/BiasAdd/ReadVariableOp?
model/conv2d_2/BiasAddBiasAddmodel/conv2d_2/Conv2D:output:0-model/conv2d_2/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????P2
model/conv2d_2/BiasAdd?
*model/batch_normalization_2/ReadVariableOpReadVariableOp3model_batch_normalization_2_readvariableop_resource*
_output_shapes
:P*
dtype02,
*model/batch_normalization_2/ReadVariableOp?
,model/batch_normalization_2/ReadVariableOp_1ReadVariableOp5model_batch_normalization_2_readvariableop_1_resource*
_output_shapes
:P*
dtype02.
,model/batch_normalization_2/ReadVariableOp_1?
;model/batch_normalization_2/FusedBatchNormV3/ReadVariableOpReadVariableOpDmodel_batch_normalization_2_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:P*
dtype02=
;model/batch_normalization_2/FusedBatchNormV3/ReadVariableOp?
=model/batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpFmodel_batch_normalization_2_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:P*
dtype02?
=model/batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1?
,model/batch_normalization_2/FusedBatchNormV3FusedBatchNormV3model/conv2d_2/BiasAdd:output:02model/batch_normalization_2/ReadVariableOp:value:04model/batch_normalization_2/ReadVariableOp_1:value:0Cmodel/batch_normalization_2/FusedBatchNormV3/ReadVariableOp:value:0Emodel/batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????P:P:P:P:P:*
epsilon%o?:*
exponential_avg_factor%
?#<2.
,model/batch_normalization_2/FusedBatchNormV3?
*model/batch_normalization_2/AssignNewValueAssignVariableOpDmodel_batch_normalization_2_fusedbatchnormv3_readvariableop_resource9model/batch_normalization_2/FusedBatchNormV3:batch_mean:0<^model/batch_normalization_2/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype02,
*model/batch_normalization_2/AssignNewValue?
,model/batch_normalization_2/AssignNewValue_1AssignVariableOpFmodel_batch_normalization_2_fusedbatchnormv3_readvariableop_1_resource=model/batch_normalization_2/FusedBatchNormV3:batch_variance:0>^model/batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype02.
,model/batch_normalization_2/AssignNewValue_1?
 model/tf.__operators__.add/AddV2AddV20model/batch_normalization_2/FusedBatchNormV3:y:0model/dropout/dropout/Mul_1:z:0*
T0*/
_output_shapes
:?????????P2"
 model/tf.__operators__.add/AddV2?
model/activation_2/ReluRelu$model/tf.__operators__.add/AddV2:z:0*
T0*/
_output_shapes
:?????????P2
model/activation_2/Relu?
model/dropout_2/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
model/dropout_2/dropout/Const?
model/dropout_2/dropout/MulMul%model/activation_2/Relu:activations:0&model/dropout_2/dropout/Const:output:0*
T0*/
_output_shapes
:?????????P2
model/dropout_2/dropout/Mul?
model/dropout_2/dropout/ShapeShape%model/activation_2/Relu:activations:0*
T0*
_output_shapes
:2
model/dropout_2/dropout/Shape?
4model/dropout_2/dropout/random_uniform/RandomUniformRandomUniform&model/dropout_2/dropout/Shape:output:0*
T0*/
_output_shapes
:?????????P*
dtype026
4model/dropout_2/dropout/random_uniform/RandomUniform?
&model/dropout_2/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L>2(
&model/dropout_2/dropout/GreaterEqual/y?
$model/dropout_2/dropout/GreaterEqualGreaterEqual=model/dropout_2/dropout/random_uniform/RandomUniform:output:0/model/dropout_2/dropout/GreaterEqual/y:output:0*
T0*/
_output_shapes
:?????????P2&
$model/dropout_2/dropout/GreaterEqual?
model/dropout_2/dropout/CastCast(model/dropout_2/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*/
_output_shapes
:?????????P2
model/dropout_2/dropout/Cast?
model/dropout_2/dropout/Mul_1Mulmodel/dropout_2/dropout/Mul:z:0 model/dropout_2/dropout/Cast:y:0*
T0*/
_output_shapes
:?????????P2
model/dropout_2/dropout/Mul_1?
conv2d_11/Conv2D/ReadVariableOpReadVariableOp(conv2d_11_conv2d_readvariableop_resource*&
_output_shapes
:P*
dtype02!
conv2d_11/Conv2D/ReadVariableOp?
conv2d_11/Conv2DConv2D!model/dropout_2/dropout/Mul_1:z:0'conv2d_11/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????*
paddingVALID*
strides
2
conv2d_11/Conv2D?
 conv2d_11/BiasAdd/ReadVariableOpReadVariableOp)conv2d_11_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02"
 conv2d_11/BiasAdd/ReadVariableOp?
conv2d_11/BiasAddBiasAddconv2d_11/Conv2D:output:0(conv2d_11/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????2
conv2d_11/BiasAdd~
conv2d_11/ReluReluconv2d_11/BiasAdd:output:0*
T0*/
_output_shapes
:?????????2
conv2d_11/Reluw
dropout_9/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
dropout_9/dropout/Const?
dropout_9/dropout/MulMulconv2d_11/Relu:activations:0 dropout_9/dropout/Const:output:0*
T0*/
_output_shapes
:?????????2
dropout_9/dropout/Mul~
dropout_9/dropout/ShapeShapeconv2d_11/Relu:activations:0*
T0*
_output_shapes
:2
dropout_9/dropout/Shape?
.dropout_9/dropout/random_uniform/RandomUniformRandomUniform dropout_9/dropout/Shape:output:0*
T0*/
_output_shapes
:?????????*
dtype020
.dropout_9/dropout/random_uniform/RandomUniform?
 dropout_9/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L>2"
 dropout_9/dropout/GreaterEqual/y?
dropout_9/dropout/GreaterEqualGreaterEqual7dropout_9/dropout/random_uniform/RandomUniform:output:0)dropout_9/dropout/GreaterEqual/y:output:0*
T0*/
_output_shapes
:?????????2 
dropout_9/dropout/GreaterEqual?
dropout_9/dropout/CastCast"dropout_9/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*/
_output_shapes
:?????????2
dropout_9/dropout/Cast?
dropout_9/dropout/Mul_1Muldropout_9/dropout/Mul:z:0dropout_9/dropout/Cast:y:0*
T0*/
_output_shapes
:?????????2
dropout_9/dropout/Mul_1s
flatten_1/ConstConst*
_output_shapes
:*
dtype0*
valueB"????@   2
flatten_1/Const?
flatten_1/ReshapeReshapedropout_9/dropout/Mul_1:z:0flatten_1/Const:output:0*
T0*'
_output_shapes
:?????????@2
flatten_1/Reshape?
conv2d_10/Conv2D/ReadVariableOpReadVariableOp(conv2d_10_conv2d_readvariableop_resource*&
_output_shapes
:PI*
dtype02!
conv2d_10/Conv2D/ReadVariableOp?
conv2d_10/Conv2DConv2D!model/dropout_2/dropout/Mul_1:z:0'conv2d_10/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????I*
paddingSAME*
strides
2
conv2d_10/Conv2D?
 conv2d_10/BiasAdd/ReadVariableOpReadVariableOp)conv2d_10_biasadd_readvariableop_resource*
_output_shapes
:I*
dtype02"
 conv2d_10/BiasAdd/ReadVariableOp?
conv2d_10/BiasAddBiasAddconv2d_10/Conv2D:output:0(conv2d_10/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????I2
conv2d_10/BiasAdd?
conv2d_10/SoftmaxSoftmaxconv2d_10/BiasAdd:output:0*
T0*/
_output_shapes
:?????????I2
conv2d_10/Softmax?
dense_1/MatMul/ReadVariableOpReadVariableOp&dense_1_matmul_readvariableop_resource*
_output_shapes

:@*
dtype02
dense_1/MatMul/ReadVariableOp?
dense_1/MatMulMatMulflatten_1/Reshape:output:0%dense_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_1/MatMul?
dense_1/BiasAdd/ReadVariableOpReadVariableOp'dense_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02 
dense_1/BiasAdd/ReadVariableOp?
dense_1/BiasAddBiasAdddense_1/MatMul:product:0&dense_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_1/BiasAddp
dense_1/TanhTanhdense_1/BiasAdd:output:0*
T0*'
_output_shapes
:?????????2
dense_1/Tanhk
IdentityIdentitydense_1/Tanh:y:0^NoOp*
T0*'
_output_shapes
:?????????2

Identity?

Identity_1Identityconv2d_10/Softmax:softmax:0^NoOp*
T0*/
_output_shapes
:?????????I2

Identity_1?
NoOpNoOp!^conv2d_10/BiasAdd/ReadVariableOp ^conv2d_10/Conv2D/ReadVariableOp!^conv2d_11/BiasAdd/ReadVariableOp ^conv2d_11/Conv2D/ReadVariableOp^dense_1/BiasAdd/ReadVariableOp^dense_1/MatMul/ReadVariableOp)^model/batch_normalization/AssignNewValue+^model/batch_normalization/AssignNewValue_1:^model/batch_normalization/FusedBatchNormV3/ReadVariableOp<^model/batch_normalization/FusedBatchNormV3/ReadVariableOp_1)^model/batch_normalization/ReadVariableOp+^model/batch_normalization/ReadVariableOp_1+^model/batch_normalization_1/AssignNewValue-^model/batch_normalization_1/AssignNewValue_1<^model/batch_normalization_1/FusedBatchNormV3/ReadVariableOp>^model/batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1+^model/batch_normalization_1/ReadVariableOp-^model/batch_normalization_1/ReadVariableOp_1+^model/batch_normalization_2/AssignNewValue-^model/batch_normalization_2/AssignNewValue_1<^model/batch_normalization_2/FusedBatchNormV3/ReadVariableOp>^model/batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1+^model/batch_normalization_2/ReadVariableOp-^model/batch_normalization_2/ReadVariableOp_1$^model/conv2d/BiasAdd/ReadVariableOp#^model/conv2d/Conv2D/ReadVariableOp&^model/conv2d_1/BiasAdd/ReadVariableOp%^model/conv2d_1/Conv2D/ReadVariableOp&^model/conv2d_2/BiasAdd/ReadVariableOp%^model/conv2d_2/Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*^
_input_shapesM
K:?????????: : : : : : : : : : : : : : : : : : : : : : : : 2D
 conv2d_10/BiasAdd/ReadVariableOp conv2d_10/BiasAdd/ReadVariableOp2B
conv2d_10/Conv2D/ReadVariableOpconv2d_10/Conv2D/ReadVariableOp2D
 conv2d_11/BiasAdd/ReadVariableOp conv2d_11/BiasAdd/ReadVariableOp2B
conv2d_11/Conv2D/ReadVariableOpconv2d_11/Conv2D/ReadVariableOp2@
dense_1/BiasAdd/ReadVariableOpdense_1/BiasAdd/ReadVariableOp2>
dense_1/MatMul/ReadVariableOpdense_1/MatMul/ReadVariableOp2T
(model/batch_normalization/AssignNewValue(model/batch_normalization/AssignNewValue2X
*model/batch_normalization/AssignNewValue_1*model/batch_normalization/AssignNewValue_12v
9model/batch_normalization/FusedBatchNormV3/ReadVariableOp9model/batch_normalization/FusedBatchNormV3/ReadVariableOp2z
;model/batch_normalization/FusedBatchNormV3/ReadVariableOp_1;model/batch_normalization/FusedBatchNormV3/ReadVariableOp_12T
(model/batch_normalization/ReadVariableOp(model/batch_normalization/ReadVariableOp2X
*model/batch_normalization/ReadVariableOp_1*model/batch_normalization/ReadVariableOp_12X
*model/batch_normalization_1/AssignNewValue*model/batch_normalization_1/AssignNewValue2\
,model/batch_normalization_1/AssignNewValue_1,model/batch_normalization_1/AssignNewValue_12z
;model/batch_normalization_1/FusedBatchNormV3/ReadVariableOp;model/batch_normalization_1/FusedBatchNormV3/ReadVariableOp2~
=model/batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1=model/batch_normalization_1/FusedBatchNormV3/ReadVariableOp_12X
*model/batch_normalization_1/ReadVariableOp*model/batch_normalization_1/ReadVariableOp2\
,model/batch_normalization_1/ReadVariableOp_1,model/batch_normalization_1/ReadVariableOp_12X
*model/batch_normalization_2/AssignNewValue*model/batch_normalization_2/AssignNewValue2\
,model/batch_normalization_2/AssignNewValue_1,model/batch_normalization_2/AssignNewValue_12z
;model/batch_normalization_2/FusedBatchNormV3/ReadVariableOp;model/batch_normalization_2/FusedBatchNormV3/ReadVariableOp2~
=model/batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1=model/batch_normalization_2/FusedBatchNormV3/ReadVariableOp_12X
*model/batch_normalization_2/ReadVariableOp*model/batch_normalization_2/ReadVariableOp2\
,model/batch_normalization_2/ReadVariableOp_1,model/batch_normalization_2/ReadVariableOp_12J
#model/conv2d/BiasAdd/ReadVariableOp#model/conv2d/BiasAdd/ReadVariableOp2H
"model/conv2d/Conv2D/ReadVariableOp"model/conv2d/Conv2D/ReadVariableOp2N
%model/conv2d_1/BiasAdd/ReadVariableOp%model/conv2d_1/BiasAdd/ReadVariableOp2L
$model/conv2d_1/Conv2D/ReadVariableOp$model/conv2d_1/Conv2D/ReadVariableOp2N
%model/conv2d_2/BiasAdd/ReadVariableOp%model/conv2d_2/BiasAdd/ReadVariableOp2L
$model/conv2d_2/Conv2D/ReadVariableOp$model/conv2d_2/Conv2D/ReadVariableOp:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?
c
G__inference_activation_layer_call_and_return_conditional_losses_3594284

inputs
identityV
ReluReluinputs*
T0*/
_output_shapes
:?????????P2
Relun
IdentityIdentityRelu:activations:0*
T0*/
_output_shapes
:?????????P2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????P:W S
/
_output_shapes
:?????????P
 
_user_specified_nameinputs
?'
?
D__inference_model_3_layer_call_and_return_conditional_losses_3595319

inputs'
model_3595263:P
model_3595265:P
model_3595267:P
model_3595269:P
model_3595271:P
model_3595273:P'
model_3595275:PP
model_3595277:P
model_3595279:P
model_3595281:P
model_3595283:P
model_3595285:P'
model_3595287:PP
model_3595289:P
model_3595291:P
model_3595293:P
model_3595295:P
model_3595297:P+
conv2d_11_3595300:P
conv2d_11_3595302:+
conv2d_10_3595307:PI
conv2d_10_3595309:I!
dense_1_3595312:@
dense_1_3595314:
identity

identity_1??!conv2d_10/StatefulPartitionedCall?!conv2d_11/StatefulPartitionedCall?dense_1/StatefulPartitionedCall?!dropout_9/StatefulPartitionedCall?model/StatefulPartitionedCall?
model/StatefulPartitionedCallStatefulPartitionedCallinputsmodel_3595263model_3595265model_3595267model_3595269model_3595271model_3595273model_3595275model_3595277model_3595279model_3595281model_3595283model_3595285model_3595287model_3595289model_3595291model_3595293model_3595295model_3595297*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????P*.
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_model_layer_call_and_return_conditional_losses_35947932
model/StatefulPartitionedCall?
!conv2d_11/StatefulPartitionedCallStatefulPartitionedCall&model/StatefulPartitionedCall:output:0conv2d_11_3595300conv2d_11_3595302*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_conv2d_11_layer_call_and_return_conditional_losses_35950342#
!conv2d_11/StatefulPartitionedCall?
!dropout_9/StatefulPartitionedCallStatefulPartitionedCall*conv2d_11/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_dropout_9_layer_call_and_return_conditional_losses_35951902#
!dropout_9/StatefulPartitionedCall?
flatten_1/PartitionedCallPartitionedCall*dropout_9/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_flatten_1_layer_call_and_return_conditional_losses_35950532
flatten_1/PartitionedCall?
!conv2d_10/StatefulPartitionedCallStatefulPartitionedCall&model/StatefulPartitionedCall:output:0conv2d_10_3595307conv2d_10_3595309*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????I*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_conv2d_10_layer_call_and_return_conditional_losses_35950662#
!conv2d_10/StatefulPartitionedCall?
dense_1/StatefulPartitionedCallStatefulPartitionedCall"flatten_1/PartitionedCall:output:0dense_1_3595312dense_1_3595314*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_dense_1_layer_call_and_return_conditional_losses_35950832!
dense_1/StatefulPartitionedCall?
IdentityIdentity(dense_1/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????2

Identity?

Identity_1Identity*conv2d_10/StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:?????????I2

Identity_1?
NoOpNoOp"^conv2d_10/StatefulPartitionedCall"^conv2d_11/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall"^dropout_9/StatefulPartitionedCall^model/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*^
_input_shapesM
K:?????????: : : : : : : : : : : : : : : : : : : : : : : : 2F
!conv2d_10/StatefulPartitionedCall!conv2d_10/StatefulPartitionedCall2F
!conv2d_11/StatefulPartitionedCall!conv2d_11/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall2F
!dropout_9/StatefulPartitionedCall!dropout_9/StatefulPartitionedCall2>
model/StatefulPartitionedCallmodel/StatefulPartitionedCall:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
R__inference_batch_normalization_2_layer_call_and_return_conditional_losses_3594125

inputs%
readvariableop_resource:P'
readvariableop_1_resource:P6
(fusedbatchnormv3_readvariableop_resource:P8
*fusedbatchnormv3_readvariableop_1_resource:P
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:P*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:P*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:P*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:P*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+???????????????????????????P:P:P:P:P:*
epsilon%o?:*
is_training( 2
FusedBatchNormV3?
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+???????????????????????????P2

Identity?
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+???????????????????????????P: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+???????????????????????????P
 
_user_specified_nameinputs
?
?
F__inference_conv2d_11_layer_call_and_return_conditional_losses_3596197

inputs8
conv2d_readvariableop_resource:P-
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:P*
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????*
paddingVALID*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????2	
BiasAdd`
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:?????????2
Reluu
IdentityIdentityRelu:activations:0^NoOp*
T0*/
_output_shapes
:?????????2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????P: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:?????????P
 
_user_specified_nameinputs
?

?
D__inference_dense_1_layer_call_and_return_conditional_losses_3596255

inputs0
matmul_readvariableop_resource:@-
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:@*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2	
BiasAddX
TanhTanhBiasAdd:output:0*
T0*'
_output_shapes
:?????????2
Tanhc
IdentityIdentityTanh:y:0^NoOp*
T0*'
_output_shapes
:?????????2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:?????????@
 
_user_specified_nameinputs
?	
?
7__inference_batch_normalization_1_layer_call_fn_3596487

inputs
unknown:P
	unknown_0:P
	unknown_1:P
	unknown_2:P
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????P*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *[
fVRT
R__inference_batch_normalization_1_layer_call_and_return_conditional_losses_35939992
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+???????????????????????????P2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+???????????????????????????P: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+???????????????????????????P
 
_user_specified_nameinputs
?
?
5__inference_batch_normalization_layer_call_fn_3596333

inputs
unknown:P
	unknown_0:P
	unknown_1:P
	unknown_2:P
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????P*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *Y
fTRR
P__inference_batch_normalization_layer_call_and_return_conditional_losses_35942692
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:?????????P2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:?????????P: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????P
 
_user_specified_nameinputs
??
?
"__inference__wrapped_model_3593851
input_4M
3model_3_model_conv2d_conv2d_readvariableop_resource:PB
4model_3_model_conv2d_biasadd_readvariableop_resource:PG
9model_3_model_batch_normalization_readvariableop_resource:PI
;model_3_model_batch_normalization_readvariableop_1_resource:PX
Jmodel_3_model_batch_normalization_fusedbatchnormv3_readvariableop_resource:PZ
Lmodel_3_model_batch_normalization_fusedbatchnormv3_readvariableop_1_resource:PO
5model_3_model_conv2d_1_conv2d_readvariableop_resource:PPD
6model_3_model_conv2d_1_biasadd_readvariableop_resource:PI
;model_3_model_batch_normalization_1_readvariableop_resource:PK
=model_3_model_batch_normalization_1_readvariableop_1_resource:PZ
Lmodel_3_model_batch_normalization_1_fusedbatchnormv3_readvariableop_resource:P\
Nmodel_3_model_batch_normalization_1_fusedbatchnormv3_readvariableop_1_resource:PO
5model_3_model_conv2d_2_conv2d_readvariableop_resource:PPD
6model_3_model_conv2d_2_biasadd_readvariableop_resource:PI
;model_3_model_batch_normalization_2_readvariableop_resource:PK
=model_3_model_batch_normalization_2_readvariableop_1_resource:PZ
Lmodel_3_model_batch_normalization_2_fusedbatchnormv3_readvariableop_resource:P\
Nmodel_3_model_batch_normalization_2_fusedbatchnormv3_readvariableop_1_resource:PJ
0model_3_conv2d_11_conv2d_readvariableop_resource:P?
1model_3_conv2d_11_biasadd_readvariableop_resource:J
0model_3_conv2d_10_conv2d_readvariableop_resource:PI?
1model_3_conv2d_10_biasadd_readvariableop_resource:I@
.model_3_dense_1_matmul_readvariableop_resource:@=
/model_3_dense_1_biasadd_readvariableop_resource:
identity

identity_1??(model_3/conv2d_10/BiasAdd/ReadVariableOp?'model_3/conv2d_10/Conv2D/ReadVariableOp?(model_3/conv2d_11/BiasAdd/ReadVariableOp?'model_3/conv2d_11/Conv2D/ReadVariableOp?&model_3/dense_1/BiasAdd/ReadVariableOp?%model_3/dense_1/MatMul/ReadVariableOp?Amodel_3/model/batch_normalization/FusedBatchNormV3/ReadVariableOp?Cmodel_3/model/batch_normalization/FusedBatchNormV3/ReadVariableOp_1?0model_3/model/batch_normalization/ReadVariableOp?2model_3/model/batch_normalization/ReadVariableOp_1?Cmodel_3/model/batch_normalization_1/FusedBatchNormV3/ReadVariableOp?Emodel_3/model/batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1?2model_3/model/batch_normalization_1/ReadVariableOp?4model_3/model/batch_normalization_1/ReadVariableOp_1?Cmodel_3/model/batch_normalization_2/FusedBatchNormV3/ReadVariableOp?Emodel_3/model/batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1?2model_3/model/batch_normalization_2/ReadVariableOp?4model_3/model/batch_normalization_2/ReadVariableOp_1?+model_3/model/conv2d/BiasAdd/ReadVariableOp?*model_3/model/conv2d/Conv2D/ReadVariableOp?-model_3/model/conv2d_1/BiasAdd/ReadVariableOp?,model_3/model/conv2d_1/Conv2D/ReadVariableOp?-model_3/model/conv2d_2/BiasAdd/ReadVariableOp?,model_3/model/conv2d_2/Conv2D/ReadVariableOp?
*model_3/model/conv2d/Conv2D/ReadVariableOpReadVariableOp3model_3_model_conv2d_conv2d_readvariableop_resource*&
_output_shapes
:P*
dtype02,
*model_3/model/conv2d/Conv2D/ReadVariableOp?
model_3/model/conv2d/Conv2DConv2Dinput_42model_3/model/conv2d/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????P*
paddingSAME*
strides
2
model_3/model/conv2d/Conv2D?
+model_3/model/conv2d/BiasAdd/ReadVariableOpReadVariableOp4model_3_model_conv2d_biasadd_readvariableop_resource*
_output_shapes
:P*
dtype02-
+model_3/model/conv2d/BiasAdd/ReadVariableOp?
model_3/model/conv2d/BiasAddBiasAdd$model_3/model/conv2d/Conv2D:output:03model_3/model/conv2d/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????P2
model_3/model/conv2d/BiasAdd?
0model_3/model/batch_normalization/ReadVariableOpReadVariableOp9model_3_model_batch_normalization_readvariableop_resource*
_output_shapes
:P*
dtype022
0model_3/model/batch_normalization/ReadVariableOp?
2model_3/model/batch_normalization/ReadVariableOp_1ReadVariableOp;model_3_model_batch_normalization_readvariableop_1_resource*
_output_shapes
:P*
dtype024
2model_3/model/batch_normalization/ReadVariableOp_1?
Amodel_3/model/batch_normalization/FusedBatchNormV3/ReadVariableOpReadVariableOpJmodel_3_model_batch_normalization_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:P*
dtype02C
Amodel_3/model/batch_normalization/FusedBatchNormV3/ReadVariableOp?
Cmodel_3/model/batch_normalization/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpLmodel_3_model_batch_normalization_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:P*
dtype02E
Cmodel_3/model/batch_normalization/FusedBatchNormV3/ReadVariableOp_1?
2model_3/model/batch_normalization/FusedBatchNormV3FusedBatchNormV3%model_3/model/conv2d/BiasAdd:output:08model_3/model/batch_normalization/ReadVariableOp:value:0:model_3/model/batch_normalization/ReadVariableOp_1:value:0Imodel_3/model/batch_normalization/FusedBatchNormV3/ReadVariableOp:value:0Kmodel_3/model/batch_normalization/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????P:P:P:P:P:*
epsilon%o?:*
is_training( 24
2model_3/model/batch_normalization/FusedBatchNormV3?
model_3/model/activation/ReluRelu6model_3/model/batch_normalization/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:?????????P2
model_3/model/activation/Relu?
model_3/model/dropout/IdentityIdentity+model_3/model/activation/Relu:activations:0*
T0*/
_output_shapes
:?????????P2 
model_3/model/dropout/Identity?
,model_3/model/conv2d_1/Conv2D/ReadVariableOpReadVariableOp5model_3_model_conv2d_1_conv2d_readvariableop_resource*&
_output_shapes
:PP*
dtype02.
,model_3/model/conv2d_1/Conv2D/ReadVariableOp?
model_3/model/conv2d_1/Conv2DConv2D'model_3/model/dropout/Identity:output:04model_3/model/conv2d_1/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????P*
paddingSAME*
strides
2
model_3/model/conv2d_1/Conv2D?
-model_3/model/conv2d_1/BiasAdd/ReadVariableOpReadVariableOp6model_3_model_conv2d_1_biasadd_readvariableop_resource*
_output_shapes
:P*
dtype02/
-model_3/model/conv2d_1/BiasAdd/ReadVariableOp?
model_3/model/conv2d_1/BiasAddBiasAdd&model_3/model/conv2d_1/Conv2D:output:05model_3/model/conv2d_1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????P2 
model_3/model/conv2d_1/BiasAdd?
2model_3/model/batch_normalization_1/ReadVariableOpReadVariableOp;model_3_model_batch_normalization_1_readvariableop_resource*
_output_shapes
:P*
dtype024
2model_3/model/batch_normalization_1/ReadVariableOp?
4model_3/model/batch_normalization_1/ReadVariableOp_1ReadVariableOp=model_3_model_batch_normalization_1_readvariableop_1_resource*
_output_shapes
:P*
dtype026
4model_3/model/batch_normalization_1/ReadVariableOp_1?
Cmodel_3/model/batch_normalization_1/FusedBatchNormV3/ReadVariableOpReadVariableOpLmodel_3_model_batch_normalization_1_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:P*
dtype02E
Cmodel_3/model/batch_normalization_1/FusedBatchNormV3/ReadVariableOp?
Emodel_3/model/batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpNmodel_3_model_batch_normalization_1_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:P*
dtype02G
Emodel_3/model/batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1?
4model_3/model/batch_normalization_1/FusedBatchNormV3FusedBatchNormV3'model_3/model/conv2d_1/BiasAdd:output:0:model_3/model/batch_normalization_1/ReadVariableOp:value:0<model_3/model/batch_normalization_1/ReadVariableOp_1:value:0Kmodel_3/model/batch_normalization_1/FusedBatchNormV3/ReadVariableOp:value:0Mmodel_3/model/batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????P:P:P:P:P:*
epsilon%o?:*
is_training( 26
4model_3/model/batch_normalization_1/FusedBatchNormV3?
model_3/model/activation_1/ReluRelu8model_3/model/batch_normalization_1/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:?????????P2!
model_3/model/activation_1/Relu?
 model_3/model/dropout_1/IdentityIdentity-model_3/model/activation_1/Relu:activations:0*
T0*/
_output_shapes
:?????????P2"
 model_3/model/dropout_1/Identity?
,model_3/model/conv2d_2/Conv2D/ReadVariableOpReadVariableOp5model_3_model_conv2d_2_conv2d_readvariableop_resource*&
_output_shapes
:PP*
dtype02.
,model_3/model/conv2d_2/Conv2D/ReadVariableOp?
model_3/model/conv2d_2/Conv2DConv2D)model_3/model/dropout_1/Identity:output:04model_3/model/conv2d_2/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????P*
paddingSAME*
strides
2
model_3/model/conv2d_2/Conv2D?
-model_3/model/conv2d_2/BiasAdd/ReadVariableOpReadVariableOp6model_3_model_conv2d_2_biasadd_readvariableop_resource*
_output_shapes
:P*
dtype02/
-model_3/model/conv2d_2/BiasAdd/ReadVariableOp?
model_3/model/conv2d_2/BiasAddBiasAdd&model_3/model/conv2d_2/Conv2D:output:05model_3/model/conv2d_2/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????P2 
model_3/model/conv2d_2/BiasAdd?
2model_3/model/batch_normalization_2/ReadVariableOpReadVariableOp;model_3_model_batch_normalization_2_readvariableop_resource*
_output_shapes
:P*
dtype024
2model_3/model/batch_normalization_2/ReadVariableOp?
4model_3/model/batch_normalization_2/ReadVariableOp_1ReadVariableOp=model_3_model_batch_normalization_2_readvariableop_1_resource*
_output_shapes
:P*
dtype026
4model_3/model/batch_normalization_2/ReadVariableOp_1?
Cmodel_3/model/batch_normalization_2/FusedBatchNormV3/ReadVariableOpReadVariableOpLmodel_3_model_batch_normalization_2_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:P*
dtype02E
Cmodel_3/model/batch_normalization_2/FusedBatchNormV3/ReadVariableOp?
Emodel_3/model/batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpNmodel_3_model_batch_normalization_2_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:P*
dtype02G
Emodel_3/model/batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1?
4model_3/model/batch_normalization_2/FusedBatchNormV3FusedBatchNormV3'model_3/model/conv2d_2/BiasAdd:output:0:model_3/model/batch_normalization_2/ReadVariableOp:value:0<model_3/model/batch_normalization_2/ReadVariableOp_1:value:0Kmodel_3/model/batch_normalization_2/FusedBatchNormV3/ReadVariableOp:value:0Mmodel_3/model/batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????P:P:P:P:P:*
epsilon%o?:*
is_training( 26
4model_3/model/batch_normalization_2/FusedBatchNormV3?
(model_3/model/tf.__operators__.add/AddV2AddV28model_3/model/batch_normalization_2/FusedBatchNormV3:y:0'model_3/model/dropout/Identity:output:0*
T0*/
_output_shapes
:?????????P2*
(model_3/model/tf.__operators__.add/AddV2?
model_3/model/activation_2/ReluRelu,model_3/model/tf.__operators__.add/AddV2:z:0*
T0*/
_output_shapes
:?????????P2!
model_3/model/activation_2/Relu?
 model_3/model/dropout_2/IdentityIdentity-model_3/model/activation_2/Relu:activations:0*
T0*/
_output_shapes
:?????????P2"
 model_3/model/dropout_2/Identity?
'model_3/conv2d_11/Conv2D/ReadVariableOpReadVariableOp0model_3_conv2d_11_conv2d_readvariableop_resource*&
_output_shapes
:P*
dtype02)
'model_3/conv2d_11/Conv2D/ReadVariableOp?
model_3/conv2d_11/Conv2DConv2D)model_3/model/dropout_2/Identity:output:0/model_3/conv2d_11/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????*
paddingVALID*
strides
2
model_3/conv2d_11/Conv2D?
(model_3/conv2d_11/BiasAdd/ReadVariableOpReadVariableOp1model_3_conv2d_11_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02*
(model_3/conv2d_11/BiasAdd/ReadVariableOp?
model_3/conv2d_11/BiasAddBiasAdd!model_3/conv2d_11/Conv2D:output:00model_3/conv2d_11/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????2
model_3/conv2d_11/BiasAdd?
model_3/conv2d_11/ReluRelu"model_3/conv2d_11/BiasAdd:output:0*
T0*/
_output_shapes
:?????????2
model_3/conv2d_11/Relu?
model_3/dropout_9/IdentityIdentity$model_3/conv2d_11/Relu:activations:0*
T0*/
_output_shapes
:?????????2
model_3/dropout_9/Identity?
model_3/flatten_1/ConstConst*
_output_shapes
:*
dtype0*
valueB"????@   2
model_3/flatten_1/Const?
model_3/flatten_1/ReshapeReshape#model_3/dropout_9/Identity:output:0 model_3/flatten_1/Const:output:0*
T0*'
_output_shapes
:?????????@2
model_3/flatten_1/Reshape?
'model_3/conv2d_10/Conv2D/ReadVariableOpReadVariableOp0model_3_conv2d_10_conv2d_readvariableop_resource*&
_output_shapes
:PI*
dtype02)
'model_3/conv2d_10/Conv2D/ReadVariableOp?
model_3/conv2d_10/Conv2DConv2D)model_3/model/dropout_2/Identity:output:0/model_3/conv2d_10/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????I*
paddingSAME*
strides
2
model_3/conv2d_10/Conv2D?
(model_3/conv2d_10/BiasAdd/ReadVariableOpReadVariableOp1model_3_conv2d_10_biasadd_readvariableop_resource*
_output_shapes
:I*
dtype02*
(model_3/conv2d_10/BiasAdd/ReadVariableOp?
model_3/conv2d_10/BiasAddBiasAdd!model_3/conv2d_10/Conv2D:output:00model_3/conv2d_10/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????I2
model_3/conv2d_10/BiasAdd?
model_3/conv2d_10/SoftmaxSoftmax"model_3/conv2d_10/BiasAdd:output:0*
T0*/
_output_shapes
:?????????I2
model_3/conv2d_10/Softmax?
%model_3/dense_1/MatMul/ReadVariableOpReadVariableOp.model_3_dense_1_matmul_readvariableop_resource*
_output_shapes

:@*
dtype02'
%model_3/dense_1/MatMul/ReadVariableOp?
model_3/dense_1/MatMulMatMul"model_3/flatten_1/Reshape:output:0-model_3/dense_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
model_3/dense_1/MatMul?
&model_3/dense_1/BiasAdd/ReadVariableOpReadVariableOp/model_3_dense_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02(
&model_3/dense_1/BiasAdd/ReadVariableOp?
model_3/dense_1/BiasAddBiasAdd model_3/dense_1/MatMul:product:0.model_3/dense_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
model_3/dense_1/BiasAdd?
model_3/dense_1/TanhTanh model_3/dense_1/BiasAdd:output:0*
T0*'
_output_shapes
:?????????2
model_3/dense_1/Tanh?
IdentityIdentity#model_3/conv2d_10/Softmax:softmax:0^NoOp*
T0*/
_output_shapes
:?????????I2

Identityw

Identity_1Identitymodel_3/dense_1/Tanh:y:0^NoOp*
T0*'
_output_shapes
:?????????2

Identity_1?

NoOpNoOp)^model_3/conv2d_10/BiasAdd/ReadVariableOp(^model_3/conv2d_10/Conv2D/ReadVariableOp)^model_3/conv2d_11/BiasAdd/ReadVariableOp(^model_3/conv2d_11/Conv2D/ReadVariableOp'^model_3/dense_1/BiasAdd/ReadVariableOp&^model_3/dense_1/MatMul/ReadVariableOpB^model_3/model/batch_normalization/FusedBatchNormV3/ReadVariableOpD^model_3/model/batch_normalization/FusedBatchNormV3/ReadVariableOp_11^model_3/model/batch_normalization/ReadVariableOp3^model_3/model/batch_normalization/ReadVariableOp_1D^model_3/model/batch_normalization_1/FusedBatchNormV3/ReadVariableOpF^model_3/model/batch_normalization_1/FusedBatchNormV3/ReadVariableOp_13^model_3/model/batch_normalization_1/ReadVariableOp5^model_3/model/batch_normalization_1/ReadVariableOp_1D^model_3/model/batch_normalization_2/FusedBatchNormV3/ReadVariableOpF^model_3/model/batch_normalization_2/FusedBatchNormV3/ReadVariableOp_13^model_3/model/batch_normalization_2/ReadVariableOp5^model_3/model/batch_normalization_2/ReadVariableOp_1,^model_3/model/conv2d/BiasAdd/ReadVariableOp+^model_3/model/conv2d/Conv2D/ReadVariableOp.^model_3/model/conv2d_1/BiasAdd/ReadVariableOp-^model_3/model/conv2d_1/Conv2D/ReadVariableOp.^model_3/model/conv2d_2/BiasAdd/ReadVariableOp-^model_3/model/conv2d_2/Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*^
_input_shapesM
K:?????????: : : : : : : : : : : : : : : : : : : : : : : : 2T
(model_3/conv2d_10/BiasAdd/ReadVariableOp(model_3/conv2d_10/BiasAdd/ReadVariableOp2R
'model_3/conv2d_10/Conv2D/ReadVariableOp'model_3/conv2d_10/Conv2D/ReadVariableOp2T
(model_3/conv2d_11/BiasAdd/ReadVariableOp(model_3/conv2d_11/BiasAdd/ReadVariableOp2R
'model_3/conv2d_11/Conv2D/ReadVariableOp'model_3/conv2d_11/Conv2D/ReadVariableOp2P
&model_3/dense_1/BiasAdd/ReadVariableOp&model_3/dense_1/BiasAdd/ReadVariableOp2N
%model_3/dense_1/MatMul/ReadVariableOp%model_3/dense_1/MatMul/ReadVariableOp2?
Amodel_3/model/batch_normalization/FusedBatchNormV3/ReadVariableOpAmodel_3/model/batch_normalization/FusedBatchNormV3/ReadVariableOp2?
Cmodel_3/model/batch_normalization/FusedBatchNormV3/ReadVariableOp_1Cmodel_3/model/batch_normalization/FusedBatchNormV3/ReadVariableOp_12d
0model_3/model/batch_normalization/ReadVariableOp0model_3/model/batch_normalization/ReadVariableOp2h
2model_3/model/batch_normalization/ReadVariableOp_12model_3/model/batch_normalization/ReadVariableOp_12?
Cmodel_3/model/batch_normalization_1/FusedBatchNormV3/ReadVariableOpCmodel_3/model/batch_normalization_1/FusedBatchNormV3/ReadVariableOp2?
Emodel_3/model/batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1Emodel_3/model/batch_normalization_1/FusedBatchNormV3/ReadVariableOp_12h
2model_3/model/batch_normalization_1/ReadVariableOp2model_3/model/batch_normalization_1/ReadVariableOp2l
4model_3/model/batch_normalization_1/ReadVariableOp_14model_3/model/batch_normalization_1/ReadVariableOp_12?
Cmodel_3/model/batch_normalization_2/FusedBatchNormV3/ReadVariableOpCmodel_3/model/batch_normalization_2/FusedBatchNormV3/ReadVariableOp2?
Emodel_3/model/batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1Emodel_3/model/batch_normalization_2/FusedBatchNormV3/ReadVariableOp_12h
2model_3/model/batch_normalization_2/ReadVariableOp2model_3/model/batch_normalization_2/ReadVariableOp2l
4model_3/model/batch_normalization_2/ReadVariableOp_14model_3/model/batch_normalization_2/ReadVariableOp_12Z
+model_3/model/conv2d/BiasAdd/ReadVariableOp+model_3/model/conv2d/BiasAdd/ReadVariableOp2X
*model_3/model/conv2d/Conv2D/ReadVariableOp*model_3/model/conv2d/Conv2D/ReadVariableOp2^
-model_3/model/conv2d_1/BiasAdd/ReadVariableOp-model_3/model/conv2d_1/BiasAdd/ReadVariableOp2\
,model_3/model/conv2d_1/Conv2D/ReadVariableOp,model_3/model/conv2d_1/Conv2D/ReadVariableOp2^
-model_3/model/conv2d_2/BiasAdd/ReadVariableOp-model_3/model/conv2d_2/BiasAdd/ReadVariableOp2\
,model_3/model/conv2d_2/Conv2D/ReadVariableOp,model_3/model/conv2d_2/Conv2D/ReadVariableOp:X T
/
_output_shapes
:?????????
!
_user_specified_name	input_4
?
?
R__inference_batch_normalization_1_layer_call_and_return_conditional_losses_3594043

inputs%
readvariableop_resource:P'
readvariableop_1_resource:P6
(fusedbatchnormv3_readvariableop_resource:P8
*fusedbatchnormv3_readvariableop_1_resource:P
identity??AssignNewValue?AssignNewValue_1?FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:P*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:P*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:P*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:P*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+???????????????????????????P:P:P:P:P:*
epsilon%o?:*
exponential_avg_factor%
?#<2
FusedBatchNormV3?
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype02
AssignNewValue?
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype02
AssignNewValue_1?
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+???????????????????????????P2

Identity?
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+???????????????????????????P: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+???????????????????????????P
 
_user_specified_nameinputs
?
d
F__inference_dropout_2_layer_call_and_return_conditional_losses_3594406

inputs

identity_1b
IdentityIdentityinputs*
T0*/
_output_shapes
:?????????P2

Identityq

Identity_1IdentityIdentity:output:0*
T0*/
_output_shapes
:?????????P2

Identity_1"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????P:W S
/
_output_shapes
:?????????P
 
_user_specified_nameinputs
?
d
F__inference_dropout_9_layer_call_and_return_conditional_losses_3596212

inputs

identity_1b
IdentityIdentityinputs*
T0*/
_output_shapes
:?????????2

Identityq

Identity_1IdentityIdentity:output:0*
T0*/
_output_shapes
:?????????2

Identity_1"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?
E
)__inference_dropout_layer_call_fn_3596433

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????P* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_dropout_layer_call_and_return_conditional_losses_35942912
PartitionedCallt
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:?????????P2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????P:W S
/
_output_shapes
:?????????P
 
_user_specified_nameinputs
?
J
.__inference_activation_2_layer_call_fn_3596783

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????P* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *R
fMRK
I__inference_activation_2_layer_call_and_return_conditional_losses_35943992
PartitionedCallt
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:?????????P2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????P:W S
/
_output_shapes
:?????????P
 
_user_specified_nameinputs
?
G
+__inference_dropout_9_layer_call_fn_3596202

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_dropout_9_layer_call_and_return_conditional_losses_35950452
PartitionedCallt
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
7__inference_batch_normalization_2_layer_call_fn_3596706

inputs
unknown:P
	unknown_0:P
	unknown_1:P
	unknown_2:P
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????P*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *[
fVRT
R__inference_batch_normalization_2_layer_call_and_return_conditional_losses_35945102
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:?????????P2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:?????????P: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????P
 
_user_specified_nameinputs
?
?
R__inference_batch_normalization_1_layer_call_and_return_conditional_losses_3594326

inputs%
readvariableop_resource:P'
readvariableop_1_resource:P6
(fusedbatchnormv3_readvariableop_resource:P8
*fusedbatchnormv3_readvariableop_1_resource:P
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:P*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:P*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:P*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:P*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????P:P:P:P:P:*
epsilon%o?:*
is_training( 2
FusedBatchNormV3w
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*/
_output_shapes
:?????????P2

Identity?
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:?????????P: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:W S
/
_output_shapes
:?????????P
 
_user_specified_nameinputs
?
e
I__inference_activation_2_layer_call_and_return_conditional_losses_3596788

inputs
identityV
ReluReluinputs*
T0*/
_output_shapes
:?????????P2
Relun
IdentityIdentityRelu:activations:0*
T0*/
_output_shapes
:?????????P2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????P:W S
/
_output_shapes
:?????????P
 
_user_specified_nameinputs
?^
?
B__inference_model_layer_call_and_return_conditional_losses_3596085

inputs?
%conv2d_conv2d_readvariableop_resource:P4
&conv2d_biasadd_readvariableop_resource:P9
+batch_normalization_readvariableop_resource:P;
-batch_normalization_readvariableop_1_resource:PJ
<batch_normalization_fusedbatchnormv3_readvariableop_resource:PL
>batch_normalization_fusedbatchnormv3_readvariableop_1_resource:PA
'conv2d_1_conv2d_readvariableop_resource:PP6
(conv2d_1_biasadd_readvariableop_resource:P;
-batch_normalization_1_readvariableop_resource:P=
/batch_normalization_1_readvariableop_1_resource:PL
>batch_normalization_1_fusedbatchnormv3_readvariableop_resource:PN
@batch_normalization_1_fusedbatchnormv3_readvariableop_1_resource:PA
'conv2d_2_conv2d_readvariableop_resource:PP6
(conv2d_2_biasadd_readvariableop_resource:P;
-batch_normalization_2_readvariableop_resource:P=
/batch_normalization_2_readvariableop_1_resource:PL
>batch_normalization_2_fusedbatchnormv3_readvariableop_resource:PN
@batch_normalization_2_fusedbatchnormv3_readvariableop_1_resource:P
identity??3batch_normalization/FusedBatchNormV3/ReadVariableOp?5batch_normalization/FusedBatchNormV3/ReadVariableOp_1?"batch_normalization/ReadVariableOp?$batch_normalization/ReadVariableOp_1?5batch_normalization_1/FusedBatchNormV3/ReadVariableOp?7batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1?$batch_normalization_1/ReadVariableOp?&batch_normalization_1/ReadVariableOp_1?5batch_normalization_2/FusedBatchNormV3/ReadVariableOp?7batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1?$batch_normalization_2/ReadVariableOp?&batch_normalization_2/ReadVariableOp_1?conv2d/BiasAdd/ReadVariableOp?conv2d/Conv2D/ReadVariableOp?conv2d_1/BiasAdd/ReadVariableOp?conv2d_1/Conv2D/ReadVariableOp?conv2d_2/BiasAdd/ReadVariableOp?conv2d_2/Conv2D/ReadVariableOp?
conv2d/Conv2D/ReadVariableOpReadVariableOp%conv2d_conv2d_readvariableop_resource*&
_output_shapes
:P*
dtype02
conv2d/Conv2D/ReadVariableOp?
conv2d/Conv2DConv2Dinputs$conv2d/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????P*
paddingSAME*
strides
2
conv2d/Conv2D?
conv2d/BiasAdd/ReadVariableOpReadVariableOp&conv2d_biasadd_readvariableop_resource*
_output_shapes
:P*
dtype02
conv2d/BiasAdd/ReadVariableOp?
conv2d/BiasAddBiasAddconv2d/Conv2D:output:0%conv2d/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????P2
conv2d/BiasAdd?
"batch_normalization/ReadVariableOpReadVariableOp+batch_normalization_readvariableop_resource*
_output_shapes
:P*
dtype02$
"batch_normalization/ReadVariableOp?
$batch_normalization/ReadVariableOp_1ReadVariableOp-batch_normalization_readvariableop_1_resource*
_output_shapes
:P*
dtype02&
$batch_normalization/ReadVariableOp_1?
3batch_normalization/FusedBatchNormV3/ReadVariableOpReadVariableOp<batch_normalization_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:P*
dtype025
3batch_normalization/FusedBatchNormV3/ReadVariableOp?
5batch_normalization/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp>batch_normalization_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:P*
dtype027
5batch_normalization/FusedBatchNormV3/ReadVariableOp_1?
$batch_normalization/FusedBatchNormV3FusedBatchNormV3conv2d/BiasAdd:output:0*batch_normalization/ReadVariableOp:value:0,batch_normalization/ReadVariableOp_1:value:0;batch_normalization/FusedBatchNormV3/ReadVariableOp:value:0=batch_normalization/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????P:P:P:P:P:*
epsilon%o?:*
is_training( 2&
$batch_normalization/FusedBatchNormV3?
activation/ReluRelu(batch_normalization/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:?????????P2
activation/Relu?
dropout/IdentityIdentityactivation/Relu:activations:0*
T0*/
_output_shapes
:?????????P2
dropout/Identity?
conv2d_1/Conv2D/ReadVariableOpReadVariableOp'conv2d_1_conv2d_readvariableop_resource*&
_output_shapes
:PP*
dtype02 
conv2d_1/Conv2D/ReadVariableOp?
conv2d_1/Conv2DConv2Ddropout/Identity:output:0&conv2d_1/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????P*
paddingSAME*
strides
2
conv2d_1/Conv2D?
conv2d_1/BiasAdd/ReadVariableOpReadVariableOp(conv2d_1_biasadd_readvariableop_resource*
_output_shapes
:P*
dtype02!
conv2d_1/BiasAdd/ReadVariableOp?
conv2d_1/BiasAddBiasAddconv2d_1/Conv2D:output:0'conv2d_1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????P2
conv2d_1/BiasAdd?
$batch_normalization_1/ReadVariableOpReadVariableOp-batch_normalization_1_readvariableop_resource*
_output_shapes
:P*
dtype02&
$batch_normalization_1/ReadVariableOp?
&batch_normalization_1/ReadVariableOp_1ReadVariableOp/batch_normalization_1_readvariableop_1_resource*
_output_shapes
:P*
dtype02(
&batch_normalization_1/ReadVariableOp_1?
5batch_normalization_1/FusedBatchNormV3/ReadVariableOpReadVariableOp>batch_normalization_1_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:P*
dtype027
5batch_normalization_1/FusedBatchNormV3/ReadVariableOp?
7batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp@batch_normalization_1_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:P*
dtype029
7batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1?
&batch_normalization_1/FusedBatchNormV3FusedBatchNormV3conv2d_1/BiasAdd:output:0,batch_normalization_1/ReadVariableOp:value:0.batch_normalization_1/ReadVariableOp_1:value:0=batch_normalization_1/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????P:P:P:P:P:*
epsilon%o?:*
is_training( 2(
&batch_normalization_1/FusedBatchNormV3?
activation_1/ReluRelu*batch_normalization_1/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:?????????P2
activation_1/Relu?
dropout_1/IdentityIdentityactivation_1/Relu:activations:0*
T0*/
_output_shapes
:?????????P2
dropout_1/Identity?
conv2d_2/Conv2D/ReadVariableOpReadVariableOp'conv2d_2_conv2d_readvariableop_resource*&
_output_shapes
:PP*
dtype02 
conv2d_2/Conv2D/ReadVariableOp?
conv2d_2/Conv2DConv2Ddropout_1/Identity:output:0&conv2d_2/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????P*
paddingSAME*
strides
2
conv2d_2/Conv2D?
conv2d_2/BiasAdd/ReadVariableOpReadVariableOp(conv2d_2_biasadd_readvariableop_resource*
_output_shapes
:P*
dtype02!
conv2d_2/BiasAdd/ReadVariableOp?
conv2d_2/BiasAddBiasAddconv2d_2/Conv2D:output:0'conv2d_2/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????P2
conv2d_2/BiasAdd?
$batch_normalization_2/ReadVariableOpReadVariableOp-batch_normalization_2_readvariableop_resource*
_output_shapes
:P*
dtype02&
$batch_normalization_2/ReadVariableOp?
&batch_normalization_2/ReadVariableOp_1ReadVariableOp/batch_normalization_2_readvariableop_1_resource*
_output_shapes
:P*
dtype02(
&batch_normalization_2/ReadVariableOp_1?
5batch_normalization_2/FusedBatchNormV3/ReadVariableOpReadVariableOp>batch_normalization_2_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:P*
dtype027
5batch_normalization_2/FusedBatchNormV3/ReadVariableOp?
7batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp@batch_normalization_2_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:P*
dtype029
7batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1?
&batch_normalization_2/FusedBatchNormV3FusedBatchNormV3conv2d_2/BiasAdd:output:0,batch_normalization_2/ReadVariableOp:value:0.batch_normalization_2/ReadVariableOp_1:value:0=batch_normalization_2/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????P:P:P:P:P:*
epsilon%o?:*
is_training( 2(
&batch_normalization_2/FusedBatchNormV3?
tf.__operators__.add/AddV2AddV2*batch_normalization_2/FusedBatchNormV3:y:0dropout/Identity:output:0*
T0*/
_output_shapes
:?????????P2
tf.__operators__.add/AddV2?
activation_2/ReluRelutf.__operators__.add/AddV2:z:0*
T0*/
_output_shapes
:?????????P2
activation_2/Relu?
dropout_2/IdentityIdentityactivation_2/Relu:activations:0*
T0*/
_output_shapes
:?????????P2
dropout_2/Identity~
IdentityIdentitydropout_2/Identity:output:0^NoOp*
T0*/
_output_shapes
:?????????P2

Identity?
NoOpNoOp4^batch_normalization/FusedBatchNormV3/ReadVariableOp6^batch_normalization/FusedBatchNormV3/ReadVariableOp_1#^batch_normalization/ReadVariableOp%^batch_normalization/ReadVariableOp_16^batch_normalization_1/FusedBatchNormV3/ReadVariableOp8^batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1%^batch_normalization_1/ReadVariableOp'^batch_normalization_1/ReadVariableOp_16^batch_normalization_2/FusedBatchNormV3/ReadVariableOp8^batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1%^batch_normalization_2/ReadVariableOp'^batch_normalization_2/ReadVariableOp_1^conv2d/BiasAdd/ReadVariableOp^conv2d/Conv2D/ReadVariableOp ^conv2d_1/BiasAdd/ReadVariableOp^conv2d_1/Conv2D/ReadVariableOp ^conv2d_2/BiasAdd/ReadVariableOp^conv2d_2/Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*R
_input_shapesA
?:?????????: : : : : : : : : : : : : : : : : : 2j
3batch_normalization/FusedBatchNormV3/ReadVariableOp3batch_normalization/FusedBatchNormV3/ReadVariableOp2n
5batch_normalization/FusedBatchNormV3/ReadVariableOp_15batch_normalization/FusedBatchNormV3/ReadVariableOp_12H
"batch_normalization/ReadVariableOp"batch_normalization/ReadVariableOp2L
$batch_normalization/ReadVariableOp_1$batch_normalization/ReadVariableOp_12n
5batch_normalization_1/FusedBatchNormV3/ReadVariableOp5batch_normalization_1/FusedBatchNormV3/ReadVariableOp2r
7batch_normalization_1/FusedBatchNormV3/ReadVariableOp_17batch_normalization_1/FusedBatchNormV3/ReadVariableOp_12L
$batch_normalization_1/ReadVariableOp$batch_normalization_1/ReadVariableOp2P
&batch_normalization_1/ReadVariableOp_1&batch_normalization_1/ReadVariableOp_12n
5batch_normalization_2/FusedBatchNormV3/ReadVariableOp5batch_normalization_2/FusedBatchNormV3/ReadVariableOp2r
7batch_normalization_2/FusedBatchNormV3/ReadVariableOp_17batch_normalization_2/FusedBatchNormV3/ReadVariableOp_12L
$batch_normalization_2/ReadVariableOp$batch_normalization_2/ReadVariableOp2P
&batch_normalization_2/ReadVariableOp_1&batch_normalization_2/ReadVariableOp_12>
conv2d/BiasAdd/ReadVariableOpconv2d/BiasAdd/ReadVariableOp2<
conv2d/Conv2D/ReadVariableOpconv2d/Conv2D/ReadVariableOp2B
conv2d_1/BiasAdd/ReadVariableOpconv2d_1/BiasAdd/ReadVariableOp2@
conv2d_1/Conv2D/ReadVariableOpconv2d_1/Conv2D/ReadVariableOp2B
conv2d_2/BiasAdd/ReadVariableOpconv2d_2/BiasAdd/ReadVariableOp2@
conv2d_2/Conv2D/ReadVariableOpconv2d_2/Conv2D/ReadVariableOp:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
'__inference_model_layer_call_fn_3594448
input_1!
unknown:P
	unknown_0:P
	unknown_1:P
	unknown_2:P
	unknown_3:P
	unknown_4:P#
	unknown_5:PP
	unknown_6:P
	unknown_7:P
	unknown_8:P
	unknown_9:P

unknown_10:P$

unknown_11:PP

unknown_12:P

unknown_13:P

unknown_14:P

unknown_15:P

unknown_16:P
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????P*4
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_model_layer_call_and_return_conditional_losses_35944092
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:?????????P2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*R
_input_shapesA
?:?????????: : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
/
_output_shapes
:?????????
!
_user_specified_name	input_1
?B
?	
B__inference_model_layer_call_and_return_conditional_losses_3594979
input_1(
conv2d_3594929:P
conv2d_3594931:P)
batch_normalization_3594934:P)
batch_normalization_3594936:P)
batch_normalization_3594938:P)
batch_normalization_3594940:P*
conv2d_1_3594945:PP
conv2d_1_3594947:P+
batch_normalization_1_3594950:P+
batch_normalization_1_3594952:P+
batch_normalization_1_3594954:P+
batch_normalization_1_3594956:P*
conv2d_2_3594961:PP
conv2d_2_3594963:P+
batch_normalization_2_3594966:P+
batch_normalization_2_3594968:P+
batch_normalization_2_3594970:P+
batch_normalization_2_3594972:P
identity??+batch_normalization/StatefulPartitionedCall?-batch_normalization_1/StatefulPartitionedCall?-batch_normalization_2/StatefulPartitionedCall?conv2d/StatefulPartitionedCall? conv2d_1/StatefulPartitionedCall? conv2d_2/StatefulPartitionedCall?dropout/StatefulPartitionedCall?!dropout_1/StatefulPartitionedCall?!dropout_2/StatefulPartitionedCall?
conv2d/StatefulPartitionedCallStatefulPartitionedCallinput_1conv2d_3594929conv2d_3594931*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????P*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_conv2d_layer_call_and_return_conditional_losses_35942462 
conv2d/StatefulPartitionedCall?
+batch_normalization/StatefulPartitionedCallStatefulPartitionedCall'conv2d/StatefulPartitionedCall:output:0batch_normalization_3594934batch_normalization_3594936batch_normalization_3594938batch_normalization_3594940*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????P*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *Y
fTRR
P__inference_batch_normalization_layer_call_and_return_conditional_losses_35946762-
+batch_normalization/StatefulPartitionedCall?
activation/PartitionedCallPartitionedCall4batch_normalization/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????P* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *P
fKRI
G__inference_activation_layer_call_and_return_conditional_losses_35942842
activation/PartitionedCall?
dropout/StatefulPartitionedCallStatefulPartitionedCall#activation/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????P* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_dropout_layer_call_and_return_conditional_losses_35946342!
dropout/StatefulPartitionedCall?
 conv2d_1/StatefulPartitionedCallStatefulPartitionedCall(dropout/StatefulPartitionedCall:output:0conv2d_1_3594945conv2d_1_3594947*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????P*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_conv2d_1_layer_call_and_return_conditional_losses_35943032"
 conv2d_1/StatefulPartitionedCall?
-batch_normalization_1/StatefulPartitionedCallStatefulPartitionedCall)conv2d_1/StatefulPartitionedCall:output:0batch_normalization_1_3594950batch_normalization_1_3594952batch_normalization_1_3594954batch_normalization_1_3594956*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????P*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *[
fVRT
R__inference_batch_normalization_1_layer_call_and_return_conditional_losses_35945932/
-batch_normalization_1/StatefulPartitionedCall?
activation_1/PartitionedCallPartitionedCall6batch_normalization_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????P* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *R
fMRK
I__inference_activation_1_layer_call_and_return_conditional_losses_35943412
activation_1/PartitionedCall?
!dropout_1/StatefulPartitionedCallStatefulPartitionedCall%activation_1/PartitionedCall:output:0 ^dropout/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????P* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_dropout_1_layer_call_and_return_conditional_losses_35945512#
!dropout_1/StatefulPartitionedCall?
 conv2d_2/StatefulPartitionedCallStatefulPartitionedCall*dropout_1/StatefulPartitionedCall:output:0conv2d_2_3594961conv2d_2_3594963*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????P*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_conv2d_2_layer_call_and_return_conditional_losses_35943602"
 conv2d_2/StatefulPartitionedCall?
-batch_normalization_2/StatefulPartitionedCallStatefulPartitionedCall)conv2d_2/StatefulPartitionedCall:output:0batch_normalization_2_3594966batch_normalization_2_3594968batch_normalization_2_3594970batch_normalization_2_3594972*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????P*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *[
fVRT
R__inference_batch_normalization_2_layer_call_and_return_conditional_losses_35945102/
-batch_normalization_2/StatefulPartitionedCall?
tf.__operators__.add/AddV2AddV26batch_normalization_2/StatefulPartitionedCall:output:0(dropout/StatefulPartitionedCall:output:0*
T0*/
_output_shapes
:?????????P2
tf.__operators__.add/AddV2?
activation_2/PartitionedCallPartitionedCalltf.__operators__.add/AddV2:z:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????P* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *R
fMRK
I__inference_activation_2_layer_call_and_return_conditional_losses_35943992
activation_2/PartitionedCall?
!dropout_2/StatefulPartitionedCallStatefulPartitionedCall%activation_2/PartitionedCall:output:0"^dropout_1/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????P* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_dropout_2_layer_call_and_return_conditional_losses_35944682#
!dropout_2/StatefulPartitionedCall?
IdentityIdentity*dropout_2/StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:?????????P2

Identity?
NoOpNoOp,^batch_normalization/StatefulPartitionedCall.^batch_normalization_1/StatefulPartitionedCall.^batch_normalization_2/StatefulPartitionedCall^conv2d/StatefulPartitionedCall!^conv2d_1/StatefulPartitionedCall!^conv2d_2/StatefulPartitionedCall ^dropout/StatefulPartitionedCall"^dropout_1/StatefulPartitionedCall"^dropout_2/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*R
_input_shapesA
?:?????????: : : : : : : : : : : : : : : : : : 2Z
+batch_normalization/StatefulPartitionedCall+batch_normalization/StatefulPartitionedCall2^
-batch_normalization_1/StatefulPartitionedCall-batch_normalization_1/StatefulPartitionedCall2^
-batch_normalization_2/StatefulPartitionedCall-batch_normalization_2/StatefulPartitionedCall2@
conv2d/StatefulPartitionedCallconv2d/StatefulPartitionedCall2D
 conv2d_1/StatefulPartitionedCall conv2d_1/StatefulPartitionedCall2D
 conv2d_2/StatefulPartitionedCall conv2d_2/StatefulPartitionedCall2B
dropout/StatefulPartitionedCalldropout/StatefulPartitionedCall2F
!dropout_1/StatefulPartitionedCall!dropout_1/StatefulPartitionedCall2F
!dropout_2/StatefulPartitionedCall!dropout_2/StatefulPartitionedCall:X T
/
_output_shapes
:?????????
!
_user_specified_name	input_1
?
?
5__inference_batch_normalization_layer_call_fn_3596346

inputs
unknown:P
	unknown_0:P
	unknown_1:P
	unknown_2:P
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????P*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *Y
fTRR
P__inference_batch_normalization_layer_call_and_return_conditional_losses_35946762
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:?????????P2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:?????????P: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????P
 
_user_specified_nameinputs
?
G
+__inference_dropout_2_layer_call_fn_3596793

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????P* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_dropout_2_layer_call_and_return_conditional_losses_35944062
PartitionedCallt
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:?????????P2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????P:W S
/
_output_shapes
:?????????P
 
_user_specified_nameinputs
?
?
R__inference_batch_normalization_2_layer_call_and_return_conditional_losses_3596760

inputs%
readvariableop_resource:P'
readvariableop_1_resource:P6
(fusedbatchnormv3_readvariableop_resource:P8
*fusedbatchnormv3_readvariableop_1_resource:P
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:P*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:P*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:P*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:P*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????P:P:P:P:P:*
epsilon%o?:*
is_training( 2
FusedBatchNormV3w
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*/
_output_shapes
:?????????P2

Identity?
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:?????????P: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:W S
/
_output_shapes
:?????????P
 
_user_specified_nameinputs
?
?
F__inference_conv2d_10_layer_call_and_return_conditional_losses_3595066

inputs8
conv2d_readvariableop_resource:PI-
biasadd_readvariableop_resource:I
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:PI*
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????I*
paddingSAME*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:I*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????I2	
BiasAddi
SoftmaxSoftmaxBiasAdd:output:0*
T0*/
_output_shapes
:?????????I2	
Softmaxt
IdentityIdentitySoftmax:softmax:0^NoOp*
T0*/
_output_shapes
:?????????I2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????P: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:?????????P
 
_user_specified_nameinputs
?
?
7__inference_batch_normalization_1_layer_call_fn_3596526

inputs
unknown:P
	unknown_0:P
	unknown_1:P
	unknown_2:P
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????P*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *[
fVRT
R__inference_batch_normalization_1_layer_call_and_return_conditional_losses_35945932
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:?????????P2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:?????????P: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????P
 
_user_specified_nameinputs
?

?
D__inference_dense_1_layer_call_and_return_conditional_losses_3595083

inputs0
matmul_readvariableop_resource:@-
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:@*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2	
BiasAddX
TanhTanhBiasAdd:output:0*
T0*'
_output_shapes
:?????????2
Tanhc
IdentityIdentityTanh:y:0^NoOp*
T0*'
_output_shapes
:?????????2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:?????????@
 
_user_specified_nameinputs
?
?
(__inference_conv2d_layer_call_fn_3596284

inputs!
unknown:P
	unknown_0:P
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????P*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_conv2d_layer_call_and_return_conditional_losses_35942462
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:?????????P2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????: : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?	
?
5__inference_batch_normalization_layer_call_fn_3596320

inputs
unknown:P
	unknown_0:P
	unknown_1:P
	unknown_2:P
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????P*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *Y
fTRR
P__inference_batch_normalization_layer_call_and_return_conditional_losses_35939172
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+???????????????????????????P2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+???????????????????????????P: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+???????????????????????????P
 
_user_specified_nameinputs
?
d
F__inference_dropout_1_layer_call_and_return_conditional_losses_3596623

inputs

identity_1b
IdentityIdentityinputs*
T0*/
_output_shapes
:?????????P2

Identityq

Identity_1IdentityIdentity:output:0*
T0*/
_output_shapes
:?????????P2

Identity_1"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????P:W S
/
_output_shapes
:?????????P
 
_user_specified_nameinputs
?
?
P__inference_batch_normalization_layer_call_and_return_conditional_losses_3596382

inputs%
readvariableop_resource:P'
readvariableop_1_resource:P6
(fusedbatchnormv3_readvariableop_resource:P8
*fusedbatchnormv3_readvariableop_1_resource:P
identity??AssignNewValue?AssignNewValue_1?FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:P*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:P*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:P*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:P*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+???????????????????????????P:P:P:P:P:*
epsilon%o?:*
exponential_avg_factor%
?#<2
FusedBatchNormV3?
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype02
AssignNewValue?
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype02
AssignNewValue_1?
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+???????????????????????????P2

Identity?
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+???????????????????????????P: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+???????????????????????????P
 
_user_specified_nameinputs
?
?
C__inference_conv2d_layer_call_and_return_conditional_losses_3594246

inputs8
conv2d_readvariableop_resource:P-
biasadd_readvariableop_resource:P
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:P*
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????P*
paddingSAME*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:P*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????P2	
BiasAdds
IdentityIdentityBiasAdd:output:0^NoOp*
T0*/
_output_shapes
:?????????P2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
R__inference_batch_normalization_2_layer_call_and_return_conditional_losses_3596742

inputs%
readvariableop_resource:P'
readvariableop_1_resource:P6
(fusedbatchnormv3_readvariableop_resource:P8
*fusedbatchnormv3_readvariableop_1_resource:P
identity??AssignNewValue?AssignNewValue_1?FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:P*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:P*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:P*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:P*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+???????????????????????????P:P:P:P:P:*
epsilon%o?:*
exponential_avg_factor%
?#<2
FusedBatchNormV3?
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype02
AssignNewValue?
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype02
AssignNewValue_1?
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+???????????????????????????P2

Identity?
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+???????????????????????????P: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+???????????????????????????P
 
_user_specified_nameinputs
?
e
F__inference_dropout_9_layer_call_and_return_conditional_losses_3596224

inputs
identity?c
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
dropout/Const{
dropout/MulMulinputsdropout/Const:output:0*
T0*/
_output_shapes
:?????????2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape?
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*/
_output_shapes
:?????????*
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L>2
dropout/GreaterEqual/y?
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*/
_output_shapes
:?????????2
dropout/GreaterEqual?
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*/
_output_shapes
:?????????2
dropout/Cast?
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*/
_output_shapes
:?????????2
dropout/Mul_1m
IdentityIdentitydropout/Mul_1:z:0*
T0*/
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
'__inference_model_layer_call_fn_3595973

inputs!
unknown:P
	unknown_0:P
	unknown_1:P
	unknown_2:P
	unknown_3:P
	unknown_4:P#
	unknown_5:PP
	unknown_6:P
	unknown_7:P
	unknown_8:P
	unknown_9:P

unknown_10:P$

unknown_11:PP

unknown_12:P

unknown_13:P

unknown_14:P

unknown_15:P

unknown_16:P
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????P*4
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_model_layer_call_and_return_conditional_losses_35944092
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:?????????P2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*R
_input_shapesA
?:?????????: : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
)__inference_model_3_layer_call_fn_3595427
input_4!
unknown:P
	unknown_0:P
	unknown_1:P
	unknown_2:P
	unknown_3:P
	unknown_4:P#
	unknown_5:PP
	unknown_6:P
	unknown_7:P
	unknown_8:P
	unknown_9:P

unknown_10:P$

unknown_11:PP

unknown_12:P

unknown_13:P

unknown_14:P

unknown_15:P

unknown_16:P$

unknown_17:P

unknown_18:$

unknown_19:PI

unknown_20:I

unknown_21:@

unknown_22:
identity

identity_1??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_4unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22*$
Tin
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:?????????:?????????I*4
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_model_3_layer_call_and_return_conditional_losses_35953192
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????2

Identity?

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*/
_output_shapes
:?????????I2

Identity_1h
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*^
_input_shapesM
K:?????????: : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
/
_output_shapes
:?????????
!
_user_specified_name	input_4
?
?
P__inference_batch_normalization_layer_call_and_return_conditional_losses_3596400

inputs%
readvariableop_resource:P'
readvariableop_1_resource:P6
(fusedbatchnormv3_readvariableop_resource:P8
*fusedbatchnormv3_readvariableop_1_resource:P
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:P*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:P*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:P*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:P*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????P:P:P:P:P:*
epsilon%o?:*
is_training( 2
FusedBatchNormV3w
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*/
_output_shapes
:?????????P2

Identity?
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:?????????P: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:W S
/
_output_shapes
:?????????P
 
_user_specified_nameinputs"?L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*?
serving_default?
C
input_48
serving_default_input_4:0?????????E
	conv2d_108
StatefulPartitionedCall:0?????????I;
dense_10
StatefulPartitionedCall:1?????????tensorflow/serving/predict:??
?
layer-0
layer_with_weights-0
layer-1
layer_with_weights-1
layer-2
layer-3
layer-4
layer_with_weights-2
layer-5
layer_with_weights-3
layer-6

signatures
#	_self_saveable_object_factories

	variables
regularization_losses
trainable_variables
	keras_api
?_default_save_signature
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_network
D
#_self_saveable_object_factories"
_tf_keras_input_layer
?
layer-0
layer_with_weights-0
layer-1
layer_with_weights-1
layer-2
layer-3
layer-4
layer_with_weights-2
layer-5
layer_with_weights-3
layer-6
layer-7
layer-8
layer_with_weights-4
layer-9
layer_with_weights-5
layer-10
layer-11
layer-12
layer-13
#_self_saveable_object_factories
	variables
regularization_losses
 trainable_variables
!	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_network
?

"kernel
#bias
#$_self_saveable_object_factories
%regularization_losses
&	variables
'trainable_variables
(	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?
#)_self_saveable_object_factories
*regularization_losses
+	variables
,trainable_variables
-	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?
#._self_saveable_object_factories
/regularization_losses
0	variables
1trainable_variables
2	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?

3kernel
4bias
#5_self_saveable_object_factories
6regularization_losses
7	variables
8trainable_variables
9	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?

:kernel
;bias
#<_self_saveable_object_factories
=regularization_losses
>	variables
?trainable_variables
@	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
-
?serving_default"
signature_map
 "
trackable_dict_wrapper
?
A0
B1
C2
D3
E4
F5
G6
H7
I8
J9
K10
L11
M12
N13
O14
P15
Q16
R17
"18
#19
320
421
:22
;23"
trackable_list_wrapper
 "
trackable_list_wrapper
?
A0
B1
C2
D3
G4
H5
I6
J7
M8
N9
O10
P11
"12
#13
314
415
:16
;17"
trackable_list_wrapper
?
Slayer_metrics
Tlayer_regularization_losses
Umetrics

	variables
regularization_losses

Vlayers
trainable_variables
Wnon_trainable_variables
?__call__
?_default_save_signature
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_dict_wrapper
D
#X_self_saveable_object_factories"
_tf_keras_input_layer
?

Akernel
Bbias
#Y_self_saveable_object_factories
Zregularization_losses
[	variables
\trainable_variables
]	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?
^axis
	Cgamma
Dbeta
Emoving_mean
Fmoving_variance
#__self_saveable_object_factories
`regularization_losses
a	variables
btrainable_variables
c	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?
#d_self_saveable_object_factories
eregularization_losses
f	variables
gtrainable_variables
h	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?
#i_self_saveable_object_factories
jregularization_losses
k	variables
ltrainable_variables
m	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?

Gkernel
Hbias
#n_self_saveable_object_factories
oregularization_losses
p	variables
qtrainable_variables
r	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?
saxis
	Igamma
Jbeta
Kmoving_mean
Lmoving_variance
#t_self_saveable_object_factories
uregularization_losses
v	variables
wtrainable_variables
x	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?
#y_self_saveable_object_factories
zregularization_losses
{	variables
|trainable_variables
}	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?
#~_self_saveable_object_factories
regularization_losses
?	variables
?trainable_variables
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?

Mkernel
Nbias
$?_self_saveable_object_factories
?regularization_losses
?	variables
?trainable_variables
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?
	?axis
	Ogamma
Pbeta
Qmoving_mean
Rmoving_variance
$?_self_saveable_object_factories
?regularization_losses
?	variables
?trainable_variables
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
O
$?_self_saveable_object_factories
?	keras_api"
_tf_keras_layer
?
$?_self_saveable_object_factories
?regularization_losses
?	variables
?trainable_variables
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?
$?_self_saveable_object_factories
?regularization_losses
?	variables
?trainable_variables
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
 "
trackable_dict_wrapper
?
A0
B1
C2
D3
E4
F5
G6
H7
I8
J9
K10
L11
M12
N13
O14
P15
Q16
R17"
trackable_list_wrapper
 "
trackable_list_wrapper
v
A0
B1
C2
D3
G4
H5
I6
J7
M8
N9
O10
P11"
trackable_list_wrapper
?
?layer_metrics
 ?layer_regularization_losses
?metrics
	variables
regularization_losses
?layers
 trainable_variables
?non_trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
*:(P2conv2d_11/kernel
:2conv2d_11/bias
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
.
"0
#1"
trackable_list_wrapper
.
"0
#1"
trackable_list_wrapper
?
?layer_metrics
 ?layer_regularization_losses
?metrics
%regularization_losses
&	variables
?layers
'trainable_variables
?non_trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?layer_metrics
 ?layer_regularization_losses
?metrics
*regularization_losses
+	variables
?layers
,trainable_variables
?non_trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?layer_metrics
 ?layer_regularization_losses
?metrics
/regularization_losses
0	variables
?layers
1trainable_variables
?non_trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 :@2dense_1/kernel
:2dense_1/bias
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
.
30
41"
trackable_list_wrapper
.
30
41"
trackable_list_wrapper
?
?layer_metrics
 ?layer_regularization_losses
?metrics
6regularization_losses
7	variables
?layers
8trainable_variables
?non_trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
*:(PI2conv2d_10/kernel
:I2conv2d_10/bias
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
.
:0
;1"
trackable_list_wrapper
.
:0
;1"
trackable_list_wrapper
?
?layer_metrics
 ?layer_regularization_losses
?metrics
=regularization_losses
>	variables
?layers
?trainable_variables
?non_trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
':%P2conv2d/kernel
:P2conv2d/bias
':%P2batch_normalization/gamma
&:$P2batch_normalization/beta
/:-P (2batch_normalization/moving_mean
3:1P (2#batch_normalization/moving_variance
):'PP2conv2d_1/kernel
:P2conv2d_1/bias
):'P2batch_normalization_1/gamma
(:&P2batch_normalization_1/beta
1:/P (2!batch_normalization_1/moving_mean
5:3P (2%batch_normalization_1/moving_variance
):'PP2conv2d_2/kernel
:P2conv2d_2/bias
):'P2batch_normalization_2/gamma
(:&P2batch_normalization_2/beta
1:/P (2!batch_normalization_2/moving_mean
5:3P (2%batch_normalization_2/moving_variance
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
Q
0
1
2
3
4
5
6"
trackable_list_wrapper
J
E0
F1
K2
L3
Q4
R5"
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
.
A0
B1"
trackable_list_wrapper
.
A0
B1"
trackable_list_wrapper
?
?layer_metrics
 ?layer_regularization_losses
?metrics
Zregularization_losses
[	variables
?layers
\trainable_variables
?non_trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
<
C0
D1
E2
F3"
trackable_list_wrapper
.
C0
D1"
trackable_list_wrapper
?
?layer_metrics
 ?layer_regularization_losses
?metrics
`regularization_losses
a	variables
?layers
btrainable_variables
?non_trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?layer_metrics
 ?layer_regularization_losses
?metrics
eregularization_losses
f	variables
?layers
gtrainable_variables
?non_trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?layer_metrics
 ?layer_regularization_losses
?metrics
jregularization_losses
k	variables
?layers
ltrainable_variables
?non_trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
.
G0
H1"
trackable_list_wrapper
.
G0
H1"
trackable_list_wrapper
?
?layer_metrics
 ?layer_regularization_losses
?metrics
oregularization_losses
p	variables
?layers
qtrainable_variables
?non_trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
<
I0
J1
K2
L3"
trackable_list_wrapper
.
I0
J1"
trackable_list_wrapper
?
?layer_metrics
 ?layer_regularization_losses
?metrics
uregularization_losses
v	variables
?layers
wtrainable_variables
?non_trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?layer_metrics
 ?layer_regularization_losses
?metrics
zregularization_losses
{	variables
?layers
|trainable_variables
?non_trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?layer_metrics
 ?layer_regularization_losses
?metrics
regularization_losses
?	variables
?layers
?trainable_variables
?non_trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
.
M0
N1"
trackable_list_wrapper
.
M0
N1"
trackable_list_wrapper
?
?layer_metrics
 ?layer_regularization_losses
?metrics
?regularization_losses
?	variables
?layers
?trainable_variables
?non_trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
<
O0
P1
Q2
R3"
trackable_list_wrapper
.
O0
P1"
trackable_list_wrapper
?
?layer_metrics
 ?layer_regularization_losses
?metrics
?regularization_losses
?	variables
?layers
?trainable_variables
?non_trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_dict_wrapper
"
_generic_user_object
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?layer_metrics
 ?layer_regularization_losses
?metrics
?regularization_losses
?	variables
?layers
?trainable_variables
?non_trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?layer_metrics
 ?layer_regularization_losses
?metrics
?regularization_losses
?	variables
?layers
?trainable_variables
?non_trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
0
1
2
3
4
5
6
7
8
9
10
11
12
13"
trackable_list_wrapper
J
E0
F1
K2
L3
Q4
R5"
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
.
E0
F1"
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
.
K0
L1"
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
.
Q0
R1"
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?B?
"__inference__wrapped_model_3593851input_4"?
???
FullArgSpec
args? 
varargsjargs
varkwjkwargs
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
)__inference_model_3_layer_call_fn_3595144
)__inference_model_3_layer_call_fn_3595657
)__inference_model_3_layer_call_fn_3595712
)__inference_model_3_layer_call_fn_3595427?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
D__inference_model_3_layer_call_and_return_conditional_losses_3595808
D__inference_model_3_layer_call_and_return_conditional_losses_3595932
D__inference_model_3_layer_call_and_return_conditional_losses_3595486
D__inference_model_3_layer_call_and_return_conditional_losses_3595545?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
'__inference_model_layer_call_fn_3594448
'__inference_model_layer_call_fn_3595973
'__inference_model_layer_call_fn_3596014
'__inference_model_layer_call_fn_3594873?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
B__inference_model_layer_call_and_return_conditional_losses_3596085
B__inference_model_layer_call_and_return_conditional_losses_3596177
B__inference_model_layer_call_and_return_conditional_losses_3594926
B__inference_model_layer_call_and_return_conditional_losses_3594979?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
+__inference_conv2d_11_layer_call_fn_3596186?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
F__inference_conv2d_11_layer_call_and_return_conditional_losses_3596197?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
+__inference_dropout_9_layer_call_fn_3596202
+__inference_dropout_9_layer_call_fn_3596207?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
F__inference_dropout_9_layer_call_and_return_conditional_losses_3596212
F__inference_dropout_9_layer_call_and_return_conditional_losses_3596224?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
+__inference_flatten_1_layer_call_fn_3596229?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
F__inference_flatten_1_layer_call_and_return_conditional_losses_3596235?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
)__inference_dense_1_layer_call_fn_3596244?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
D__inference_dense_1_layer_call_and_return_conditional_losses_3596255?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
+__inference_conv2d_10_layer_call_fn_3596264?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
F__inference_conv2d_10_layer_call_and_return_conditional_losses_3596275?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?B?
%__inference_signature_wrapper_3595602input_4"?
???
FullArgSpec
args? 
varargs
 
varkwjkwargs
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
(__inference_conv2d_layer_call_fn_3596284?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
C__inference_conv2d_layer_call_and_return_conditional_losses_3596294?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
5__inference_batch_normalization_layer_call_fn_3596307
5__inference_batch_normalization_layer_call_fn_3596320
5__inference_batch_normalization_layer_call_fn_3596333
5__inference_batch_normalization_layer_call_fn_3596346?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
P__inference_batch_normalization_layer_call_and_return_conditional_losses_3596364
P__inference_batch_normalization_layer_call_and_return_conditional_losses_3596382
P__inference_batch_normalization_layer_call_and_return_conditional_losses_3596400
P__inference_batch_normalization_layer_call_and_return_conditional_losses_3596418?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
,__inference_activation_layer_call_fn_3596423?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
G__inference_activation_layer_call_and_return_conditional_losses_3596428?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
)__inference_dropout_layer_call_fn_3596433
)__inference_dropout_layer_call_fn_3596438?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
D__inference_dropout_layer_call_and_return_conditional_losses_3596443
D__inference_dropout_layer_call_and_return_conditional_losses_3596455?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
*__inference_conv2d_1_layer_call_fn_3596464?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
E__inference_conv2d_1_layer_call_and_return_conditional_losses_3596474?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
7__inference_batch_normalization_1_layer_call_fn_3596487
7__inference_batch_normalization_1_layer_call_fn_3596500
7__inference_batch_normalization_1_layer_call_fn_3596513
7__inference_batch_normalization_1_layer_call_fn_3596526?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
R__inference_batch_normalization_1_layer_call_and_return_conditional_losses_3596544
R__inference_batch_normalization_1_layer_call_and_return_conditional_losses_3596562
R__inference_batch_normalization_1_layer_call_and_return_conditional_losses_3596580
R__inference_batch_normalization_1_layer_call_and_return_conditional_losses_3596598?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
.__inference_activation_1_layer_call_fn_3596603?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
I__inference_activation_1_layer_call_and_return_conditional_losses_3596608?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
+__inference_dropout_1_layer_call_fn_3596613
+__inference_dropout_1_layer_call_fn_3596618?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
F__inference_dropout_1_layer_call_and_return_conditional_losses_3596623
F__inference_dropout_1_layer_call_and_return_conditional_losses_3596635?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
*__inference_conv2d_2_layer_call_fn_3596644?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
E__inference_conv2d_2_layer_call_and_return_conditional_losses_3596654?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
7__inference_batch_normalization_2_layer_call_fn_3596667
7__inference_batch_normalization_2_layer_call_fn_3596680
7__inference_batch_normalization_2_layer_call_fn_3596693
7__inference_batch_normalization_2_layer_call_fn_3596706?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
R__inference_batch_normalization_2_layer_call_and_return_conditional_losses_3596724
R__inference_batch_normalization_2_layer_call_and_return_conditional_losses_3596742
R__inference_batch_normalization_2_layer_call_and_return_conditional_losses_3596760
R__inference_batch_normalization_2_layer_call_and_return_conditional_losses_3596778?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
.__inference_activation_2_layer_call_fn_3596783?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
I__inference_activation_2_layer_call_and_return_conditional_losses_3596788?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
+__inference_dropout_2_layer_call_fn_3596793
+__inference_dropout_2_layer_call_fn_3596798?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
F__inference_dropout_2_layer_call_and_return_conditional_losses_3596803
F__inference_dropout_2_layer_call_and_return_conditional_losses_3596815?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 ?
"__inference__wrapped_model_3593851?ABCDEFGHIJKLMNOPQR"#:;348?5
.?+
)?&
input_4?????????
? "k?h
8
	conv2d_10+?(
	conv2d_10?????????I
,
dense_1!?
dense_1??????????
I__inference_activation_1_layer_call_and_return_conditional_losses_3596608h7?4
-?*
(?%
inputs?????????P
? "-?*
#? 
0?????????P
? ?
.__inference_activation_1_layer_call_fn_3596603[7?4
-?*
(?%
inputs?????????P
? " ??????????P?
I__inference_activation_2_layer_call_and_return_conditional_losses_3596788h7?4
-?*
(?%
inputs?????????P
? "-?*
#? 
0?????????P
? ?
.__inference_activation_2_layer_call_fn_3596783[7?4
-?*
(?%
inputs?????????P
? " ??????????P?
G__inference_activation_layer_call_and_return_conditional_losses_3596428h7?4
-?*
(?%
inputs?????????P
? "-?*
#? 
0?????????P
? ?
,__inference_activation_layer_call_fn_3596423[7?4
-?*
(?%
inputs?????????P
? " ??????????P?
R__inference_batch_normalization_1_layer_call_and_return_conditional_losses_3596544?IJKLM?J
C?@
:?7
inputs+???????????????????????????P
p 
? "??<
5?2
0+???????????????????????????P
? ?
R__inference_batch_normalization_1_layer_call_and_return_conditional_losses_3596562?IJKLM?J
C?@
:?7
inputs+???????????????????????????P
p
? "??<
5?2
0+???????????????????????????P
? ?
R__inference_batch_normalization_1_layer_call_and_return_conditional_losses_3596580rIJKL;?8
1?.
(?%
inputs?????????P
p 
? "-?*
#? 
0?????????P
? ?
R__inference_batch_normalization_1_layer_call_and_return_conditional_losses_3596598rIJKL;?8
1?.
(?%
inputs?????????P
p
? "-?*
#? 
0?????????P
? ?
7__inference_batch_normalization_1_layer_call_fn_3596487?IJKLM?J
C?@
:?7
inputs+???????????????????????????P
p 
? "2?/+???????????????????????????P?
7__inference_batch_normalization_1_layer_call_fn_3596500?IJKLM?J
C?@
:?7
inputs+???????????????????????????P
p
? "2?/+???????????????????????????P?
7__inference_batch_normalization_1_layer_call_fn_3596513eIJKL;?8
1?.
(?%
inputs?????????P
p 
? " ??????????P?
7__inference_batch_normalization_1_layer_call_fn_3596526eIJKL;?8
1?.
(?%
inputs?????????P
p
? " ??????????P?
R__inference_batch_normalization_2_layer_call_and_return_conditional_losses_3596724?OPQRM?J
C?@
:?7
inputs+???????????????????????????P
p 
? "??<
5?2
0+???????????????????????????P
? ?
R__inference_batch_normalization_2_layer_call_and_return_conditional_losses_3596742?OPQRM?J
C?@
:?7
inputs+???????????????????????????P
p
? "??<
5?2
0+???????????????????????????P
? ?
R__inference_batch_normalization_2_layer_call_and_return_conditional_losses_3596760rOPQR;?8
1?.
(?%
inputs?????????P
p 
? "-?*
#? 
0?????????P
? ?
R__inference_batch_normalization_2_layer_call_and_return_conditional_losses_3596778rOPQR;?8
1?.
(?%
inputs?????????P
p
? "-?*
#? 
0?????????P
? ?
7__inference_batch_normalization_2_layer_call_fn_3596667?OPQRM?J
C?@
:?7
inputs+???????????????????????????P
p 
? "2?/+???????????????????????????P?
7__inference_batch_normalization_2_layer_call_fn_3596680?OPQRM?J
C?@
:?7
inputs+???????????????????????????P
p
? "2?/+???????????????????????????P?
7__inference_batch_normalization_2_layer_call_fn_3596693eOPQR;?8
1?.
(?%
inputs?????????P
p 
? " ??????????P?
7__inference_batch_normalization_2_layer_call_fn_3596706eOPQR;?8
1?.
(?%
inputs?????????P
p
? " ??????????P?
P__inference_batch_normalization_layer_call_and_return_conditional_losses_3596364?CDEFM?J
C?@
:?7
inputs+???????????????????????????P
p 
? "??<
5?2
0+???????????????????????????P
? ?
P__inference_batch_normalization_layer_call_and_return_conditional_losses_3596382?CDEFM?J
C?@
:?7
inputs+???????????????????????????P
p
? "??<
5?2
0+???????????????????????????P
? ?
P__inference_batch_normalization_layer_call_and_return_conditional_losses_3596400rCDEF;?8
1?.
(?%
inputs?????????P
p 
? "-?*
#? 
0?????????P
? ?
P__inference_batch_normalization_layer_call_and_return_conditional_losses_3596418rCDEF;?8
1?.
(?%
inputs?????????P
p
? "-?*
#? 
0?????????P
? ?
5__inference_batch_normalization_layer_call_fn_3596307?CDEFM?J
C?@
:?7
inputs+???????????????????????????P
p 
? "2?/+???????????????????????????P?
5__inference_batch_normalization_layer_call_fn_3596320?CDEFM?J
C?@
:?7
inputs+???????????????????????????P
p
? "2?/+???????????????????????????P?
5__inference_batch_normalization_layer_call_fn_3596333eCDEF;?8
1?.
(?%
inputs?????????P
p 
? " ??????????P?
5__inference_batch_normalization_layer_call_fn_3596346eCDEF;?8
1?.
(?%
inputs?????????P
p
? " ??????????P?
F__inference_conv2d_10_layer_call_and_return_conditional_losses_3596275l:;7?4
-?*
(?%
inputs?????????P
? "-?*
#? 
0?????????I
? ?
+__inference_conv2d_10_layer_call_fn_3596264_:;7?4
-?*
(?%
inputs?????????P
? " ??????????I?
F__inference_conv2d_11_layer_call_and_return_conditional_losses_3596197l"#7?4
-?*
(?%
inputs?????????P
? "-?*
#? 
0?????????
? ?
+__inference_conv2d_11_layer_call_fn_3596186_"#7?4
-?*
(?%
inputs?????????P
? " ???????????
E__inference_conv2d_1_layer_call_and_return_conditional_losses_3596474lGH7?4
-?*
(?%
inputs?????????P
? "-?*
#? 
0?????????P
? ?
*__inference_conv2d_1_layer_call_fn_3596464_GH7?4
-?*
(?%
inputs?????????P
? " ??????????P?
E__inference_conv2d_2_layer_call_and_return_conditional_losses_3596654lMN7?4
-?*
(?%
inputs?????????P
? "-?*
#? 
0?????????P
? ?
*__inference_conv2d_2_layer_call_fn_3596644_MN7?4
-?*
(?%
inputs?????????P
? " ??????????P?
C__inference_conv2d_layer_call_and_return_conditional_losses_3596294lAB7?4
-?*
(?%
inputs?????????
? "-?*
#? 
0?????????P
? ?
(__inference_conv2d_layer_call_fn_3596284_AB7?4
-?*
(?%
inputs?????????
? " ??????????P?
D__inference_dense_1_layer_call_and_return_conditional_losses_3596255\34/?,
%?"
 ?
inputs?????????@
? "%?"
?
0?????????
? |
)__inference_dense_1_layer_call_fn_3596244O34/?,
%?"
 ?
inputs?????????@
? "???????????
F__inference_dropout_1_layer_call_and_return_conditional_losses_3596623l;?8
1?.
(?%
inputs?????????P
p 
? "-?*
#? 
0?????????P
? ?
F__inference_dropout_1_layer_call_and_return_conditional_losses_3596635l;?8
1?.
(?%
inputs?????????P
p
? "-?*
#? 
0?????????P
? ?
+__inference_dropout_1_layer_call_fn_3596613_;?8
1?.
(?%
inputs?????????P
p 
? " ??????????P?
+__inference_dropout_1_layer_call_fn_3596618_;?8
1?.
(?%
inputs?????????P
p
? " ??????????P?
F__inference_dropout_2_layer_call_and_return_conditional_losses_3596803l;?8
1?.
(?%
inputs?????????P
p 
? "-?*
#? 
0?????????P
? ?
F__inference_dropout_2_layer_call_and_return_conditional_losses_3596815l;?8
1?.
(?%
inputs?????????P
p
? "-?*
#? 
0?????????P
? ?
+__inference_dropout_2_layer_call_fn_3596793_;?8
1?.
(?%
inputs?????????P
p 
? " ??????????P?
+__inference_dropout_2_layer_call_fn_3596798_;?8
1?.
(?%
inputs?????????P
p
? " ??????????P?
F__inference_dropout_9_layer_call_and_return_conditional_losses_3596212l;?8
1?.
(?%
inputs?????????
p 
? "-?*
#? 
0?????????
? ?
F__inference_dropout_9_layer_call_and_return_conditional_losses_3596224l;?8
1?.
(?%
inputs?????????
p
? "-?*
#? 
0?????????
? ?
+__inference_dropout_9_layer_call_fn_3596202_;?8
1?.
(?%
inputs?????????
p 
? " ???????????
+__inference_dropout_9_layer_call_fn_3596207_;?8
1?.
(?%
inputs?????????
p
? " ???????????
D__inference_dropout_layer_call_and_return_conditional_losses_3596443l;?8
1?.
(?%
inputs?????????P
p 
? "-?*
#? 
0?????????P
? ?
D__inference_dropout_layer_call_and_return_conditional_losses_3596455l;?8
1?.
(?%
inputs?????????P
p
? "-?*
#? 
0?????????P
? ?
)__inference_dropout_layer_call_fn_3596433_;?8
1?.
(?%
inputs?????????P
p 
? " ??????????P?
)__inference_dropout_layer_call_fn_3596438_;?8
1?.
(?%
inputs?????????P
p
? " ??????????P?
F__inference_flatten_1_layer_call_and_return_conditional_losses_3596235`7?4
-?*
(?%
inputs?????????
? "%?"
?
0?????????@
? ?
+__inference_flatten_1_layer_call_fn_3596229S7?4
-?*
(?%
inputs?????????
? "??????????@?
D__inference_model_3_layer_call_and_return_conditional_losses_3595486?ABCDEFGHIJKLMNOPQR"#:;34@?=
6?3
)?&
input_4?????????
p 

 
? "S?P
I?F
?
0/0?????????
%?"
0/1?????????I
? ?
D__inference_model_3_layer_call_and_return_conditional_losses_3595545?ABCDEFGHIJKLMNOPQR"#:;34@?=
6?3
)?&
input_4?????????
p

 
? "S?P
I?F
?
0/0?????????
%?"
0/1?????????I
? ?
D__inference_model_3_layer_call_and_return_conditional_losses_3595808?ABCDEFGHIJKLMNOPQR"#:;34??<
5?2
(?%
inputs?????????
p 

 
? "S?P
I?F
?
0/0?????????
%?"
0/1?????????I
? ?
D__inference_model_3_layer_call_and_return_conditional_losses_3595932?ABCDEFGHIJKLMNOPQR"#:;34??<
5?2
(?%
inputs?????????
p

 
? "S?P
I?F
?
0/0?????????
%?"
0/1?????????I
? ?
)__inference_model_3_layer_call_fn_3595144?ABCDEFGHIJKLMNOPQR"#:;34@?=
6?3
)?&
input_4?????????
p 

 
? "E?B
?
0?????????
#? 
1?????????I?
)__inference_model_3_layer_call_fn_3595427?ABCDEFGHIJKLMNOPQR"#:;34@?=
6?3
)?&
input_4?????????
p

 
? "E?B
?
0?????????
#? 
1?????????I?
)__inference_model_3_layer_call_fn_3595657?ABCDEFGHIJKLMNOPQR"#:;34??<
5?2
(?%
inputs?????????
p 

 
? "E?B
?
0?????????
#? 
1?????????I?
)__inference_model_3_layer_call_fn_3595712?ABCDEFGHIJKLMNOPQR"#:;34??<
5?2
(?%
inputs?????????
p

 
? "E?B
?
0?????????
#? 
1?????????I?
B__inference_model_layer_call_and_return_conditional_losses_3594926?ABCDEFGHIJKLMNOPQR@?=
6?3
)?&
input_1?????????
p 

 
? "-?*
#? 
0?????????P
? ?
B__inference_model_layer_call_and_return_conditional_losses_3594979?ABCDEFGHIJKLMNOPQR@?=
6?3
)?&
input_1?????????
p

 
? "-?*
#? 
0?????????P
? ?
B__inference_model_layer_call_and_return_conditional_losses_3596085?ABCDEFGHIJKLMNOPQR??<
5?2
(?%
inputs?????????
p 

 
? "-?*
#? 
0?????????P
? ?
B__inference_model_layer_call_and_return_conditional_losses_3596177?ABCDEFGHIJKLMNOPQR??<
5?2
(?%
inputs?????????
p

 
? "-?*
#? 
0?????????P
? ?
'__inference_model_layer_call_fn_3594448xABCDEFGHIJKLMNOPQR@?=
6?3
)?&
input_1?????????
p 

 
? " ??????????P?
'__inference_model_layer_call_fn_3594873xABCDEFGHIJKLMNOPQR@?=
6?3
)?&
input_1?????????
p

 
? " ??????????P?
'__inference_model_layer_call_fn_3595973wABCDEFGHIJKLMNOPQR??<
5?2
(?%
inputs?????????
p 

 
? " ??????????P?
'__inference_model_layer_call_fn_3596014wABCDEFGHIJKLMNOPQR??<
5?2
(?%
inputs?????????
p

 
? " ??????????P?
%__inference_signature_wrapper_3595602?ABCDEFGHIJKLMNOPQR"#:;34C?@
? 
9?6
4
input_4)?&
input_4?????????"k?h
8
	conv2d_10+?(
	conv2d_10?????????I
,
dense_1!?
dense_1?????????