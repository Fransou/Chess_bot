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
conv2d_51/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*!
shared_nameconv2d_51/kernel
~
$conv2d_51/kernel/Read/ReadVariableOpReadVariableOpconv2d_51/kernel*'
_output_shapes
:?*
dtype0
t
conv2d_51/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameconv2d_51/bias
m
"conv2d_51/bias/Read/ReadVariableOpReadVariableOpconv2d_51/bias*
_output_shapes
:*
dtype0
?
conv2d_49/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:??*!
shared_nameconv2d_49/kernel

$conv2d_49/kernel/Read/ReadVariableOpReadVariableOpconv2d_49/kernel*(
_output_shapes
:??*
dtype0
u
conv2d_49/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*
shared_nameconv2d_49/bias
n
"conv2d_49/bias/Read/ReadVariableOpReadVariableOpconv2d_49/bias*
_output_shapes	
:?*
dtype0
x
dense_5/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@*
shared_namedense_5/kernel
q
"dense_5/kernel/Read/ReadVariableOpReadVariableOpdense_5/kernel*
_output_shapes

:@*
dtype0
?
conv2d_50/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:?I*!
shared_nameconv2d_50/kernel
~
$conv2d_50/kernel/Read/ReadVariableOpReadVariableOpconv2d_50/kernel*'
_output_shapes
:?I*
dtype0
t
conv2d_50/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:I*
shared_nameconv2d_50/bias
m
"conv2d_50/bias/Read/ReadVariableOpReadVariableOpconv2d_50/bias*
_output_shapes
:I*
dtype0
?
conv2d_40/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*!
shared_nameconv2d_40/kernel
~
$conv2d_40/kernel/Read/ReadVariableOpReadVariableOpconv2d_40/kernel*'
_output_shapes
:?*
dtype0
u
conv2d_40/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*
shared_nameconv2d_40/bias
n
"conv2d_40/bias/Read/ReadVariableOpReadVariableOpconv2d_40/bias*
_output_shapes	
:?*
dtype0
?
batch_normalization_28/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*-
shared_namebatch_normalization_28/gamma
?
0batch_normalization_28/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_28/gamma*
_output_shapes	
:?*
dtype0
?
batch_normalization_28/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*,
shared_namebatch_normalization_28/beta
?
/batch_normalization_28/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_28/beta*
_output_shapes	
:?*
dtype0
?
"batch_normalization_28/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*3
shared_name$"batch_normalization_28/moving_mean
?
6batch_normalization_28/moving_mean/Read/ReadVariableOpReadVariableOp"batch_normalization_28/moving_mean*
_output_shapes	
:?*
dtype0
?
&batch_normalization_28/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*7
shared_name(&batch_normalization_28/moving_variance
?
:batch_normalization_28/moving_variance/Read/ReadVariableOpReadVariableOp&batch_normalization_28/moving_variance*
_output_shapes	
:?*
dtype0
?
conv2d_41/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:??*!
shared_nameconv2d_41/kernel

$conv2d_41/kernel/Read/ReadVariableOpReadVariableOpconv2d_41/kernel*(
_output_shapes
:??*
dtype0
u
conv2d_41/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*
shared_nameconv2d_41/bias
n
"conv2d_41/bias/Read/ReadVariableOpReadVariableOpconv2d_41/bias*
_output_shapes	
:?*
dtype0
?
batch_normalization_29/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*-
shared_namebatch_normalization_29/gamma
?
0batch_normalization_29/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_29/gamma*
_output_shapes	
:?*
dtype0
?
batch_normalization_29/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*,
shared_namebatch_normalization_29/beta
?
/batch_normalization_29/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_29/beta*
_output_shapes	
:?*
dtype0
?
"batch_normalization_29/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*3
shared_name$"batch_normalization_29/moving_mean
?
6batch_normalization_29/moving_mean/Read/ReadVariableOpReadVariableOp"batch_normalization_29/moving_mean*
_output_shapes	
:?*
dtype0
?
&batch_normalization_29/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*7
shared_name(&batch_normalization_29/moving_variance
?
:batch_normalization_29/moving_variance/Read/ReadVariableOpReadVariableOp&batch_normalization_29/moving_variance*
_output_shapes	
:?*
dtype0
?
conv2d_42/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:??*!
shared_nameconv2d_42/kernel

$conv2d_42/kernel/Read/ReadVariableOpReadVariableOpconv2d_42/kernel*(
_output_shapes
:??*
dtype0
u
conv2d_42/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*
shared_nameconv2d_42/bias
n
"conv2d_42/bias/Read/ReadVariableOpReadVariableOpconv2d_42/bias*
_output_shapes	
:?*
dtype0
?
batch_normalization_30/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*-
shared_namebatch_normalization_30/gamma
?
0batch_normalization_30/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_30/gamma*
_output_shapes	
:?*
dtype0
?
batch_normalization_30/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*,
shared_namebatch_normalization_30/beta
?
/batch_normalization_30/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_30/beta*
_output_shapes	
:?*
dtype0
?
"batch_normalization_30/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*3
shared_name$"batch_normalization_30/moving_mean
?
6batch_normalization_30/moving_mean/Read/ReadVariableOpReadVariableOp"batch_normalization_30/moving_mean*
_output_shapes	
:?*
dtype0
?
&batch_normalization_30/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*7
shared_name(&batch_normalization_30/moving_variance
?
:batch_normalization_30/moving_variance/Read/ReadVariableOpReadVariableOp&batch_normalization_30/moving_variance*
_output_shapes	
:?*
dtype0

NoOpNoOp
?Q
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*?P
value?PB?P B?P
?
layer-0
layer_with_weights-0
layer-1
layer_with_weights-1
layer-2
layer-3
layer_with_weights-2
layer-4
layer-5
layer-6
layer_with_weights-3
layer-7
	layer_with_weights-4
	layer-8

regularization_losses
	variables
trainable_variables
	keras_api

signatures
 
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
regularization_losses
	variables
trainable_variables
 	keras_api
h

!kernel
"bias
#regularization_losses
$	variables
%trainable_variables
&	keras_api
R
'regularization_losses
(	variables
)trainable_variables
*	keras_api
h

+kernel
,bias
-regularization_losses
.	variables
/trainable_variables
0	keras_api
R
1regularization_losses
2	variables
3trainable_variables
4	keras_api
R
5regularization_losses
6	variables
7trainable_variables
8	keras_api
^

9kernel
:regularization_losses
;	variables
<trainable_variables
=	keras_api
h

>kernel
?bias
@regularization_losses
A	variables
Btrainable_variables
C	keras_api
 
?
D0
E1
F2
G3
H4
I5
J6
K7
L8
M9
N10
O11
P12
Q13
R14
S15
T16
U17
!18
"19
+20
,21
922
>23
?24
?
D0
E1
F2
G3
J4
K5
L6
M7
P8
Q9
R10
S11
!12
"13
+14
,15
916
>17
?18
?

regularization_losses
Vlayer_metrics
	variables

Wlayers
Xnon_trainable_variables
trainable_variables
Ymetrics
Zlayer_regularization_losses
 
 
h

Dkernel
Ebias
[regularization_losses
\	variables
]trainable_variables
^	keras_api
?
_axis
	Fgamma
Gbeta
Hmoving_mean
Imoving_variance
`regularization_losses
a	variables
btrainable_variables
c	keras_api
R
dregularization_losses
e	variables
ftrainable_variables
g	keras_api
R
hregularization_losses
i	variables
jtrainable_variables
k	keras_api
h

Jkernel
Kbias
lregularization_losses
m	variables
ntrainable_variables
o	keras_api
?
paxis
	Lgamma
Mbeta
Nmoving_mean
Omoving_variance
qregularization_losses
r	variables
strainable_variables
t	keras_api
R
uregularization_losses
v	variables
wtrainable_variables
x	keras_api
R
yregularization_losses
z	variables
{trainable_variables
|	keras_api
i

Pkernel
Qbias
}regularization_losses
~	variables
trainable_variables
?	keras_api
?
	?axis
	Rgamma
Sbeta
Tmoving_mean
Umoving_variance
?regularization_losses
?	variables
?trainable_variables
?	keras_api

?	keras_api
V
?regularization_losses
?	variables
?trainable_variables
?	keras_api
V
?regularization_losses
?	variables
?trainable_variables
?	keras_api
 
?
D0
E1
F2
G3
H4
I5
J6
K7
L8
M9
N10
O11
P12
Q13
R14
S15
T16
U17
V
D0
E1
F2
G3
J4
K5
L6
M7
P8
Q9
R10
S11
?
regularization_losses
?layer_metrics
	variables
?layers
?non_trainable_variables
trainable_variables
?metrics
 ?layer_regularization_losses
\Z
VARIABLE_VALUEconv2d_51/kernel6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE
XV
VARIABLE_VALUEconv2d_51/bias4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE
 

!0
"1

!0
"1
?
#regularization_losses
?layer_metrics
$	variables
?non_trainable_variables
?layers
%trainable_variables
?metrics
 ?layer_regularization_losses
 
 
 
?
'regularization_losses
?layer_metrics
(	variables
?non_trainable_variables
?layers
)trainable_variables
?metrics
 ?layer_regularization_losses
\Z
VARIABLE_VALUEconv2d_49/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE
XV
VARIABLE_VALUEconv2d_49/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE
 

+0
,1

+0
,1
?
-regularization_losses
?layer_metrics
.	variables
?non_trainable_variables
?layers
/trainable_variables
?metrics
 ?layer_regularization_losses
 
 
 
?
1regularization_losses
?layer_metrics
2	variables
?non_trainable_variables
?layers
3trainable_variables
?metrics
 ?layer_regularization_losses
 
 
 
?
5regularization_losses
?layer_metrics
6	variables
?non_trainable_variables
?layers
7trainable_variables
?metrics
 ?layer_regularization_losses
ZX
VARIABLE_VALUEdense_5/kernel6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUE
 

90

90
?
:regularization_losses
?layer_metrics
;	variables
?non_trainable_variables
?layers
<trainable_variables
?metrics
 ?layer_regularization_losses
\Z
VARIABLE_VALUEconv2d_50/kernel6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUE
XV
VARIABLE_VALUEconv2d_50/bias4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUE
 

>0
?1

>0
?1
?
@regularization_losses
?layer_metrics
A	variables
?non_trainable_variables
?layers
Btrainable_variables
?metrics
 ?layer_regularization_losses
LJ
VARIABLE_VALUEconv2d_40/kernel&variables/0/.ATTRIBUTES/VARIABLE_VALUE
JH
VARIABLE_VALUEconv2d_40/bias&variables/1/.ATTRIBUTES/VARIABLE_VALUE
XV
VARIABLE_VALUEbatch_normalization_28/gamma&variables/2/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEbatch_normalization_28/beta&variables/3/.ATTRIBUTES/VARIABLE_VALUE
^\
VARIABLE_VALUE"batch_normalization_28/moving_mean&variables/4/.ATTRIBUTES/VARIABLE_VALUE
b`
VARIABLE_VALUE&batch_normalization_28/moving_variance&variables/5/.ATTRIBUTES/VARIABLE_VALUE
LJ
VARIABLE_VALUEconv2d_41/kernel&variables/6/.ATTRIBUTES/VARIABLE_VALUE
JH
VARIABLE_VALUEconv2d_41/bias&variables/7/.ATTRIBUTES/VARIABLE_VALUE
XV
VARIABLE_VALUEbatch_normalization_29/gamma&variables/8/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEbatch_normalization_29/beta&variables/9/.ATTRIBUTES/VARIABLE_VALUE
_]
VARIABLE_VALUE"batch_normalization_29/moving_mean'variables/10/.ATTRIBUTES/VARIABLE_VALUE
ca
VARIABLE_VALUE&batch_normalization_29/moving_variance'variables/11/.ATTRIBUTES/VARIABLE_VALUE
MK
VARIABLE_VALUEconv2d_42/kernel'variables/12/.ATTRIBUTES/VARIABLE_VALUE
KI
VARIABLE_VALUEconv2d_42/bias'variables/13/.ATTRIBUTES/VARIABLE_VALUE
YW
VARIABLE_VALUEbatch_normalization_30/gamma'variables/14/.ATTRIBUTES/VARIABLE_VALUE
XV
VARIABLE_VALUEbatch_normalization_30/beta'variables/15/.ATTRIBUTES/VARIABLE_VALUE
_]
VARIABLE_VALUE"batch_normalization_30/moving_mean'variables/16/.ATTRIBUTES/VARIABLE_VALUE
ca
VARIABLE_VALUE&batch_normalization_30/moving_variance'variables/17/.ATTRIBUTES/VARIABLE_VALUE
 
?
0
1
2
3
4
5
6
7
	8
*
H0
I1
N2
O3
T4
U5
 
 
 

D0
E1

D0
E1
?
[regularization_losses
?layer_metrics
\	variables
?non_trainable_variables
?layers
]trainable_variables
?metrics
 ?layer_regularization_losses
 
 

F0
G1
H2
I3

F0
G1
?
`regularization_losses
?layer_metrics
a	variables
?non_trainable_variables
?layers
btrainable_variables
?metrics
 ?layer_regularization_losses
 
 
 
?
dregularization_losses
?layer_metrics
e	variables
?non_trainable_variables
?layers
ftrainable_variables
?metrics
 ?layer_regularization_losses
 
 
 
?
hregularization_losses
?layer_metrics
i	variables
?non_trainable_variables
?layers
jtrainable_variables
?metrics
 ?layer_regularization_losses
 

J0
K1

J0
K1
?
lregularization_losses
?layer_metrics
m	variables
?non_trainable_variables
?layers
ntrainable_variables
?metrics
 ?layer_regularization_losses
 
 

L0
M1
N2
O3

L0
M1
?
qregularization_losses
?layer_metrics
r	variables
?non_trainable_variables
?layers
strainable_variables
?metrics
 ?layer_regularization_losses
 
 
 
?
uregularization_losses
?layer_metrics
v	variables
?non_trainable_variables
?layers
wtrainable_variables
?metrics
 ?layer_regularization_losses
 
 
 
?
yregularization_losses
?layer_metrics
z	variables
?non_trainable_variables
?layers
{trainable_variables
?metrics
 ?layer_regularization_losses
 

P0
Q1

P0
Q1
?
}regularization_losses
?layer_metrics
~	variables
?non_trainable_variables
?layers
trainable_variables
?metrics
 ?layer_regularization_losses
 
 

R0
S1
T2
U3

R0
S1
?
?regularization_losses
?layer_metrics
?	variables
?non_trainable_variables
?layers
?trainable_variables
?metrics
 ?layer_regularization_losses
 
 
 
 
?
?regularization_losses
?layer_metrics
?	variables
?non_trainable_variables
?layers
?trainable_variables
?metrics
 ?layer_regularization_losses
 
 
 
?
?regularization_losses
?layer_metrics
?	variables
?non_trainable_variables
?layers
?trainable_variables
?metrics
 ?layer_regularization_losses
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
H0
I1
N2
O3
T4
U5
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
H0
I1
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
N0
O1
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
T0
U1
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
?
serving_default_input_12Placeholder*/
_output_shapes
:?????????*
dtype0*$
shape:?????????
?
StatefulPartitionedCallStatefulPartitionedCallserving_default_input_12conv2d_40/kernelconv2d_40/biasbatch_normalization_28/gammabatch_normalization_28/beta"batch_normalization_28/moving_mean&batch_normalization_28/moving_varianceconv2d_41/kernelconv2d_41/biasbatch_normalization_29/gammabatch_normalization_29/beta"batch_normalization_29/moving_mean&batch_normalization_29/moving_varianceconv2d_42/kernelconv2d_42/biasbatch_normalization_30/gammabatch_normalization_30/beta"batch_normalization_30/moving_mean&batch_normalization_30/moving_varianceconv2d_51/kernelconv2d_51/biasconv2d_49/kernelconv2d_49/biasconv2d_50/kernelconv2d_50/biasdense_5/kernel*%
Tin
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:?????????I:?????????*;
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8? *-
f(R&
$__inference_signature_wrapper_779649
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
?
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename$conv2d_51/kernel/Read/ReadVariableOp"conv2d_51/bias/Read/ReadVariableOp$conv2d_49/kernel/Read/ReadVariableOp"conv2d_49/bias/Read/ReadVariableOp"dense_5/kernel/Read/ReadVariableOp$conv2d_50/kernel/Read/ReadVariableOp"conv2d_50/bias/Read/ReadVariableOp$conv2d_40/kernel/Read/ReadVariableOp"conv2d_40/bias/Read/ReadVariableOp0batch_normalization_28/gamma/Read/ReadVariableOp/batch_normalization_28/beta/Read/ReadVariableOp6batch_normalization_28/moving_mean/Read/ReadVariableOp:batch_normalization_28/moving_variance/Read/ReadVariableOp$conv2d_41/kernel/Read/ReadVariableOp"conv2d_41/bias/Read/ReadVariableOp0batch_normalization_29/gamma/Read/ReadVariableOp/batch_normalization_29/beta/Read/ReadVariableOp6batch_normalization_29/moving_mean/Read/ReadVariableOp:batch_normalization_29/moving_variance/Read/ReadVariableOp$conv2d_42/kernel/Read/ReadVariableOp"conv2d_42/bias/Read/ReadVariableOp0batch_normalization_30/gamma/Read/ReadVariableOp/batch_normalization_30/beta/Read/ReadVariableOp6batch_normalization_30/moving_mean/Read/ReadVariableOp:batch_normalization_30/moving_variance/Read/ReadVariableOpConst*&
Tin
2*
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
GPU 2J 8? *(
f#R!
__inference__traced_save_780920
?
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenameconv2d_51/kernelconv2d_51/biasconv2d_49/kernelconv2d_49/biasdense_5/kernelconv2d_50/kernelconv2d_50/biasconv2d_40/kernelconv2d_40/biasbatch_normalization_28/gammabatch_normalization_28/beta"batch_normalization_28/moving_mean&batch_normalization_28/moving_varianceconv2d_41/kernelconv2d_41/biasbatch_normalization_29/gammabatch_normalization_29/beta"batch_normalization_29/moving_mean&batch_normalization_29/moving_varianceconv2d_42/kernelconv2d_42/biasbatch_normalization_30/gammabatch_normalization_30/beta"batch_normalization_30/moving_mean&batch_normalization_30/moving_variance*%
Tin
2*
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
GPU 2J 8? *+
f&R$
"__inference__traced_restore_781005??
?
?
E__inference_conv2d_50_layer_call_and_return_conditional_losses_779081

inputs9
conv2d_readvariableop_resource:?I-
biasadd_readvariableop_resource:I
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*'
_output_shapes
:?I*
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
_construction_contextkEagerRuntime*3
_input_shapes"
 :??????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
*__inference_conv2d_51_layer_call_fn_780190

inputs"
unknown:?
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
GPU 2J 8? *N
fIRG
E__inference_conv2d_51_layer_call_and_return_conditional_losses_7790252
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
_construction_contextkEagerRuntime*3
_input_shapes"
 :??????????: : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
*__inference_conv2d_41_layer_call_fn_780486

inputs#
unknown:??
	unknown_0:	?
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_conv2d_41_layer_call_and_return_conditional_losses_7783182
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*0
_output_shapes
:??????????2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :??????????: : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:??????????
 
_user_specified_nameinputs
?
d
F__inference_dropout_36_layer_call_and_return_conditional_losses_780473

inputs

identity_1c
IdentityIdentityinputs*
T0*0
_output_shapes
:??????????2

Identityr

Identity_1IdentityIdentity:output:0*
T0*0
_output_shapes
:??????????2

Identity_1"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:??????????:X T
0
_output_shapes
:??????????
 
_user_specified_nameinputs
?
e
I__inference_activation_28_layer_call_and_return_conditional_losses_778299

inputs
identityW
ReluReluinputs*
T0*0
_output_shapes
:??????????2
Reluo
IdentityIdentityRelu:activations:0*
T0*0
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:??????????:X T
0
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
R__inference_batch_normalization_28_layer_call_and_return_conditional_losses_778284

inputs&
readvariableop_resource:	?(
readvariableop_1_resource:	?7
(fusedbatchnormv3_readvariableop_resource:	?9
*fusedbatchnormv3_readvariableop_1_resource:	?
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1u
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:?*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:?*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:?*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:?*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:??????????:?:?:?:?:*
epsilon%o?:*
is_training( 2
FusedBatchNormV3x
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*0
_output_shapes
:??????????2

Identity?
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*7
_input_shapes&
$:??????????: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:X T
0
_output_shapes
:??????????
 
_user_specified_nameinputs
?
e
I__inference_activation_30_layer_call_and_return_conditional_losses_778414

inputs
identityW
ReluReluinputs*
T0*0
_output_shapes
:??????????2
Reluo
IdentityIdentityRelu:activations:0*
T0*0
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:??????????:X T
0
_output_shapes
:??????????
 
_user_specified_nameinputs
?
F
*__inference_flatten_5_layer_call_fn_780245

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
GPU 2J 8? *N
fIRG
E__inference_flatten_5_layer_call_and_return_conditional_losses_7790682
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
?
?
R__inference_batch_normalization_29_layer_call_and_return_conditional_losses_778058

inputs&
readvariableop_resource:	?(
readvariableop_1_resource:	?7
(fusedbatchnormv3_readvariableop_resource:	?9
*fusedbatchnormv3_readvariableop_1_resource:	?
identity??AssignNewValue?AssignNewValue_1?FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1u
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:?*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:?*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:?*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:?*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,????????????????????????????:?:?:?:?:*
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
T0*B
_output_shapes0
.:,????????????????????????????2

Identity?
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,????????????????????????????: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:j f
B
_output_shapes0
.:,????????????????????????????
 
_user_specified_nameinputs
?
?
R__inference_batch_normalization_29_layer_call_and_return_conditional_losses_780602

inputs&
readvariableop_resource:	?(
readvariableop_1_resource:	?7
(fusedbatchnormv3_readvariableop_resource:	?9
*fusedbatchnormv3_readvariableop_1_resource:	?
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1u
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:?*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:?*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:?*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:?*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:??????????:?:?:?:?:*
epsilon%o?:*
is_training( 2
FusedBatchNormV3x
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*0
_output_shapes
:??????????2

Identity?
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*7
_input_shapes&
$:??????????: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:X T
0
_output_shapes
:??????????
 
_user_specified_nameinputs
?
|
(__inference_dense_5_layer_call_fn_780277

inputs
unknown:@
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_dense_5_layer_call_and_return_conditional_losses_7790952
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
_construction_contextkEagerRuntime*(
_input_shapes
:?????????@: 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????@
 
_user_specified_nameinputs
?	
?
7__inference_batch_normalization_29_layer_call_fn_780548

inputs
unknown:	?
	unknown_0:	?
	unknown_1:	?
	unknown_2:	?
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *[
fVRT
R__inference_batch_normalization_29_layer_call_and_return_conditional_losses_7785922
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*0
_output_shapes
:??????????2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*7
_input_shapes&
$:??????????: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:??????????
 
_user_specified_nameinputs
?	
?
7__inference_batch_normalization_30_layer_call_fn_780694

inputs
unknown:	?
	unknown_0:	?
	unknown_1:	?
	unknown_2:	?
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,????????????????????????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *[
fVRT
R__inference_batch_normalization_30_layer_call_and_return_conditional_losses_7781842
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*B
_output_shapes0
.:,????????????????????????????2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,????????????????????????????: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:j f
B
_output_shapes0
.:,????????????????????????????
 
_user_specified_nameinputs
?
d
F__inference_dropout_45_layer_call_and_return_conditional_losses_780216

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
?
d
F__inference_dropout_44_layer_call_and_return_conditional_losses_779060

inputs

identity_1c
IdentityIdentityinputs*
T0*0
_output_shapes
:??????????2

Identityr

Identity_1IdentityIdentity:output:0*
T0*0
_output_shapes
:??????????2

Identity_1"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:??????????:X T
0
_output_shapes
:??????????
 
_user_specified_nameinputs
?
G
+__inference_dropout_38_layer_call_fn_780812

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_dropout_38_layer_call_and_return_conditional_losses_7784752
PartitionedCallu
IdentityIdentityPartitionedCall:output:0*
T0*0
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:??????????:X T
0
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
*__inference_conv2d_49_layer_call_fn_780229

inputs#
unknown:??
	unknown_0:	?
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_conv2d_49_layer_call_and_return_conditional_losses_7790422
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*0
_output_shapes
:??????????2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :??????????: : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
E__inference_conv2d_49_layer_call_and_return_conditional_losses_779042

inputs:
conv2d_readvariableop_resource:??.
biasadd_readvariableop_resource:	?
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:??*
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????*
paddingSAME*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????2	
BiasAdda
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:??????????2
Reluv
IdentityIdentityRelu:activations:0^NoOp*
T0*0
_output_shapes
:??????????2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :??????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:??????????
 
_user_specified_nameinputs
?
G
+__inference_dropout_37_layer_call_fn_780640

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_dropout_37_layer_call_and_return_conditional_losses_7785502
PartitionedCallu
IdentityIdentityPartitionedCall:output:0*
T0*0
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:??????????:X T
0
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
)__inference_model_11_layer_call_fn_779903

inputs"
unknown:?
	unknown_0:	?
	unknown_1:	?
	unknown_2:	?
	unknown_3:	?
	unknown_4:	?%
	unknown_5:??
	unknown_6:	?
	unknown_7:	?
	unknown_8:	?
	unknown_9:	?

unknown_10:	?&

unknown_11:??

unknown_12:	?

unknown_13:	?

unknown_14:	?

unknown_15:	?

unknown_16:	?%

unknown_17:?

unknown_18:&

unknown_19:??

unknown_20:	?%

unknown_21:?I

unknown_22:I

unknown_23:@
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
unknown_22
unknown_23*%
Tin
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:?????????:?????????I*;
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_model_11_layer_call_and_return_conditional_losses_7791012
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
_construction_contextkEagerRuntime*`
_input_shapesO
M:?????????: : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
(__inference_model_8_layer_call_fn_778864
input_9"
unknown:?
	unknown_0:	?
	unknown_1:	?
	unknown_2:	?
	unknown_3:	?
	unknown_4:	?%
	unknown_5:??
	unknown_6:	?
	unknown_7:	?
	unknown_8:	?
	unknown_9:	?

unknown_10:	?&

unknown_11:??

unknown_12:	?

unknown_13:	?

unknown_14:	?

unknown_15:	?

unknown_16:	?
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_9unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
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
 *0
_output_shapes
:??????????*.
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_model_8_layer_call_and_return_conditional_losses_7787842
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*0
_output_shapes
:??????????2

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
_user_specified_name	input_9
?.
?	
D__inference_model_11_layer_call_and_return_conditional_losses_779101

inputs)
model_8_778977:?
model_8_778979:	?
model_8_778981:	?
model_8_778983:	?
model_8_778985:	?
model_8_778987:	?*
model_8_778989:??
model_8_778991:	?
model_8_778993:	?
model_8_778995:	?
model_8_778997:	?
model_8_778999:	?*
model_8_779001:??
model_8_779003:	?
model_8_779005:	?
model_8_779007:	?
model_8_779009:	?
model_8_779011:	?+
conv2d_51_779026:?
conv2d_51_779028:,
conv2d_49_779043:??
conv2d_49_779045:	?+
conv2d_50_779082:?I
conv2d_50_779084:I 
dense_5_779096:@
identity

identity_1??!conv2d_49/StatefulPartitionedCall?!conv2d_50/StatefulPartitionedCall?!conv2d_51/StatefulPartitionedCall?dense_5/StatefulPartitionedCall?model_8/StatefulPartitionedCall?
model_8/StatefulPartitionedCallStatefulPartitionedCallinputsmodel_8_778977model_8_778979model_8_778981model_8_778983model_8_778985model_8_778987model_8_778989model_8_778991model_8_778993model_8_778995model_8_778997model_8_778999model_8_779001model_8_779003model_8_779005model_8_779007model_8_779009model_8_779011*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????*4
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_model_8_layer_call_and_return_conditional_losses_7784242!
model_8/StatefulPartitionedCall?
!conv2d_51/StatefulPartitionedCallStatefulPartitionedCall(model_8/StatefulPartitionedCall:output:0conv2d_51_779026conv2d_51_779028*
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
GPU 2J 8? *N
fIRG
E__inference_conv2d_51_layer_call_and_return_conditional_losses_7790252#
!conv2d_51/StatefulPartitionedCall?
!conv2d_49/StatefulPartitionedCallStatefulPartitionedCall(model_8/StatefulPartitionedCall:output:0conv2d_49_779043conv2d_49_779045*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_conv2d_49_layer_call_and_return_conditional_losses_7790422#
!conv2d_49/StatefulPartitionedCall?
dropout_45/PartitionedCallPartitionedCall*conv2d_51/StatefulPartitionedCall:output:0*
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
F__inference_dropout_45_layer_call_and_return_conditional_losses_7790532
dropout_45/PartitionedCall?
dropout_44/PartitionedCallPartitionedCall*conv2d_49/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_dropout_44_layer_call_and_return_conditional_losses_7790602
dropout_44/PartitionedCall?
flatten_5/PartitionedCallPartitionedCall#dropout_45/PartitionedCall:output:0*
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
GPU 2J 8? *N
fIRG
E__inference_flatten_5_layer_call_and_return_conditional_losses_7790682
flatten_5/PartitionedCall?
!conv2d_50/StatefulPartitionedCallStatefulPartitionedCall#dropout_44/PartitionedCall:output:0conv2d_50_779082conv2d_50_779084*
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
GPU 2J 8? *N
fIRG
E__inference_conv2d_50_layer_call_and_return_conditional_losses_7790812#
!conv2d_50/StatefulPartitionedCall?
dense_5/StatefulPartitionedCallStatefulPartitionedCall"flatten_5/PartitionedCall:output:0dense_5_779096*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_dense_5_layer_call_and_return_conditional_losses_7790952!
dense_5/StatefulPartitionedCall?
IdentityIdentity(dense_5/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????2

Identity?

Identity_1Identity*conv2d_50/StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:?????????I2

Identity_1?
NoOpNoOp"^conv2d_49/StatefulPartitionedCall"^conv2d_50/StatefulPartitionedCall"^conv2d_51/StatefulPartitionedCall ^dense_5/StatefulPartitionedCall ^model_8/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*`
_input_shapesO
M:?????????: : : : : : : : : : : : : : : : : : : : : : : : : 2F
!conv2d_49/StatefulPartitionedCall!conv2d_49/StatefulPartitionedCall2F
!conv2d_50/StatefulPartitionedCall!conv2d_50/StatefulPartitionedCall2F
!conv2d_51/StatefulPartitionedCall!conv2d_51/StatefulPartitionedCall2B
dense_5/StatefulPartitionedCalldense_5/StatefulPartitionedCall2B
model_8/StatefulPartitionedCallmodel_8/StatefulPartitionedCall:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?
G
+__inference_dropout_36_layer_call_fn_780468

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_dropout_36_layer_call_and_return_conditional_losses_7786252
PartitionedCallu
IdentityIdentityPartitionedCall:output:0*
T0*0
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:??????????:X T
0
_output_shapes
:??????????
 
_user_specified_nameinputs
?
d
F__inference_dropout_38_layer_call_and_return_conditional_losses_780817

inputs

identity_1c
IdentityIdentityinputs*
T0*0
_output_shapes
:??????????2

Identityr

Identity_1IdentityIdentity:output:0*
T0*0
_output_shapes
:??????????2

Identity_1"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:??????????:X T
0
_output_shapes
:??????????
 
_user_specified_nameinputs
??
?
C__inference_model_8_layer_call_and_return_conditional_losses_778917
input_9+
conv2d_40_778867:?
conv2d_40_778869:	?,
batch_normalization_28_778872:	?,
batch_normalization_28_778874:	?,
batch_normalization_28_778876:	?,
batch_normalization_28_778878:	?,
conv2d_41_778883:??
conv2d_41_778885:	?,
batch_normalization_29_778888:	?,
batch_normalization_29_778890:	?,
batch_normalization_29_778892:	?,
batch_normalization_29_778894:	?,
conv2d_42_778899:??
conv2d_42_778901:	?,
batch_normalization_30_778904:	?,
batch_normalization_30_778906:	?,
batch_normalization_30_778908:	?,
batch_normalization_30_778910:	?
identity??.batch_normalization_28/StatefulPartitionedCall?.batch_normalization_29/StatefulPartitionedCall?.batch_normalization_30/StatefulPartitionedCall?!conv2d_40/StatefulPartitionedCall?!conv2d_41/StatefulPartitionedCall?!conv2d_42/StatefulPartitionedCall?
!conv2d_40/StatefulPartitionedCallStatefulPartitionedCallinput_9conv2d_40_778867conv2d_40_778869*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_conv2d_40_layer_call_and_return_conditional_losses_7782612#
!conv2d_40/StatefulPartitionedCall?
.batch_normalization_28/StatefulPartitionedCallStatefulPartitionedCall*conv2d_40/StatefulPartitionedCall:output:0batch_normalization_28_778872batch_normalization_28_778874batch_normalization_28_778876batch_normalization_28_778878*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *[
fVRT
R__inference_batch_normalization_28_layer_call_and_return_conditional_losses_77828420
.batch_normalization_28/StatefulPartitionedCall?
activation_28/PartitionedCallPartitionedCall7batch_normalization_28/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *R
fMRK
I__inference_activation_28_layer_call_and_return_conditional_losses_7782992
activation_28/PartitionedCall?
dropout_36/PartitionedCallPartitionedCall&activation_28/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_dropout_36_layer_call_and_return_conditional_losses_7783062
dropout_36/PartitionedCall?
!conv2d_41/StatefulPartitionedCallStatefulPartitionedCall#dropout_36/PartitionedCall:output:0conv2d_41_778883conv2d_41_778885*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_conv2d_41_layer_call_and_return_conditional_losses_7783182#
!conv2d_41/StatefulPartitionedCall?
.batch_normalization_29/StatefulPartitionedCallStatefulPartitionedCall*conv2d_41/StatefulPartitionedCall:output:0batch_normalization_29_778888batch_normalization_29_778890batch_normalization_29_778892batch_normalization_29_778894*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *[
fVRT
R__inference_batch_normalization_29_layer_call_and_return_conditional_losses_77834120
.batch_normalization_29/StatefulPartitionedCall?
activation_29/PartitionedCallPartitionedCall7batch_normalization_29/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *R
fMRK
I__inference_activation_29_layer_call_and_return_conditional_losses_7783562
activation_29/PartitionedCall?
dropout_37/PartitionedCallPartitionedCall&activation_29/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_dropout_37_layer_call_and_return_conditional_losses_7783632
dropout_37/PartitionedCall?
!conv2d_42/StatefulPartitionedCallStatefulPartitionedCall#dropout_37/PartitionedCall:output:0conv2d_42_778899conv2d_42_778901*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_conv2d_42_layer_call_and_return_conditional_losses_7783752#
!conv2d_42/StatefulPartitionedCall?
.batch_normalization_30/StatefulPartitionedCallStatefulPartitionedCall*conv2d_42/StatefulPartitionedCall:output:0batch_normalization_30_778904batch_normalization_30_778906batch_normalization_30_778908batch_normalization_30_778910*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *[
fVRT
R__inference_batch_normalization_30_layer_call_and_return_conditional_losses_77839820
.batch_normalization_30/StatefulPartitionedCall?
tf.__operators__.add_12/AddV2AddV27batch_normalization_30/StatefulPartitionedCall:output:0#dropout_36/PartitionedCall:output:0*
T0*0
_output_shapes
:??????????2
tf.__operators__.add_12/AddV2?
activation_30/PartitionedCallPartitionedCall!tf.__operators__.add_12/AddV2:z:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *R
fMRK
I__inference_activation_30_layer_call_and_return_conditional_losses_7784142
activation_30/PartitionedCall?
dropout_38/PartitionedCallPartitionedCall&activation_30/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_dropout_38_layer_call_and_return_conditional_losses_7784212
dropout_38/PartitionedCall?
IdentityIdentity#dropout_38/PartitionedCall:output:0^NoOp*
T0*0
_output_shapes
:??????????2

Identity?
NoOpNoOp/^batch_normalization_28/StatefulPartitionedCall/^batch_normalization_29/StatefulPartitionedCall/^batch_normalization_30/StatefulPartitionedCall"^conv2d_40/StatefulPartitionedCall"^conv2d_41/StatefulPartitionedCall"^conv2d_42/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*R
_input_shapesA
?:?????????: : : : : : : : : : : : : : : : : : 2`
.batch_normalization_28/StatefulPartitionedCall.batch_normalization_28/StatefulPartitionedCall2`
.batch_normalization_29/StatefulPartitionedCall.batch_normalization_29/StatefulPartitionedCall2`
.batch_normalization_30/StatefulPartitionedCall.batch_normalization_30/StatefulPartitionedCall2F
!conv2d_40/StatefulPartitionedCall!conv2d_40/StatefulPartitionedCall2F
!conv2d_41/StatefulPartitionedCall!conv2d_41/StatefulPartitionedCall2F
!conv2d_42/StatefulPartitionedCall!conv2d_42/StatefulPartitionedCall:X T
/
_output_shapes
:?????????
!
_user_specified_name	input_9
?
G
+__inference_dropout_36_layer_call_fn_780463

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_dropout_36_layer_call_and_return_conditional_losses_7783062
PartitionedCallu
IdentityIdentityPartitionedCall:output:0*
T0*0
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:??????????:X T
0
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
E__inference_conv2d_41_layer_call_and_return_conditional_losses_780496

inputs:
conv2d_readvariableop_resource:??.
biasadd_readvariableop_resource:	?
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:??*
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????*
paddingSAME*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????2	
BiasAddt
IdentityIdentityBiasAdd:output:0^NoOp*
T0*0
_output_shapes
:??????????2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :??????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
E__inference_conv2d_42_layer_call_and_return_conditional_losses_780668

inputs:
conv2d_readvariableop_resource:??.
biasadd_readvariableop_resource:	?
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:??*
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????*
paddingSAME*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????2	
BiasAddt
IdentityIdentityBiasAdd:output:0^NoOp*
T0*0
_output_shapes
:??????????2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :??????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:??????????
 
_user_specified_nameinputs
?	
?
7__inference_batch_normalization_29_layer_call_fn_780535

inputs
unknown:	?
	unknown_0:	?
	unknown_1:	?
	unknown_2:	?
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *[
fVRT
R__inference_batch_normalization_29_layer_call_and_return_conditional_losses_7783412
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*0
_output_shapes
:??????????2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*7
_input_shapes&
$:??????????: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:??????????
 
_user_specified_nameinputs
?
G
+__inference_dropout_38_layer_call_fn_780807

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_dropout_38_layer_call_and_return_conditional_losses_7784212
PartitionedCallu
IdentityIdentityPartitionedCall:output:0*
T0*0
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:??????????:X T
0
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
E__inference_conv2d_40_layer_call_and_return_conditional_losses_780324

inputs9
conv2d_readvariableop_resource:?.
biasadd_readvariableop_resource:	?
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*'
_output_shapes
:?*
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????*
paddingSAME*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????2	
BiasAddt
IdentityIdentityBiasAdd:output:0^NoOp*
T0*0
_output_shapes
:??????????2

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
?
?
*__inference_conv2d_42_layer_call_fn_780658

inputs#
unknown:??
	unknown_0:	?
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_conv2d_42_layer_call_and_return_conditional_losses_7783752
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*0
_output_shapes
:??????????2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :??????????: : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:??????????
 
_user_specified_nameinputs
?
b
F__inference_dropout_44_layer_call_and_return_conditional_losses_779192

inputs
identityc
IdentityIdentityinputs*
T0*0
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:??????????:X T
0
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
C__inference_dense_5_layer_call_and_return_conditional_losses_780285

inputs0
matmul_readvariableop_resource:@
identity??MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:@*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
MatMulX
TanhTanhMatMul:product:0*
T0*'
_output_shapes
:?????????2
Tanhc
IdentityIdentityTanh:y:0^NoOp*
T0*'
_output_shapes
:?????????2

Identityf
NoOpNoOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*(
_input_shapes
:?????????@: 2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:?????????@
 
_user_specified_nameinputs
?
?
R__inference_batch_normalization_30_layer_call_and_return_conditional_losses_780756

inputs&
readvariableop_resource:	?(
readvariableop_1_resource:	?7
(fusedbatchnormv3_readvariableop_resource:	?9
*fusedbatchnormv3_readvariableop_1_resource:	?
identity??AssignNewValue?AssignNewValue_1?FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1u
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:?*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:?*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:?*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:?*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,????????????????????????????:?:?:?:?:*
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
T0*B
_output_shapes0
.:,????????????????????????????2

Identity?
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,????????????????????????????: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:j f
B
_output_shapes0
.:,????????????????????????????
 
_user_specified_nameinputs
?
?
(__inference_model_8_layer_call_fn_780181

inputs"
unknown:?
	unknown_0:	?
	unknown_1:	?
	unknown_2:	?
	unknown_3:	?
	unknown_4:	?%
	unknown_5:??
	unknown_6:	?
	unknown_7:	?
	unknown_8:	?
	unknown_9:	?

unknown_10:	?&

unknown_11:??

unknown_12:	?

unknown_13:	?

unknown_14:	?

unknown_15:	?

unknown_16:	?
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
 *0
_output_shapes
:??????????*.
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_model_8_layer_call_and_return_conditional_losses_7787842
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*0
_output_shapes
:??????????2

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
?
?
R__inference_batch_normalization_30_layer_call_and_return_conditional_losses_780738

inputs&
readvariableop_resource:	?(
readvariableop_1_resource:	?7
(fusedbatchnormv3_readvariableop_resource:	?9
*fusedbatchnormv3_readvariableop_1_resource:	?
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1u
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:?*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:?*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:?*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:?*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,????????????????????????????:?:?:?:?:*
epsilon%o?:*
is_training( 2
FusedBatchNormV3?
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*B
_output_shapes0
.:,????????????????????????????2

Identity?
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,????????????????????????????: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:j f
B
_output_shapes0
.:,????????????????????????????
 
_user_specified_nameinputs
?
?
$__inference_signature_wrapper_779649
input_12"
unknown:?
	unknown_0:	?
	unknown_1:	?
	unknown_2:	?
	unknown_3:	?
	unknown_4:	?%
	unknown_5:??
	unknown_6:	?
	unknown_7:	?
	unknown_8:	?
	unknown_9:	?

unknown_10:	?&

unknown_11:??

unknown_12:	?

unknown_13:	?

unknown_14:	?

unknown_15:	?

unknown_16:	?%

unknown_17:?

unknown_18:&

unknown_19:??

unknown_20:	?%

unknown_21:?I

unknown_22:I

unknown_23:@
identity

identity_1??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_12unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
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
unknown_22
unknown_23*%
Tin
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:?????????I:?????????*;
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8? **
f%R#
!__inference__wrapped_model_7778662
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
_construction_contextkEagerRuntime*`
_input_shapesO
M:?????????: : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Y U
/
_output_shapes
:?????????
"
_user_specified_name
input_12
?
?
R__inference_batch_normalization_28_layer_call_and_return_conditional_losses_780448

inputs&
readvariableop_resource:	?(
readvariableop_1_resource:	?7
(fusedbatchnormv3_readvariableop_resource:	?9
*fusedbatchnormv3_readvariableop_1_resource:	?
identity??AssignNewValue?AssignNewValue_1?FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1u
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:?*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:?*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:?*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:?*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:??????????:?:?:?:?:*
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
AssignNewValue_1x
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*0
_output_shapes
:??????????2

Identity?
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*7
_input_shapes&
$:??????????: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:X T
0
_output_shapes
:??????????
 
_user_specified_nameinputs
?t
?
C__inference_model_8_layer_call_and_return_conditional_losses_780099

inputsC
(conv2d_40_conv2d_readvariableop_resource:?8
)conv2d_40_biasadd_readvariableop_resource:	?=
.batch_normalization_28_readvariableop_resource:	??
0batch_normalization_28_readvariableop_1_resource:	?N
?batch_normalization_28_fusedbatchnormv3_readvariableop_resource:	?P
Abatch_normalization_28_fusedbatchnormv3_readvariableop_1_resource:	?D
(conv2d_41_conv2d_readvariableop_resource:??8
)conv2d_41_biasadd_readvariableop_resource:	?=
.batch_normalization_29_readvariableop_resource:	??
0batch_normalization_29_readvariableop_1_resource:	?N
?batch_normalization_29_fusedbatchnormv3_readvariableop_resource:	?P
Abatch_normalization_29_fusedbatchnormv3_readvariableop_1_resource:	?D
(conv2d_42_conv2d_readvariableop_resource:??8
)conv2d_42_biasadd_readvariableop_resource:	?=
.batch_normalization_30_readvariableop_resource:	??
0batch_normalization_30_readvariableop_1_resource:	?N
?batch_normalization_30_fusedbatchnormv3_readvariableop_resource:	?P
Abatch_normalization_30_fusedbatchnormv3_readvariableop_1_resource:	?
identity??%batch_normalization_28/AssignNewValue?'batch_normalization_28/AssignNewValue_1?6batch_normalization_28/FusedBatchNormV3/ReadVariableOp?8batch_normalization_28/FusedBatchNormV3/ReadVariableOp_1?%batch_normalization_28/ReadVariableOp?'batch_normalization_28/ReadVariableOp_1?%batch_normalization_29/AssignNewValue?'batch_normalization_29/AssignNewValue_1?6batch_normalization_29/FusedBatchNormV3/ReadVariableOp?8batch_normalization_29/FusedBatchNormV3/ReadVariableOp_1?%batch_normalization_29/ReadVariableOp?'batch_normalization_29/ReadVariableOp_1?%batch_normalization_30/AssignNewValue?'batch_normalization_30/AssignNewValue_1?6batch_normalization_30/FusedBatchNormV3/ReadVariableOp?8batch_normalization_30/FusedBatchNormV3/ReadVariableOp_1?%batch_normalization_30/ReadVariableOp?'batch_normalization_30/ReadVariableOp_1? conv2d_40/BiasAdd/ReadVariableOp?conv2d_40/Conv2D/ReadVariableOp? conv2d_41/BiasAdd/ReadVariableOp?conv2d_41/Conv2D/ReadVariableOp? conv2d_42/BiasAdd/ReadVariableOp?conv2d_42/Conv2D/ReadVariableOp?
conv2d_40/Conv2D/ReadVariableOpReadVariableOp(conv2d_40_conv2d_readvariableop_resource*'
_output_shapes
:?*
dtype02!
conv2d_40/Conv2D/ReadVariableOp?
conv2d_40/Conv2DConv2Dinputs'conv2d_40/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????*
paddingSAME*
strides
2
conv2d_40/Conv2D?
 conv2d_40/BiasAdd/ReadVariableOpReadVariableOp)conv2d_40_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02"
 conv2d_40/BiasAdd/ReadVariableOp?
conv2d_40/BiasAddBiasAddconv2d_40/Conv2D:output:0(conv2d_40/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????2
conv2d_40/BiasAdd?
%batch_normalization_28/ReadVariableOpReadVariableOp.batch_normalization_28_readvariableop_resource*
_output_shapes	
:?*
dtype02'
%batch_normalization_28/ReadVariableOp?
'batch_normalization_28/ReadVariableOp_1ReadVariableOp0batch_normalization_28_readvariableop_1_resource*
_output_shapes	
:?*
dtype02)
'batch_normalization_28/ReadVariableOp_1?
6batch_normalization_28/FusedBatchNormV3/ReadVariableOpReadVariableOp?batch_normalization_28_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:?*
dtype028
6batch_normalization_28/FusedBatchNormV3/ReadVariableOp?
8batch_normalization_28/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpAbatch_normalization_28_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:?*
dtype02:
8batch_normalization_28/FusedBatchNormV3/ReadVariableOp_1?
'batch_normalization_28/FusedBatchNormV3FusedBatchNormV3conv2d_40/BiasAdd:output:0-batch_normalization_28/ReadVariableOp:value:0/batch_normalization_28/ReadVariableOp_1:value:0>batch_normalization_28/FusedBatchNormV3/ReadVariableOp:value:0@batch_normalization_28/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:??????????:?:?:?:?:*
epsilon%o?:*
exponential_avg_factor%
?#<2)
'batch_normalization_28/FusedBatchNormV3?
%batch_normalization_28/AssignNewValueAssignVariableOp?batch_normalization_28_fusedbatchnormv3_readvariableop_resource4batch_normalization_28/FusedBatchNormV3:batch_mean:07^batch_normalization_28/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype02'
%batch_normalization_28/AssignNewValue?
'batch_normalization_28/AssignNewValue_1AssignVariableOpAbatch_normalization_28_fusedbatchnormv3_readvariableop_1_resource8batch_normalization_28/FusedBatchNormV3:batch_variance:09^batch_normalization_28/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype02)
'batch_normalization_28/AssignNewValue_1?
activation_28/ReluRelu+batch_normalization_28/FusedBatchNormV3:y:0*
T0*0
_output_shapes
:??????????2
activation_28/Relu?
conv2d_41/Conv2D/ReadVariableOpReadVariableOp(conv2d_41_conv2d_readvariableop_resource*(
_output_shapes
:??*
dtype02!
conv2d_41/Conv2D/ReadVariableOp?
conv2d_41/Conv2DConv2D activation_28/Relu:activations:0'conv2d_41/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????*
paddingSAME*
strides
2
conv2d_41/Conv2D?
 conv2d_41/BiasAdd/ReadVariableOpReadVariableOp)conv2d_41_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02"
 conv2d_41/BiasAdd/ReadVariableOp?
conv2d_41/BiasAddBiasAddconv2d_41/Conv2D:output:0(conv2d_41/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????2
conv2d_41/BiasAdd?
%batch_normalization_29/ReadVariableOpReadVariableOp.batch_normalization_29_readvariableop_resource*
_output_shapes	
:?*
dtype02'
%batch_normalization_29/ReadVariableOp?
'batch_normalization_29/ReadVariableOp_1ReadVariableOp0batch_normalization_29_readvariableop_1_resource*
_output_shapes	
:?*
dtype02)
'batch_normalization_29/ReadVariableOp_1?
6batch_normalization_29/FusedBatchNormV3/ReadVariableOpReadVariableOp?batch_normalization_29_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:?*
dtype028
6batch_normalization_29/FusedBatchNormV3/ReadVariableOp?
8batch_normalization_29/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpAbatch_normalization_29_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:?*
dtype02:
8batch_normalization_29/FusedBatchNormV3/ReadVariableOp_1?
'batch_normalization_29/FusedBatchNormV3FusedBatchNormV3conv2d_41/BiasAdd:output:0-batch_normalization_29/ReadVariableOp:value:0/batch_normalization_29/ReadVariableOp_1:value:0>batch_normalization_29/FusedBatchNormV3/ReadVariableOp:value:0@batch_normalization_29/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:??????????:?:?:?:?:*
epsilon%o?:*
exponential_avg_factor%
?#<2)
'batch_normalization_29/FusedBatchNormV3?
%batch_normalization_29/AssignNewValueAssignVariableOp?batch_normalization_29_fusedbatchnormv3_readvariableop_resource4batch_normalization_29/FusedBatchNormV3:batch_mean:07^batch_normalization_29/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype02'
%batch_normalization_29/AssignNewValue?
'batch_normalization_29/AssignNewValue_1AssignVariableOpAbatch_normalization_29_fusedbatchnormv3_readvariableop_1_resource8batch_normalization_29/FusedBatchNormV3:batch_variance:09^batch_normalization_29/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype02)
'batch_normalization_29/AssignNewValue_1?
activation_29/ReluRelu+batch_normalization_29/FusedBatchNormV3:y:0*
T0*0
_output_shapes
:??????????2
activation_29/Relu?
conv2d_42/Conv2D/ReadVariableOpReadVariableOp(conv2d_42_conv2d_readvariableop_resource*(
_output_shapes
:??*
dtype02!
conv2d_42/Conv2D/ReadVariableOp?
conv2d_42/Conv2DConv2D activation_29/Relu:activations:0'conv2d_42/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????*
paddingSAME*
strides
2
conv2d_42/Conv2D?
 conv2d_42/BiasAdd/ReadVariableOpReadVariableOp)conv2d_42_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02"
 conv2d_42/BiasAdd/ReadVariableOp?
conv2d_42/BiasAddBiasAddconv2d_42/Conv2D:output:0(conv2d_42/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????2
conv2d_42/BiasAdd?
%batch_normalization_30/ReadVariableOpReadVariableOp.batch_normalization_30_readvariableop_resource*
_output_shapes	
:?*
dtype02'
%batch_normalization_30/ReadVariableOp?
'batch_normalization_30/ReadVariableOp_1ReadVariableOp0batch_normalization_30_readvariableop_1_resource*
_output_shapes	
:?*
dtype02)
'batch_normalization_30/ReadVariableOp_1?
6batch_normalization_30/FusedBatchNormV3/ReadVariableOpReadVariableOp?batch_normalization_30_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:?*
dtype028
6batch_normalization_30/FusedBatchNormV3/ReadVariableOp?
8batch_normalization_30/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpAbatch_normalization_30_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:?*
dtype02:
8batch_normalization_30/FusedBatchNormV3/ReadVariableOp_1?
'batch_normalization_30/FusedBatchNormV3FusedBatchNormV3conv2d_42/BiasAdd:output:0-batch_normalization_30/ReadVariableOp:value:0/batch_normalization_30/ReadVariableOp_1:value:0>batch_normalization_30/FusedBatchNormV3/ReadVariableOp:value:0@batch_normalization_30/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:??????????:?:?:?:?:*
epsilon%o?:*
exponential_avg_factor%
?#<2)
'batch_normalization_30/FusedBatchNormV3?
%batch_normalization_30/AssignNewValueAssignVariableOp?batch_normalization_30_fusedbatchnormv3_readvariableop_resource4batch_normalization_30/FusedBatchNormV3:batch_mean:07^batch_normalization_30/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype02'
%batch_normalization_30/AssignNewValue?
'batch_normalization_30/AssignNewValue_1AssignVariableOpAbatch_normalization_30_fusedbatchnormv3_readvariableop_1_resource8batch_normalization_30/FusedBatchNormV3:batch_variance:09^batch_normalization_30/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype02)
'batch_normalization_30/AssignNewValue_1?
tf.__operators__.add_12/AddV2AddV2+batch_normalization_30/FusedBatchNormV3:y:0 activation_28/Relu:activations:0*
T0*0
_output_shapes
:??????????2
tf.__operators__.add_12/AddV2?
activation_30/ReluRelu!tf.__operators__.add_12/AddV2:z:0*
T0*0
_output_shapes
:??????????2
activation_30/Relu?
IdentityIdentity activation_30/Relu:activations:0^NoOp*
T0*0
_output_shapes
:??????????2

Identity?
NoOpNoOp&^batch_normalization_28/AssignNewValue(^batch_normalization_28/AssignNewValue_17^batch_normalization_28/FusedBatchNormV3/ReadVariableOp9^batch_normalization_28/FusedBatchNormV3/ReadVariableOp_1&^batch_normalization_28/ReadVariableOp(^batch_normalization_28/ReadVariableOp_1&^batch_normalization_29/AssignNewValue(^batch_normalization_29/AssignNewValue_17^batch_normalization_29/FusedBatchNormV3/ReadVariableOp9^batch_normalization_29/FusedBatchNormV3/ReadVariableOp_1&^batch_normalization_29/ReadVariableOp(^batch_normalization_29/ReadVariableOp_1&^batch_normalization_30/AssignNewValue(^batch_normalization_30/AssignNewValue_17^batch_normalization_30/FusedBatchNormV3/ReadVariableOp9^batch_normalization_30/FusedBatchNormV3/ReadVariableOp_1&^batch_normalization_30/ReadVariableOp(^batch_normalization_30/ReadVariableOp_1!^conv2d_40/BiasAdd/ReadVariableOp ^conv2d_40/Conv2D/ReadVariableOp!^conv2d_41/BiasAdd/ReadVariableOp ^conv2d_41/Conv2D/ReadVariableOp!^conv2d_42/BiasAdd/ReadVariableOp ^conv2d_42/Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*R
_input_shapesA
?:?????????: : : : : : : : : : : : : : : : : : 2N
%batch_normalization_28/AssignNewValue%batch_normalization_28/AssignNewValue2R
'batch_normalization_28/AssignNewValue_1'batch_normalization_28/AssignNewValue_12p
6batch_normalization_28/FusedBatchNormV3/ReadVariableOp6batch_normalization_28/FusedBatchNormV3/ReadVariableOp2t
8batch_normalization_28/FusedBatchNormV3/ReadVariableOp_18batch_normalization_28/FusedBatchNormV3/ReadVariableOp_12N
%batch_normalization_28/ReadVariableOp%batch_normalization_28/ReadVariableOp2R
'batch_normalization_28/ReadVariableOp_1'batch_normalization_28/ReadVariableOp_12N
%batch_normalization_29/AssignNewValue%batch_normalization_29/AssignNewValue2R
'batch_normalization_29/AssignNewValue_1'batch_normalization_29/AssignNewValue_12p
6batch_normalization_29/FusedBatchNormV3/ReadVariableOp6batch_normalization_29/FusedBatchNormV3/ReadVariableOp2t
8batch_normalization_29/FusedBatchNormV3/ReadVariableOp_18batch_normalization_29/FusedBatchNormV3/ReadVariableOp_12N
%batch_normalization_29/ReadVariableOp%batch_normalization_29/ReadVariableOp2R
'batch_normalization_29/ReadVariableOp_1'batch_normalization_29/ReadVariableOp_12N
%batch_normalization_30/AssignNewValue%batch_normalization_30/AssignNewValue2R
'batch_normalization_30/AssignNewValue_1'batch_normalization_30/AssignNewValue_12p
6batch_normalization_30/FusedBatchNormV3/ReadVariableOp6batch_normalization_30/FusedBatchNormV3/ReadVariableOp2t
8batch_normalization_30/FusedBatchNormV3/ReadVariableOp_18batch_normalization_30/FusedBatchNormV3/ReadVariableOp_12N
%batch_normalization_30/ReadVariableOp%batch_normalization_30/ReadVariableOp2R
'batch_normalization_30/ReadVariableOp_1'batch_normalization_30/ReadVariableOp_12D
 conv2d_40/BiasAdd/ReadVariableOp conv2d_40/BiasAdd/ReadVariableOp2B
conv2d_40/Conv2D/ReadVariableOpconv2d_40/Conv2D/ReadVariableOp2D
 conv2d_41/BiasAdd/ReadVariableOp conv2d_41/BiasAdd/ReadVariableOp2B
conv2d_41/Conv2D/ReadVariableOpconv2d_41/Conv2D/ReadVariableOp2D
 conv2d_42/BiasAdd/ReadVariableOp conv2d_42/BiasAdd/ReadVariableOp2B
conv2d_42/Conv2D/ReadVariableOpconv2d_42/Conv2D/ReadVariableOp:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
??
?
D__inference_model_11_layer_call_and_return_conditional_losses_779846

inputsK
0model_8_conv2d_40_conv2d_readvariableop_resource:?@
1model_8_conv2d_40_biasadd_readvariableop_resource:	?E
6model_8_batch_normalization_28_readvariableop_resource:	?G
8model_8_batch_normalization_28_readvariableop_1_resource:	?V
Gmodel_8_batch_normalization_28_fusedbatchnormv3_readvariableop_resource:	?X
Imodel_8_batch_normalization_28_fusedbatchnormv3_readvariableop_1_resource:	?L
0model_8_conv2d_41_conv2d_readvariableop_resource:??@
1model_8_conv2d_41_biasadd_readvariableop_resource:	?E
6model_8_batch_normalization_29_readvariableop_resource:	?G
8model_8_batch_normalization_29_readvariableop_1_resource:	?V
Gmodel_8_batch_normalization_29_fusedbatchnormv3_readvariableop_resource:	?X
Imodel_8_batch_normalization_29_fusedbatchnormv3_readvariableop_1_resource:	?L
0model_8_conv2d_42_conv2d_readvariableop_resource:??@
1model_8_conv2d_42_biasadd_readvariableop_resource:	?E
6model_8_batch_normalization_30_readvariableop_resource:	?G
8model_8_batch_normalization_30_readvariableop_1_resource:	?V
Gmodel_8_batch_normalization_30_fusedbatchnormv3_readvariableop_resource:	?X
Imodel_8_batch_normalization_30_fusedbatchnormv3_readvariableop_1_resource:	?C
(conv2d_51_conv2d_readvariableop_resource:?7
)conv2d_51_biasadd_readvariableop_resource:D
(conv2d_49_conv2d_readvariableop_resource:??8
)conv2d_49_biasadd_readvariableop_resource:	?C
(conv2d_50_conv2d_readvariableop_resource:?I7
)conv2d_50_biasadd_readvariableop_resource:I8
&dense_5_matmul_readvariableop_resource:@
identity

identity_1?? conv2d_49/BiasAdd/ReadVariableOp?conv2d_49/Conv2D/ReadVariableOp? conv2d_50/BiasAdd/ReadVariableOp?conv2d_50/Conv2D/ReadVariableOp? conv2d_51/BiasAdd/ReadVariableOp?conv2d_51/Conv2D/ReadVariableOp?dense_5/MatMul/ReadVariableOp?-model_8/batch_normalization_28/AssignNewValue?/model_8/batch_normalization_28/AssignNewValue_1?>model_8/batch_normalization_28/FusedBatchNormV3/ReadVariableOp?@model_8/batch_normalization_28/FusedBatchNormV3/ReadVariableOp_1?-model_8/batch_normalization_28/ReadVariableOp?/model_8/batch_normalization_28/ReadVariableOp_1?-model_8/batch_normalization_29/AssignNewValue?/model_8/batch_normalization_29/AssignNewValue_1?>model_8/batch_normalization_29/FusedBatchNormV3/ReadVariableOp?@model_8/batch_normalization_29/FusedBatchNormV3/ReadVariableOp_1?-model_8/batch_normalization_29/ReadVariableOp?/model_8/batch_normalization_29/ReadVariableOp_1?-model_8/batch_normalization_30/AssignNewValue?/model_8/batch_normalization_30/AssignNewValue_1?>model_8/batch_normalization_30/FusedBatchNormV3/ReadVariableOp?@model_8/batch_normalization_30/FusedBatchNormV3/ReadVariableOp_1?-model_8/batch_normalization_30/ReadVariableOp?/model_8/batch_normalization_30/ReadVariableOp_1?(model_8/conv2d_40/BiasAdd/ReadVariableOp?'model_8/conv2d_40/Conv2D/ReadVariableOp?(model_8/conv2d_41/BiasAdd/ReadVariableOp?'model_8/conv2d_41/Conv2D/ReadVariableOp?(model_8/conv2d_42/BiasAdd/ReadVariableOp?'model_8/conv2d_42/Conv2D/ReadVariableOp?
'model_8/conv2d_40/Conv2D/ReadVariableOpReadVariableOp0model_8_conv2d_40_conv2d_readvariableop_resource*'
_output_shapes
:?*
dtype02)
'model_8/conv2d_40/Conv2D/ReadVariableOp?
model_8/conv2d_40/Conv2DConv2Dinputs/model_8/conv2d_40/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????*
paddingSAME*
strides
2
model_8/conv2d_40/Conv2D?
(model_8/conv2d_40/BiasAdd/ReadVariableOpReadVariableOp1model_8_conv2d_40_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02*
(model_8/conv2d_40/BiasAdd/ReadVariableOp?
model_8/conv2d_40/BiasAddBiasAdd!model_8/conv2d_40/Conv2D:output:00model_8/conv2d_40/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????2
model_8/conv2d_40/BiasAdd?
-model_8/batch_normalization_28/ReadVariableOpReadVariableOp6model_8_batch_normalization_28_readvariableop_resource*
_output_shapes	
:?*
dtype02/
-model_8/batch_normalization_28/ReadVariableOp?
/model_8/batch_normalization_28/ReadVariableOp_1ReadVariableOp8model_8_batch_normalization_28_readvariableop_1_resource*
_output_shapes	
:?*
dtype021
/model_8/batch_normalization_28/ReadVariableOp_1?
>model_8/batch_normalization_28/FusedBatchNormV3/ReadVariableOpReadVariableOpGmodel_8_batch_normalization_28_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:?*
dtype02@
>model_8/batch_normalization_28/FusedBatchNormV3/ReadVariableOp?
@model_8/batch_normalization_28/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpImodel_8_batch_normalization_28_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:?*
dtype02B
@model_8/batch_normalization_28/FusedBatchNormV3/ReadVariableOp_1?
/model_8/batch_normalization_28/FusedBatchNormV3FusedBatchNormV3"model_8/conv2d_40/BiasAdd:output:05model_8/batch_normalization_28/ReadVariableOp:value:07model_8/batch_normalization_28/ReadVariableOp_1:value:0Fmodel_8/batch_normalization_28/FusedBatchNormV3/ReadVariableOp:value:0Hmodel_8/batch_normalization_28/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:??????????:?:?:?:?:*
epsilon%o?:*
exponential_avg_factor%
?#<21
/model_8/batch_normalization_28/FusedBatchNormV3?
-model_8/batch_normalization_28/AssignNewValueAssignVariableOpGmodel_8_batch_normalization_28_fusedbatchnormv3_readvariableop_resource<model_8/batch_normalization_28/FusedBatchNormV3:batch_mean:0?^model_8/batch_normalization_28/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype02/
-model_8/batch_normalization_28/AssignNewValue?
/model_8/batch_normalization_28/AssignNewValue_1AssignVariableOpImodel_8_batch_normalization_28_fusedbatchnormv3_readvariableop_1_resource@model_8/batch_normalization_28/FusedBatchNormV3:batch_variance:0A^model_8/batch_normalization_28/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype021
/model_8/batch_normalization_28/AssignNewValue_1?
model_8/activation_28/ReluRelu3model_8/batch_normalization_28/FusedBatchNormV3:y:0*
T0*0
_output_shapes
:??????????2
model_8/activation_28/Relu?
'model_8/conv2d_41/Conv2D/ReadVariableOpReadVariableOp0model_8_conv2d_41_conv2d_readvariableop_resource*(
_output_shapes
:??*
dtype02)
'model_8/conv2d_41/Conv2D/ReadVariableOp?
model_8/conv2d_41/Conv2DConv2D(model_8/activation_28/Relu:activations:0/model_8/conv2d_41/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????*
paddingSAME*
strides
2
model_8/conv2d_41/Conv2D?
(model_8/conv2d_41/BiasAdd/ReadVariableOpReadVariableOp1model_8_conv2d_41_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02*
(model_8/conv2d_41/BiasAdd/ReadVariableOp?
model_8/conv2d_41/BiasAddBiasAdd!model_8/conv2d_41/Conv2D:output:00model_8/conv2d_41/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????2
model_8/conv2d_41/BiasAdd?
-model_8/batch_normalization_29/ReadVariableOpReadVariableOp6model_8_batch_normalization_29_readvariableop_resource*
_output_shapes	
:?*
dtype02/
-model_8/batch_normalization_29/ReadVariableOp?
/model_8/batch_normalization_29/ReadVariableOp_1ReadVariableOp8model_8_batch_normalization_29_readvariableop_1_resource*
_output_shapes	
:?*
dtype021
/model_8/batch_normalization_29/ReadVariableOp_1?
>model_8/batch_normalization_29/FusedBatchNormV3/ReadVariableOpReadVariableOpGmodel_8_batch_normalization_29_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:?*
dtype02@
>model_8/batch_normalization_29/FusedBatchNormV3/ReadVariableOp?
@model_8/batch_normalization_29/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpImodel_8_batch_normalization_29_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:?*
dtype02B
@model_8/batch_normalization_29/FusedBatchNormV3/ReadVariableOp_1?
/model_8/batch_normalization_29/FusedBatchNormV3FusedBatchNormV3"model_8/conv2d_41/BiasAdd:output:05model_8/batch_normalization_29/ReadVariableOp:value:07model_8/batch_normalization_29/ReadVariableOp_1:value:0Fmodel_8/batch_normalization_29/FusedBatchNormV3/ReadVariableOp:value:0Hmodel_8/batch_normalization_29/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:??????????:?:?:?:?:*
epsilon%o?:*
exponential_avg_factor%
?#<21
/model_8/batch_normalization_29/FusedBatchNormV3?
-model_8/batch_normalization_29/AssignNewValueAssignVariableOpGmodel_8_batch_normalization_29_fusedbatchnormv3_readvariableop_resource<model_8/batch_normalization_29/FusedBatchNormV3:batch_mean:0?^model_8/batch_normalization_29/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype02/
-model_8/batch_normalization_29/AssignNewValue?
/model_8/batch_normalization_29/AssignNewValue_1AssignVariableOpImodel_8_batch_normalization_29_fusedbatchnormv3_readvariableop_1_resource@model_8/batch_normalization_29/FusedBatchNormV3:batch_variance:0A^model_8/batch_normalization_29/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype021
/model_8/batch_normalization_29/AssignNewValue_1?
model_8/activation_29/ReluRelu3model_8/batch_normalization_29/FusedBatchNormV3:y:0*
T0*0
_output_shapes
:??????????2
model_8/activation_29/Relu?
'model_8/conv2d_42/Conv2D/ReadVariableOpReadVariableOp0model_8_conv2d_42_conv2d_readvariableop_resource*(
_output_shapes
:??*
dtype02)
'model_8/conv2d_42/Conv2D/ReadVariableOp?
model_8/conv2d_42/Conv2DConv2D(model_8/activation_29/Relu:activations:0/model_8/conv2d_42/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????*
paddingSAME*
strides
2
model_8/conv2d_42/Conv2D?
(model_8/conv2d_42/BiasAdd/ReadVariableOpReadVariableOp1model_8_conv2d_42_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02*
(model_8/conv2d_42/BiasAdd/ReadVariableOp?
model_8/conv2d_42/BiasAddBiasAdd!model_8/conv2d_42/Conv2D:output:00model_8/conv2d_42/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????2
model_8/conv2d_42/BiasAdd?
-model_8/batch_normalization_30/ReadVariableOpReadVariableOp6model_8_batch_normalization_30_readvariableop_resource*
_output_shapes	
:?*
dtype02/
-model_8/batch_normalization_30/ReadVariableOp?
/model_8/batch_normalization_30/ReadVariableOp_1ReadVariableOp8model_8_batch_normalization_30_readvariableop_1_resource*
_output_shapes	
:?*
dtype021
/model_8/batch_normalization_30/ReadVariableOp_1?
>model_8/batch_normalization_30/FusedBatchNormV3/ReadVariableOpReadVariableOpGmodel_8_batch_normalization_30_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:?*
dtype02@
>model_8/batch_normalization_30/FusedBatchNormV3/ReadVariableOp?
@model_8/batch_normalization_30/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpImodel_8_batch_normalization_30_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:?*
dtype02B
@model_8/batch_normalization_30/FusedBatchNormV3/ReadVariableOp_1?
/model_8/batch_normalization_30/FusedBatchNormV3FusedBatchNormV3"model_8/conv2d_42/BiasAdd:output:05model_8/batch_normalization_30/ReadVariableOp:value:07model_8/batch_normalization_30/ReadVariableOp_1:value:0Fmodel_8/batch_normalization_30/FusedBatchNormV3/ReadVariableOp:value:0Hmodel_8/batch_normalization_30/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:??????????:?:?:?:?:*
epsilon%o?:*
exponential_avg_factor%
?#<21
/model_8/batch_normalization_30/FusedBatchNormV3?
-model_8/batch_normalization_30/AssignNewValueAssignVariableOpGmodel_8_batch_normalization_30_fusedbatchnormv3_readvariableop_resource<model_8/batch_normalization_30/FusedBatchNormV3:batch_mean:0?^model_8/batch_normalization_30/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype02/
-model_8/batch_normalization_30/AssignNewValue?
/model_8/batch_normalization_30/AssignNewValue_1AssignVariableOpImodel_8_batch_normalization_30_fusedbatchnormv3_readvariableop_1_resource@model_8/batch_normalization_30/FusedBatchNormV3:batch_variance:0A^model_8/batch_normalization_30/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype021
/model_8/batch_normalization_30/AssignNewValue_1?
%model_8/tf.__operators__.add_12/AddV2AddV23model_8/batch_normalization_30/FusedBatchNormV3:y:0(model_8/activation_28/Relu:activations:0*
T0*0
_output_shapes
:??????????2'
%model_8/tf.__operators__.add_12/AddV2?
model_8/activation_30/ReluRelu)model_8/tf.__operators__.add_12/AddV2:z:0*
T0*0
_output_shapes
:??????????2
model_8/activation_30/Relu?
conv2d_51/Conv2D/ReadVariableOpReadVariableOp(conv2d_51_conv2d_readvariableop_resource*'
_output_shapes
:?*
dtype02!
conv2d_51/Conv2D/ReadVariableOp?
conv2d_51/Conv2DConv2D(model_8/activation_30/Relu:activations:0'conv2d_51/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????*
paddingVALID*
strides
2
conv2d_51/Conv2D?
 conv2d_51/BiasAdd/ReadVariableOpReadVariableOp)conv2d_51_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02"
 conv2d_51/BiasAdd/ReadVariableOp?
conv2d_51/BiasAddBiasAddconv2d_51/Conv2D:output:0(conv2d_51/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????2
conv2d_51/BiasAdd~
conv2d_51/ReluReluconv2d_51/BiasAdd:output:0*
T0*/
_output_shapes
:?????????2
conv2d_51/Relu?
conv2d_49/Conv2D/ReadVariableOpReadVariableOp(conv2d_49_conv2d_readvariableop_resource*(
_output_shapes
:??*
dtype02!
conv2d_49/Conv2D/ReadVariableOp?
conv2d_49/Conv2DConv2D(model_8/activation_30/Relu:activations:0'conv2d_49/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????*
paddingSAME*
strides
2
conv2d_49/Conv2D?
 conv2d_49/BiasAdd/ReadVariableOpReadVariableOp)conv2d_49_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02"
 conv2d_49/BiasAdd/ReadVariableOp?
conv2d_49/BiasAddBiasAddconv2d_49/Conv2D:output:0(conv2d_49/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????2
conv2d_49/BiasAdd
conv2d_49/ReluReluconv2d_49/BiasAdd:output:0*
T0*0
_output_shapes
:??????????2
conv2d_49/Relus
flatten_5/ConstConst*
_output_shapes
:*
dtype0*
valueB"????@   2
flatten_5/Const?
flatten_5/ReshapeReshapeconv2d_51/Relu:activations:0flatten_5/Const:output:0*
T0*'
_output_shapes
:?????????@2
flatten_5/Reshape?
conv2d_50/Conv2D/ReadVariableOpReadVariableOp(conv2d_50_conv2d_readvariableop_resource*'
_output_shapes
:?I*
dtype02!
conv2d_50/Conv2D/ReadVariableOp?
conv2d_50/Conv2DConv2Dconv2d_49/Relu:activations:0'conv2d_50/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????I*
paddingSAME*
strides
2
conv2d_50/Conv2D?
 conv2d_50/BiasAdd/ReadVariableOpReadVariableOp)conv2d_50_biasadd_readvariableop_resource*
_output_shapes
:I*
dtype02"
 conv2d_50/BiasAdd/ReadVariableOp?
conv2d_50/BiasAddBiasAddconv2d_50/Conv2D:output:0(conv2d_50/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????I2
conv2d_50/BiasAdd?
conv2d_50/SoftmaxSoftmaxconv2d_50/BiasAdd:output:0*
T0*/
_output_shapes
:?????????I2
conv2d_50/Softmax?
dense_5/MatMul/ReadVariableOpReadVariableOp&dense_5_matmul_readvariableop_resource*
_output_shapes

:@*
dtype02
dense_5/MatMul/ReadVariableOp?
dense_5/MatMulMatMulflatten_5/Reshape:output:0%dense_5/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_5/MatMulp
dense_5/TanhTanhdense_5/MatMul:product:0*
T0*'
_output_shapes
:?????????2
dense_5/Tanhk
IdentityIdentitydense_5/Tanh:y:0^NoOp*
T0*'
_output_shapes
:?????????2

Identity?

Identity_1Identityconv2d_50/Softmax:softmax:0^NoOp*
T0*/
_output_shapes
:?????????I2

Identity_1?
NoOpNoOp!^conv2d_49/BiasAdd/ReadVariableOp ^conv2d_49/Conv2D/ReadVariableOp!^conv2d_50/BiasAdd/ReadVariableOp ^conv2d_50/Conv2D/ReadVariableOp!^conv2d_51/BiasAdd/ReadVariableOp ^conv2d_51/Conv2D/ReadVariableOp^dense_5/MatMul/ReadVariableOp.^model_8/batch_normalization_28/AssignNewValue0^model_8/batch_normalization_28/AssignNewValue_1?^model_8/batch_normalization_28/FusedBatchNormV3/ReadVariableOpA^model_8/batch_normalization_28/FusedBatchNormV3/ReadVariableOp_1.^model_8/batch_normalization_28/ReadVariableOp0^model_8/batch_normalization_28/ReadVariableOp_1.^model_8/batch_normalization_29/AssignNewValue0^model_8/batch_normalization_29/AssignNewValue_1?^model_8/batch_normalization_29/FusedBatchNormV3/ReadVariableOpA^model_8/batch_normalization_29/FusedBatchNormV3/ReadVariableOp_1.^model_8/batch_normalization_29/ReadVariableOp0^model_8/batch_normalization_29/ReadVariableOp_1.^model_8/batch_normalization_30/AssignNewValue0^model_8/batch_normalization_30/AssignNewValue_1?^model_8/batch_normalization_30/FusedBatchNormV3/ReadVariableOpA^model_8/batch_normalization_30/FusedBatchNormV3/ReadVariableOp_1.^model_8/batch_normalization_30/ReadVariableOp0^model_8/batch_normalization_30/ReadVariableOp_1)^model_8/conv2d_40/BiasAdd/ReadVariableOp(^model_8/conv2d_40/Conv2D/ReadVariableOp)^model_8/conv2d_41/BiasAdd/ReadVariableOp(^model_8/conv2d_41/Conv2D/ReadVariableOp)^model_8/conv2d_42/BiasAdd/ReadVariableOp(^model_8/conv2d_42/Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*`
_input_shapesO
M:?????????: : : : : : : : : : : : : : : : : : : : : : : : : 2D
 conv2d_49/BiasAdd/ReadVariableOp conv2d_49/BiasAdd/ReadVariableOp2B
conv2d_49/Conv2D/ReadVariableOpconv2d_49/Conv2D/ReadVariableOp2D
 conv2d_50/BiasAdd/ReadVariableOp conv2d_50/BiasAdd/ReadVariableOp2B
conv2d_50/Conv2D/ReadVariableOpconv2d_50/Conv2D/ReadVariableOp2D
 conv2d_51/BiasAdd/ReadVariableOp conv2d_51/BiasAdd/ReadVariableOp2B
conv2d_51/Conv2D/ReadVariableOpconv2d_51/Conv2D/ReadVariableOp2>
dense_5/MatMul/ReadVariableOpdense_5/MatMul/ReadVariableOp2^
-model_8/batch_normalization_28/AssignNewValue-model_8/batch_normalization_28/AssignNewValue2b
/model_8/batch_normalization_28/AssignNewValue_1/model_8/batch_normalization_28/AssignNewValue_12?
>model_8/batch_normalization_28/FusedBatchNormV3/ReadVariableOp>model_8/batch_normalization_28/FusedBatchNormV3/ReadVariableOp2?
@model_8/batch_normalization_28/FusedBatchNormV3/ReadVariableOp_1@model_8/batch_normalization_28/FusedBatchNormV3/ReadVariableOp_12^
-model_8/batch_normalization_28/ReadVariableOp-model_8/batch_normalization_28/ReadVariableOp2b
/model_8/batch_normalization_28/ReadVariableOp_1/model_8/batch_normalization_28/ReadVariableOp_12^
-model_8/batch_normalization_29/AssignNewValue-model_8/batch_normalization_29/AssignNewValue2b
/model_8/batch_normalization_29/AssignNewValue_1/model_8/batch_normalization_29/AssignNewValue_12?
>model_8/batch_normalization_29/FusedBatchNormV3/ReadVariableOp>model_8/batch_normalization_29/FusedBatchNormV3/ReadVariableOp2?
@model_8/batch_normalization_29/FusedBatchNormV3/ReadVariableOp_1@model_8/batch_normalization_29/FusedBatchNormV3/ReadVariableOp_12^
-model_8/batch_normalization_29/ReadVariableOp-model_8/batch_normalization_29/ReadVariableOp2b
/model_8/batch_normalization_29/ReadVariableOp_1/model_8/batch_normalization_29/ReadVariableOp_12^
-model_8/batch_normalization_30/AssignNewValue-model_8/batch_normalization_30/AssignNewValue2b
/model_8/batch_normalization_30/AssignNewValue_1/model_8/batch_normalization_30/AssignNewValue_12?
>model_8/batch_normalization_30/FusedBatchNormV3/ReadVariableOp>model_8/batch_normalization_30/FusedBatchNormV3/ReadVariableOp2?
@model_8/batch_normalization_30/FusedBatchNormV3/ReadVariableOp_1@model_8/batch_normalization_30/FusedBatchNormV3/ReadVariableOp_12^
-model_8/batch_normalization_30/ReadVariableOp-model_8/batch_normalization_30/ReadVariableOp2b
/model_8/batch_normalization_30/ReadVariableOp_1/model_8/batch_normalization_30/ReadVariableOp_12T
(model_8/conv2d_40/BiasAdd/ReadVariableOp(model_8/conv2d_40/BiasAdd/ReadVariableOp2R
'model_8/conv2d_40/Conv2D/ReadVariableOp'model_8/conv2d_40/Conv2D/ReadVariableOp2T
(model_8/conv2d_41/BiasAdd/ReadVariableOp(model_8/conv2d_41/BiasAdd/ReadVariableOp2R
'model_8/conv2d_41/Conv2D/ReadVariableOp'model_8/conv2d_41/Conv2D/ReadVariableOp2T
(model_8/conv2d_42/BiasAdd/ReadVariableOp(model_8/conv2d_42/BiasAdd/ReadVariableOp2R
'model_8/conv2d_42/Conv2D/ReadVariableOp'model_8/conv2d_42/Conv2D/ReadVariableOp:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?;
?
__inference__traced_save_780920
file_prefix/
+savev2_conv2d_51_kernel_read_readvariableop-
)savev2_conv2d_51_bias_read_readvariableop/
+savev2_conv2d_49_kernel_read_readvariableop-
)savev2_conv2d_49_bias_read_readvariableop-
)savev2_dense_5_kernel_read_readvariableop/
+savev2_conv2d_50_kernel_read_readvariableop-
)savev2_conv2d_50_bias_read_readvariableop/
+savev2_conv2d_40_kernel_read_readvariableop-
)savev2_conv2d_40_bias_read_readvariableop;
7savev2_batch_normalization_28_gamma_read_readvariableop:
6savev2_batch_normalization_28_beta_read_readvariableopA
=savev2_batch_normalization_28_moving_mean_read_readvariableopE
Asavev2_batch_normalization_28_moving_variance_read_readvariableop/
+savev2_conv2d_41_kernel_read_readvariableop-
)savev2_conv2d_41_bias_read_readvariableop;
7savev2_batch_normalization_29_gamma_read_readvariableop:
6savev2_batch_normalization_29_beta_read_readvariableopA
=savev2_batch_normalization_29_moving_mean_read_readvariableopE
Asavev2_batch_normalization_29_moving_variance_read_readvariableop/
+savev2_conv2d_42_kernel_read_readvariableop-
)savev2_conv2d_42_bias_read_readvariableop;
7savev2_batch_normalization_30_gamma_read_readvariableop:
6savev2_batch_normalization_30_beta_read_readvariableopA
=savev2_batch_normalization_30_moving_mean_read_readvariableopE
Asavev2_batch_normalization_30_moving_variance_read_readvariableop
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
:*
dtype0*?	
value?	B?	B6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB&variables/0/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB&variables/3/.ATTRIBUTES/VARIABLE_VALUEB&variables/4/.ATTRIBUTES/VARIABLE_VALUEB&variables/5/.ATTRIBUTES/VARIABLE_VALUEB&variables/6/.ATTRIBUTES/VARIABLE_VALUEB&variables/7/.ATTRIBUTES/VARIABLE_VALUEB&variables/8/.ATTRIBUTES/VARIABLE_VALUEB&variables/9/.ATTRIBUTES/VARIABLE_VALUEB'variables/10/.ATTRIBUTES/VARIABLE_VALUEB'variables/11/.ATTRIBUTES/VARIABLE_VALUEB'variables/12/.ATTRIBUTES/VARIABLE_VALUEB'variables/13/.ATTRIBUTES/VARIABLE_VALUEB'variables/14/.ATTRIBUTES/VARIABLE_VALUEB'variables/15/.ATTRIBUTES/VARIABLE_VALUEB'variables/16/.ATTRIBUTES/VARIABLE_VALUEB'variables/17/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2/tensor_names?
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*G
value>B<B B B B B B B B B B B B B B B B B B B B B B B B B B 2
SaveV2/shape_and_slices?
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0+savev2_conv2d_51_kernel_read_readvariableop)savev2_conv2d_51_bias_read_readvariableop+savev2_conv2d_49_kernel_read_readvariableop)savev2_conv2d_49_bias_read_readvariableop)savev2_dense_5_kernel_read_readvariableop+savev2_conv2d_50_kernel_read_readvariableop)savev2_conv2d_50_bias_read_readvariableop+savev2_conv2d_40_kernel_read_readvariableop)savev2_conv2d_40_bias_read_readvariableop7savev2_batch_normalization_28_gamma_read_readvariableop6savev2_batch_normalization_28_beta_read_readvariableop=savev2_batch_normalization_28_moving_mean_read_readvariableopAsavev2_batch_normalization_28_moving_variance_read_readvariableop+savev2_conv2d_41_kernel_read_readvariableop)savev2_conv2d_41_bias_read_readvariableop7savev2_batch_normalization_29_gamma_read_readvariableop6savev2_batch_normalization_29_beta_read_readvariableop=savev2_batch_normalization_29_moving_mean_read_readvariableopAsavev2_batch_normalization_29_moving_variance_read_readvariableop+savev2_conv2d_42_kernel_read_readvariableop)savev2_conv2d_42_bias_read_readvariableop7savev2_batch_normalization_30_gamma_read_readvariableop6savev2_batch_normalization_30_beta_read_readvariableop=savev2_batch_normalization_30_moving_mean_read_readvariableopAsavev2_batch_normalization_30_moving_variance_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *(
dtypes
22
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

identity_1Identity_1:output:0*?
_input_shapes?
?: :?::??:?:@:?I:I:?:?:?:?:?:?:??:?:?:?:?:?:??:?:?:?:?:?: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:-)
'
_output_shapes
:?: 

_output_shapes
::.*
(
_output_shapes
:??:!

_output_shapes	
:?:$ 

_output_shapes

:@:-)
'
_output_shapes
:?I: 

_output_shapes
:I:-)
'
_output_shapes
:?:!	

_output_shapes	
:?:!


_output_shapes	
:?:!

_output_shapes	
:?:!

_output_shapes	
:?:!

_output_shapes	
:?:.*
(
_output_shapes
:??:!

_output_shapes	
:?:!

_output_shapes	
:?:!

_output_shapes	
:?:!

_output_shapes	
:?:!

_output_shapes	
:?:.*
(
_output_shapes
:??:!

_output_shapes	
:?:!

_output_shapes	
:?:!

_output_shapes	
:?:!

_output_shapes	
:?:!

_output_shapes	
:?:

_output_shapes
: 
?
?
*__inference_conv2d_40_layer_call_fn_780314

inputs"
unknown:?
	unknown_0:	?
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_conv2d_40_layer_call_and_return_conditional_losses_7782612
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*0
_output_shapes
:??????????2

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
?.
?	
D__inference_model_11_layer_call_and_return_conditional_losses_779352

inputs)
model_8_779292:?
model_8_779294:	?
model_8_779296:	?
model_8_779298:	?
model_8_779300:	?
model_8_779302:	?*
model_8_779304:??
model_8_779306:	?
model_8_779308:	?
model_8_779310:	?
model_8_779312:	?
model_8_779314:	?*
model_8_779316:??
model_8_779318:	?
model_8_779320:	?
model_8_779322:	?
model_8_779324:	?
model_8_779326:	?+
conv2d_51_779329:?
conv2d_51_779331:,
conv2d_49_779334:??
conv2d_49_779336:	?+
conv2d_50_779342:?I
conv2d_50_779344:I 
dense_5_779347:@
identity

identity_1??!conv2d_49/StatefulPartitionedCall?!conv2d_50/StatefulPartitionedCall?!conv2d_51/StatefulPartitionedCall?dense_5/StatefulPartitionedCall?model_8/StatefulPartitionedCall?
model_8/StatefulPartitionedCallStatefulPartitionedCallinputsmodel_8_779292model_8_779294model_8_779296model_8_779298model_8_779300model_8_779302model_8_779304model_8_779306model_8_779308model_8_779310model_8_779312model_8_779314model_8_779316model_8_779318model_8_779320model_8_779322model_8_779324model_8_779326*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????*.
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_model_8_layer_call_and_return_conditional_losses_7787842!
model_8/StatefulPartitionedCall?
!conv2d_51/StatefulPartitionedCallStatefulPartitionedCall(model_8/StatefulPartitionedCall:output:0conv2d_51_779329conv2d_51_779331*
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
GPU 2J 8? *N
fIRG
E__inference_conv2d_51_layer_call_and_return_conditional_losses_7790252#
!conv2d_51/StatefulPartitionedCall?
!conv2d_49/StatefulPartitionedCallStatefulPartitionedCall(model_8/StatefulPartitionedCall:output:0conv2d_49_779334conv2d_49_779336*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_conv2d_49_layer_call_and_return_conditional_losses_7790422#
!conv2d_49/StatefulPartitionedCall?
dropout_45/PartitionedCallPartitionedCall*conv2d_51/StatefulPartitionedCall:output:0*
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
F__inference_dropout_45_layer_call_and_return_conditional_losses_7792072
dropout_45/PartitionedCall?
dropout_44/PartitionedCallPartitionedCall*conv2d_49/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_dropout_44_layer_call_and_return_conditional_losses_7791922
dropout_44/PartitionedCall?
flatten_5/PartitionedCallPartitionedCall#dropout_45/PartitionedCall:output:0*
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
GPU 2J 8? *N
fIRG
E__inference_flatten_5_layer_call_and_return_conditional_losses_7790682
flatten_5/PartitionedCall?
!conv2d_50/StatefulPartitionedCallStatefulPartitionedCall#dropout_44/PartitionedCall:output:0conv2d_50_779342conv2d_50_779344*
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
GPU 2J 8? *N
fIRG
E__inference_conv2d_50_layer_call_and_return_conditional_losses_7790812#
!conv2d_50/StatefulPartitionedCall?
dense_5/StatefulPartitionedCallStatefulPartitionedCall"flatten_5/PartitionedCall:output:0dense_5_779347*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_dense_5_layer_call_and_return_conditional_losses_7790952!
dense_5/StatefulPartitionedCall?
IdentityIdentity(dense_5/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????2

Identity?

Identity_1Identity*conv2d_50/StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:?????????I2

Identity_1?
NoOpNoOp"^conv2d_49/StatefulPartitionedCall"^conv2d_50/StatefulPartitionedCall"^conv2d_51/StatefulPartitionedCall ^dense_5/StatefulPartitionedCall ^model_8/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*`
_input_shapesO
M:?????????: : : : : : : : : : : : : : : : : : : : : : : : : 2F
!conv2d_49/StatefulPartitionedCall!conv2d_49/StatefulPartitionedCall2F
!conv2d_50/StatefulPartitionedCall!conv2d_50/StatefulPartitionedCall2F
!conv2d_51/StatefulPartitionedCall!conv2d_51/StatefulPartitionedCall2B
dense_5/StatefulPartitionedCalldense_5/StatefulPartitionedCall2B
model_8/StatefulPartitionedCallmodel_8/StatefulPartitionedCall:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
E__inference_conv2d_50_layer_call_and_return_conditional_losses_780305

inputs9
conv2d_readvariableop_resource:?I-
biasadd_readvariableop_resource:I
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*'
_output_shapes
:?I*
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
_construction_contextkEagerRuntime*3
_input_shapes"
 :??????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:??????????
 
_user_specified_nameinputs
?	
?
7__inference_batch_normalization_28_layer_call_fn_780376

inputs
unknown:	?
	unknown_0:	?
	unknown_1:	?
	unknown_2:	?
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *[
fVRT
R__inference_batch_normalization_28_layer_call_and_return_conditional_losses_7786672
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*0
_output_shapes
:??????????2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*7
_input_shapes&
$:??????????: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:??????????
 
_user_specified_nameinputs
??
?
!__inference__wrapped_model_777866
input_12T
9model_11_model_8_conv2d_40_conv2d_readvariableop_resource:?I
:model_11_model_8_conv2d_40_biasadd_readvariableop_resource:	?N
?model_11_model_8_batch_normalization_28_readvariableop_resource:	?P
Amodel_11_model_8_batch_normalization_28_readvariableop_1_resource:	?_
Pmodel_11_model_8_batch_normalization_28_fusedbatchnormv3_readvariableop_resource:	?a
Rmodel_11_model_8_batch_normalization_28_fusedbatchnormv3_readvariableop_1_resource:	?U
9model_11_model_8_conv2d_41_conv2d_readvariableop_resource:??I
:model_11_model_8_conv2d_41_biasadd_readvariableop_resource:	?N
?model_11_model_8_batch_normalization_29_readvariableop_resource:	?P
Amodel_11_model_8_batch_normalization_29_readvariableop_1_resource:	?_
Pmodel_11_model_8_batch_normalization_29_fusedbatchnormv3_readvariableop_resource:	?a
Rmodel_11_model_8_batch_normalization_29_fusedbatchnormv3_readvariableop_1_resource:	?U
9model_11_model_8_conv2d_42_conv2d_readvariableop_resource:??I
:model_11_model_8_conv2d_42_biasadd_readvariableop_resource:	?N
?model_11_model_8_batch_normalization_30_readvariableop_resource:	?P
Amodel_11_model_8_batch_normalization_30_readvariableop_1_resource:	?_
Pmodel_11_model_8_batch_normalization_30_fusedbatchnormv3_readvariableop_resource:	?a
Rmodel_11_model_8_batch_normalization_30_fusedbatchnormv3_readvariableop_1_resource:	?L
1model_11_conv2d_51_conv2d_readvariableop_resource:?@
2model_11_conv2d_51_biasadd_readvariableop_resource:M
1model_11_conv2d_49_conv2d_readvariableop_resource:??A
2model_11_conv2d_49_biasadd_readvariableop_resource:	?L
1model_11_conv2d_50_conv2d_readvariableop_resource:?I@
2model_11_conv2d_50_biasadd_readvariableop_resource:IA
/model_11_dense_5_matmul_readvariableop_resource:@
identity

identity_1??)model_11/conv2d_49/BiasAdd/ReadVariableOp?(model_11/conv2d_49/Conv2D/ReadVariableOp?)model_11/conv2d_50/BiasAdd/ReadVariableOp?(model_11/conv2d_50/Conv2D/ReadVariableOp?)model_11/conv2d_51/BiasAdd/ReadVariableOp?(model_11/conv2d_51/Conv2D/ReadVariableOp?&model_11/dense_5/MatMul/ReadVariableOp?Gmodel_11/model_8/batch_normalization_28/FusedBatchNormV3/ReadVariableOp?Imodel_11/model_8/batch_normalization_28/FusedBatchNormV3/ReadVariableOp_1?6model_11/model_8/batch_normalization_28/ReadVariableOp?8model_11/model_8/batch_normalization_28/ReadVariableOp_1?Gmodel_11/model_8/batch_normalization_29/FusedBatchNormV3/ReadVariableOp?Imodel_11/model_8/batch_normalization_29/FusedBatchNormV3/ReadVariableOp_1?6model_11/model_8/batch_normalization_29/ReadVariableOp?8model_11/model_8/batch_normalization_29/ReadVariableOp_1?Gmodel_11/model_8/batch_normalization_30/FusedBatchNormV3/ReadVariableOp?Imodel_11/model_8/batch_normalization_30/FusedBatchNormV3/ReadVariableOp_1?6model_11/model_8/batch_normalization_30/ReadVariableOp?8model_11/model_8/batch_normalization_30/ReadVariableOp_1?1model_11/model_8/conv2d_40/BiasAdd/ReadVariableOp?0model_11/model_8/conv2d_40/Conv2D/ReadVariableOp?1model_11/model_8/conv2d_41/BiasAdd/ReadVariableOp?0model_11/model_8/conv2d_41/Conv2D/ReadVariableOp?1model_11/model_8/conv2d_42/BiasAdd/ReadVariableOp?0model_11/model_8/conv2d_42/Conv2D/ReadVariableOp?
0model_11/model_8/conv2d_40/Conv2D/ReadVariableOpReadVariableOp9model_11_model_8_conv2d_40_conv2d_readvariableop_resource*'
_output_shapes
:?*
dtype022
0model_11/model_8/conv2d_40/Conv2D/ReadVariableOp?
!model_11/model_8/conv2d_40/Conv2DConv2Dinput_128model_11/model_8/conv2d_40/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????*
paddingSAME*
strides
2#
!model_11/model_8/conv2d_40/Conv2D?
1model_11/model_8/conv2d_40/BiasAdd/ReadVariableOpReadVariableOp:model_11_model_8_conv2d_40_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype023
1model_11/model_8/conv2d_40/BiasAdd/ReadVariableOp?
"model_11/model_8/conv2d_40/BiasAddBiasAdd*model_11/model_8/conv2d_40/Conv2D:output:09model_11/model_8/conv2d_40/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????2$
"model_11/model_8/conv2d_40/BiasAdd?
6model_11/model_8/batch_normalization_28/ReadVariableOpReadVariableOp?model_11_model_8_batch_normalization_28_readvariableop_resource*
_output_shapes	
:?*
dtype028
6model_11/model_8/batch_normalization_28/ReadVariableOp?
8model_11/model_8/batch_normalization_28/ReadVariableOp_1ReadVariableOpAmodel_11_model_8_batch_normalization_28_readvariableop_1_resource*
_output_shapes	
:?*
dtype02:
8model_11/model_8/batch_normalization_28/ReadVariableOp_1?
Gmodel_11/model_8/batch_normalization_28/FusedBatchNormV3/ReadVariableOpReadVariableOpPmodel_11_model_8_batch_normalization_28_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:?*
dtype02I
Gmodel_11/model_8/batch_normalization_28/FusedBatchNormV3/ReadVariableOp?
Imodel_11/model_8/batch_normalization_28/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpRmodel_11_model_8_batch_normalization_28_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:?*
dtype02K
Imodel_11/model_8/batch_normalization_28/FusedBatchNormV3/ReadVariableOp_1?
8model_11/model_8/batch_normalization_28/FusedBatchNormV3FusedBatchNormV3+model_11/model_8/conv2d_40/BiasAdd:output:0>model_11/model_8/batch_normalization_28/ReadVariableOp:value:0@model_11/model_8/batch_normalization_28/ReadVariableOp_1:value:0Omodel_11/model_8/batch_normalization_28/FusedBatchNormV3/ReadVariableOp:value:0Qmodel_11/model_8/batch_normalization_28/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:??????????:?:?:?:?:*
epsilon%o?:*
is_training( 2:
8model_11/model_8/batch_normalization_28/FusedBatchNormV3?
#model_11/model_8/activation_28/ReluRelu<model_11/model_8/batch_normalization_28/FusedBatchNormV3:y:0*
T0*0
_output_shapes
:??????????2%
#model_11/model_8/activation_28/Relu?
$model_11/model_8/dropout_36/IdentityIdentity1model_11/model_8/activation_28/Relu:activations:0*
T0*0
_output_shapes
:??????????2&
$model_11/model_8/dropout_36/Identity?
0model_11/model_8/conv2d_41/Conv2D/ReadVariableOpReadVariableOp9model_11_model_8_conv2d_41_conv2d_readvariableop_resource*(
_output_shapes
:??*
dtype022
0model_11/model_8/conv2d_41/Conv2D/ReadVariableOp?
!model_11/model_8/conv2d_41/Conv2DConv2D-model_11/model_8/dropout_36/Identity:output:08model_11/model_8/conv2d_41/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????*
paddingSAME*
strides
2#
!model_11/model_8/conv2d_41/Conv2D?
1model_11/model_8/conv2d_41/BiasAdd/ReadVariableOpReadVariableOp:model_11_model_8_conv2d_41_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype023
1model_11/model_8/conv2d_41/BiasAdd/ReadVariableOp?
"model_11/model_8/conv2d_41/BiasAddBiasAdd*model_11/model_8/conv2d_41/Conv2D:output:09model_11/model_8/conv2d_41/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????2$
"model_11/model_8/conv2d_41/BiasAdd?
6model_11/model_8/batch_normalization_29/ReadVariableOpReadVariableOp?model_11_model_8_batch_normalization_29_readvariableop_resource*
_output_shapes	
:?*
dtype028
6model_11/model_8/batch_normalization_29/ReadVariableOp?
8model_11/model_8/batch_normalization_29/ReadVariableOp_1ReadVariableOpAmodel_11_model_8_batch_normalization_29_readvariableop_1_resource*
_output_shapes	
:?*
dtype02:
8model_11/model_8/batch_normalization_29/ReadVariableOp_1?
Gmodel_11/model_8/batch_normalization_29/FusedBatchNormV3/ReadVariableOpReadVariableOpPmodel_11_model_8_batch_normalization_29_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:?*
dtype02I
Gmodel_11/model_8/batch_normalization_29/FusedBatchNormV3/ReadVariableOp?
Imodel_11/model_8/batch_normalization_29/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpRmodel_11_model_8_batch_normalization_29_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:?*
dtype02K
Imodel_11/model_8/batch_normalization_29/FusedBatchNormV3/ReadVariableOp_1?
8model_11/model_8/batch_normalization_29/FusedBatchNormV3FusedBatchNormV3+model_11/model_8/conv2d_41/BiasAdd:output:0>model_11/model_8/batch_normalization_29/ReadVariableOp:value:0@model_11/model_8/batch_normalization_29/ReadVariableOp_1:value:0Omodel_11/model_8/batch_normalization_29/FusedBatchNormV3/ReadVariableOp:value:0Qmodel_11/model_8/batch_normalization_29/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:??????????:?:?:?:?:*
epsilon%o?:*
is_training( 2:
8model_11/model_8/batch_normalization_29/FusedBatchNormV3?
#model_11/model_8/activation_29/ReluRelu<model_11/model_8/batch_normalization_29/FusedBatchNormV3:y:0*
T0*0
_output_shapes
:??????????2%
#model_11/model_8/activation_29/Relu?
$model_11/model_8/dropout_37/IdentityIdentity1model_11/model_8/activation_29/Relu:activations:0*
T0*0
_output_shapes
:??????????2&
$model_11/model_8/dropout_37/Identity?
0model_11/model_8/conv2d_42/Conv2D/ReadVariableOpReadVariableOp9model_11_model_8_conv2d_42_conv2d_readvariableop_resource*(
_output_shapes
:??*
dtype022
0model_11/model_8/conv2d_42/Conv2D/ReadVariableOp?
!model_11/model_8/conv2d_42/Conv2DConv2D-model_11/model_8/dropout_37/Identity:output:08model_11/model_8/conv2d_42/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????*
paddingSAME*
strides
2#
!model_11/model_8/conv2d_42/Conv2D?
1model_11/model_8/conv2d_42/BiasAdd/ReadVariableOpReadVariableOp:model_11_model_8_conv2d_42_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype023
1model_11/model_8/conv2d_42/BiasAdd/ReadVariableOp?
"model_11/model_8/conv2d_42/BiasAddBiasAdd*model_11/model_8/conv2d_42/Conv2D:output:09model_11/model_8/conv2d_42/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????2$
"model_11/model_8/conv2d_42/BiasAdd?
6model_11/model_8/batch_normalization_30/ReadVariableOpReadVariableOp?model_11_model_8_batch_normalization_30_readvariableop_resource*
_output_shapes	
:?*
dtype028
6model_11/model_8/batch_normalization_30/ReadVariableOp?
8model_11/model_8/batch_normalization_30/ReadVariableOp_1ReadVariableOpAmodel_11_model_8_batch_normalization_30_readvariableop_1_resource*
_output_shapes	
:?*
dtype02:
8model_11/model_8/batch_normalization_30/ReadVariableOp_1?
Gmodel_11/model_8/batch_normalization_30/FusedBatchNormV3/ReadVariableOpReadVariableOpPmodel_11_model_8_batch_normalization_30_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:?*
dtype02I
Gmodel_11/model_8/batch_normalization_30/FusedBatchNormV3/ReadVariableOp?
Imodel_11/model_8/batch_normalization_30/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpRmodel_11_model_8_batch_normalization_30_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:?*
dtype02K
Imodel_11/model_8/batch_normalization_30/FusedBatchNormV3/ReadVariableOp_1?
8model_11/model_8/batch_normalization_30/FusedBatchNormV3FusedBatchNormV3+model_11/model_8/conv2d_42/BiasAdd:output:0>model_11/model_8/batch_normalization_30/ReadVariableOp:value:0@model_11/model_8/batch_normalization_30/ReadVariableOp_1:value:0Omodel_11/model_8/batch_normalization_30/FusedBatchNormV3/ReadVariableOp:value:0Qmodel_11/model_8/batch_normalization_30/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:??????????:?:?:?:?:*
epsilon%o?:*
is_training( 2:
8model_11/model_8/batch_normalization_30/FusedBatchNormV3?
.model_11/model_8/tf.__operators__.add_12/AddV2AddV2<model_11/model_8/batch_normalization_30/FusedBatchNormV3:y:0-model_11/model_8/dropout_36/Identity:output:0*
T0*0
_output_shapes
:??????????20
.model_11/model_8/tf.__operators__.add_12/AddV2?
#model_11/model_8/activation_30/ReluRelu2model_11/model_8/tf.__operators__.add_12/AddV2:z:0*
T0*0
_output_shapes
:??????????2%
#model_11/model_8/activation_30/Relu?
$model_11/model_8/dropout_38/IdentityIdentity1model_11/model_8/activation_30/Relu:activations:0*
T0*0
_output_shapes
:??????????2&
$model_11/model_8/dropout_38/Identity?
(model_11/conv2d_51/Conv2D/ReadVariableOpReadVariableOp1model_11_conv2d_51_conv2d_readvariableop_resource*'
_output_shapes
:?*
dtype02*
(model_11/conv2d_51/Conv2D/ReadVariableOp?
model_11/conv2d_51/Conv2DConv2D-model_11/model_8/dropout_38/Identity:output:00model_11/conv2d_51/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????*
paddingVALID*
strides
2
model_11/conv2d_51/Conv2D?
)model_11/conv2d_51/BiasAdd/ReadVariableOpReadVariableOp2model_11_conv2d_51_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02+
)model_11/conv2d_51/BiasAdd/ReadVariableOp?
model_11/conv2d_51/BiasAddBiasAdd"model_11/conv2d_51/Conv2D:output:01model_11/conv2d_51/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????2
model_11/conv2d_51/BiasAdd?
model_11/conv2d_51/ReluRelu#model_11/conv2d_51/BiasAdd:output:0*
T0*/
_output_shapes
:?????????2
model_11/conv2d_51/Relu?
(model_11/conv2d_49/Conv2D/ReadVariableOpReadVariableOp1model_11_conv2d_49_conv2d_readvariableop_resource*(
_output_shapes
:??*
dtype02*
(model_11/conv2d_49/Conv2D/ReadVariableOp?
model_11/conv2d_49/Conv2DConv2D-model_11/model_8/dropout_38/Identity:output:00model_11/conv2d_49/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????*
paddingSAME*
strides
2
model_11/conv2d_49/Conv2D?
)model_11/conv2d_49/BiasAdd/ReadVariableOpReadVariableOp2model_11_conv2d_49_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02+
)model_11/conv2d_49/BiasAdd/ReadVariableOp?
model_11/conv2d_49/BiasAddBiasAdd"model_11/conv2d_49/Conv2D:output:01model_11/conv2d_49/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????2
model_11/conv2d_49/BiasAdd?
model_11/conv2d_49/ReluRelu#model_11/conv2d_49/BiasAdd:output:0*
T0*0
_output_shapes
:??????????2
model_11/conv2d_49/Relu?
model_11/dropout_45/IdentityIdentity%model_11/conv2d_51/Relu:activations:0*
T0*/
_output_shapes
:?????????2
model_11/dropout_45/Identity?
model_11/dropout_44/IdentityIdentity%model_11/conv2d_49/Relu:activations:0*
T0*0
_output_shapes
:??????????2
model_11/dropout_44/Identity?
model_11/flatten_5/ConstConst*
_output_shapes
:*
dtype0*
valueB"????@   2
model_11/flatten_5/Const?
model_11/flatten_5/ReshapeReshape%model_11/dropout_45/Identity:output:0!model_11/flatten_5/Const:output:0*
T0*'
_output_shapes
:?????????@2
model_11/flatten_5/Reshape?
(model_11/conv2d_50/Conv2D/ReadVariableOpReadVariableOp1model_11_conv2d_50_conv2d_readvariableop_resource*'
_output_shapes
:?I*
dtype02*
(model_11/conv2d_50/Conv2D/ReadVariableOp?
model_11/conv2d_50/Conv2DConv2D%model_11/dropout_44/Identity:output:00model_11/conv2d_50/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????I*
paddingSAME*
strides
2
model_11/conv2d_50/Conv2D?
)model_11/conv2d_50/BiasAdd/ReadVariableOpReadVariableOp2model_11_conv2d_50_biasadd_readvariableop_resource*
_output_shapes
:I*
dtype02+
)model_11/conv2d_50/BiasAdd/ReadVariableOp?
model_11/conv2d_50/BiasAddBiasAdd"model_11/conv2d_50/Conv2D:output:01model_11/conv2d_50/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????I2
model_11/conv2d_50/BiasAdd?
model_11/conv2d_50/SoftmaxSoftmax#model_11/conv2d_50/BiasAdd:output:0*
T0*/
_output_shapes
:?????????I2
model_11/conv2d_50/Softmax?
&model_11/dense_5/MatMul/ReadVariableOpReadVariableOp/model_11_dense_5_matmul_readvariableop_resource*
_output_shapes

:@*
dtype02(
&model_11/dense_5/MatMul/ReadVariableOp?
model_11/dense_5/MatMulMatMul#model_11/flatten_5/Reshape:output:0.model_11/dense_5/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
model_11/dense_5/MatMul?
model_11/dense_5/TanhTanh!model_11/dense_5/MatMul:product:0*
T0*'
_output_shapes
:?????????2
model_11/dense_5/Tanh?
IdentityIdentity$model_11/conv2d_50/Softmax:softmax:0^NoOp*
T0*/
_output_shapes
:?????????I2

Identityx

Identity_1Identitymodel_11/dense_5/Tanh:y:0^NoOp*
T0*'
_output_shapes
:?????????2

Identity_1?
NoOpNoOp*^model_11/conv2d_49/BiasAdd/ReadVariableOp)^model_11/conv2d_49/Conv2D/ReadVariableOp*^model_11/conv2d_50/BiasAdd/ReadVariableOp)^model_11/conv2d_50/Conv2D/ReadVariableOp*^model_11/conv2d_51/BiasAdd/ReadVariableOp)^model_11/conv2d_51/Conv2D/ReadVariableOp'^model_11/dense_5/MatMul/ReadVariableOpH^model_11/model_8/batch_normalization_28/FusedBatchNormV3/ReadVariableOpJ^model_11/model_8/batch_normalization_28/FusedBatchNormV3/ReadVariableOp_17^model_11/model_8/batch_normalization_28/ReadVariableOp9^model_11/model_8/batch_normalization_28/ReadVariableOp_1H^model_11/model_8/batch_normalization_29/FusedBatchNormV3/ReadVariableOpJ^model_11/model_8/batch_normalization_29/FusedBatchNormV3/ReadVariableOp_17^model_11/model_8/batch_normalization_29/ReadVariableOp9^model_11/model_8/batch_normalization_29/ReadVariableOp_1H^model_11/model_8/batch_normalization_30/FusedBatchNormV3/ReadVariableOpJ^model_11/model_8/batch_normalization_30/FusedBatchNormV3/ReadVariableOp_17^model_11/model_8/batch_normalization_30/ReadVariableOp9^model_11/model_8/batch_normalization_30/ReadVariableOp_12^model_11/model_8/conv2d_40/BiasAdd/ReadVariableOp1^model_11/model_8/conv2d_40/Conv2D/ReadVariableOp2^model_11/model_8/conv2d_41/BiasAdd/ReadVariableOp1^model_11/model_8/conv2d_41/Conv2D/ReadVariableOp2^model_11/model_8/conv2d_42/BiasAdd/ReadVariableOp1^model_11/model_8/conv2d_42/Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*`
_input_shapesO
M:?????????: : : : : : : : : : : : : : : : : : : : : : : : : 2V
)model_11/conv2d_49/BiasAdd/ReadVariableOp)model_11/conv2d_49/BiasAdd/ReadVariableOp2T
(model_11/conv2d_49/Conv2D/ReadVariableOp(model_11/conv2d_49/Conv2D/ReadVariableOp2V
)model_11/conv2d_50/BiasAdd/ReadVariableOp)model_11/conv2d_50/BiasAdd/ReadVariableOp2T
(model_11/conv2d_50/Conv2D/ReadVariableOp(model_11/conv2d_50/Conv2D/ReadVariableOp2V
)model_11/conv2d_51/BiasAdd/ReadVariableOp)model_11/conv2d_51/BiasAdd/ReadVariableOp2T
(model_11/conv2d_51/Conv2D/ReadVariableOp(model_11/conv2d_51/Conv2D/ReadVariableOp2P
&model_11/dense_5/MatMul/ReadVariableOp&model_11/dense_5/MatMul/ReadVariableOp2?
Gmodel_11/model_8/batch_normalization_28/FusedBatchNormV3/ReadVariableOpGmodel_11/model_8/batch_normalization_28/FusedBatchNormV3/ReadVariableOp2?
Imodel_11/model_8/batch_normalization_28/FusedBatchNormV3/ReadVariableOp_1Imodel_11/model_8/batch_normalization_28/FusedBatchNormV3/ReadVariableOp_12p
6model_11/model_8/batch_normalization_28/ReadVariableOp6model_11/model_8/batch_normalization_28/ReadVariableOp2t
8model_11/model_8/batch_normalization_28/ReadVariableOp_18model_11/model_8/batch_normalization_28/ReadVariableOp_12?
Gmodel_11/model_8/batch_normalization_29/FusedBatchNormV3/ReadVariableOpGmodel_11/model_8/batch_normalization_29/FusedBatchNormV3/ReadVariableOp2?
Imodel_11/model_8/batch_normalization_29/FusedBatchNormV3/ReadVariableOp_1Imodel_11/model_8/batch_normalization_29/FusedBatchNormV3/ReadVariableOp_12p
6model_11/model_8/batch_normalization_29/ReadVariableOp6model_11/model_8/batch_normalization_29/ReadVariableOp2t
8model_11/model_8/batch_normalization_29/ReadVariableOp_18model_11/model_8/batch_normalization_29/ReadVariableOp_12?
Gmodel_11/model_8/batch_normalization_30/FusedBatchNormV3/ReadVariableOpGmodel_11/model_8/batch_normalization_30/FusedBatchNormV3/ReadVariableOp2?
Imodel_11/model_8/batch_normalization_30/FusedBatchNormV3/ReadVariableOp_1Imodel_11/model_8/batch_normalization_30/FusedBatchNormV3/ReadVariableOp_12p
6model_11/model_8/batch_normalization_30/ReadVariableOp6model_11/model_8/batch_normalization_30/ReadVariableOp2t
8model_11/model_8/batch_normalization_30/ReadVariableOp_18model_11/model_8/batch_normalization_30/ReadVariableOp_12f
1model_11/model_8/conv2d_40/BiasAdd/ReadVariableOp1model_11/model_8/conv2d_40/BiasAdd/ReadVariableOp2d
0model_11/model_8/conv2d_40/Conv2D/ReadVariableOp0model_11/model_8/conv2d_40/Conv2D/ReadVariableOp2f
1model_11/model_8/conv2d_41/BiasAdd/ReadVariableOp1model_11/model_8/conv2d_41/BiasAdd/ReadVariableOp2d
0model_11/model_8/conv2d_41/Conv2D/ReadVariableOp0model_11/model_8/conv2d_41/Conv2D/ReadVariableOp2f
1model_11/model_8/conv2d_42/BiasAdd/ReadVariableOp1model_11/model_8/conv2d_42/BiasAdd/ReadVariableOp2d
0model_11/model_8/conv2d_42/Conv2D/ReadVariableOp0model_11/model_8/conv2d_42/Conv2D/ReadVariableOp:Y U
/
_output_shapes
:?????????
"
_user_specified_name
input_12
?
J
.__inference_activation_29_layer_call_fn_780625

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *R
fMRK
I__inference_activation_29_layer_call_and_return_conditional_losses_7783562
PartitionedCallu
IdentityIdentityPartitionedCall:output:0*
T0*0
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:??????????:X T
0
_output_shapes
:??????????
 
_user_specified_nameinputs
?
G
+__inference_dropout_45_layer_call_fn_780211

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
F__inference_dropout_45_layer_call_and_return_conditional_losses_7792072
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
?.
?	
D__inference_model_11_layer_call_and_return_conditional_losses_779527
input_12)
model_8_779467:?
model_8_779469:	?
model_8_779471:	?
model_8_779473:	?
model_8_779475:	?
model_8_779477:	?*
model_8_779479:??
model_8_779481:	?
model_8_779483:	?
model_8_779485:	?
model_8_779487:	?
model_8_779489:	?*
model_8_779491:??
model_8_779493:	?
model_8_779495:	?
model_8_779497:	?
model_8_779499:	?
model_8_779501:	?+
conv2d_51_779504:?
conv2d_51_779506:,
conv2d_49_779509:??
conv2d_49_779511:	?+
conv2d_50_779517:?I
conv2d_50_779519:I 
dense_5_779522:@
identity

identity_1??!conv2d_49/StatefulPartitionedCall?!conv2d_50/StatefulPartitionedCall?!conv2d_51/StatefulPartitionedCall?dense_5/StatefulPartitionedCall?model_8/StatefulPartitionedCall?
model_8/StatefulPartitionedCallStatefulPartitionedCallinput_12model_8_779467model_8_779469model_8_779471model_8_779473model_8_779475model_8_779477model_8_779479model_8_779481model_8_779483model_8_779485model_8_779487model_8_779489model_8_779491model_8_779493model_8_779495model_8_779497model_8_779499model_8_779501*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????*4
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_model_8_layer_call_and_return_conditional_losses_7784242!
model_8/StatefulPartitionedCall?
!conv2d_51/StatefulPartitionedCallStatefulPartitionedCall(model_8/StatefulPartitionedCall:output:0conv2d_51_779504conv2d_51_779506*
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
GPU 2J 8? *N
fIRG
E__inference_conv2d_51_layer_call_and_return_conditional_losses_7790252#
!conv2d_51/StatefulPartitionedCall?
!conv2d_49/StatefulPartitionedCallStatefulPartitionedCall(model_8/StatefulPartitionedCall:output:0conv2d_49_779509conv2d_49_779511*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_conv2d_49_layer_call_and_return_conditional_losses_7790422#
!conv2d_49/StatefulPartitionedCall?
dropout_45/PartitionedCallPartitionedCall*conv2d_51/StatefulPartitionedCall:output:0*
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
F__inference_dropout_45_layer_call_and_return_conditional_losses_7790532
dropout_45/PartitionedCall?
dropout_44/PartitionedCallPartitionedCall*conv2d_49/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_dropout_44_layer_call_and_return_conditional_losses_7790602
dropout_44/PartitionedCall?
flatten_5/PartitionedCallPartitionedCall#dropout_45/PartitionedCall:output:0*
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
GPU 2J 8? *N
fIRG
E__inference_flatten_5_layer_call_and_return_conditional_losses_7790682
flatten_5/PartitionedCall?
!conv2d_50/StatefulPartitionedCallStatefulPartitionedCall#dropout_44/PartitionedCall:output:0conv2d_50_779517conv2d_50_779519*
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
GPU 2J 8? *N
fIRG
E__inference_conv2d_50_layer_call_and_return_conditional_losses_7790812#
!conv2d_50/StatefulPartitionedCall?
dense_5/StatefulPartitionedCallStatefulPartitionedCall"flatten_5/PartitionedCall:output:0dense_5_779522*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_dense_5_layer_call_and_return_conditional_losses_7790952!
dense_5/StatefulPartitionedCall?
IdentityIdentity(dense_5/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????2

Identity?

Identity_1Identity*conv2d_50/StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:?????????I2

Identity_1?
NoOpNoOp"^conv2d_49/StatefulPartitionedCall"^conv2d_50/StatefulPartitionedCall"^conv2d_51/StatefulPartitionedCall ^dense_5/StatefulPartitionedCall ^model_8/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*`
_input_shapesO
M:?????????: : : : : : : : : : : : : : : : : : : : : : : : : 2F
!conv2d_49/StatefulPartitionedCall!conv2d_49/StatefulPartitionedCall2F
!conv2d_50/StatefulPartitionedCall!conv2d_50/StatefulPartitionedCall2F
!conv2d_51/StatefulPartitionedCall!conv2d_51/StatefulPartitionedCall2B
dense_5/StatefulPartitionedCalldense_5/StatefulPartitionedCall2B
model_8/StatefulPartitionedCallmodel_8/StatefulPartitionedCall:Y U
/
_output_shapes
:?????????
"
_user_specified_name
input_12
?
d
F__inference_dropout_37_layer_call_and_return_conditional_losses_778363

inputs

identity_1c
IdentityIdentityinputs*
T0*0
_output_shapes
:??????????2

Identityr

Identity_1IdentityIdentity:output:0*
T0*0
_output_shapes
:??????????2

Identity_1"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:??????????:X T
0
_output_shapes
:??????????
 
_user_specified_nameinputs
?
b
F__inference_dropout_45_layer_call_and_return_conditional_losses_780220

inputs
identityb
IdentityIdentityinputs*
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
?
b
F__inference_dropout_36_layer_call_and_return_conditional_losses_778625

inputs
identityc
IdentityIdentityinputs*
T0*0
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:??????????:X T
0
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
R__inference_batch_normalization_30_layer_call_and_return_conditional_losses_778398

inputs&
readvariableop_resource:	?(
readvariableop_1_resource:	?7
(fusedbatchnormv3_readvariableop_resource:	?9
*fusedbatchnormv3_readvariableop_1_resource:	?
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1u
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:?*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:?*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:?*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:?*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:??????????:?:?:?:?:*
epsilon%o?:*
is_training( 2
FusedBatchNormV3x
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*0
_output_shapes
:??????????2

Identity?
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*7
_input_shapes&
$:??????????: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:X T
0
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
R__inference_batch_normalization_29_layer_call_and_return_conditional_losses_780620

inputs&
readvariableop_resource:	?(
readvariableop_1_resource:	?7
(fusedbatchnormv3_readvariableop_resource:	?9
*fusedbatchnormv3_readvariableop_1_resource:	?
identity??AssignNewValue?AssignNewValue_1?FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1u
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:?*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:?*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:?*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:?*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:??????????:?:?:?:?:*
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
AssignNewValue_1x
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*0
_output_shapes
:??????????2

Identity?
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*7
_input_shapes&
$:??????????: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:X T
0
_output_shapes
:??????????
 
_user_specified_nameinputs
?
e
I__inference_activation_29_layer_call_and_return_conditional_losses_778356

inputs
identityW
ReluReluinputs*
T0*0
_output_shapes
:??????????2
Reluo
IdentityIdentityRelu:activations:0*
T0*0
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:??????????:X T
0
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
R__inference_batch_normalization_28_layer_call_and_return_conditional_losses_780394

inputs&
readvariableop_resource:	?(
readvariableop_1_resource:	?7
(fusedbatchnormv3_readvariableop_resource:	?9
*fusedbatchnormv3_readvariableop_1_resource:	?
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1u
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:?*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:?*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:?*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:?*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,????????????????????????????:?:?:?:?:*
epsilon%o?:*
is_training( 2
FusedBatchNormV3?
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*B
_output_shapes0
.:,????????????????????????????2

Identity?
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,????????????????????????????: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:j f
B
_output_shapes0
.:,????????????????????????????
 
_user_specified_nameinputs
?
?
(__inference_model_8_layer_call_fn_780140

inputs"
unknown:?
	unknown_0:	?
	unknown_1:	?
	unknown_2:	?
	unknown_3:	?
	unknown_4:	?%
	unknown_5:??
	unknown_6:	?
	unknown_7:	?
	unknown_8:	?
	unknown_9:	?

unknown_10:	?&

unknown_11:??

unknown_12:	?

unknown_13:	?

unknown_14:	?

unknown_15:	?

unknown_16:	?
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
 *0
_output_shapes
:??????????*4
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_model_8_layer_call_and_return_conditional_losses_7784242
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*0
_output_shapes
:??????????2

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
?
b
F__inference_dropout_36_layer_call_and_return_conditional_losses_780477

inputs
identityc
IdentityIdentityinputs*
T0*0
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:??????????:X T
0
_output_shapes
:??????????
 
_user_specified_nameinputs
?
G
+__inference_dropout_45_layer_call_fn_780206

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
F__inference_dropout_45_layer_call_and_return_conditional_losses_7790532
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
?
?
(__inference_model_8_layer_call_fn_778463
input_9"
unknown:?
	unknown_0:	?
	unknown_1:	?
	unknown_2:	?
	unknown_3:	?
	unknown_4:	?%
	unknown_5:??
	unknown_6:	?
	unknown_7:	?
	unknown_8:	?
	unknown_9:	?

unknown_10:	?&

unknown_11:??

unknown_12:	?

unknown_13:	?

unknown_14:	?

unknown_15:	?

unknown_16:	?
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_9unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
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
 *0
_output_shapes
:??????????*4
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_model_8_layer_call_and_return_conditional_losses_7784242
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*0
_output_shapes
:??????????2

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
_user_specified_name	input_9
?
d
F__inference_dropout_38_layer_call_and_return_conditional_losses_778421

inputs

identity_1c
IdentityIdentityinputs*
T0*0
_output_shapes
:??????????2

Identityr

Identity_1IdentityIdentity:output:0*
T0*0
_output_shapes
:??????????2

Identity_1"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:??????????:X T
0
_output_shapes
:??????????
 
_user_specified_nameinputs
?
d
F__inference_dropout_45_layer_call_and_return_conditional_losses_779053

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
?
R__inference_batch_normalization_28_layer_call_and_return_conditional_losses_777888

inputs&
readvariableop_resource:	?(
readvariableop_1_resource:	?7
(fusedbatchnormv3_readvariableop_resource:	?9
*fusedbatchnormv3_readvariableop_1_resource:	?
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1u
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:?*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:?*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:?*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:?*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,????????????????????????????:?:?:?:?:*
epsilon%o?:*
is_training( 2
FusedBatchNormV3?
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*B
_output_shapes0
.:,????????????????????????????2

Identity?
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,????????????????????????????: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:j f
B
_output_shapes0
.:,????????????????????????????
 
_user_specified_nameinputs
?
J
.__inference_activation_30_layer_call_fn_780797

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *R
fMRK
I__inference_activation_30_layer_call_and_return_conditional_losses_7784142
PartitionedCallu
IdentityIdentityPartitionedCall:output:0*
T0*0
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:??????????:X T
0
_output_shapes
:??????????
 
_user_specified_nameinputs
?	
?
7__inference_batch_normalization_28_layer_call_fn_780350

inputs
unknown:	?
	unknown_0:	?
	unknown_1:	?
	unknown_2:	?
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,????????????????????????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *[
fVRT
R__inference_batch_normalization_28_layer_call_and_return_conditional_losses_7779322
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*B
_output_shapes0
.:,????????????????????????????2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,????????????????????????????: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:j f
B
_output_shapes0
.:,????????????????????????????
 
_user_specified_nameinputs
?l
?
"__inference__traced_restore_781005
file_prefix<
!assignvariableop_conv2d_51_kernel:?/
!assignvariableop_1_conv2d_51_bias:?
#assignvariableop_2_conv2d_49_kernel:??0
!assignvariableop_3_conv2d_49_bias:	?3
!assignvariableop_4_dense_5_kernel:@>
#assignvariableop_5_conv2d_50_kernel:?I/
!assignvariableop_6_conv2d_50_bias:I>
#assignvariableop_7_conv2d_40_kernel:?0
!assignvariableop_8_conv2d_40_bias:	?>
/assignvariableop_9_batch_normalization_28_gamma:	?>
/assignvariableop_10_batch_normalization_28_beta:	?E
6assignvariableop_11_batch_normalization_28_moving_mean:	?I
:assignvariableop_12_batch_normalization_28_moving_variance:	?@
$assignvariableop_13_conv2d_41_kernel:??1
"assignvariableop_14_conv2d_41_bias:	??
0assignvariableop_15_batch_normalization_29_gamma:	?>
/assignvariableop_16_batch_normalization_29_beta:	?E
6assignvariableop_17_batch_normalization_29_moving_mean:	?I
:assignvariableop_18_batch_normalization_29_moving_variance:	?@
$assignvariableop_19_conv2d_42_kernel:??1
"assignvariableop_20_conv2d_42_bias:	??
0assignvariableop_21_batch_normalization_30_gamma:	?>
/assignvariableop_22_batch_normalization_30_beta:	?E
6assignvariableop_23_batch_normalization_30_moving_mean:	?I
:assignvariableop_24_batch_normalization_30_moving_variance:	?
identity_26??AssignVariableOp?AssignVariableOp_1?AssignVariableOp_10?AssignVariableOp_11?AssignVariableOp_12?AssignVariableOp_13?AssignVariableOp_14?AssignVariableOp_15?AssignVariableOp_16?AssignVariableOp_17?AssignVariableOp_18?AssignVariableOp_19?AssignVariableOp_2?AssignVariableOp_20?AssignVariableOp_21?AssignVariableOp_22?AssignVariableOp_23?AssignVariableOp_24?AssignVariableOp_3?AssignVariableOp_4?AssignVariableOp_5?AssignVariableOp_6?AssignVariableOp_7?AssignVariableOp_8?AssignVariableOp_9?

RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*?	
value?	B?	B6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB&variables/0/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB&variables/3/.ATTRIBUTES/VARIABLE_VALUEB&variables/4/.ATTRIBUTES/VARIABLE_VALUEB&variables/5/.ATTRIBUTES/VARIABLE_VALUEB&variables/6/.ATTRIBUTES/VARIABLE_VALUEB&variables/7/.ATTRIBUTES/VARIABLE_VALUEB&variables/8/.ATTRIBUTES/VARIABLE_VALUEB&variables/9/.ATTRIBUTES/VARIABLE_VALUEB'variables/10/.ATTRIBUTES/VARIABLE_VALUEB'variables/11/.ATTRIBUTES/VARIABLE_VALUEB'variables/12/.ATTRIBUTES/VARIABLE_VALUEB'variables/13/.ATTRIBUTES/VARIABLE_VALUEB'variables/14/.ATTRIBUTES/VARIABLE_VALUEB'variables/15/.ATTRIBUTES/VARIABLE_VALUEB'variables/16/.ATTRIBUTES/VARIABLE_VALUEB'variables/17/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2/tensor_names?
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*G
value>B<B B B B B B B B B B B B B B B B B B B B B B B B B B 2
RestoreV2/shape_and_slices?
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*|
_output_shapesj
h::::::::::::::::::::::::::*(
dtypes
22
	RestoreV2g
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:2

Identity?
AssignVariableOpAssignVariableOp!assignvariableop_conv2d_51_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOpk

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:2

Identity_1?
AssignVariableOp_1AssignVariableOp!assignvariableop_1_conv2d_51_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_1k

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:2

Identity_2?
AssignVariableOp_2AssignVariableOp#assignvariableop_2_conv2d_49_kernelIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_2k

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:2

Identity_3?
AssignVariableOp_3AssignVariableOp!assignvariableop_3_conv2d_49_biasIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_3k

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:2

Identity_4?
AssignVariableOp_4AssignVariableOp!assignvariableop_4_dense_5_kernelIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_4k

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:2

Identity_5?
AssignVariableOp_5AssignVariableOp#assignvariableop_5_conv2d_50_kernelIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_5k

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:2

Identity_6?
AssignVariableOp_6AssignVariableOp!assignvariableop_6_conv2d_50_biasIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_6k

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:2

Identity_7?
AssignVariableOp_7AssignVariableOp#assignvariableop_7_conv2d_40_kernelIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_7k

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:2

Identity_8?
AssignVariableOp_8AssignVariableOp!assignvariableop_8_conv2d_40_biasIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_8k

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:2

Identity_9?
AssignVariableOp_9AssignVariableOp/assignvariableop_9_batch_normalization_28_gammaIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_9n
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:2
Identity_10?
AssignVariableOp_10AssignVariableOp/assignvariableop_10_batch_normalization_28_betaIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_10n
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:2
Identity_11?
AssignVariableOp_11AssignVariableOp6assignvariableop_11_batch_normalization_28_moving_meanIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_11n
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:2
Identity_12?
AssignVariableOp_12AssignVariableOp:assignvariableop_12_batch_normalization_28_moving_varianceIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_12n
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:2
Identity_13?
AssignVariableOp_13AssignVariableOp$assignvariableop_13_conv2d_41_kernelIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_13n
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:2
Identity_14?
AssignVariableOp_14AssignVariableOp"assignvariableop_14_conv2d_41_biasIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_14n
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:2
Identity_15?
AssignVariableOp_15AssignVariableOp0assignvariableop_15_batch_normalization_29_gammaIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_15n
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:2
Identity_16?
AssignVariableOp_16AssignVariableOp/assignvariableop_16_batch_normalization_29_betaIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_16n
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:2
Identity_17?
AssignVariableOp_17AssignVariableOp6assignvariableop_17_batch_normalization_29_moving_meanIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_17n
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:2
Identity_18?
AssignVariableOp_18AssignVariableOp:assignvariableop_18_batch_normalization_29_moving_varianceIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_18n
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:2
Identity_19?
AssignVariableOp_19AssignVariableOp$assignvariableop_19_conv2d_42_kernelIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_19n
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:2
Identity_20?
AssignVariableOp_20AssignVariableOp"assignvariableop_20_conv2d_42_biasIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_20n
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:2
Identity_21?
AssignVariableOp_21AssignVariableOp0assignvariableop_21_batch_normalization_30_gammaIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_21n
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:2
Identity_22?
AssignVariableOp_22AssignVariableOp/assignvariableop_22_batch_normalization_30_betaIdentity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_22n
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:2
Identity_23?
AssignVariableOp_23AssignVariableOp6assignvariableop_23_batch_normalization_30_moving_meanIdentity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_23n
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:2
Identity_24?
AssignVariableOp_24AssignVariableOp:assignvariableop_24_batch_normalization_30_moving_varianceIdentity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_249
NoOpNoOp"/device:CPU:0*
_output_shapes
 2
NoOp?
Identity_25Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2
Identity_25f
Identity_26IdentityIdentity_25:output:0^NoOp_1*
T0*
_output_shapes
: 2
Identity_26?
NoOp_1NoOp^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*"
_acd_function_control_output(*
_output_shapes
 2
NoOp_1"#
identity_26Identity_26:output:0*G
_input_shapes6
4: : : : : : : : : : : : : : : : : : : : : : : : : : 2$
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
AssignVariableOp_23AssignVariableOp_232*
AssignVariableOp_24AssignVariableOp_242(
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
?
?
R__inference_batch_normalization_30_layer_call_and_return_conditional_losses_778517

inputs&
readvariableop_resource:	?(
readvariableop_1_resource:	?7
(fusedbatchnormv3_readvariableop_resource:	?9
*fusedbatchnormv3_readvariableop_1_resource:	?
identity??AssignNewValue?AssignNewValue_1?FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1u
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:?*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:?*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:?*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:?*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:??????????:?:?:?:?:*
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
AssignNewValue_1x
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*0
_output_shapes
:??????????2

Identity?
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*7
_input_shapes&
$:??????????: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:X T
0
_output_shapes
:??????????
 
_user_specified_nameinputs
?
e
I__inference_activation_28_layer_call_and_return_conditional_losses_780458

inputs
identityW
ReluReluinputs*
T0*0
_output_shapes
:??????????2
Reluo
IdentityIdentityRelu:activations:0*
T0*0
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:??????????:X T
0
_output_shapes
:??????????
 
_user_specified_nameinputs
??
?
C__inference_model_8_layer_call_and_return_conditional_losses_778784

inputs+
conv2d_40_778734:?
conv2d_40_778736:	?,
batch_normalization_28_778739:	?,
batch_normalization_28_778741:	?,
batch_normalization_28_778743:	?,
batch_normalization_28_778745:	?,
conv2d_41_778750:??
conv2d_41_778752:	?,
batch_normalization_29_778755:	?,
batch_normalization_29_778757:	?,
batch_normalization_29_778759:	?,
batch_normalization_29_778761:	?,
conv2d_42_778766:??
conv2d_42_778768:	?,
batch_normalization_30_778771:	?,
batch_normalization_30_778773:	?,
batch_normalization_30_778775:	?,
batch_normalization_30_778777:	?
identity??.batch_normalization_28/StatefulPartitionedCall?.batch_normalization_29/StatefulPartitionedCall?.batch_normalization_30/StatefulPartitionedCall?!conv2d_40/StatefulPartitionedCall?!conv2d_41/StatefulPartitionedCall?!conv2d_42/StatefulPartitionedCall?
!conv2d_40/StatefulPartitionedCallStatefulPartitionedCallinputsconv2d_40_778734conv2d_40_778736*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_conv2d_40_layer_call_and_return_conditional_losses_7782612#
!conv2d_40/StatefulPartitionedCall?
.batch_normalization_28/StatefulPartitionedCallStatefulPartitionedCall*conv2d_40/StatefulPartitionedCall:output:0batch_normalization_28_778739batch_normalization_28_778741batch_normalization_28_778743batch_normalization_28_778745*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *[
fVRT
R__inference_batch_normalization_28_layer_call_and_return_conditional_losses_77866720
.batch_normalization_28/StatefulPartitionedCall?
activation_28/PartitionedCallPartitionedCall7batch_normalization_28/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *R
fMRK
I__inference_activation_28_layer_call_and_return_conditional_losses_7782992
activation_28/PartitionedCall?
dropout_36/PartitionedCallPartitionedCall&activation_28/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_dropout_36_layer_call_and_return_conditional_losses_7786252
dropout_36/PartitionedCall?
!conv2d_41/StatefulPartitionedCallStatefulPartitionedCall#dropout_36/PartitionedCall:output:0conv2d_41_778750conv2d_41_778752*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_conv2d_41_layer_call_and_return_conditional_losses_7783182#
!conv2d_41/StatefulPartitionedCall?
.batch_normalization_29/StatefulPartitionedCallStatefulPartitionedCall*conv2d_41/StatefulPartitionedCall:output:0batch_normalization_29_778755batch_normalization_29_778757batch_normalization_29_778759batch_normalization_29_778761*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *[
fVRT
R__inference_batch_normalization_29_layer_call_and_return_conditional_losses_77859220
.batch_normalization_29/StatefulPartitionedCall?
activation_29/PartitionedCallPartitionedCall7batch_normalization_29/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *R
fMRK
I__inference_activation_29_layer_call_and_return_conditional_losses_7783562
activation_29/PartitionedCall?
dropout_37/PartitionedCallPartitionedCall&activation_29/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_dropout_37_layer_call_and_return_conditional_losses_7785502
dropout_37/PartitionedCall?
!conv2d_42/StatefulPartitionedCallStatefulPartitionedCall#dropout_37/PartitionedCall:output:0conv2d_42_778766conv2d_42_778768*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_conv2d_42_layer_call_and_return_conditional_losses_7783752#
!conv2d_42/StatefulPartitionedCall?
.batch_normalization_30/StatefulPartitionedCallStatefulPartitionedCall*conv2d_42/StatefulPartitionedCall:output:0batch_normalization_30_778771batch_normalization_30_778773batch_normalization_30_778775batch_normalization_30_778777*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *[
fVRT
R__inference_batch_normalization_30_layer_call_and_return_conditional_losses_77851720
.batch_normalization_30/StatefulPartitionedCall?
tf.__operators__.add_12/AddV2AddV27batch_normalization_30/StatefulPartitionedCall:output:0#dropout_36/PartitionedCall:output:0*
T0*0
_output_shapes
:??????????2
tf.__operators__.add_12/AddV2?
activation_30/PartitionedCallPartitionedCall!tf.__operators__.add_12/AddV2:z:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *R
fMRK
I__inference_activation_30_layer_call_and_return_conditional_losses_7784142
activation_30/PartitionedCall?
dropout_38/PartitionedCallPartitionedCall&activation_30/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_dropout_38_layer_call_and_return_conditional_losses_7784752
dropout_38/PartitionedCall?
IdentityIdentity#dropout_38/PartitionedCall:output:0^NoOp*
T0*0
_output_shapes
:??????????2

Identity?
NoOpNoOp/^batch_normalization_28/StatefulPartitionedCall/^batch_normalization_29/StatefulPartitionedCall/^batch_normalization_30/StatefulPartitionedCall"^conv2d_40/StatefulPartitionedCall"^conv2d_41/StatefulPartitionedCall"^conv2d_42/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*R
_input_shapesA
?:?????????: : : : : : : : : : : : : : : : : : 2`
.batch_normalization_28/StatefulPartitionedCall.batch_normalization_28/StatefulPartitionedCall2`
.batch_normalization_29/StatefulPartitionedCall.batch_normalization_29/StatefulPartitionedCall2`
.batch_normalization_30/StatefulPartitionedCall.batch_normalization_30/StatefulPartitionedCall2F
!conv2d_40/StatefulPartitionedCall!conv2d_40/StatefulPartitionedCall2F
!conv2d_41/StatefulPartitionedCall!conv2d_41/StatefulPartitionedCall2F
!conv2d_42/StatefulPartitionedCall!conv2d_42/StatefulPartitionedCall:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?	
?
7__inference_batch_normalization_30_layer_call_fn_780720

inputs
unknown:	?
	unknown_0:	?
	unknown_1:	?
	unknown_2:	?
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *[
fVRT
R__inference_batch_normalization_30_layer_call_and_return_conditional_losses_7785172
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*0
_output_shapes
:??????????2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*7
_input_shapes&
$:??????????: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
E__inference_conv2d_41_layer_call_and_return_conditional_losses_778318

inputs:
conv2d_readvariableop_resource:??.
biasadd_readvariableop_resource:	?
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:??*
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????*
paddingSAME*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????2	
BiasAddt
IdentityIdentityBiasAdd:output:0^NoOp*
T0*0
_output_shapes
:??????????2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :??????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:??????????
 
_user_specified_nameinputs
?
a
E__inference_flatten_5_layer_call_and_return_conditional_losses_779068

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
?
?
R__inference_batch_normalization_29_layer_call_and_return_conditional_losses_778592

inputs&
readvariableop_resource:	?(
readvariableop_1_resource:	?7
(fusedbatchnormv3_readvariableop_resource:	?9
*fusedbatchnormv3_readvariableop_1_resource:	?
identity??AssignNewValue?AssignNewValue_1?FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1u
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:?*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:?*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:?*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:?*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:??????????:?:?:?:?:*
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
AssignNewValue_1x
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*0
_output_shapes
:??????????2

Identity?
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*7
_input_shapes&
$:??????????: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:X T
0
_output_shapes
:??????????
 
_user_specified_nameinputs
?
G
+__inference_dropout_37_layer_call_fn_780635

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_dropout_37_layer_call_and_return_conditional_losses_7783632
PartitionedCallu
IdentityIdentityPartitionedCall:output:0*
T0*0
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:??????????:X T
0
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
R__inference_batch_normalization_28_layer_call_and_return_conditional_losses_780412

inputs&
readvariableop_resource:	?(
readvariableop_1_resource:	?7
(fusedbatchnormv3_readvariableop_resource:	?9
*fusedbatchnormv3_readvariableop_1_resource:	?
identity??AssignNewValue?AssignNewValue_1?FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1u
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:?*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:?*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:?*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:?*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,????????????????????????????:?:?:?:?:*
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
T0*B
_output_shapes0
.:,????????????????????????????2

Identity?
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,????????????????????????????: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:j f
B
_output_shapes0
.:,????????????????????????????
 
_user_specified_nameinputs
?
?
E__inference_conv2d_49_layer_call_and_return_conditional_losses_780240

inputs:
conv2d_readvariableop_resource:??.
biasadd_readvariableop_resource:	?
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:??*
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????*
paddingSAME*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????2	
BiasAdda
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:??????????2
Reluv
IdentityIdentityRelu:activations:0^NoOp*
T0*0
_output_shapes
:??????????2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :??????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
R__inference_batch_normalization_29_layer_call_and_return_conditional_losses_780584

inputs&
readvariableop_resource:	?(
readvariableop_1_resource:	?7
(fusedbatchnormv3_readvariableop_resource:	?9
*fusedbatchnormv3_readvariableop_1_resource:	?
identity??AssignNewValue?AssignNewValue_1?FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1u
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:?*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:?*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:?*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:?*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,????????????????????????????:?:?:?:?:*
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
T0*B
_output_shapes0
.:,????????????????????????????2

Identity?
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,????????????????????????????: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:j f
B
_output_shapes0
.:,????????????????????????????
 
_user_specified_nameinputs
?
?
E__inference_conv2d_51_layer_call_and_return_conditional_losses_779025

inputs9
conv2d_readvariableop_resource:?-
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*'
_output_shapes
:?*
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
_construction_contextkEagerRuntime*3
_input_shapes"
 :??????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
R__inference_batch_normalization_28_layer_call_and_return_conditional_losses_778667

inputs&
readvariableop_resource:	?(
readvariableop_1_resource:	?7
(fusedbatchnormv3_readvariableop_resource:	?9
*fusedbatchnormv3_readvariableop_1_resource:	?
identity??AssignNewValue?AssignNewValue_1?FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1u
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:?*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:?*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:?*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:?*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:??????????:?:?:?:?:*
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
AssignNewValue_1x
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*0
_output_shapes
:??????????2

Identity?
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*7
_input_shapes&
$:??????????: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:X T
0
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
C__inference_dense_5_layer_call_and_return_conditional_losses_779095

inputs0
matmul_readvariableop_resource:@
identity??MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:@*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
MatMulX
TanhTanhMatMul:product:0*
T0*'
_output_shapes
:?????????2
Tanhc
IdentityIdentityTanh:y:0^NoOp*
T0*'
_output_shapes
:?????????2

Identityf
NoOpNoOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*(
_input_shapes
:?????????@: 2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:?????????@
 
_user_specified_nameinputs
?
?
)__inference_model_11_layer_call_fn_779960

inputs"
unknown:?
	unknown_0:	?
	unknown_1:	?
	unknown_2:	?
	unknown_3:	?
	unknown_4:	?%
	unknown_5:??
	unknown_6:	?
	unknown_7:	?
	unknown_8:	?
	unknown_9:	?

unknown_10:	?&

unknown_11:??

unknown_12:	?

unknown_13:	?

unknown_14:	?

unknown_15:	?

unknown_16:	?%

unknown_17:?

unknown_18:&

unknown_19:??

unknown_20:	?%

unknown_21:?I

unknown_22:I

unknown_23:@
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
unknown_22
unknown_23*%
Tin
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:?????????:?????????I*5
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_model_11_layer_call_and_return_conditional_losses_7793522
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
_construction_contextkEagerRuntime*`
_input_shapesO
M:?????????: : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?	
?
7__inference_batch_normalization_30_layer_call_fn_780681

inputs
unknown:	?
	unknown_0:	?
	unknown_1:	?
	unknown_2:	?
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,????????????????????????????*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *[
fVRT
R__inference_batch_normalization_30_layer_call_and_return_conditional_losses_7781402
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*B
_output_shapes0
.:,????????????????????????????2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,????????????????????????????: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:j f
B
_output_shapes0
.:,????????????????????????????
 
_user_specified_nameinputs
ٗ
?
D__inference_model_11_layer_call_and_return_conditional_losses_779750

inputsK
0model_8_conv2d_40_conv2d_readvariableop_resource:?@
1model_8_conv2d_40_biasadd_readvariableop_resource:	?E
6model_8_batch_normalization_28_readvariableop_resource:	?G
8model_8_batch_normalization_28_readvariableop_1_resource:	?V
Gmodel_8_batch_normalization_28_fusedbatchnormv3_readvariableop_resource:	?X
Imodel_8_batch_normalization_28_fusedbatchnormv3_readvariableop_1_resource:	?L
0model_8_conv2d_41_conv2d_readvariableop_resource:??@
1model_8_conv2d_41_biasadd_readvariableop_resource:	?E
6model_8_batch_normalization_29_readvariableop_resource:	?G
8model_8_batch_normalization_29_readvariableop_1_resource:	?V
Gmodel_8_batch_normalization_29_fusedbatchnormv3_readvariableop_resource:	?X
Imodel_8_batch_normalization_29_fusedbatchnormv3_readvariableop_1_resource:	?L
0model_8_conv2d_42_conv2d_readvariableop_resource:??@
1model_8_conv2d_42_biasadd_readvariableop_resource:	?E
6model_8_batch_normalization_30_readvariableop_resource:	?G
8model_8_batch_normalization_30_readvariableop_1_resource:	?V
Gmodel_8_batch_normalization_30_fusedbatchnormv3_readvariableop_resource:	?X
Imodel_8_batch_normalization_30_fusedbatchnormv3_readvariableop_1_resource:	?C
(conv2d_51_conv2d_readvariableop_resource:?7
)conv2d_51_biasadd_readvariableop_resource:D
(conv2d_49_conv2d_readvariableop_resource:??8
)conv2d_49_biasadd_readvariableop_resource:	?C
(conv2d_50_conv2d_readvariableop_resource:?I7
)conv2d_50_biasadd_readvariableop_resource:I8
&dense_5_matmul_readvariableop_resource:@
identity

identity_1?? conv2d_49/BiasAdd/ReadVariableOp?conv2d_49/Conv2D/ReadVariableOp? conv2d_50/BiasAdd/ReadVariableOp?conv2d_50/Conv2D/ReadVariableOp? conv2d_51/BiasAdd/ReadVariableOp?conv2d_51/Conv2D/ReadVariableOp?dense_5/MatMul/ReadVariableOp?>model_8/batch_normalization_28/FusedBatchNormV3/ReadVariableOp?@model_8/batch_normalization_28/FusedBatchNormV3/ReadVariableOp_1?-model_8/batch_normalization_28/ReadVariableOp?/model_8/batch_normalization_28/ReadVariableOp_1?>model_8/batch_normalization_29/FusedBatchNormV3/ReadVariableOp?@model_8/batch_normalization_29/FusedBatchNormV3/ReadVariableOp_1?-model_8/batch_normalization_29/ReadVariableOp?/model_8/batch_normalization_29/ReadVariableOp_1?>model_8/batch_normalization_30/FusedBatchNormV3/ReadVariableOp?@model_8/batch_normalization_30/FusedBatchNormV3/ReadVariableOp_1?-model_8/batch_normalization_30/ReadVariableOp?/model_8/batch_normalization_30/ReadVariableOp_1?(model_8/conv2d_40/BiasAdd/ReadVariableOp?'model_8/conv2d_40/Conv2D/ReadVariableOp?(model_8/conv2d_41/BiasAdd/ReadVariableOp?'model_8/conv2d_41/Conv2D/ReadVariableOp?(model_8/conv2d_42/BiasAdd/ReadVariableOp?'model_8/conv2d_42/Conv2D/ReadVariableOp?
'model_8/conv2d_40/Conv2D/ReadVariableOpReadVariableOp0model_8_conv2d_40_conv2d_readvariableop_resource*'
_output_shapes
:?*
dtype02)
'model_8/conv2d_40/Conv2D/ReadVariableOp?
model_8/conv2d_40/Conv2DConv2Dinputs/model_8/conv2d_40/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????*
paddingSAME*
strides
2
model_8/conv2d_40/Conv2D?
(model_8/conv2d_40/BiasAdd/ReadVariableOpReadVariableOp1model_8_conv2d_40_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02*
(model_8/conv2d_40/BiasAdd/ReadVariableOp?
model_8/conv2d_40/BiasAddBiasAdd!model_8/conv2d_40/Conv2D:output:00model_8/conv2d_40/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????2
model_8/conv2d_40/BiasAdd?
-model_8/batch_normalization_28/ReadVariableOpReadVariableOp6model_8_batch_normalization_28_readvariableop_resource*
_output_shapes	
:?*
dtype02/
-model_8/batch_normalization_28/ReadVariableOp?
/model_8/batch_normalization_28/ReadVariableOp_1ReadVariableOp8model_8_batch_normalization_28_readvariableop_1_resource*
_output_shapes	
:?*
dtype021
/model_8/batch_normalization_28/ReadVariableOp_1?
>model_8/batch_normalization_28/FusedBatchNormV3/ReadVariableOpReadVariableOpGmodel_8_batch_normalization_28_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:?*
dtype02@
>model_8/batch_normalization_28/FusedBatchNormV3/ReadVariableOp?
@model_8/batch_normalization_28/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpImodel_8_batch_normalization_28_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:?*
dtype02B
@model_8/batch_normalization_28/FusedBatchNormV3/ReadVariableOp_1?
/model_8/batch_normalization_28/FusedBatchNormV3FusedBatchNormV3"model_8/conv2d_40/BiasAdd:output:05model_8/batch_normalization_28/ReadVariableOp:value:07model_8/batch_normalization_28/ReadVariableOp_1:value:0Fmodel_8/batch_normalization_28/FusedBatchNormV3/ReadVariableOp:value:0Hmodel_8/batch_normalization_28/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:??????????:?:?:?:?:*
epsilon%o?:*
is_training( 21
/model_8/batch_normalization_28/FusedBatchNormV3?
model_8/activation_28/ReluRelu3model_8/batch_normalization_28/FusedBatchNormV3:y:0*
T0*0
_output_shapes
:??????????2
model_8/activation_28/Relu?
model_8/dropout_36/IdentityIdentity(model_8/activation_28/Relu:activations:0*
T0*0
_output_shapes
:??????????2
model_8/dropout_36/Identity?
'model_8/conv2d_41/Conv2D/ReadVariableOpReadVariableOp0model_8_conv2d_41_conv2d_readvariableop_resource*(
_output_shapes
:??*
dtype02)
'model_8/conv2d_41/Conv2D/ReadVariableOp?
model_8/conv2d_41/Conv2DConv2D$model_8/dropout_36/Identity:output:0/model_8/conv2d_41/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????*
paddingSAME*
strides
2
model_8/conv2d_41/Conv2D?
(model_8/conv2d_41/BiasAdd/ReadVariableOpReadVariableOp1model_8_conv2d_41_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02*
(model_8/conv2d_41/BiasAdd/ReadVariableOp?
model_8/conv2d_41/BiasAddBiasAdd!model_8/conv2d_41/Conv2D:output:00model_8/conv2d_41/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????2
model_8/conv2d_41/BiasAdd?
-model_8/batch_normalization_29/ReadVariableOpReadVariableOp6model_8_batch_normalization_29_readvariableop_resource*
_output_shapes	
:?*
dtype02/
-model_8/batch_normalization_29/ReadVariableOp?
/model_8/batch_normalization_29/ReadVariableOp_1ReadVariableOp8model_8_batch_normalization_29_readvariableop_1_resource*
_output_shapes	
:?*
dtype021
/model_8/batch_normalization_29/ReadVariableOp_1?
>model_8/batch_normalization_29/FusedBatchNormV3/ReadVariableOpReadVariableOpGmodel_8_batch_normalization_29_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:?*
dtype02@
>model_8/batch_normalization_29/FusedBatchNormV3/ReadVariableOp?
@model_8/batch_normalization_29/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpImodel_8_batch_normalization_29_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:?*
dtype02B
@model_8/batch_normalization_29/FusedBatchNormV3/ReadVariableOp_1?
/model_8/batch_normalization_29/FusedBatchNormV3FusedBatchNormV3"model_8/conv2d_41/BiasAdd:output:05model_8/batch_normalization_29/ReadVariableOp:value:07model_8/batch_normalization_29/ReadVariableOp_1:value:0Fmodel_8/batch_normalization_29/FusedBatchNormV3/ReadVariableOp:value:0Hmodel_8/batch_normalization_29/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:??????????:?:?:?:?:*
epsilon%o?:*
is_training( 21
/model_8/batch_normalization_29/FusedBatchNormV3?
model_8/activation_29/ReluRelu3model_8/batch_normalization_29/FusedBatchNormV3:y:0*
T0*0
_output_shapes
:??????????2
model_8/activation_29/Relu?
model_8/dropout_37/IdentityIdentity(model_8/activation_29/Relu:activations:0*
T0*0
_output_shapes
:??????????2
model_8/dropout_37/Identity?
'model_8/conv2d_42/Conv2D/ReadVariableOpReadVariableOp0model_8_conv2d_42_conv2d_readvariableop_resource*(
_output_shapes
:??*
dtype02)
'model_8/conv2d_42/Conv2D/ReadVariableOp?
model_8/conv2d_42/Conv2DConv2D$model_8/dropout_37/Identity:output:0/model_8/conv2d_42/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????*
paddingSAME*
strides
2
model_8/conv2d_42/Conv2D?
(model_8/conv2d_42/BiasAdd/ReadVariableOpReadVariableOp1model_8_conv2d_42_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02*
(model_8/conv2d_42/BiasAdd/ReadVariableOp?
model_8/conv2d_42/BiasAddBiasAdd!model_8/conv2d_42/Conv2D:output:00model_8/conv2d_42/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????2
model_8/conv2d_42/BiasAdd?
-model_8/batch_normalization_30/ReadVariableOpReadVariableOp6model_8_batch_normalization_30_readvariableop_resource*
_output_shapes	
:?*
dtype02/
-model_8/batch_normalization_30/ReadVariableOp?
/model_8/batch_normalization_30/ReadVariableOp_1ReadVariableOp8model_8_batch_normalization_30_readvariableop_1_resource*
_output_shapes	
:?*
dtype021
/model_8/batch_normalization_30/ReadVariableOp_1?
>model_8/batch_normalization_30/FusedBatchNormV3/ReadVariableOpReadVariableOpGmodel_8_batch_normalization_30_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:?*
dtype02@
>model_8/batch_normalization_30/FusedBatchNormV3/ReadVariableOp?
@model_8/batch_normalization_30/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpImodel_8_batch_normalization_30_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:?*
dtype02B
@model_8/batch_normalization_30/FusedBatchNormV3/ReadVariableOp_1?
/model_8/batch_normalization_30/FusedBatchNormV3FusedBatchNormV3"model_8/conv2d_42/BiasAdd:output:05model_8/batch_normalization_30/ReadVariableOp:value:07model_8/batch_normalization_30/ReadVariableOp_1:value:0Fmodel_8/batch_normalization_30/FusedBatchNormV3/ReadVariableOp:value:0Hmodel_8/batch_normalization_30/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:??????????:?:?:?:?:*
epsilon%o?:*
is_training( 21
/model_8/batch_normalization_30/FusedBatchNormV3?
%model_8/tf.__operators__.add_12/AddV2AddV23model_8/batch_normalization_30/FusedBatchNormV3:y:0$model_8/dropout_36/Identity:output:0*
T0*0
_output_shapes
:??????????2'
%model_8/tf.__operators__.add_12/AddV2?
model_8/activation_30/ReluRelu)model_8/tf.__operators__.add_12/AddV2:z:0*
T0*0
_output_shapes
:??????????2
model_8/activation_30/Relu?
model_8/dropout_38/IdentityIdentity(model_8/activation_30/Relu:activations:0*
T0*0
_output_shapes
:??????????2
model_8/dropout_38/Identity?
conv2d_51/Conv2D/ReadVariableOpReadVariableOp(conv2d_51_conv2d_readvariableop_resource*'
_output_shapes
:?*
dtype02!
conv2d_51/Conv2D/ReadVariableOp?
conv2d_51/Conv2DConv2D$model_8/dropout_38/Identity:output:0'conv2d_51/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????*
paddingVALID*
strides
2
conv2d_51/Conv2D?
 conv2d_51/BiasAdd/ReadVariableOpReadVariableOp)conv2d_51_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02"
 conv2d_51/BiasAdd/ReadVariableOp?
conv2d_51/BiasAddBiasAddconv2d_51/Conv2D:output:0(conv2d_51/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????2
conv2d_51/BiasAdd~
conv2d_51/ReluReluconv2d_51/BiasAdd:output:0*
T0*/
_output_shapes
:?????????2
conv2d_51/Relu?
conv2d_49/Conv2D/ReadVariableOpReadVariableOp(conv2d_49_conv2d_readvariableop_resource*(
_output_shapes
:??*
dtype02!
conv2d_49/Conv2D/ReadVariableOp?
conv2d_49/Conv2DConv2D$model_8/dropout_38/Identity:output:0'conv2d_49/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????*
paddingSAME*
strides
2
conv2d_49/Conv2D?
 conv2d_49/BiasAdd/ReadVariableOpReadVariableOp)conv2d_49_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02"
 conv2d_49/BiasAdd/ReadVariableOp?
conv2d_49/BiasAddBiasAddconv2d_49/Conv2D:output:0(conv2d_49/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????2
conv2d_49/BiasAdd
conv2d_49/ReluReluconv2d_49/BiasAdd:output:0*
T0*0
_output_shapes
:??????????2
conv2d_49/Relu?
dropout_45/IdentityIdentityconv2d_51/Relu:activations:0*
T0*/
_output_shapes
:?????????2
dropout_45/Identity?
dropout_44/IdentityIdentityconv2d_49/Relu:activations:0*
T0*0
_output_shapes
:??????????2
dropout_44/Identitys
flatten_5/ConstConst*
_output_shapes
:*
dtype0*
valueB"????@   2
flatten_5/Const?
flatten_5/ReshapeReshapedropout_45/Identity:output:0flatten_5/Const:output:0*
T0*'
_output_shapes
:?????????@2
flatten_5/Reshape?
conv2d_50/Conv2D/ReadVariableOpReadVariableOp(conv2d_50_conv2d_readvariableop_resource*'
_output_shapes
:?I*
dtype02!
conv2d_50/Conv2D/ReadVariableOp?
conv2d_50/Conv2DConv2Ddropout_44/Identity:output:0'conv2d_50/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????I*
paddingSAME*
strides
2
conv2d_50/Conv2D?
 conv2d_50/BiasAdd/ReadVariableOpReadVariableOp)conv2d_50_biasadd_readvariableop_resource*
_output_shapes
:I*
dtype02"
 conv2d_50/BiasAdd/ReadVariableOp?
conv2d_50/BiasAddBiasAddconv2d_50/Conv2D:output:0(conv2d_50/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????I2
conv2d_50/BiasAdd?
conv2d_50/SoftmaxSoftmaxconv2d_50/BiasAdd:output:0*
T0*/
_output_shapes
:?????????I2
conv2d_50/Softmax?
dense_5/MatMul/ReadVariableOpReadVariableOp&dense_5_matmul_readvariableop_resource*
_output_shapes

:@*
dtype02
dense_5/MatMul/ReadVariableOp?
dense_5/MatMulMatMulflatten_5/Reshape:output:0%dense_5/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_5/MatMulp
dense_5/TanhTanhdense_5/MatMul:product:0*
T0*'
_output_shapes
:?????????2
dense_5/Tanhk
IdentityIdentitydense_5/Tanh:y:0^NoOp*
T0*'
_output_shapes
:?????????2

Identity?

Identity_1Identityconv2d_50/Softmax:softmax:0^NoOp*
T0*/
_output_shapes
:?????????I2

Identity_1?	
NoOpNoOp!^conv2d_49/BiasAdd/ReadVariableOp ^conv2d_49/Conv2D/ReadVariableOp!^conv2d_50/BiasAdd/ReadVariableOp ^conv2d_50/Conv2D/ReadVariableOp!^conv2d_51/BiasAdd/ReadVariableOp ^conv2d_51/Conv2D/ReadVariableOp^dense_5/MatMul/ReadVariableOp?^model_8/batch_normalization_28/FusedBatchNormV3/ReadVariableOpA^model_8/batch_normalization_28/FusedBatchNormV3/ReadVariableOp_1.^model_8/batch_normalization_28/ReadVariableOp0^model_8/batch_normalization_28/ReadVariableOp_1?^model_8/batch_normalization_29/FusedBatchNormV3/ReadVariableOpA^model_8/batch_normalization_29/FusedBatchNormV3/ReadVariableOp_1.^model_8/batch_normalization_29/ReadVariableOp0^model_8/batch_normalization_29/ReadVariableOp_1?^model_8/batch_normalization_30/FusedBatchNormV3/ReadVariableOpA^model_8/batch_normalization_30/FusedBatchNormV3/ReadVariableOp_1.^model_8/batch_normalization_30/ReadVariableOp0^model_8/batch_normalization_30/ReadVariableOp_1)^model_8/conv2d_40/BiasAdd/ReadVariableOp(^model_8/conv2d_40/Conv2D/ReadVariableOp)^model_8/conv2d_41/BiasAdd/ReadVariableOp(^model_8/conv2d_41/Conv2D/ReadVariableOp)^model_8/conv2d_42/BiasAdd/ReadVariableOp(^model_8/conv2d_42/Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*`
_input_shapesO
M:?????????: : : : : : : : : : : : : : : : : : : : : : : : : 2D
 conv2d_49/BiasAdd/ReadVariableOp conv2d_49/BiasAdd/ReadVariableOp2B
conv2d_49/Conv2D/ReadVariableOpconv2d_49/Conv2D/ReadVariableOp2D
 conv2d_50/BiasAdd/ReadVariableOp conv2d_50/BiasAdd/ReadVariableOp2B
conv2d_50/Conv2D/ReadVariableOpconv2d_50/Conv2D/ReadVariableOp2D
 conv2d_51/BiasAdd/ReadVariableOp conv2d_51/BiasAdd/ReadVariableOp2B
conv2d_51/Conv2D/ReadVariableOpconv2d_51/Conv2D/ReadVariableOp2>
dense_5/MatMul/ReadVariableOpdense_5/MatMul/ReadVariableOp2?
>model_8/batch_normalization_28/FusedBatchNormV3/ReadVariableOp>model_8/batch_normalization_28/FusedBatchNormV3/ReadVariableOp2?
@model_8/batch_normalization_28/FusedBatchNormV3/ReadVariableOp_1@model_8/batch_normalization_28/FusedBatchNormV3/ReadVariableOp_12^
-model_8/batch_normalization_28/ReadVariableOp-model_8/batch_normalization_28/ReadVariableOp2b
/model_8/batch_normalization_28/ReadVariableOp_1/model_8/batch_normalization_28/ReadVariableOp_12?
>model_8/batch_normalization_29/FusedBatchNormV3/ReadVariableOp>model_8/batch_normalization_29/FusedBatchNormV3/ReadVariableOp2?
@model_8/batch_normalization_29/FusedBatchNormV3/ReadVariableOp_1@model_8/batch_normalization_29/FusedBatchNormV3/ReadVariableOp_12^
-model_8/batch_normalization_29/ReadVariableOp-model_8/batch_normalization_29/ReadVariableOp2b
/model_8/batch_normalization_29/ReadVariableOp_1/model_8/batch_normalization_29/ReadVariableOp_12?
>model_8/batch_normalization_30/FusedBatchNormV3/ReadVariableOp>model_8/batch_normalization_30/FusedBatchNormV3/ReadVariableOp2?
@model_8/batch_normalization_30/FusedBatchNormV3/ReadVariableOp_1@model_8/batch_normalization_30/FusedBatchNormV3/ReadVariableOp_12^
-model_8/batch_normalization_30/ReadVariableOp-model_8/batch_normalization_30/ReadVariableOp2b
/model_8/batch_normalization_30/ReadVariableOp_1/model_8/batch_normalization_30/ReadVariableOp_12T
(model_8/conv2d_40/BiasAdd/ReadVariableOp(model_8/conv2d_40/BiasAdd/ReadVariableOp2R
'model_8/conv2d_40/Conv2D/ReadVariableOp'model_8/conv2d_40/Conv2D/ReadVariableOp2T
(model_8/conv2d_41/BiasAdd/ReadVariableOp(model_8/conv2d_41/BiasAdd/ReadVariableOp2R
'model_8/conv2d_41/Conv2D/ReadVariableOp'model_8/conv2d_41/Conv2D/ReadVariableOp2T
(model_8/conv2d_42/BiasAdd/ReadVariableOp(model_8/conv2d_42/BiasAdd/ReadVariableOp2R
'model_8/conv2d_42/Conv2D/ReadVariableOp'model_8/conv2d_42/Conv2D/ReadVariableOp:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
R__inference_batch_normalization_28_layer_call_and_return_conditional_losses_777932

inputs&
readvariableop_resource:	?(
readvariableop_1_resource:	?7
(fusedbatchnormv3_readvariableop_resource:	?9
*fusedbatchnormv3_readvariableop_1_resource:	?
identity??AssignNewValue?AssignNewValue_1?FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1u
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:?*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:?*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:?*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:?*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,????????????????????????????:?:?:?:?:*
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
T0*B
_output_shapes0
.:,????????????????????????????2

Identity?
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,????????????????????????????: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:j f
B
_output_shapes0
.:,????????????????????????????
 
_user_specified_nameinputs
?
e
I__inference_activation_29_layer_call_and_return_conditional_losses_780630

inputs
identityW
ReluReluinputs*
T0*0
_output_shapes
:??????????2
Reluo
IdentityIdentityRelu:activations:0*
T0*0
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:??????????:X T
0
_output_shapes
:??????????
 
_user_specified_nameinputs
?.
?	
D__inference_model_11_layer_call_and_return_conditional_losses_779590
input_12)
model_8_779530:?
model_8_779532:	?
model_8_779534:	?
model_8_779536:	?
model_8_779538:	?
model_8_779540:	?*
model_8_779542:??
model_8_779544:	?
model_8_779546:	?
model_8_779548:	?
model_8_779550:	?
model_8_779552:	?*
model_8_779554:??
model_8_779556:	?
model_8_779558:	?
model_8_779560:	?
model_8_779562:	?
model_8_779564:	?+
conv2d_51_779567:?
conv2d_51_779569:,
conv2d_49_779572:??
conv2d_49_779574:	?+
conv2d_50_779580:?I
conv2d_50_779582:I 
dense_5_779585:@
identity

identity_1??!conv2d_49/StatefulPartitionedCall?!conv2d_50/StatefulPartitionedCall?!conv2d_51/StatefulPartitionedCall?dense_5/StatefulPartitionedCall?model_8/StatefulPartitionedCall?
model_8/StatefulPartitionedCallStatefulPartitionedCallinput_12model_8_779530model_8_779532model_8_779534model_8_779536model_8_779538model_8_779540model_8_779542model_8_779544model_8_779546model_8_779548model_8_779550model_8_779552model_8_779554model_8_779556model_8_779558model_8_779560model_8_779562model_8_779564*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????*.
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_model_8_layer_call_and_return_conditional_losses_7787842!
model_8/StatefulPartitionedCall?
!conv2d_51/StatefulPartitionedCallStatefulPartitionedCall(model_8/StatefulPartitionedCall:output:0conv2d_51_779567conv2d_51_779569*
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
GPU 2J 8? *N
fIRG
E__inference_conv2d_51_layer_call_and_return_conditional_losses_7790252#
!conv2d_51/StatefulPartitionedCall?
!conv2d_49/StatefulPartitionedCallStatefulPartitionedCall(model_8/StatefulPartitionedCall:output:0conv2d_49_779572conv2d_49_779574*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_conv2d_49_layer_call_and_return_conditional_losses_7790422#
!conv2d_49/StatefulPartitionedCall?
dropout_45/PartitionedCallPartitionedCall*conv2d_51/StatefulPartitionedCall:output:0*
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
F__inference_dropout_45_layer_call_and_return_conditional_losses_7792072
dropout_45/PartitionedCall?
dropout_44/PartitionedCallPartitionedCall*conv2d_49/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_dropout_44_layer_call_and_return_conditional_losses_7791922
dropout_44/PartitionedCall?
flatten_5/PartitionedCallPartitionedCall#dropout_45/PartitionedCall:output:0*
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
GPU 2J 8? *N
fIRG
E__inference_flatten_5_layer_call_and_return_conditional_losses_7790682
flatten_5/PartitionedCall?
!conv2d_50/StatefulPartitionedCallStatefulPartitionedCall#dropout_44/PartitionedCall:output:0conv2d_50_779580conv2d_50_779582*
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
GPU 2J 8? *N
fIRG
E__inference_conv2d_50_layer_call_and_return_conditional_losses_7790812#
!conv2d_50/StatefulPartitionedCall?
dense_5/StatefulPartitionedCallStatefulPartitionedCall"flatten_5/PartitionedCall:output:0dense_5_779585*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_dense_5_layer_call_and_return_conditional_losses_7790952!
dense_5/StatefulPartitionedCall?
IdentityIdentity(dense_5/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????2

Identity?

Identity_1Identity*conv2d_50/StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:?????????I2

Identity_1?
NoOpNoOp"^conv2d_49/StatefulPartitionedCall"^conv2d_50/StatefulPartitionedCall"^conv2d_51/StatefulPartitionedCall ^dense_5/StatefulPartitionedCall ^model_8/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*`
_input_shapesO
M:?????????: : : : : : : : : : : : : : : : : : : : : : : : : 2F
!conv2d_49/StatefulPartitionedCall!conv2d_49/StatefulPartitionedCall2F
!conv2d_50/StatefulPartitionedCall!conv2d_50/StatefulPartitionedCall2F
!conv2d_51/StatefulPartitionedCall!conv2d_51/StatefulPartitionedCall2B
dense_5/StatefulPartitionedCalldense_5/StatefulPartitionedCall2B
model_8/StatefulPartitionedCallmodel_8/StatefulPartitionedCall:Y U
/
_output_shapes
:?????????
"
_user_specified_name
input_12
?
d
F__inference_dropout_36_layer_call_and_return_conditional_losses_778306

inputs

identity_1c
IdentityIdentityinputs*
T0*0
_output_shapes
:??????????2

Identityr

Identity_1IdentityIdentity:output:0*
T0*0
_output_shapes
:??????????2

Identity_1"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:??????????:X T
0
_output_shapes
:??????????
 
_user_specified_nameinputs
?
b
F__inference_dropout_45_layer_call_and_return_conditional_losses_779207

inputs
identityb
IdentityIdentityinputs*
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
?
E__inference_conv2d_51_layer_call_and_return_conditional_losses_780201

inputs9
conv2d_readvariableop_resource:?-
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*'
_output_shapes
:?*
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
_construction_contextkEagerRuntime*3
_input_shapes"
 :??????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:??????????
 
_user_specified_nameinputs
?	
?
7__inference_batch_normalization_30_layer_call_fn_780707

inputs
unknown:	?
	unknown_0:	?
	unknown_1:	?
	unknown_2:	?
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *[
fVRT
R__inference_batch_normalization_30_layer_call_and_return_conditional_losses_7783982
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*0
_output_shapes
:??????????2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*7
_input_shapes&
$:??????????: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:??????????
 
_user_specified_nameinputs
?
G
+__inference_dropout_44_layer_call_fn_780256

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_dropout_44_layer_call_and_return_conditional_losses_7790602
PartitionedCallu
IdentityIdentityPartitionedCall:output:0*
T0*0
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:??????????:X T
0
_output_shapes
:??????????
 
_user_specified_nameinputs
?a
?
C__inference_model_8_layer_call_and_return_conditional_losses_780031

inputsC
(conv2d_40_conv2d_readvariableop_resource:?8
)conv2d_40_biasadd_readvariableop_resource:	?=
.batch_normalization_28_readvariableop_resource:	??
0batch_normalization_28_readvariableop_1_resource:	?N
?batch_normalization_28_fusedbatchnormv3_readvariableop_resource:	?P
Abatch_normalization_28_fusedbatchnormv3_readvariableop_1_resource:	?D
(conv2d_41_conv2d_readvariableop_resource:??8
)conv2d_41_biasadd_readvariableop_resource:	?=
.batch_normalization_29_readvariableop_resource:	??
0batch_normalization_29_readvariableop_1_resource:	?N
?batch_normalization_29_fusedbatchnormv3_readvariableop_resource:	?P
Abatch_normalization_29_fusedbatchnormv3_readvariableop_1_resource:	?D
(conv2d_42_conv2d_readvariableop_resource:??8
)conv2d_42_biasadd_readvariableop_resource:	?=
.batch_normalization_30_readvariableop_resource:	??
0batch_normalization_30_readvariableop_1_resource:	?N
?batch_normalization_30_fusedbatchnormv3_readvariableop_resource:	?P
Abatch_normalization_30_fusedbatchnormv3_readvariableop_1_resource:	?
identity??6batch_normalization_28/FusedBatchNormV3/ReadVariableOp?8batch_normalization_28/FusedBatchNormV3/ReadVariableOp_1?%batch_normalization_28/ReadVariableOp?'batch_normalization_28/ReadVariableOp_1?6batch_normalization_29/FusedBatchNormV3/ReadVariableOp?8batch_normalization_29/FusedBatchNormV3/ReadVariableOp_1?%batch_normalization_29/ReadVariableOp?'batch_normalization_29/ReadVariableOp_1?6batch_normalization_30/FusedBatchNormV3/ReadVariableOp?8batch_normalization_30/FusedBatchNormV3/ReadVariableOp_1?%batch_normalization_30/ReadVariableOp?'batch_normalization_30/ReadVariableOp_1? conv2d_40/BiasAdd/ReadVariableOp?conv2d_40/Conv2D/ReadVariableOp? conv2d_41/BiasAdd/ReadVariableOp?conv2d_41/Conv2D/ReadVariableOp? conv2d_42/BiasAdd/ReadVariableOp?conv2d_42/Conv2D/ReadVariableOp?
conv2d_40/Conv2D/ReadVariableOpReadVariableOp(conv2d_40_conv2d_readvariableop_resource*'
_output_shapes
:?*
dtype02!
conv2d_40/Conv2D/ReadVariableOp?
conv2d_40/Conv2DConv2Dinputs'conv2d_40/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????*
paddingSAME*
strides
2
conv2d_40/Conv2D?
 conv2d_40/BiasAdd/ReadVariableOpReadVariableOp)conv2d_40_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02"
 conv2d_40/BiasAdd/ReadVariableOp?
conv2d_40/BiasAddBiasAddconv2d_40/Conv2D:output:0(conv2d_40/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????2
conv2d_40/BiasAdd?
%batch_normalization_28/ReadVariableOpReadVariableOp.batch_normalization_28_readvariableop_resource*
_output_shapes	
:?*
dtype02'
%batch_normalization_28/ReadVariableOp?
'batch_normalization_28/ReadVariableOp_1ReadVariableOp0batch_normalization_28_readvariableop_1_resource*
_output_shapes	
:?*
dtype02)
'batch_normalization_28/ReadVariableOp_1?
6batch_normalization_28/FusedBatchNormV3/ReadVariableOpReadVariableOp?batch_normalization_28_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:?*
dtype028
6batch_normalization_28/FusedBatchNormV3/ReadVariableOp?
8batch_normalization_28/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpAbatch_normalization_28_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:?*
dtype02:
8batch_normalization_28/FusedBatchNormV3/ReadVariableOp_1?
'batch_normalization_28/FusedBatchNormV3FusedBatchNormV3conv2d_40/BiasAdd:output:0-batch_normalization_28/ReadVariableOp:value:0/batch_normalization_28/ReadVariableOp_1:value:0>batch_normalization_28/FusedBatchNormV3/ReadVariableOp:value:0@batch_normalization_28/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:??????????:?:?:?:?:*
epsilon%o?:*
is_training( 2)
'batch_normalization_28/FusedBatchNormV3?
activation_28/ReluRelu+batch_normalization_28/FusedBatchNormV3:y:0*
T0*0
_output_shapes
:??????????2
activation_28/Relu?
dropout_36/IdentityIdentity activation_28/Relu:activations:0*
T0*0
_output_shapes
:??????????2
dropout_36/Identity?
conv2d_41/Conv2D/ReadVariableOpReadVariableOp(conv2d_41_conv2d_readvariableop_resource*(
_output_shapes
:??*
dtype02!
conv2d_41/Conv2D/ReadVariableOp?
conv2d_41/Conv2DConv2Ddropout_36/Identity:output:0'conv2d_41/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????*
paddingSAME*
strides
2
conv2d_41/Conv2D?
 conv2d_41/BiasAdd/ReadVariableOpReadVariableOp)conv2d_41_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02"
 conv2d_41/BiasAdd/ReadVariableOp?
conv2d_41/BiasAddBiasAddconv2d_41/Conv2D:output:0(conv2d_41/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????2
conv2d_41/BiasAdd?
%batch_normalization_29/ReadVariableOpReadVariableOp.batch_normalization_29_readvariableop_resource*
_output_shapes	
:?*
dtype02'
%batch_normalization_29/ReadVariableOp?
'batch_normalization_29/ReadVariableOp_1ReadVariableOp0batch_normalization_29_readvariableop_1_resource*
_output_shapes	
:?*
dtype02)
'batch_normalization_29/ReadVariableOp_1?
6batch_normalization_29/FusedBatchNormV3/ReadVariableOpReadVariableOp?batch_normalization_29_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:?*
dtype028
6batch_normalization_29/FusedBatchNormV3/ReadVariableOp?
8batch_normalization_29/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpAbatch_normalization_29_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:?*
dtype02:
8batch_normalization_29/FusedBatchNormV3/ReadVariableOp_1?
'batch_normalization_29/FusedBatchNormV3FusedBatchNormV3conv2d_41/BiasAdd:output:0-batch_normalization_29/ReadVariableOp:value:0/batch_normalization_29/ReadVariableOp_1:value:0>batch_normalization_29/FusedBatchNormV3/ReadVariableOp:value:0@batch_normalization_29/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:??????????:?:?:?:?:*
epsilon%o?:*
is_training( 2)
'batch_normalization_29/FusedBatchNormV3?
activation_29/ReluRelu+batch_normalization_29/FusedBatchNormV3:y:0*
T0*0
_output_shapes
:??????????2
activation_29/Relu?
dropout_37/IdentityIdentity activation_29/Relu:activations:0*
T0*0
_output_shapes
:??????????2
dropout_37/Identity?
conv2d_42/Conv2D/ReadVariableOpReadVariableOp(conv2d_42_conv2d_readvariableop_resource*(
_output_shapes
:??*
dtype02!
conv2d_42/Conv2D/ReadVariableOp?
conv2d_42/Conv2DConv2Ddropout_37/Identity:output:0'conv2d_42/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????*
paddingSAME*
strides
2
conv2d_42/Conv2D?
 conv2d_42/BiasAdd/ReadVariableOpReadVariableOp)conv2d_42_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02"
 conv2d_42/BiasAdd/ReadVariableOp?
conv2d_42/BiasAddBiasAddconv2d_42/Conv2D:output:0(conv2d_42/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????2
conv2d_42/BiasAdd?
%batch_normalization_30/ReadVariableOpReadVariableOp.batch_normalization_30_readvariableop_resource*
_output_shapes	
:?*
dtype02'
%batch_normalization_30/ReadVariableOp?
'batch_normalization_30/ReadVariableOp_1ReadVariableOp0batch_normalization_30_readvariableop_1_resource*
_output_shapes	
:?*
dtype02)
'batch_normalization_30/ReadVariableOp_1?
6batch_normalization_30/FusedBatchNormV3/ReadVariableOpReadVariableOp?batch_normalization_30_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:?*
dtype028
6batch_normalization_30/FusedBatchNormV3/ReadVariableOp?
8batch_normalization_30/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpAbatch_normalization_30_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:?*
dtype02:
8batch_normalization_30/FusedBatchNormV3/ReadVariableOp_1?
'batch_normalization_30/FusedBatchNormV3FusedBatchNormV3conv2d_42/BiasAdd:output:0-batch_normalization_30/ReadVariableOp:value:0/batch_normalization_30/ReadVariableOp_1:value:0>batch_normalization_30/FusedBatchNormV3/ReadVariableOp:value:0@batch_normalization_30/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:??????????:?:?:?:?:*
epsilon%o?:*
is_training( 2)
'batch_normalization_30/FusedBatchNormV3?
tf.__operators__.add_12/AddV2AddV2+batch_normalization_30/FusedBatchNormV3:y:0dropout_36/Identity:output:0*
T0*0
_output_shapes
:??????????2
tf.__operators__.add_12/AddV2?
activation_30/ReluRelu!tf.__operators__.add_12/AddV2:z:0*
T0*0
_output_shapes
:??????????2
activation_30/Relu?
dropout_38/IdentityIdentity activation_30/Relu:activations:0*
T0*0
_output_shapes
:??????????2
dropout_38/Identity?
IdentityIdentitydropout_38/Identity:output:0^NoOp*
T0*0
_output_shapes
:??????????2

Identity?
NoOpNoOp7^batch_normalization_28/FusedBatchNormV3/ReadVariableOp9^batch_normalization_28/FusedBatchNormV3/ReadVariableOp_1&^batch_normalization_28/ReadVariableOp(^batch_normalization_28/ReadVariableOp_17^batch_normalization_29/FusedBatchNormV3/ReadVariableOp9^batch_normalization_29/FusedBatchNormV3/ReadVariableOp_1&^batch_normalization_29/ReadVariableOp(^batch_normalization_29/ReadVariableOp_17^batch_normalization_30/FusedBatchNormV3/ReadVariableOp9^batch_normalization_30/FusedBatchNormV3/ReadVariableOp_1&^batch_normalization_30/ReadVariableOp(^batch_normalization_30/ReadVariableOp_1!^conv2d_40/BiasAdd/ReadVariableOp ^conv2d_40/Conv2D/ReadVariableOp!^conv2d_41/BiasAdd/ReadVariableOp ^conv2d_41/Conv2D/ReadVariableOp!^conv2d_42/BiasAdd/ReadVariableOp ^conv2d_42/Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*R
_input_shapesA
?:?????????: : : : : : : : : : : : : : : : : : 2p
6batch_normalization_28/FusedBatchNormV3/ReadVariableOp6batch_normalization_28/FusedBatchNormV3/ReadVariableOp2t
8batch_normalization_28/FusedBatchNormV3/ReadVariableOp_18batch_normalization_28/FusedBatchNormV3/ReadVariableOp_12N
%batch_normalization_28/ReadVariableOp%batch_normalization_28/ReadVariableOp2R
'batch_normalization_28/ReadVariableOp_1'batch_normalization_28/ReadVariableOp_12p
6batch_normalization_29/FusedBatchNormV3/ReadVariableOp6batch_normalization_29/FusedBatchNormV3/ReadVariableOp2t
8batch_normalization_29/FusedBatchNormV3/ReadVariableOp_18batch_normalization_29/FusedBatchNormV3/ReadVariableOp_12N
%batch_normalization_29/ReadVariableOp%batch_normalization_29/ReadVariableOp2R
'batch_normalization_29/ReadVariableOp_1'batch_normalization_29/ReadVariableOp_12p
6batch_normalization_30/FusedBatchNormV3/ReadVariableOp6batch_normalization_30/FusedBatchNormV3/ReadVariableOp2t
8batch_normalization_30/FusedBatchNormV3/ReadVariableOp_18batch_normalization_30/FusedBatchNormV3/ReadVariableOp_12N
%batch_normalization_30/ReadVariableOp%batch_normalization_30/ReadVariableOp2R
'batch_normalization_30/ReadVariableOp_1'batch_normalization_30/ReadVariableOp_12D
 conv2d_40/BiasAdd/ReadVariableOp conv2d_40/BiasAdd/ReadVariableOp2B
conv2d_40/Conv2D/ReadVariableOpconv2d_40/Conv2D/ReadVariableOp2D
 conv2d_41/BiasAdd/ReadVariableOp conv2d_41/BiasAdd/ReadVariableOp2B
conv2d_41/Conv2D/ReadVariableOpconv2d_41/Conv2D/ReadVariableOp2D
 conv2d_42/BiasAdd/ReadVariableOp conv2d_42/BiasAdd/ReadVariableOp2B
conv2d_42/Conv2D/ReadVariableOpconv2d_42/Conv2D/ReadVariableOp:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
R__inference_batch_normalization_29_layer_call_and_return_conditional_losses_778014

inputs&
readvariableop_resource:	?(
readvariableop_1_resource:	?7
(fusedbatchnormv3_readvariableop_resource:	?9
*fusedbatchnormv3_readvariableop_1_resource:	?
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1u
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:?*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:?*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:?*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:?*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,????????????????????????????:?:?:?:?:*
epsilon%o?:*
is_training( 2
FusedBatchNormV3?
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*B
_output_shapes0
.:,????????????????????????????2

Identity?
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,????????????????????????????: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:j f
B
_output_shapes0
.:,????????????????????????????
 
_user_specified_nameinputs
?
?
E__inference_conv2d_42_layer_call_and_return_conditional_losses_778375

inputs:
conv2d_readvariableop_resource:??.
biasadd_readvariableop_resource:	?
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:??*
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????*
paddingSAME*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????2	
BiasAddt
IdentityIdentityBiasAdd:output:0^NoOp*
T0*0
_output_shapes
:??????????2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :??????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:??????????
 
_user_specified_nameinputs
?
b
F__inference_dropout_37_layer_call_and_return_conditional_losses_780649

inputs
identityc
IdentityIdentityinputs*
T0*0
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:??????????:X T
0
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
R__inference_batch_normalization_30_layer_call_and_return_conditional_losses_780774

inputs&
readvariableop_resource:	?(
readvariableop_1_resource:	?7
(fusedbatchnormv3_readvariableop_resource:	?9
*fusedbatchnormv3_readvariableop_1_resource:	?
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1u
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:?*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:?*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:?*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:?*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:??????????:?:?:?:?:*
epsilon%o?:*
is_training( 2
FusedBatchNormV3x
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*0
_output_shapes
:??????????2

Identity?
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*7
_input_shapes&
$:??????????: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:X T
0
_output_shapes
:??????????
 
_user_specified_nameinputs
?
a
E__inference_flatten_5_layer_call_and_return_conditional_losses_780251

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
?
b
F__inference_dropout_38_layer_call_and_return_conditional_losses_778475

inputs
identityc
IdentityIdentityinputs*
T0*0
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:??????????:X T
0
_output_shapes
:??????????
 
_user_specified_nameinputs
?
e
I__inference_activation_30_layer_call_and_return_conditional_losses_780802

inputs
identityW
ReluReluinputs*
T0*0
_output_shapes
:??????????2
Reluo
IdentityIdentityRelu:activations:0*
T0*0
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:??????????:X T
0
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
*__inference_conv2d_50_layer_call_fn_780294

inputs"
unknown:?I
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
GPU 2J 8? *N
fIRG
E__inference_conv2d_50_layer_call_and_return_conditional_losses_7790812
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
_construction_contextkEagerRuntime*3
_input_shapes"
 :??????????: : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
R__inference_batch_normalization_30_layer_call_and_return_conditional_losses_778140

inputs&
readvariableop_resource:	?(
readvariableop_1_resource:	?7
(fusedbatchnormv3_readvariableop_resource:	?9
*fusedbatchnormv3_readvariableop_1_resource:	?
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1u
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:?*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:?*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:?*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:?*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,????????????????????????????:?:?:?:?:*
epsilon%o?:*
is_training( 2
FusedBatchNormV3?
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*B
_output_shapes0
.:,????????????????????????????2

Identity?
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,????????????????????????????: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:j f
B
_output_shapes0
.:,????????????????????????????
 
_user_specified_nameinputs
??
?
C__inference_model_8_layer_call_and_return_conditional_losses_778970
input_9+
conv2d_40_778920:?
conv2d_40_778922:	?,
batch_normalization_28_778925:	?,
batch_normalization_28_778927:	?,
batch_normalization_28_778929:	?,
batch_normalization_28_778931:	?,
conv2d_41_778936:??
conv2d_41_778938:	?,
batch_normalization_29_778941:	?,
batch_normalization_29_778943:	?,
batch_normalization_29_778945:	?,
batch_normalization_29_778947:	?,
conv2d_42_778952:??
conv2d_42_778954:	?,
batch_normalization_30_778957:	?,
batch_normalization_30_778959:	?,
batch_normalization_30_778961:	?,
batch_normalization_30_778963:	?
identity??.batch_normalization_28/StatefulPartitionedCall?.batch_normalization_29/StatefulPartitionedCall?.batch_normalization_30/StatefulPartitionedCall?!conv2d_40/StatefulPartitionedCall?!conv2d_41/StatefulPartitionedCall?!conv2d_42/StatefulPartitionedCall?
!conv2d_40/StatefulPartitionedCallStatefulPartitionedCallinput_9conv2d_40_778920conv2d_40_778922*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_conv2d_40_layer_call_and_return_conditional_losses_7782612#
!conv2d_40/StatefulPartitionedCall?
.batch_normalization_28/StatefulPartitionedCallStatefulPartitionedCall*conv2d_40/StatefulPartitionedCall:output:0batch_normalization_28_778925batch_normalization_28_778927batch_normalization_28_778929batch_normalization_28_778931*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *[
fVRT
R__inference_batch_normalization_28_layer_call_and_return_conditional_losses_77866720
.batch_normalization_28/StatefulPartitionedCall?
activation_28/PartitionedCallPartitionedCall7batch_normalization_28/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *R
fMRK
I__inference_activation_28_layer_call_and_return_conditional_losses_7782992
activation_28/PartitionedCall?
dropout_36/PartitionedCallPartitionedCall&activation_28/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_dropout_36_layer_call_and_return_conditional_losses_7786252
dropout_36/PartitionedCall?
!conv2d_41/StatefulPartitionedCallStatefulPartitionedCall#dropout_36/PartitionedCall:output:0conv2d_41_778936conv2d_41_778938*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_conv2d_41_layer_call_and_return_conditional_losses_7783182#
!conv2d_41/StatefulPartitionedCall?
.batch_normalization_29/StatefulPartitionedCallStatefulPartitionedCall*conv2d_41/StatefulPartitionedCall:output:0batch_normalization_29_778941batch_normalization_29_778943batch_normalization_29_778945batch_normalization_29_778947*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *[
fVRT
R__inference_batch_normalization_29_layer_call_and_return_conditional_losses_77859220
.batch_normalization_29/StatefulPartitionedCall?
activation_29/PartitionedCallPartitionedCall7batch_normalization_29/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *R
fMRK
I__inference_activation_29_layer_call_and_return_conditional_losses_7783562
activation_29/PartitionedCall?
dropout_37/PartitionedCallPartitionedCall&activation_29/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_dropout_37_layer_call_and_return_conditional_losses_7785502
dropout_37/PartitionedCall?
!conv2d_42/StatefulPartitionedCallStatefulPartitionedCall#dropout_37/PartitionedCall:output:0conv2d_42_778952conv2d_42_778954*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_conv2d_42_layer_call_and_return_conditional_losses_7783752#
!conv2d_42/StatefulPartitionedCall?
.batch_normalization_30/StatefulPartitionedCallStatefulPartitionedCall*conv2d_42/StatefulPartitionedCall:output:0batch_normalization_30_778957batch_normalization_30_778959batch_normalization_30_778961batch_normalization_30_778963*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *[
fVRT
R__inference_batch_normalization_30_layer_call_and_return_conditional_losses_77851720
.batch_normalization_30/StatefulPartitionedCall?
tf.__operators__.add_12/AddV2AddV27batch_normalization_30/StatefulPartitionedCall:output:0#dropout_36/PartitionedCall:output:0*
T0*0
_output_shapes
:??????????2
tf.__operators__.add_12/AddV2?
activation_30/PartitionedCallPartitionedCall!tf.__operators__.add_12/AddV2:z:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *R
fMRK
I__inference_activation_30_layer_call_and_return_conditional_losses_7784142
activation_30/PartitionedCall?
dropout_38/PartitionedCallPartitionedCall&activation_30/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_dropout_38_layer_call_and_return_conditional_losses_7784752
dropout_38/PartitionedCall?
IdentityIdentity#dropout_38/PartitionedCall:output:0^NoOp*
T0*0
_output_shapes
:??????????2

Identity?
NoOpNoOp/^batch_normalization_28/StatefulPartitionedCall/^batch_normalization_29/StatefulPartitionedCall/^batch_normalization_30/StatefulPartitionedCall"^conv2d_40/StatefulPartitionedCall"^conv2d_41/StatefulPartitionedCall"^conv2d_42/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*R
_input_shapesA
?:?????????: : : : : : : : : : : : : : : : : : 2`
.batch_normalization_28/StatefulPartitionedCall.batch_normalization_28/StatefulPartitionedCall2`
.batch_normalization_29/StatefulPartitionedCall.batch_normalization_29/StatefulPartitionedCall2`
.batch_normalization_30/StatefulPartitionedCall.batch_normalization_30/StatefulPartitionedCall2F
!conv2d_40/StatefulPartitionedCall!conv2d_40/StatefulPartitionedCall2F
!conv2d_41/StatefulPartitionedCall!conv2d_41/StatefulPartitionedCall2F
!conv2d_42/StatefulPartitionedCall!conv2d_42/StatefulPartitionedCall:X T
/
_output_shapes
:?????????
!
_user_specified_name	input_9
?
b
F__inference_dropout_37_layer_call_and_return_conditional_losses_778550

inputs
identityc
IdentityIdentityinputs*
T0*0
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:??????????:X T
0
_output_shapes
:??????????
 
_user_specified_nameinputs
?	
?
7__inference_batch_normalization_28_layer_call_fn_780337

inputs
unknown:	?
	unknown_0:	?
	unknown_1:	?
	unknown_2:	?
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,????????????????????????????*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *[
fVRT
R__inference_batch_normalization_28_layer_call_and_return_conditional_losses_7778882
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*B
_output_shapes0
.:,????????????????????????????2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,????????????????????????????: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:j f
B
_output_shapes0
.:,????????????????????????????
 
_user_specified_nameinputs
?
J
.__inference_activation_28_layer_call_fn_780453

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *R
fMRK
I__inference_activation_28_layer_call_and_return_conditional_losses_7782992
PartitionedCallu
IdentityIdentityPartitionedCall:output:0*
T0*0
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:??????????:X T
0
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
R__inference_batch_normalization_30_layer_call_and_return_conditional_losses_778184

inputs&
readvariableop_resource:	?(
readvariableop_1_resource:	?7
(fusedbatchnormv3_readvariableop_resource:	?9
*fusedbatchnormv3_readvariableop_1_resource:	?
identity??AssignNewValue?AssignNewValue_1?FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1u
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:?*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:?*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:?*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:?*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,????????????????????????????:?:?:?:?:*
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
T0*B
_output_shapes0
.:,????????????????????????????2

Identity?
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,????????????????????????????: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:j f
B
_output_shapes0
.:,????????????????????????????
 
_user_specified_nameinputs
?
d
F__inference_dropout_44_layer_call_and_return_conditional_losses_780266

inputs

identity_1c
IdentityIdentityinputs*
T0*0
_output_shapes
:??????????2

Identityr

Identity_1IdentityIdentity:output:0*
T0*0
_output_shapes
:??????????2

Identity_1"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:??????????:X T
0
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
R__inference_batch_normalization_30_layer_call_and_return_conditional_losses_780792

inputs&
readvariableop_resource:	?(
readvariableop_1_resource:	?7
(fusedbatchnormv3_readvariableop_resource:	?9
*fusedbatchnormv3_readvariableop_1_resource:	?
identity??AssignNewValue?AssignNewValue_1?FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1u
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:?*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:?*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:?*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:?*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:??????????:?:?:?:?:*
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
AssignNewValue_1x
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*0
_output_shapes
:??????????2

Identity?
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*7
_input_shapes&
$:??????????: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:X T
0
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
R__inference_batch_normalization_29_layer_call_and_return_conditional_losses_780566

inputs&
readvariableop_resource:	?(
readvariableop_1_resource:	?7
(fusedbatchnormv3_readvariableop_resource:	?9
*fusedbatchnormv3_readvariableop_1_resource:	?
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1u
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:?*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:?*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:?*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:?*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,????????????????????????????:?:?:?:?:*
epsilon%o?:*
is_training( 2
FusedBatchNormV3?
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*B
_output_shapes0
.:,????????????????????????????2

Identity?
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,????????????????????????????: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:j f
B
_output_shapes0
.:,????????????????????????????
 
_user_specified_nameinputs
?
b
F__inference_dropout_38_layer_call_and_return_conditional_losses_780821

inputs
identityc
IdentityIdentityinputs*
T0*0
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:??????????:X T
0
_output_shapes
:??????????
 
_user_specified_nameinputs
??
?
C__inference_model_8_layer_call_and_return_conditional_losses_778424

inputs+
conv2d_40_778262:?
conv2d_40_778264:	?,
batch_normalization_28_778285:	?,
batch_normalization_28_778287:	?,
batch_normalization_28_778289:	?,
batch_normalization_28_778291:	?,
conv2d_41_778319:??
conv2d_41_778321:	?,
batch_normalization_29_778342:	?,
batch_normalization_29_778344:	?,
batch_normalization_29_778346:	?,
batch_normalization_29_778348:	?,
conv2d_42_778376:??
conv2d_42_778378:	?,
batch_normalization_30_778399:	?,
batch_normalization_30_778401:	?,
batch_normalization_30_778403:	?,
batch_normalization_30_778405:	?
identity??.batch_normalization_28/StatefulPartitionedCall?.batch_normalization_29/StatefulPartitionedCall?.batch_normalization_30/StatefulPartitionedCall?!conv2d_40/StatefulPartitionedCall?!conv2d_41/StatefulPartitionedCall?!conv2d_42/StatefulPartitionedCall?
!conv2d_40/StatefulPartitionedCallStatefulPartitionedCallinputsconv2d_40_778262conv2d_40_778264*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_conv2d_40_layer_call_and_return_conditional_losses_7782612#
!conv2d_40/StatefulPartitionedCall?
.batch_normalization_28/StatefulPartitionedCallStatefulPartitionedCall*conv2d_40/StatefulPartitionedCall:output:0batch_normalization_28_778285batch_normalization_28_778287batch_normalization_28_778289batch_normalization_28_778291*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *[
fVRT
R__inference_batch_normalization_28_layer_call_and_return_conditional_losses_77828420
.batch_normalization_28/StatefulPartitionedCall?
activation_28/PartitionedCallPartitionedCall7batch_normalization_28/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *R
fMRK
I__inference_activation_28_layer_call_and_return_conditional_losses_7782992
activation_28/PartitionedCall?
dropout_36/PartitionedCallPartitionedCall&activation_28/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_dropout_36_layer_call_and_return_conditional_losses_7783062
dropout_36/PartitionedCall?
!conv2d_41/StatefulPartitionedCallStatefulPartitionedCall#dropout_36/PartitionedCall:output:0conv2d_41_778319conv2d_41_778321*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_conv2d_41_layer_call_and_return_conditional_losses_7783182#
!conv2d_41/StatefulPartitionedCall?
.batch_normalization_29/StatefulPartitionedCallStatefulPartitionedCall*conv2d_41/StatefulPartitionedCall:output:0batch_normalization_29_778342batch_normalization_29_778344batch_normalization_29_778346batch_normalization_29_778348*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *[
fVRT
R__inference_batch_normalization_29_layer_call_and_return_conditional_losses_77834120
.batch_normalization_29/StatefulPartitionedCall?
activation_29/PartitionedCallPartitionedCall7batch_normalization_29/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *R
fMRK
I__inference_activation_29_layer_call_and_return_conditional_losses_7783562
activation_29/PartitionedCall?
dropout_37/PartitionedCallPartitionedCall&activation_29/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_dropout_37_layer_call_and_return_conditional_losses_7783632
dropout_37/PartitionedCall?
!conv2d_42/StatefulPartitionedCallStatefulPartitionedCall#dropout_37/PartitionedCall:output:0conv2d_42_778376conv2d_42_778378*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_conv2d_42_layer_call_and_return_conditional_losses_7783752#
!conv2d_42/StatefulPartitionedCall?
.batch_normalization_30/StatefulPartitionedCallStatefulPartitionedCall*conv2d_42/StatefulPartitionedCall:output:0batch_normalization_30_778399batch_normalization_30_778401batch_normalization_30_778403batch_normalization_30_778405*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *[
fVRT
R__inference_batch_normalization_30_layer_call_and_return_conditional_losses_77839820
.batch_normalization_30/StatefulPartitionedCall?
tf.__operators__.add_12/AddV2AddV27batch_normalization_30/StatefulPartitionedCall:output:0#dropout_36/PartitionedCall:output:0*
T0*0
_output_shapes
:??????????2
tf.__operators__.add_12/AddV2?
activation_30/PartitionedCallPartitionedCall!tf.__operators__.add_12/AddV2:z:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *R
fMRK
I__inference_activation_30_layer_call_and_return_conditional_losses_7784142
activation_30/PartitionedCall?
dropout_38/PartitionedCallPartitionedCall&activation_30/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_dropout_38_layer_call_and_return_conditional_losses_7784212
dropout_38/PartitionedCall?
IdentityIdentity#dropout_38/PartitionedCall:output:0^NoOp*
T0*0
_output_shapes
:??????????2

Identity?
NoOpNoOp/^batch_normalization_28/StatefulPartitionedCall/^batch_normalization_29/StatefulPartitionedCall/^batch_normalization_30/StatefulPartitionedCall"^conv2d_40/StatefulPartitionedCall"^conv2d_41/StatefulPartitionedCall"^conv2d_42/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*R
_input_shapesA
?:?????????: : : : : : : : : : : : : : : : : : 2`
.batch_normalization_28/StatefulPartitionedCall.batch_normalization_28/StatefulPartitionedCall2`
.batch_normalization_29/StatefulPartitionedCall.batch_normalization_29/StatefulPartitionedCall2`
.batch_normalization_30/StatefulPartitionedCall.batch_normalization_30/StatefulPartitionedCall2F
!conv2d_40/StatefulPartitionedCall!conv2d_40/StatefulPartitionedCall2F
!conv2d_41/StatefulPartitionedCall!conv2d_41/StatefulPartitionedCall2F
!conv2d_42/StatefulPartitionedCall!conv2d_42/StatefulPartitionedCall:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?	
?
7__inference_batch_normalization_28_layer_call_fn_780363

inputs
unknown:	?
	unknown_0:	?
	unknown_1:	?
	unknown_2:	?
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *[
fVRT
R__inference_batch_normalization_28_layer_call_and_return_conditional_losses_7782842
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*0
_output_shapes
:??????????2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*7
_input_shapes&
$:??????????: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:??????????
 
_user_specified_nameinputs
?
G
+__inference_dropout_44_layer_call_fn_780261

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_dropout_44_layer_call_and_return_conditional_losses_7791922
PartitionedCallu
IdentityIdentityPartitionedCall:output:0*
T0*0
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:??????????:X T
0
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
)__inference_model_11_layer_call_fn_779464
input_12"
unknown:?
	unknown_0:	?
	unknown_1:	?
	unknown_2:	?
	unknown_3:	?
	unknown_4:	?%
	unknown_5:??
	unknown_6:	?
	unknown_7:	?
	unknown_8:	?
	unknown_9:	?

unknown_10:	?&

unknown_11:??

unknown_12:	?

unknown_13:	?

unknown_14:	?

unknown_15:	?

unknown_16:	?%

unknown_17:?

unknown_18:&

unknown_19:??

unknown_20:	?%

unknown_21:?I

unknown_22:I

unknown_23:@
identity

identity_1??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_12unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
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
unknown_22
unknown_23*%
Tin
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:?????????:?????????I*5
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_model_11_layer_call_and_return_conditional_losses_7793522
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
_construction_contextkEagerRuntime*`
_input_shapesO
M:?????????: : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Y U
/
_output_shapes
:?????????
"
_user_specified_name
input_12
?
?
R__inference_batch_normalization_28_layer_call_and_return_conditional_losses_780430

inputs&
readvariableop_resource:	?(
readvariableop_1_resource:	?7
(fusedbatchnormv3_readvariableop_resource:	?9
*fusedbatchnormv3_readvariableop_1_resource:	?
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1u
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:?*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:?*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:?*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:?*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:??????????:?:?:?:?:*
epsilon%o?:*
is_training( 2
FusedBatchNormV3x
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*0
_output_shapes
:??????????2

Identity?
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*7
_input_shapes&
$:??????????: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:X T
0
_output_shapes
:??????????
 
_user_specified_nameinputs
?	
?
7__inference_batch_normalization_29_layer_call_fn_780509

inputs
unknown:	?
	unknown_0:	?
	unknown_1:	?
	unknown_2:	?
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,????????????????????????????*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *[
fVRT
R__inference_batch_normalization_29_layer_call_and_return_conditional_losses_7780142
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*B
_output_shapes0
.:,????????????????????????????2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,????????????????????????????: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:j f
B
_output_shapes0
.:,????????????????????????????
 
_user_specified_nameinputs
?
d
F__inference_dropout_37_layer_call_and_return_conditional_losses_780645

inputs

identity_1c
IdentityIdentityinputs*
T0*0
_output_shapes
:??????????2

Identityr

Identity_1IdentityIdentity:output:0*
T0*0
_output_shapes
:??????????2

Identity_1"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:??????????:X T
0
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
)__inference_model_11_layer_call_fn_779156
input_12"
unknown:?
	unknown_0:	?
	unknown_1:	?
	unknown_2:	?
	unknown_3:	?
	unknown_4:	?%
	unknown_5:??
	unknown_6:	?
	unknown_7:	?
	unknown_8:	?
	unknown_9:	?

unknown_10:	?&

unknown_11:??

unknown_12:	?

unknown_13:	?

unknown_14:	?

unknown_15:	?

unknown_16:	?%

unknown_17:?

unknown_18:&

unknown_19:??

unknown_20:	?%

unknown_21:?I

unknown_22:I

unknown_23:@
identity

identity_1??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_12unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
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
unknown_22
unknown_23*%
Tin
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:?????????:?????????I*;
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_model_11_layer_call_and_return_conditional_losses_7791012
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
_construction_contextkEagerRuntime*`
_input_shapesO
M:?????????: : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Y U
/
_output_shapes
:?????????
"
_user_specified_name
input_12
?
?
E__inference_conv2d_40_layer_call_and_return_conditional_losses_778261

inputs9
conv2d_readvariableop_resource:?.
biasadd_readvariableop_resource:	?
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*'
_output_shapes
:?*
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????*
paddingSAME*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????2	
BiasAddt
IdentityIdentityBiasAdd:output:0^NoOp*
T0*0
_output_shapes
:??????????2

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
?
?
R__inference_batch_normalization_29_layer_call_and_return_conditional_losses_778341

inputs&
readvariableop_resource:	?(
readvariableop_1_resource:	?7
(fusedbatchnormv3_readvariableop_resource:	?9
*fusedbatchnormv3_readvariableop_1_resource:	?
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1u
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:?*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:?*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:?*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:?*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:??????????:?:?:?:?:*
epsilon%o?:*
is_training( 2
FusedBatchNormV3x
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*0
_output_shapes
:??????????2

Identity?
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*7
_input_shapes&
$:??????????: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:X T
0
_output_shapes
:??????????
 
_user_specified_nameinputs
?
b
F__inference_dropout_44_layer_call_and_return_conditional_losses_780270

inputs
identityc
IdentityIdentityinputs*
T0*0
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:??????????:X T
0
_output_shapes
:??????????
 
_user_specified_nameinputs
?	
?
7__inference_batch_normalization_29_layer_call_fn_780522

inputs
unknown:	?
	unknown_0:	?
	unknown_1:	?
	unknown_2:	?
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,????????????????????????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *[
fVRT
R__inference_batch_normalization_29_layer_call_and_return_conditional_losses_7780582
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*B
_output_shapes0
.:,????????????????????????????2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,????????????????????????????: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:j f
B
_output_shapes0
.:,????????????????????????????
 
_user_specified_nameinputs"?L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*?
serving_default?
E
input_129
serving_default_input_12:0?????????E
	conv2d_508
StatefulPartitionedCall:0?????????I;
dense_50
StatefulPartitionedCall:1?????????tensorflow/serving/predict:??
?
layer-0
layer_with_weights-0
layer-1
layer_with_weights-1
layer-2
layer-3
layer_with_weights-2
layer-4
layer-5
layer-6
layer_with_weights-3
layer-7
	layer_with_weights-4
	layer-8

regularization_losses
	variables
trainable_variables
	keras_api

signatures
+?&call_and_return_all_conditional_losses
?__call__
?_default_save_signature"
_tf_keras_network
"
_tf_keras_input_layer
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
regularization_losses
	variables
trainable_variables
 	keras_api
+?&call_and_return_all_conditional_losses
?__call__"
_tf_keras_network
?

!kernel
"bias
#regularization_losses
$	variables
%trainable_variables
&	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?
'regularization_losses
(	variables
)trainable_variables
*	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?

+kernel
,bias
-regularization_losses
.	variables
/trainable_variables
0	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?
1regularization_losses
2	variables
3trainable_variables
4	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?
5regularization_losses
6	variables
7trainable_variables
8	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?

9kernel
:regularization_losses
;	variables
<trainable_variables
=	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?

>kernel
?bias
@regularization_losses
A	variables
Btrainable_variables
C	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
 "
trackable_list_wrapper
?
D0
E1
F2
G3
H4
I5
J6
K7
L8
M9
N10
O11
P12
Q13
R14
S15
T16
U17
!18
"19
+20
,21
922
>23
?24"
trackable_list_wrapper
?
D0
E1
F2
G3
J4
K5
L6
M7
P8
Q9
R10
S11
!12
"13
+14
,15
916
>17
?18"
trackable_list_wrapper
?

regularization_losses
Vlayer_metrics
	variables

Wlayers
Xnon_trainable_variables
trainable_variables
Ymetrics
Zlayer_regularization_losses
?__call__
?_default_save_signature
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
-
?serving_default"
signature_map
"
_tf_keras_input_layer
?

Dkernel
Ebias
[regularization_losses
\	variables
]trainable_variables
^	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?
_axis
	Fgamma
Gbeta
Hmoving_mean
Imoving_variance
`regularization_losses
a	variables
btrainable_variables
c	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?
dregularization_losses
e	variables
ftrainable_variables
g	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?
hregularization_losses
i	variables
jtrainable_variables
k	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?

Jkernel
Kbias
lregularization_losses
m	variables
ntrainable_variables
o	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?
paxis
	Lgamma
Mbeta
Nmoving_mean
Omoving_variance
qregularization_losses
r	variables
strainable_variables
t	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?
uregularization_losses
v	variables
wtrainable_variables
x	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?
yregularization_losses
z	variables
{trainable_variables
|	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?

Pkernel
Qbias
}regularization_losses
~	variables
trainable_variables
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?
	?axis
	Rgamma
Sbeta
Tmoving_mean
Umoving_variance
?regularization_losses
?	variables
?trainable_variables
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
)
?	keras_api"
_tf_keras_layer
?
?regularization_losses
?	variables
?trainable_variables
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?
?regularization_losses
?	variables
?trainable_variables
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
 "
trackable_list_wrapper
?
D0
E1
F2
G3
H4
I5
J6
K7
L8
M9
N10
O11
P12
Q13
R14
S15
T16
U17"
trackable_list_wrapper
v
D0
E1
F2
G3
J4
K5
L6
M7
P8
Q9
R10
S11"
trackable_list_wrapper
?
regularization_losses
?layer_metrics
	variables
?layers
?non_trainable_variables
trainable_variables
?metrics
 ?layer_regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
+:)?2conv2d_51/kernel
:2conv2d_51/bias
 "
trackable_list_wrapper
.
!0
"1"
trackable_list_wrapper
.
!0
"1"
trackable_list_wrapper
?
#regularization_losses
?layer_metrics
$	variables
?non_trainable_variables
?layers
%trainable_variables
?metrics
 ?layer_regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
'regularization_losses
?layer_metrics
(	variables
?non_trainable_variables
?layers
)trainable_variables
?metrics
 ?layer_regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
,:*??2conv2d_49/kernel
:?2conv2d_49/bias
 "
trackable_list_wrapper
.
+0
,1"
trackable_list_wrapper
.
+0
,1"
trackable_list_wrapper
?
-regularization_losses
?layer_metrics
.	variables
?non_trainable_variables
?layers
/trainable_variables
?metrics
 ?layer_regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
1regularization_losses
?layer_metrics
2	variables
?non_trainable_variables
?layers
3trainable_variables
?metrics
 ?layer_regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
5regularization_losses
?layer_metrics
6	variables
?non_trainable_variables
?layers
7trainable_variables
?metrics
 ?layer_regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 :@2dense_5/kernel
 "
trackable_list_wrapper
'
90"
trackable_list_wrapper
'
90"
trackable_list_wrapper
?
:regularization_losses
?layer_metrics
;	variables
?non_trainable_variables
?layers
<trainable_variables
?metrics
 ?layer_regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
+:)?I2conv2d_50/kernel
:I2conv2d_50/bias
 "
trackable_list_wrapper
.
>0
?1"
trackable_list_wrapper
.
>0
?1"
trackable_list_wrapper
?
@regularization_losses
?layer_metrics
A	variables
?non_trainable_variables
?layers
Btrainable_variables
?metrics
 ?layer_regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
+:)?2conv2d_40/kernel
:?2conv2d_40/bias
+:)?2batch_normalization_28/gamma
*:(?2batch_normalization_28/beta
3:1? (2"batch_normalization_28/moving_mean
7:5? (2&batch_normalization_28/moving_variance
,:*??2conv2d_41/kernel
:?2conv2d_41/bias
+:)?2batch_normalization_29/gamma
*:(?2batch_normalization_29/beta
3:1? (2"batch_normalization_29/moving_mean
7:5? (2&batch_normalization_29/moving_variance
,:*??2conv2d_42/kernel
:?2conv2d_42/bias
+:)?2batch_normalization_30/gamma
*:(?2batch_normalization_30/beta
3:1? (2"batch_normalization_30/moving_mean
7:5? (2&batch_normalization_30/moving_variance
 "
trackable_dict_wrapper
_
0
1
2
3
4
5
6
7
	8"
trackable_list_wrapper
J
H0
I1
N2
O3
T4
U5"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
.
D0
E1"
trackable_list_wrapper
.
D0
E1"
trackable_list_wrapper
?
[regularization_losses
?layer_metrics
\	variables
?non_trainable_variables
?layers
]trainable_variables
?metrics
 ?layer_regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
<
F0
G1
H2
I3"
trackable_list_wrapper
.
F0
G1"
trackable_list_wrapper
?
`regularization_losses
?layer_metrics
a	variables
?non_trainable_variables
?layers
btrainable_variables
?metrics
 ?layer_regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
dregularization_losses
?layer_metrics
e	variables
?non_trainable_variables
?layers
ftrainable_variables
?metrics
 ?layer_regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
hregularization_losses
?layer_metrics
i	variables
?non_trainable_variables
?layers
jtrainable_variables
?metrics
 ?layer_regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
.
J0
K1"
trackable_list_wrapper
.
J0
K1"
trackable_list_wrapper
?
lregularization_losses
?layer_metrics
m	variables
?non_trainable_variables
?layers
ntrainable_variables
?metrics
 ?layer_regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
<
L0
M1
N2
O3"
trackable_list_wrapper
.
L0
M1"
trackable_list_wrapper
?
qregularization_losses
?layer_metrics
r	variables
?non_trainable_variables
?layers
strainable_variables
?metrics
 ?layer_regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
uregularization_losses
?layer_metrics
v	variables
?non_trainable_variables
?layers
wtrainable_variables
?metrics
 ?layer_regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
yregularization_losses
?layer_metrics
z	variables
?non_trainable_variables
?layers
{trainable_variables
?metrics
 ?layer_regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
.
P0
Q1"
trackable_list_wrapper
.
P0
Q1"
trackable_list_wrapper
?
}regularization_losses
?layer_metrics
~	variables
?non_trainable_variables
?layers
trainable_variables
?metrics
 ?layer_regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
<
R0
S1
T2
U3"
trackable_list_wrapper
.
R0
S1"
trackable_list_wrapper
?
?regularization_losses
?layer_metrics
?	variables
?non_trainable_variables
?layers
?trainable_variables
?metrics
 ?layer_regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?regularization_losses
?layer_metrics
?	variables
?non_trainable_variables
?layers
?trainable_variables
?metrics
 ?layer_regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?regularization_losses
?layer_metrics
?	variables
?non_trainable_variables
?layers
?trainable_variables
?metrics
 ?layer_regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_dict_wrapper
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
H0
I1
N2
O3
T4
U5"
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
.
H0
I1"
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
.
N0
O1"
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
.
T0
U1"
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
?2?
D__inference_model_11_layer_call_and_return_conditional_losses_779750
D__inference_model_11_layer_call_and_return_conditional_losses_779846
D__inference_model_11_layer_call_and_return_conditional_losses_779527
D__inference_model_11_layer_call_and_return_conditional_losses_779590?
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
)__inference_model_11_layer_call_fn_779156
)__inference_model_11_layer_call_fn_779903
)__inference_model_11_layer_call_fn_779960
)__inference_model_11_layer_call_fn_779464?
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
?B?
!__inference__wrapped_model_777866input_12"?
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
?2?
C__inference_model_8_layer_call_and_return_conditional_losses_780031
C__inference_model_8_layer_call_and_return_conditional_losses_780099
C__inference_model_8_layer_call_and_return_conditional_losses_778917
C__inference_model_8_layer_call_and_return_conditional_losses_778970?
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
(__inference_model_8_layer_call_fn_778463
(__inference_model_8_layer_call_fn_780140
(__inference_model_8_layer_call_fn_780181
(__inference_model_8_layer_call_fn_778864?
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
*__inference_conv2d_51_layer_call_fn_780190?
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
E__inference_conv2d_51_layer_call_and_return_conditional_losses_780201?
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
+__inference_dropout_45_layer_call_fn_780206
+__inference_dropout_45_layer_call_fn_780211?
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
F__inference_dropout_45_layer_call_and_return_conditional_losses_780216
F__inference_dropout_45_layer_call_and_return_conditional_losses_780220?
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
*__inference_conv2d_49_layer_call_fn_780229?
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
E__inference_conv2d_49_layer_call_and_return_conditional_losses_780240?
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
*__inference_flatten_5_layer_call_fn_780245?
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
E__inference_flatten_5_layer_call_and_return_conditional_losses_780251?
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
+__inference_dropout_44_layer_call_fn_780256
+__inference_dropout_44_layer_call_fn_780261?
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
F__inference_dropout_44_layer_call_and_return_conditional_losses_780266
F__inference_dropout_44_layer_call_and_return_conditional_losses_780270?
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
(__inference_dense_5_layer_call_fn_780277?
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
C__inference_dense_5_layer_call_and_return_conditional_losses_780285?
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
*__inference_conv2d_50_layer_call_fn_780294?
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
E__inference_conv2d_50_layer_call_and_return_conditional_losses_780305?
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
$__inference_signature_wrapper_779649input_12"?
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
*__inference_conv2d_40_layer_call_fn_780314?
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
E__inference_conv2d_40_layer_call_and_return_conditional_losses_780324?
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
7__inference_batch_normalization_28_layer_call_fn_780337
7__inference_batch_normalization_28_layer_call_fn_780350
7__inference_batch_normalization_28_layer_call_fn_780363
7__inference_batch_normalization_28_layer_call_fn_780376?
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
R__inference_batch_normalization_28_layer_call_and_return_conditional_losses_780394
R__inference_batch_normalization_28_layer_call_and_return_conditional_losses_780412
R__inference_batch_normalization_28_layer_call_and_return_conditional_losses_780430
R__inference_batch_normalization_28_layer_call_and_return_conditional_losses_780448?
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
.__inference_activation_28_layer_call_fn_780453?
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
I__inference_activation_28_layer_call_and_return_conditional_losses_780458?
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
+__inference_dropout_36_layer_call_fn_780463
+__inference_dropout_36_layer_call_fn_780468?
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
F__inference_dropout_36_layer_call_and_return_conditional_losses_780473
F__inference_dropout_36_layer_call_and_return_conditional_losses_780477?
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
*__inference_conv2d_41_layer_call_fn_780486?
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
E__inference_conv2d_41_layer_call_and_return_conditional_losses_780496?
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
7__inference_batch_normalization_29_layer_call_fn_780509
7__inference_batch_normalization_29_layer_call_fn_780522
7__inference_batch_normalization_29_layer_call_fn_780535
7__inference_batch_normalization_29_layer_call_fn_780548?
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
R__inference_batch_normalization_29_layer_call_and_return_conditional_losses_780566
R__inference_batch_normalization_29_layer_call_and_return_conditional_losses_780584
R__inference_batch_normalization_29_layer_call_and_return_conditional_losses_780602
R__inference_batch_normalization_29_layer_call_and_return_conditional_losses_780620?
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
.__inference_activation_29_layer_call_fn_780625?
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
I__inference_activation_29_layer_call_and_return_conditional_losses_780630?
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
+__inference_dropout_37_layer_call_fn_780635
+__inference_dropout_37_layer_call_fn_780640?
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
F__inference_dropout_37_layer_call_and_return_conditional_losses_780645
F__inference_dropout_37_layer_call_and_return_conditional_losses_780649?
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
*__inference_conv2d_42_layer_call_fn_780658?
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
E__inference_conv2d_42_layer_call_and_return_conditional_losses_780668?
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
7__inference_batch_normalization_30_layer_call_fn_780681
7__inference_batch_normalization_30_layer_call_fn_780694
7__inference_batch_normalization_30_layer_call_fn_780707
7__inference_batch_normalization_30_layer_call_fn_780720?
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
R__inference_batch_normalization_30_layer_call_and_return_conditional_losses_780738
R__inference_batch_normalization_30_layer_call_and_return_conditional_losses_780756
R__inference_batch_normalization_30_layer_call_and_return_conditional_losses_780774
R__inference_batch_normalization_30_layer_call_and_return_conditional_losses_780792?
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
.__inference_activation_30_layer_call_fn_780797?
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
I__inference_activation_30_layer_call_and_return_conditional_losses_780802?
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
+__inference_dropout_38_layer_call_fn_780807
+__inference_dropout_38_layer_call_fn_780812?
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
F__inference_dropout_38_layer_call_and_return_conditional_losses_780817
F__inference_dropout_38_layer_call_and_return_conditional_losses_780821?
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
!__inference__wrapped_model_777866?DEFGHIJKLMNOPQRSTU!"+,>?99?6
/?,
*?'
input_12?????????
? "k?h
8
	conv2d_50+?(
	conv2d_50?????????I
,
dense_5!?
dense_5??????????
I__inference_activation_28_layer_call_and_return_conditional_losses_780458j8?5
.?+
)?&
inputs??????????
? ".?+
$?!
0??????????
? ?
.__inference_activation_28_layer_call_fn_780453]8?5
.?+
)?&
inputs??????????
? "!????????????
I__inference_activation_29_layer_call_and_return_conditional_losses_780630j8?5
.?+
)?&
inputs??????????
? ".?+
$?!
0??????????
? ?
.__inference_activation_29_layer_call_fn_780625]8?5
.?+
)?&
inputs??????????
? "!????????????
I__inference_activation_30_layer_call_and_return_conditional_losses_780802j8?5
.?+
)?&
inputs??????????
? ".?+
$?!
0??????????
? ?
.__inference_activation_30_layer_call_fn_780797]8?5
.?+
)?&
inputs??????????
? "!????????????
R__inference_batch_normalization_28_layer_call_and_return_conditional_losses_780394?FGHIN?K
D?A
;?8
inputs,????????????????????????????
p 
? "@?=
6?3
0,????????????????????????????
? ?
R__inference_batch_normalization_28_layer_call_and_return_conditional_losses_780412?FGHIN?K
D?A
;?8
inputs,????????????????????????????
p
? "@?=
6?3
0,????????????????????????????
? ?
R__inference_batch_normalization_28_layer_call_and_return_conditional_losses_780430tFGHI<?9
2?/
)?&
inputs??????????
p 
? ".?+
$?!
0??????????
? ?
R__inference_batch_normalization_28_layer_call_and_return_conditional_losses_780448tFGHI<?9
2?/
)?&
inputs??????????
p
? ".?+
$?!
0??????????
? ?
7__inference_batch_normalization_28_layer_call_fn_780337?FGHIN?K
D?A
;?8
inputs,????????????????????????????
p 
? "3?0,?????????????????????????????
7__inference_batch_normalization_28_layer_call_fn_780350?FGHIN?K
D?A
;?8
inputs,????????????????????????????
p
? "3?0,?????????????????????????????
7__inference_batch_normalization_28_layer_call_fn_780363gFGHI<?9
2?/
)?&
inputs??????????
p 
? "!????????????
7__inference_batch_normalization_28_layer_call_fn_780376gFGHI<?9
2?/
)?&
inputs??????????
p
? "!????????????
R__inference_batch_normalization_29_layer_call_and_return_conditional_losses_780566?LMNON?K
D?A
;?8
inputs,????????????????????????????
p 
? "@?=
6?3
0,????????????????????????????
? ?
R__inference_batch_normalization_29_layer_call_and_return_conditional_losses_780584?LMNON?K
D?A
;?8
inputs,????????????????????????????
p
? "@?=
6?3
0,????????????????????????????
? ?
R__inference_batch_normalization_29_layer_call_and_return_conditional_losses_780602tLMNO<?9
2?/
)?&
inputs??????????
p 
? ".?+
$?!
0??????????
? ?
R__inference_batch_normalization_29_layer_call_and_return_conditional_losses_780620tLMNO<?9
2?/
)?&
inputs??????????
p
? ".?+
$?!
0??????????
? ?
7__inference_batch_normalization_29_layer_call_fn_780509?LMNON?K
D?A
;?8
inputs,????????????????????????????
p 
? "3?0,?????????????????????????????
7__inference_batch_normalization_29_layer_call_fn_780522?LMNON?K
D?A
;?8
inputs,????????????????????????????
p
? "3?0,?????????????????????????????
7__inference_batch_normalization_29_layer_call_fn_780535gLMNO<?9
2?/
)?&
inputs??????????
p 
? "!????????????
7__inference_batch_normalization_29_layer_call_fn_780548gLMNO<?9
2?/
)?&
inputs??????????
p
? "!????????????
R__inference_batch_normalization_30_layer_call_and_return_conditional_losses_780738?RSTUN?K
D?A
;?8
inputs,????????????????????????????
p 
? "@?=
6?3
0,????????????????????????????
? ?
R__inference_batch_normalization_30_layer_call_and_return_conditional_losses_780756?RSTUN?K
D?A
;?8
inputs,????????????????????????????
p
? "@?=
6?3
0,????????????????????????????
? ?
R__inference_batch_normalization_30_layer_call_and_return_conditional_losses_780774tRSTU<?9
2?/
)?&
inputs??????????
p 
? ".?+
$?!
0??????????
? ?
R__inference_batch_normalization_30_layer_call_and_return_conditional_losses_780792tRSTU<?9
2?/
)?&
inputs??????????
p
? ".?+
$?!
0??????????
? ?
7__inference_batch_normalization_30_layer_call_fn_780681?RSTUN?K
D?A
;?8
inputs,????????????????????????????
p 
? "3?0,?????????????????????????????
7__inference_batch_normalization_30_layer_call_fn_780694?RSTUN?K
D?A
;?8
inputs,????????????????????????????
p
? "3?0,?????????????????????????????
7__inference_batch_normalization_30_layer_call_fn_780707gRSTU<?9
2?/
)?&
inputs??????????
p 
? "!????????????
7__inference_batch_normalization_30_layer_call_fn_780720gRSTU<?9
2?/
)?&
inputs??????????
p
? "!????????????
E__inference_conv2d_40_layer_call_and_return_conditional_losses_780324mDE7?4
-?*
(?%
inputs?????????
? ".?+
$?!
0??????????
? ?
*__inference_conv2d_40_layer_call_fn_780314`DE7?4
-?*
(?%
inputs?????????
? "!????????????
E__inference_conv2d_41_layer_call_and_return_conditional_losses_780496nJK8?5
.?+
)?&
inputs??????????
? ".?+
$?!
0??????????
? ?
*__inference_conv2d_41_layer_call_fn_780486aJK8?5
.?+
)?&
inputs??????????
? "!????????????
E__inference_conv2d_42_layer_call_and_return_conditional_losses_780668nPQ8?5
.?+
)?&
inputs??????????
? ".?+
$?!
0??????????
? ?
*__inference_conv2d_42_layer_call_fn_780658aPQ8?5
.?+
)?&
inputs??????????
? "!????????????
E__inference_conv2d_49_layer_call_and_return_conditional_losses_780240n+,8?5
.?+
)?&
inputs??????????
? ".?+
$?!
0??????????
? ?
*__inference_conv2d_49_layer_call_fn_780229a+,8?5
.?+
)?&
inputs??????????
? "!????????????
E__inference_conv2d_50_layer_call_and_return_conditional_losses_780305m>?8?5
.?+
)?&
inputs??????????
? "-?*
#? 
0?????????I
? ?
*__inference_conv2d_50_layer_call_fn_780294`>?8?5
.?+
)?&
inputs??????????
? " ??????????I?
E__inference_conv2d_51_layer_call_and_return_conditional_losses_780201m!"8?5
.?+
)?&
inputs??????????
? "-?*
#? 
0?????????
? ?
*__inference_conv2d_51_layer_call_fn_780190`!"8?5
.?+
)?&
inputs??????????
? " ???????????
C__inference_dense_5_layer_call_and_return_conditional_losses_780285[9/?,
%?"
 ?
inputs?????????@
? "%?"
?
0?????????
? z
(__inference_dense_5_layer_call_fn_780277N9/?,
%?"
 ?
inputs?????????@
? "???????????
F__inference_dropout_36_layer_call_and_return_conditional_losses_780473n<?9
2?/
)?&
inputs??????????
p 
? ".?+
$?!
0??????????
? ?
F__inference_dropout_36_layer_call_and_return_conditional_losses_780477n<?9
2?/
)?&
inputs??????????
p
? ".?+
$?!
0??????????
? ?
+__inference_dropout_36_layer_call_fn_780463a<?9
2?/
)?&
inputs??????????
p 
? "!????????????
+__inference_dropout_36_layer_call_fn_780468a<?9
2?/
)?&
inputs??????????
p
? "!????????????
F__inference_dropout_37_layer_call_and_return_conditional_losses_780645n<?9
2?/
)?&
inputs??????????
p 
? ".?+
$?!
0??????????
? ?
F__inference_dropout_37_layer_call_and_return_conditional_losses_780649n<?9
2?/
)?&
inputs??????????
p
? ".?+
$?!
0??????????
? ?
+__inference_dropout_37_layer_call_fn_780635a<?9
2?/
)?&
inputs??????????
p 
? "!????????????
+__inference_dropout_37_layer_call_fn_780640a<?9
2?/
)?&
inputs??????????
p
? "!????????????
F__inference_dropout_38_layer_call_and_return_conditional_losses_780817n<?9
2?/
)?&
inputs??????????
p 
? ".?+
$?!
0??????????
? ?
F__inference_dropout_38_layer_call_and_return_conditional_losses_780821n<?9
2?/
)?&
inputs??????????
p
? ".?+
$?!
0??????????
? ?
+__inference_dropout_38_layer_call_fn_780807a<?9
2?/
)?&
inputs??????????
p 
? "!????????????
+__inference_dropout_38_layer_call_fn_780812a<?9
2?/
)?&
inputs??????????
p
? "!????????????
F__inference_dropout_44_layer_call_and_return_conditional_losses_780266n<?9
2?/
)?&
inputs??????????
p 
? ".?+
$?!
0??????????
? ?
F__inference_dropout_44_layer_call_and_return_conditional_losses_780270n<?9
2?/
)?&
inputs??????????
p
? ".?+
$?!
0??????????
? ?
+__inference_dropout_44_layer_call_fn_780256a<?9
2?/
)?&
inputs??????????
p 
? "!????????????
+__inference_dropout_44_layer_call_fn_780261a<?9
2?/
)?&
inputs??????????
p
? "!????????????
F__inference_dropout_45_layer_call_and_return_conditional_losses_780216l;?8
1?.
(?%
inputs?????????
p 
? "-?*
#? 
0?????????
? ?
F__inference_dropout_45_layer_call_and_return_conditional_losses_780220l;?8
1?.
(?%
inputs?????????
p
? "-?*
#? 
0?????????
? ?
+__inference_dropout_45_layer_call_fn_780206_;?8
1?.
(?%
inputs?????????
p 
? " ???????????
+__inference_dropout_45_layer_call_fn_780211_;?8
1?.
(?%
inputs?????????
p
? " ???????????
E__inference_flatten_5_layer_call_and_return_conditional_losses_780251`7?4
-?*
(?%
inputs?????????
? "%?"
?
0?????????@
? ?
*__inference_flatten_5_layer_call_fn_780245S7?4
-?*
(?%
inputs?????????
? "??????????@?
D__inference_model_11_layer_call_and_return_conditional_losses_779527?DEFGHIJKLMNOPQRSTU!"+,>?9A?>
7?4
*?'
input_12?????????
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
D__inference_model_11_layer_call_and_return_conditional_losses_779590?DEFGHIJKLMNOPQRSTU!"+,>?9A?>
7?4
*?'
input_12?????????
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
D__inference_model_11_layer_call_and_return_conditional_losses_779750?DEFGHIJKLMNOPQRSTU!"+,>?9??<
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
D__inference_model_11_layer_call_and_return_conditional_losses_779846?DEFGHIJKLMNOPQRSTU!"+,>?9??<
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
)__inference_model_11_layer_call_fn_779156?DEFGHIJKLMNOPQRSTU!"+,>?9A?>
7?4
*?'
input_12?????????
p 

 
? "E?B
?
0?????????
#? 
1?????????I?
)__inference_model_11_layer_call_fn_779464?DEFGHIJKLMNOPQRSTU!"+,>?9A?>
7?4
*?'
input_12?????????
p

 
? "E?B
?
0?????????
#? 
1?????????I?
)__inference_model_11_layer_call_fn_779903?DEFGHIJKLMNOPQRSTU!"+,>?9??<
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
)__inference_model_11_layer_call_fn_779960?DEFGHIJKLMNOPQRSTU!"+,>?9??<
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
C__inference_model_8_layer_call_and_return_conditional_losses_778917?DEFGHIJKLMNOPQRSTU@?=
6?3
)?&
input_9?????????
p 

 
? ".?+
$?!
0??????????
? ?
C__inference_model_8_layer_call_and_return_conditional_losses_778970?DEFGHIJKLMNOPQRSTU@?=
6?3
)?&
input_9?????????
p

 
? ".?+
$?!
0??????????
? ?
C__inference_model_8_layer_call_and_return_conditional_losses_780031?DEFGHIJKLMNOPQRSTU??<
5?2
(?%
inputs?????????
p 

 
? ".?+
$?!
0??????????
? ?
C__inference_model_8_layer_call_and_return_conditional_losses_780099?DEFGHIJKLMNOPQRSTU??<
5?2
(?%
inputs?????????
p

 
? ".?+
$?!
0??????????
? ?
(__inference_model_8_layer_call_fn_778463yDEFGHIJKLMNOPQRSTU@?=
6?3
)?&
input_9?????????
p 

 
? "!????????????
(__inference_model_8_layer_call_fn_778864yDEFGHIJKLMNOPQRSTU@?=
6?3
)?&
input_9?????????
p

 
? "!????????????
(__inference_model_8_layer_call_fn_780140xDEFGHIJKLMNOPQRSTU??<
5?2
(?%
inputs?????????
p 

 
? "!????????????
(__inference_model_8_layer_call_fn_780181xDEFGHIJKLMNOPQRSTU??<
5?2
(?%
inputs?????????
p

 
? "!????????????
$__inference_signature_wrapper_779649?DEFGHIJKLMNOPQRSTU!"+,>?9E?B
? 
;?8
6
input_12*?'
input_12?????????"k?h
8
	conv2d_50+?(
	conv2d_50?????????I
,
dense_5!?
dense_5?????????