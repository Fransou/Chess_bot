??2
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
 ?"serve*2.6.12v2.6.0-101-g3aa40c3ce9d8??+
?
conv2d_55/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*!
shared_nameconv2d_55/kernel
~
$conv2d_55/kernel/Read/ReadVariableOpReadVariableOpconv2d_55/kernel*'
_output_shapes
:?*
dtype0
t
conv2d_55/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameconv2d_55/bias
m
"conv2d_55/bias/Read/ReadVariableOpReadVariableOpconv2d_55/bias*
_output_shapes
:*
dtype0
?
conv2d_53/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:??*!
shared_nameconv2d_53/kernel

$conv2d_53/kernel/Read/ReadVariableOpReadVariableOpconv2d_53/kernel*(
_output_shapes
:??*
dtype0
u
conv2d_53/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*
shared_nameconv2d_53/bias
n
"conv2d_53/bias/Read/ReadVariableOpReadVariableOpconv2d_53/bias*
_output_shapes	
:?*
dtype0
x
dense_3/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@*
shared_namedense_3/kernel
q
"dense_3/kernel/Read/ReadVariableOpReadVariableOpdense_3/kernel*
_output_shapes

:@*
dtype0
?
conv2d_54/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:?I*!
shared_nameconv2d_54/kernel
~
$conv2d_54/kernel/Read/ReadVariableOpReadVariableOpconv2d_54/kernel*'
_output_shapes
:?I*
dtype0
t
conv2d_54/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:I*
shared_nameconv2d_54/bias
m
"conv2d_54/bias/Read/ReadVariableOpReadVariableOpconv2d_54/bias*
_output_shapes
:I*
dtype0
?
conv2d_28/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*!
shared_nameconv2d_28/kernel
~
$conv2d_28/kernel/Read/ReadVariableOpReadVariableOpconv2d_28/kernel*'
_output_shapes
:?*
dtype0
u
conv2d_28/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*
shared_nameconv2d_28/bias
n
"conv2d_28/bias/Read/ReadVariableOpReadVariableOpconv2d_28/bias*
_output_shapes	
:?*
dtype0
?
batch_normalization_12/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*-
shared_namebatch_normalization_12/gamma
?
0batch_normalization_12/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_12/gamma*
_output_shapes	
:?*
dtype0
?
batch_normalization_12/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*,
shared_namebatch_normalization_12/beta
?
/batch_normalization_12/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_12/beta*
_output_shapes	
:?*
dtype0
?
"batch_normalization_12/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*3
shared_name$"batch_normalization_12/moving_mean
?
6batch_normalization_12/moving_mean/Read/ReadVariableOpReadVariableOp"batch_normalization_12/moving_mean*
_output_shapes	
:?*
dtype0
?
&batch_normalization_12/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*7
shared_name(&batch_normalization_12/moving_variance
?
:batch_normalization_12/moving_variance/Read/ReadVariableOpReadVariableOp&batch_normalization_12/moving_variance*
_output_shapes	
:?*
dtype0
?
conv2d_29/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:??*!
shared_nameconv2d_29/kernel

$conv2d_29/kernel/Read/ReadVariableOpReadVariableOpconv2d_29/kernel*(
_output_shapes
:??*
dtype0
u
conv2d_29/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*
shared_nameconv2d_29/bias
n
"conv2d_29/bias/Read/ReadVariableOpReadVariableOpconv2d_29/bias*
_output_shapes	
:?*
dtype0
?
conv2d_30/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:??*!
shared_nameconv2d_30/kernel

$conv2d_30/kernel/Read/ReadVariableOpReadVariableOpconv2d_30/kernel*(
_output_shapes
:??*
dtype0
u
conv2d_30/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*
shared_nameconv2d_30/bias
n
"conv2d_30/bias/Read/ReadVariableOpReadVariableOpconv2d_30/bias*
_output_shapes	
:?*
dtype0
?
batch_normalization_13/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*-
shared_namebatch_normalization_13/gamma
?
0batch_normalization_13/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_13/gamma*
_output_shapes	
:?*
dtype0
?
batch_normalization_13/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*,
shared_namebatch_normalization_13/beta
?
/batch_normalization_13/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_13/beta*
_output_shapes	
:?*
dtype0
?
"batch_normalization_13/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*3
shared_name$"batch_normalization_13/moving_mean
?
6batch_normalization_13/moving_mean/Read/ReadVariableOpReadVariableOp"batch_normalization_13/moving_mean*
_output_shapes	
:?*
dtype0
?
&batch_normalization_13/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*7
shared_name(&batch_normalization_13/moving_variance
?
:batch_normalization_13/moving_variance/Read/ReadVariableOpReadVariableOp&batch_normalization_13/moving_variance*
_output_shapes	
:?*
dtype0
?
conv2d_31/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:??*!
shared_nameconv2d_31/kernel

$conv2d_31/kernel/Read/ReadVariableOpReadVariableOpconv2d_31/kernel*(
_output_shapes
:??*
dtype0
u
conv2d_31/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*
shared_nameconv2d_31/bias
n
"conv2d_31/bias/Read/ReadVariableOpReadVariableOpconv2d_31/bias*
_output_shapes	
:?*
dtype0
?
conv2d_32/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:??*!
shared_nameconv2d_32/kernel

$conv2d_32/kernel/Read/ReadVariableOpReadVariableOpconv2d_32/kernel*(
_output_shapes
:??*
dtype0
u
conv2d_32/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*
shared_nameconv2d_32/bias
n
"conv2d_32/bias/Read/ReadVariableOpReadVariableOpconv2d_32/bias*
_output_shapes	
:?*
dtype0
?
batch_normalization_14/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*-
shared_namebatch_normalization_14/gamma
?
0batch_normalization_14/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_14/gamma*
_output_shapes	
:?*
dtype0
?
batch_normalization_14/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*,
shared_namebatch_normalization_14/beta
?
/batch_normalization_14/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_14/beta*
_output_shapes	
:?*
dtype0
?
"batch_normalization_14/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*3
shared_name$"batch_normalization_14/moving_mean
?
6batch_normalization_14/moving_mean/Read/ReadVariableOpReadVariableOp"batch_normalization_14/moving_mean*
_output_shapes	
:?*
dtype0
?
&batch_normalization_14/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*7
shared_name(&batch_normalization_14/moving_variance
?
:batch_normalization_14/moving_variance/Read/ReadVariableOpReadVariableOp&batch_normalization_14/moving_variance*
_output_shapes	
:?*
dtype0
?
conv2d_33/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:??*!
shared_nameconv2d_33/kernel

$conv2d_33/kernel/Read/ReadVariableOpReadVariableOpconv2d_33/kernel*(
_output_shapes
:??*
dtype0
u
conv2d_33/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*
shared_nameconv2d_33/bias
n
"conv2d_33/bias/Read/ReadVariableOpReadVariableOpconv2d_33/bias*
_output_shapes	
:?*
dtype0
?
conv2d_34/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:??*!
shared_nameconv2d_34/kernel

$conv2d_34/kernel/Read/ReadVariableOpReadVariableOpconv2d_34/kernel*(
_output_shapes
:??*
dtype0
u
conv2d_34/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*
shared_nameconv2d_34/bias
n
"conv2d_34/bias/Read/ReadVariableOpReadVariableOpconv2d_34/bias*
_output_shapes	
:?*
dtype0
?
batch_normalization_15/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*-
shared_namebatch_normalization_15/gamma
?
0batch_normalization_15/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_15/gamma*
_output_shapes	
:?*
dtype0
?
batch_normalization_15/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*,
shared_namebatch_normalization_15/beta
?
/batch_normalization_15/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_15/beta*
_output_shapes	
:?*
dtype0
?
"batch_normalization_15/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*3
shared_name$"batch_normalization_15/moving_mean
?
6batch_normalization_15/moving_mean/Read/ReadVariableOpReadVariableOp"batch_normalization_15/moving_mean*
_output_shapes	
:?*
dtype0
?
&batch_normalization_15/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*7
shared_name(&batch_normalization_15/moving_variance
?
:batch_normalization_15/moving_variance/Read/ReadVariableOpReadVariableOp&batch_normalization_15/moving_variance*
_output_shapes	
:?*
dtype0
?
conv2d_35/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:??*!
shared_nameconv2d_35/kernel

$conv2d_35/kernel/Read/ReadVariableOpReadVariableOpconv2d_35/kernel*(
_output_shapes
:??*
dtype0
u
conv2d_35/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*
shared_nameconv2d_35/bias
n
"conv2d_35/bias/Read/ReadVariableOpReadVariableOpconv2d_35/bias*
_output_shapes	
:?*
dtype0
?
conv2d_36/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:??*!
shared_nameconv2d_36/kernel

$conv2d_36/kernel/Read/ReadVariableOpReadVariableOpconv2d_36/kernel*(
_output_shapes
:??*
dtype0
u
conv2d_36/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*
shared_nameconv2d_36/bias
n
"conv2d_36/bias/Read/ReadVariableOpReadVariableOpconv2d_36/bias*
_output_shapes	
:?*
dtype0
?
batch_normalization_16/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*-
shared_namebatch_normalization_16/gamma
?
0batch_normalization_16/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_16/gamma*
_output_shapes	
:?*
dtype0
?
batch_normalization_16/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*,
shared_namebatch_normalization_16/beta
?
/batch_normalization_16/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_16/beta*
_output_shapes	
:?*
dtype0
?
"batch_normalization_16/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*3
shared_name$"batch_normalization_16/moving_mean
?
6batch_normalization_16/moving_mean/Read/ReadVariableOpReadVariableOp"batch_normalization_16/moving_mean*
_output_shapes	
:?*
dtype0
?
&batch_normalization_16/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*7
shared_name(&batch_normalization_16/moving_variance
?
:batch_normalization_16/moving_variance/Read/ReadVariableOpReadVariableOp&batch_normalization_16/moving_variance*
_output_shapes	
:?*
dtype0
?
conv2d_37/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:??*!
shared_nameconv2d_37/kernel

$conv2d_37/kernel/Read/ReadVariableOpReadVariableOpconv2d_37/kernel*(
_output_shapes
:??*
dtype0
u
conv2d_37/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*
shared_nameconv2d_37/bias
n
"conv2d_37/bias/Read/ReadVariableOpReadVariableOpconv2d_37/bias*
_output_shapes	
:?*
dtype0
?
conv2d_38/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:??*!
shared_nameconv2d_38/kernel

$conv2d_38/kernel/Read/ReadVariableOpReadVariableOpconv2d_38/kernel*(
_output_shapes
:??*
dtype0
u
conv2d_38/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*
shared_nameconv2d_38/bias
n
"conv2d_38/bias/Read/ReadVariableOpReadVariableOpconv2d_38/bias*
_output_shapes	
:?*
dtype0
?
batch_normalization_17/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*-
shared_namebatch_normalization_17/gamma
?
0batch_normalization_17/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_17/gamma*
_output_shapes	
:?*
dtype0
?
batch_normalization_17/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*,
shared_namebatch_normalization_17/beta
?
/batch_normalization_17/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_17/beta*
_output_shapes	
:?*
dtype0
?
"batch_normalization_17/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*3
shared_name$"batch_normalization_17/moving_mean
?
6batch_normalization_17/moving_mean/Read/ReadVariableOpReadVariableOp"batch_normalization_17/moving_mean*
_output_shapes	
:?*
dtype0
?
&batch_normalization_17/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*7
shared_name(&batch_normalization_17/moving_variance
?
:batch_normalization_17/moving_variance/Read/ReadVariableOpReadVariableOp&batch_normalization_17/moving_variance*
_output_shapes	
:?*
dtype0

NoOpNoOp
??
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*??
value??B?? Bڶ
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

	variables
regularization_losses
trainable_variables
	keras_api

signatures
 
?
layer-0
layer_with_weights-0
layer-1
layer_with_weights-1
layer-2
layer-3
layer-4
layer_with_weights-2
layer-5
layer-6
layer-7
layer_with_weights-3
layer-8
layer-9
layer-10
layer_with_weights-4
layer-11
layer-12
layer_with_weights-5
layer-13
layer-14
layer-15
layer_with_weights-6
layer-16
 layer-17
!layer-18
"layer_with_weights-7
"layer-19
#layer-20
$layer_with_weights-8
$layer-21
%layer-22
&layer-23
'layer_with_weights-9
'layer-24
(layer-25
)layer-26
*layer_with_weights-10
*layer-27
+layer-28
,layer_with_weights-11
,layer-29
-layer-30
.layer-31
/layer_with_weights-12
/layer-32
0layer-33
1layer-34
2layer_with_weights-13
2layer-35
3layer-36
4layer_with_weights-14
4layer-37
5layer-38
6layer-39
7layer_with_weights-15
7layer-40
8layer-41
9layer-42
:layer_with_weights-16
:layer-43
;layer-44
<	variables
=regularization_losses
>trainable_variables
?	keras_api
h

@kernel
Abias
B	variables
Cregularization_losses
Dtrainable_variables
E	keras_api
R
F	variables
Gregularization_losses
Htrainable_variables
I	keras_api
h

Jkernel
Kbias
L	variables
Mregularization_losses
Ntrainable_variables
O	keras_api
R
P	variables
Qregularization_losses
Rtrainable_variables
S	keras_api
R
T	variables
Uregularization_losses
Vtrainable_variables
W	keras_api
^

Xkernel
Y	variables
Zregularization_losses
[trainable_variables
\	keras_api
h

]kernel
^bias
_	variables
`regularization_losses
atrainable_variables
b	keras_api
?
c0
d1
e2
f3
g4
h5
i6
j7
k8
l9
m10
n11
o12
p13
q14
r15
s16
t17
u18
v19
w20
x21
y22
z23
{24
|25
}26
~27
28
?29
?30
?31
?32
?33
?34
?35
?36
?37
?38
?39
?40
?41
?42
?43
?44
?45
@46
A47
J48
K49
X50
]51
^52
 
?
c0
d1
e2
f3
i4
j5
k6
l7
m8
n9
q10
r11
s12
t13
u14
v15
y16
z17
{18
|19
}20
~21
?22
?23
?24
?25
?26
?27
?28
?29
?30
?31
?32
?33
@34
A35
J36
K37
X38
]39
^40
?

	variables
?layer_metrics
regularization_losses
 ?layer_regularization_losses
trainable_variables
?metrics
?layers
?non_trainable_variables
 
 
l

ckernel
dbias
?	variables
?regularization_losses
?trainable_variables
?	keras_api
?
	?axis
	egamma
fbeta
gmoving_mean
hmoving_variance
?	variables
?regularization_losses
?trainable_variables
?	keras_api
V
?	variables
?regularization_losses
?trainable_variables
?	keras_api
V
?	variables
?regularization_losses
?trainable_variables
?	keras_api
l

ikernel
jbias
?	variables
?regularization_losses
?trainable_variables
?	keras_api
V
?	variables
?regularization_losses
?trainable_variables
?	keras_api
V
?	variables
?regularization_losses
?trainable_variables
?	keras_api
l

kkernel
lbias
?	variables
?regularization_losses
?trainable_variables
?	keras_api

?	keras_api
V
?	variables
?regularization_losses
?trainable_variables
?	keras_api
?
	?axis
	mgamma
nbeta
omoving_mean
pmoving_variance
?	variables
?regularization_losses
?trainable_variables
?	keras_api
V
?	variables
?regularization_losses
?trainable_variables
?	keras_api
l

qkernel
rbias
?	variables
?regularization_losses
?trainable_variables
?	keras_api
V
?	variables
?regularization_losses
?trainable_variables
?	keras_api
V
?	variables
?regularization_losses
?trainable_variables
?	keras_api
l

skernel
tbias
?	variables
?regularization_losses
?trainable_variables
?	keras_api

?	keras_api
V
?	variables
?regularization_losses
?trainable_variables
?	keras_api
?
	?axis
	ugamma
vbeta
wmoving_mean
xmoving_variance
?	variables
?regularization_losses
?trainable_variables
?	keras_api
V
?	variables
?regularization_losses
?trainable_variables
?	keras_api
l

ykernel
zbias
?	variables
?regularization_losses
?trainable_variables
?	keras_api
V
?	variables
?regularization_losses
?trainable_variables
?	keras_api
V
?	variables
?regularization_losses
?trainable_variables
?	keras_api
l

{kernel
|bias
?	variables
?regularization_losses
?trainable_variables
?	keras_api

?	keras_api
V
?	variables
?regularization_losses
?trainable_variables
?	keras_api
?
	?axis
	}gamma
~beta
moving_mean
?moving_variance
?	variables
?regularization_losses
?trainable_variables
?	keras_api
V
?	variables
?regularization_losses
?trainable_variables
?	keras_api
n
?kernel
	?bias
?	variables
?regularization_losses
?trainable_variables
?	keras_api
V
?	variables
?regularization_losses
?trainable_variables
?	keras_api
V
?	variables
?regularization_losses
?trainable_variables
?	keras_api
n
?kernel
	?bias
?	variables
?regularization_losses
?trainable_variables
?	keras_api

?	keras_api
V
?	variables
?regularization_losses
?trainable_variables
?	keras_api
?
	?axis

?gamma
	?beta
?moving_mean
?moving_variance
?	variables
?regularization_losses
?trainable_variables
?	keras_api
V
?	variables
?regularization_losses
?trainable_variables
?	keras_api
n
?kernel
	?bias
?	variables
?regularization_losses
?trainable_variables
?	keras_api
V
?	variables
?regularization_losses
?trainable_variables
?	keras_api
V
?	variables
?regularization_losses
?trainable_variables
?	keras_api
n
?kernel
	?bias
?	variables
?regularization_losses
?trainable_variables
?	keras_api

?	keras_api
V
?	variables
?regularization_losses
?trainable_variables
?	keras_api
?
	?axis

?gamma
	?beta
?moving_mean
?moving_variance
?	variables
?regularization_losses
?trainable_variables
?	keras_api
V
?	variables
?regularization_losses
?trainable_variables
?	keras_api
?
c0
d1
e2
f3
g4
h5
i6
j7
k8
l9
m10
n11
o12
p13
q14
r15
s16
t17
u18
v19
w20
x21
y22
z23
{24
|25
}26
~27
28
?29
?30
?31
?32
?33
?34
?35
?36
?37
?38
?39
?40
?41
?42
?43
?44
?45
 
?
c0
d1
e2
f3
i4
j5
k6
l7
m8
n9
q10
r11
s12
t13
u14
v15
y16
z17
{18
|19
}20
~21
?22
?23
?24
?25
?26
?27
?28
?29
?30
?31
?32
?33
?
<	variables
?layer_metrics
=regularization_losses
 ?layer_regularization_losses
>trainable_variables
?metrics
?layers
?non_trainable_variables
\Z
VARIABLE_VALUEconv2d_55/kernel6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE
XV
VARIABLE_VALUEconv2d_55/bias4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE

@0
A1
 

@0
A1
?
B	variables
?layer_metrics
Cregularization_losses
 ?layer_regularization_losses
Dtrainable_variables
?metrics
?layers
?non_trainable_variables
 
 
 
?
F	variables
?layer_metrics
Gregularization_losses
 ?layer_regularization_losses
Htrainable_variables
?metrics
?layers
?non_trainable_variables
\Z
VARIABLE_VALUEconv2d_53/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE
XV
VARIABLE_VALUEconv2d_53/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE

J0
K1
 

J0
K1
?
L	variables
?layer_metrics
Mregularization_losses
 ?layer_regularization_losses
Ntrainable_variables
?metrics
?layers
?non_trainable_variables
 
 
 
?
P	variables
?layer_metrics
Qregularization_losses
 ?layer_regularization_losses
Rtrainable_variables
?metrics
?layers
?non_trainable_variables
 
 
 
?
T	variables
?layer_metrics
Uregularization_losses
 ?layer_regularization_losses
Vtrainable_variables
?metrics
?layers
?non_trainable_variables
ZX
VARIABLE_VALUEdense_3/kernel6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUE

X0
 

X0
?
Y	variables
?layer_metrics
Zregularization_losses
 ?layer_regularization_losses
[trainable_variables
?metrics
?layers
?non_trainable_variables
\Z
VARIABLE_VALUEconv2d_54/kernel6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUE
XV
VARIABLE_VALUEconv2d_54/bias4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUE

]0
^1
 

]0
^1
?
_	variables
?layer_metrics
`regularization_losses
 ?layer_regularization_losses
atrainable_variables
?metrics
?layers
?non_trainable_variables
LJ
VARIABLE_VALUEconv2d_28/kernel&variables/0/.ATTRIBUTES/VARIABLE_VALUE
JH
VARIABLE_VALUEconv2d_28/bias&variables/1/.ATTRIBUTES/VARIABLE_VALUE
XV
VARIABLE_VALUEbatch_normalization_12/gamma&variables/2/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEbatch_normalization_12/beta&variables/3/.ATTRIBUTES/VARIABLE_VALUE
^\
VARIABLE_VALUE"batch_normalization_12/moving_mean&variables/4/.ATTRIBUTES/VARIABLE_VALUE
b`
VARIABLE_VALUE&batch_normalization_12/moving_variance&variables/5/.ATTRIBUTES/VARIABLE_VALUE
LJ
VARIABLE_VALUEconv2d_29/kernel&variables/6/.ATTRIBUTES/VARIABLE_VALUE
JH
VARIABLE_VALUEconv2d_29/bias&variables/7/.ATTRIBUTES/VARIABLE_VALUE
LJ
VARIABLE_VALUEconv2d_30/kernel&variables/8/.ATTRIBUTES/VARIABLE_VALUE
JH
VARIABLE_VALUEconv2d_30/bias&variables/9/.ATTRIBUTES/VARIABLE_VALUE
YW
VARIABLE_VALUEbatch_normalization_13/gamma'variables/10/.ATTRIBUTES/VARIABLE_VALUE
XV
VARIABLE_VALUEbatch_normalization_13/beta'variables/11/.ATTRIBUTES/VARIABLE_VALUE
_]
VARIABLE_VALUE"batch_normalization_13/moving_mean'variables/12/.ATTRIBUTES/VARIABLE_VALUE
ca
VARIABLE_VALUE&batch_normalization_13/moving_variance'variables/13/.ATTRIBUTES/VARIABLE_VALUE
MK
VARIABLE_VALUEconv2d_31/kernel'variables/14/.ATTRIBUTES/VARIABLE_VALUE
KI
VARIABLE_VALUEconv2d_31/bias'variables/15/.ATTRIBUTES/VARIABLE_VALUE
MK
VARIABLE_VALUEconv2d_32/kernel'variables/16/.ATTRIBUTES/VARIABLE_VALUE
KI
VARIABLE_VALUEconv2d_32/bias'variables/17/.ATTRIBUTES/VARIABLE_VALUE
YW
VARIABLE_VALUEbatch_normalization_14/gamma'variables/18/.ATTRIBUTES/VARIABLE_VALUE
XV
VARIABLE_VALUEbatch_normalization_14/beta'variables/19/.ATTRIBUTES/VARIABLE_VALUE
_]
VARIABLE_VALUE"batch_normalization_14/moving_mean'variables/20/.ATTRIBUTES/VARIABLE_VALUE
ca
VARIABLE_VALUE&batch_normalization_14/moving_variance'variables/21/.ATTRIBUTES/VARIABLE_VALUE
MK
VARIABLE_VALUEconv2d_33/kernel'variables/22/.ATTRIBUTES/VARIABLE_VALUE
KI
VARIABLE_VALUEconv2d_33/bias'variables/23/.ATTRIBUTES/VARIABLE_VALUE
MK
VARIABLE_VALUEconv2d_34/kernel'variables/24/.ATTRIBUTES/VARIABLE_VALUE
KI
VARIABLE_VALUEconv2d_34/bias'variables/25/.ATTRIBUTES/VARIABLE_VALUE
YW
VARIABLE_VALUEbatch_normalization_15/gamma'variables/26/.ATTRIBUTES/VARIABLE_VALUE
XV
VARIABLE_VALUEbatch_normalization_15/beta'variables/27/.ATTRIBUTES/VARIABLE_VALUE
_]
VARIABLE_VALUE"batch_normalization_15/moving_mean'variables/28/.ATTRIBUTES/VARIABLE_VALUE
ca
VARIABLE_VALUE&batch_normalization_15/moving_variance'variables/29/.ATTRIBUTES/VARIABLE_VALUE
MK
VARIABLE_VALUEconv2d_35/kernel'variables/30/.ATTRIBUTES/VARIABLE_VALUE
KI
VARIABLE_VALUEconv2d_35/bias'variables/31/.ATTRIBUTES/VARIABLE_VALUE
MK
VARIABLE_VALUEconv2d_36/kernel'variables/32/.ATTRIBUTES/VARIABLE_VALUE
KI
VARIABLE_VALUEconv2d_36/bias'variables/33/.ATTRIBUTES/VARIABLE_VALUE
YW
VARIABLE_VALUEbatch_normalization_16/gamma'variables/34/.ATTRIBUTES/VARIABLE_VALUE
XV
VARIABLE_VALUEbatch_normalization_16/beta'variables/35/.ATTRIBUTES/VARIABLE_VALUE
_]
VARIABLE_VALUE"batch_normalization_16/moving_mean'variables/36/.ATTRIBUTES/VARIABLE_VALUE
ca
VARIABLE_VALUE&batch_normalization_16/moving_variance'variables/37/.ATTRIBUTES/VARIABLE_VALUE
MK
VARIABLE_VALUEconv2d_37/kernel'variables/38/.ATTRIBUTES/VARIABLE_VALUE
KI
VARIABLE_VALUEconv2d_37/bias'variables/39/.ATTRIBUTES/VARIABLE_VALUE
MK
VARIABLE_VALUEconv2d_38/kernel'variables/40/.ATTRIBUTES/VARIABLE_VALUE
KI
VARIABLE_VALUEconv2d_38/bias'variables/41/.ATTRIBUTES/VARIABLE_VALUE
YW
VARIABLE_VALUEbatch_normalization_17/gamma'variables/42/.ATTRIBUTES/VARIABLE_VALUE
XV
VARIABLE_VALUEbatch_normalization_17/beta'variables/43/.ATTRIBUTES/VARIABLE_VALUE
_]
VARIABLE_VALUE"batch_normalization_17/moving_mean'variables/44/.ATTRIBUTES/VARIABLE_VALUE
ca
VARIABLE_VALUE&batch_normalization_17/moving_variance'variables/45/.ATTRIBUTES/VARIABLE_VALUE
 
 
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
[
g0
h1
o2
p3
w4
x5
6
?7
?8
?9
?10
?11

c0
d1
 

c0
d1
?
?	variables
?layer_metrics
?regularization_losses
 ?layer_regularization_losses
?trainable_variables
?metrics
?layers
?non_trainable_variables
 

e0
f1
g2
h3
 

e0
f1
?
?	variables
?layer_metrics
?regularization_losses
 ?layer_regularization_losses
?trainable_variables
?metrics
?layers
?non_trainable_variables
 
 
 
?
?	variables
?layer_metrics
?regularization_losses
 ?layer_regularization_losses
?trainable_variables
?metrics
?layers
?non_trainable_variables
 
 
 
?
?	variables
?layer_metrics
?regularization_losses
 ?layer_regularization_losses
?trainable_variables
?metrics
?layers
?non_trainable_variables

i0
j1
 

i0
j1
?
?	variables
?layer_metrics
?regularization_losses
 ?layer_regularization_losses
?trainable_variables
?metrics
?layers
?non_trainable_variables
 
 
 
?
?	variables
?layer_metrics
?regularization_losses
 ?layer_regularization_losses
?trainable_variables
?metrics
?layers
?non_trainable_variables
 
 
 
?
?	variables
?layer_metrics
?regularization_losses
 ?layer_regularization_losses
?trainable_variables
?metrics
?layers
?non_trainable_variables

k0
l1
 

k0
l1
?
?	variables
?layer_metrics
?regularization_losses
 ?layer_regularization_losses
?trainable_variables
?metrics
?layers
?non_trainable_variables
 
 
 
 
?
?	variables
?layer_metrics
?regularization_losses
 ?layer_regularization_losses
?trainable_variables
?metrics
?layers
?non_trainable_variables
 

m0
n1
o2
p3
 

m0
n1
?
?	variables
?layer_metrics
?regularization_losses
 ?layer_regularization_losses
?trainable_variables
?metrics
?layers
?non_trainable_variables
 
 
 
?
?	variables
?layer_metrics
?regularization_losses
 ?layer_regularization_losses
?trainable_variables
?metrics
?layers
?non_trainable_variables

q0
r1
 

q0
r1
?
?	variables
?layer_metrics
?regularization_losses
 ?layer_regularization_losses
?trainable_variables
?metrics
?layers
?non_trainable_variables
 
 
 
?
?	variables
?layer_metrics
?regularization_losses
 ?layer_regularization_losses
?trainable_variables
?metrics
?layers
?non_trainable_variables
 
 
 
?
?	variables
?layer_metrics
?regularization_losses
 ?layer_regularization_losses
?trainable_variables
?metrics
?layers
?non_trainable_variables

s0
t1
 

s0
t1
?
?	variables
?layer_metrics
?regularization_losses
 ?layer_regularization_losses
?trainable_variables
?metrics
?layers
?non_trainable_variables
 
 
 
 
?
?	variables
?layer_metrics
?regularization_losses
 ?layer_regularization_losses
?trainable_variables
?metrics
?layers
?non_trainable_variables
 

u0
v1
w2
x3
 

u0
v1
?
?	variables
?layer_metrics
?regularization_losses
 ?layer_regularization_losses
?trainable_variables
?metrics
?layers
?non_trainable_variables
 
 
 
?
?	variables
?layer_metrics
?regularization_losses
 ?layer_regularization_losses
?trainable_variables
?metrics
?layers
?non_trainable_variables

y0
z1
 

y0
z1
?
?	variables
?layer_metrics
?regularization_losses
 ?layer_regularization_losses
?trainable_variables
?metrics
?layers
?non_trainable_variables
 
 
 
?
?	variables
?layer_metrics
?regularization_losses
 ?layer_regularization_losses
?trainable_variables
?metrics
?layers
?non_trainable_variables
 
 
 
?
?	variables
?layer_metrics
?regularization_losses
 ?layer_regularization_losses
?trainable_variables
?metrics
?layers
?non_trainable_variables

{0
|1
 

{0
|1
?
?	variables
?layer_metrics
?regularization_losses
 ?layer_regularization_losses
?trainable_variables
?metrics
?layers
?non_trainable_variables
 
 
 
 
?
?	variables
?layer_metrics
?regularization_losses
 ?layer_regularization_losses
?trainable_variables
?metrics
?layers
?non_trainable_variables
 

}0
~1
2
?3
 

}0
~1
?
?	variables
?layer_metrics
?regularization_losses
 ?layer_regularization_losses
?trainable_variables
?metrics
?layers
?non_trainable_variables
 
 
 
?
?	variables
?layer_metrics
?regularization_losses
 ?layer_regularization_losses
?trainable_variables
?metrics
?layers
?non_trainable_variables

?0
?1
 

?0
?1
?
?	variables
?layer_metrics
?regularization_losses
 ?layer_regularization_losses
?trainable_variables
?metrics
?layers
?non_trainable_variables
 
 
 
?
?	variables
?layer_metrics
?regularization_losses
 ?layer_regularization_losses
?trainable_variables
?metrics
?layers
?non_trainable_variables
 
 
 
?
?	variables
?layer_metrics
?regularization_losses
 ?layer_regularization_losses
?trainable_variables
?metrics
?layers
?non_trainable_variables

?0
?1
 

?0
?1
?
?	variables
?layer_metrics
?regularization_losses
 ?layer_regularization_losses
?trainable_variables
?metrics
?layers
?non_trainable_variables
 
 
 
 
?
?	variables
?layer_metrics
?regularization_losses
 ?layer_regularization_losses
?trainable_variables
?metrics
?layers
?non_trainable_variables
 
 
?0
?1
?2
?3
 

?0
?1
?
?	variables
?layer_metrics
?regularization_losses
 ?layer_regularization_losses
?trainable_variables
?metrics
?layers
?non_trainable_variables
 
 
 
?
?	variables
?layer_metrics
?regularization_losses
 ?layer_regularization_losses
?trainable_variables
?metrics
?layers
?non_trainable_variables

?0
?1
 

?0
?1
?
?	variables
?layer_metrics
?regularization_losses
 ?layer_regularization_losses
?trainable_variables
?metrics
?layers
?non_trainable_variables
 
 
 
?
?	variables
?layer_metrics
?regularization_losses
 ?layer_regularization_losses
?trainable_variables
?metrics
?layers
?non_trainable_variables
 
 
 
?
?	variables
?layer_metrics
?regularization_losses
 ?layer_regularization_losses
?trainable_variables
?metrics
?layers
?non_trainable_variables

?0
?1
 

?0
?1
?
?	variables
?layer_metrics
?regularization_losses
 ?layer_regularization_losses
?trainable_variables
?metrics
?layers
?non_trainable_variables
 
 
 
 
?
?	variables
?layer_metrics
?regularization_losses
 ?layer_regularization_losses
?trainable_variables
?metrics
?layers
?non_trainable_variables
 
 
?0
?1
?2
?3
 

?0
?1
?
?	variables
?layer_metrics
?regularization_losses
 ?layer_regularization_losses
?trainable_variables
?metrics
?layers
?non_trainable_variables
 
 
 
?
?	variables
?layer_metrics
?regularization_losses
 ?layer_regularization_losses
?trainable_variables
?metrics
?layers
?non_trainable_variables
 
 
 
?
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
14
15
16
 17
!18
"19
#20
$21
%22
&23
'24
(25
)26
*27
+28
,29
-30
.31
/32
033
134
235
336
437
538
639
740
841
942
:43
;44
[
g0
h1
o2
p3
w4
x5
6
?7
?8
?9
?10
?11
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
 

g0
h1
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
o0
p1
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
w0
x1
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

0
?1
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

?0
?1
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

?0
?1
 
 
 
 
 
?
serving_default_input_8Placeholder*/
_output_shapes
:?????????*
dtype0*$
shape:?????????
?
StatefulPartitionedCallStatefulPartitionedCallserving_default_input_8conv2d_28/kernelconv2d_28/biasbatch_normalization_12/gammabatch_normalization_12/beta"batch_normalization_12/moving_mean&batch_normalization_12/moving_varianceconv2d_29/kernelconv2d_29/biasconv2d_30/kernelconv2d_30/biasbatch_normalization_13/gammabatch_normalization_13/beta"batch_normalization_13/moving_mean&batch_normalization_13/moving_varianceconv2d_31/kernelconv2d_31/biasconv2d_32/kernelconv2d_32/biasbatch_normalization_14/gammabatch_normalization_14/beta"batch_normalization_14/moving_mean&batch_normalization_14/moving_varianceconv2d_33/kernelconv2d_33/biasconv2d_34/kernelconv2d_34/biasbatch_normalization_15/gammabatch_normalization_15/beta"batch_normalization_15/moving_mean&batch_normalization_15/moving_varianceconv2d_35/kernelconv2d_35/biasconv2d_36/kernelconv2d_36/biasbatch_normalization_16/gammabatch_normalization_16/beta"batch_normalization_16/moving_mean&batch_normalization_16/moving_varianceconv2d_37/kernelconv2d_37/biasconv2d_38/kernelconv2d_38/biasbatch_normalization_17/gammabatch_normalization_17/beta"batch_normalization_17/moving_mean&batch_normalization_17/moving_varianceconv2d_55/kernelconv2d_55/biasconv2d_53/kernelconv2d_53/biasconv2d_54/kernelconv2d_54/biasdense_3/kernel*A
Tin:
826*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:?????????I:?????????*W
_read_only_resource_inputs9
75	
 !"#$%&'()*+,-./012345*-
config_proto

CPU

GPU 2J 8? */
f*R(
&__inference_signature_wrapper_22006163
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
?
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename$conv2d_55/kernel/Read/ReadVariableOp"conv2d_55/bias/Read/ReadVariableOp$conv2d_53/kernel/Read/ReadVariableOp"conv2d_53/bias/Read/ReadVariableOp"dense_3/kernel/Read/ReadVariableOp$conv2d_54/kernel/Read/ReadVariableOp"conv2d_54/bias/Read/ReadVariableOp$conv2d_28/kernel/Read/ReadVariableOp"conv2d_28/bias/Read/ReadVariableOp0batch_normalization_12/gamma/Read/ReadVariableOp/batch_normalization_12/beta/Read/ReadVariableOp6batch_normalization_12/moving_mean/Read/ReadVariableOp:batch_normalization_12/moving_variance/Read/ReadVariableOp$conv2d_29/kernel/Read/ReadVariableOp"conv2d_29/bias/Read/ReadVariableOp$conv2d_30/kernel/Read/ReadVariableOp"conv2d_30/bias/Read/ReadVariableOp0batch_normalization_13/gamma/Read/ReadVariableOp/batch_normalization_13/beta/Read/ReadVariableOp6batch_normalization_13/moving_mean/Read/ReadVariableOp:batch_normalization_13/moving_variance/Read/ReadVariableOp$conv2d_31/kernel/Read/ReadVariableOp"conv2d_31/bias/Read/ReadVariableOp$conv2d_32/kernel/Read/ReadVariableOp"conv2d_32/bias/Read/ReadVariableOp0batch_normalization_14/gamma/Read/ReadVariableOp/batch_normalization_14/beta/Read/ReadVariableOp6batch_normalization_14/moving_mean/Read/ReadVariableOp:batch_normalization_14/moving_variance/Read/ReadVariableOp$conv2d_33/kernel/Read/ReadVariableOp"conv2d_33/bias/Read/ReadVariableOp$conv2d_34/kernel/Read/ReadVariableOp"conv2d_34/bias/Read/ReadVariableOp0batch_normalization_15/gamma/Read/ReadVariableOp/batch_normalization_15/beta/Read/ReadVariableOp6batch_normalization_15/moving_mean/Read/ReadVariableOp:batch_normalization_15/moving_variance/Read/ReadVariableOp$conv2d_35/kernel/Read/ReadVariableOp"conv2d_35/bias/Read/ReadVariableOp$conv2d_36/kernel/Read/ReadVariableOp"conv2d_36/bias/Read/ReadVariableOp0batch_normalization_16/gamma/Read/ReadVariableOp/batch_normalization_16/beta/Read/ReadVariableOp6batch_normalization_16/moving_mean/Read/ReadVariableOp:batch_normalization_16/moving_variance/Read/ReadVariableOp$conv2d_37/kernel/Read/ReadVariableOp"conv2d_37/bias/Read/ReadVariableOp$conv2d_38/kernel/Read/ReadVariableOp"conv2d_38/bias/Read/ReadVariableOp0batch_normalization_17/gamma/Read/ReadVariableOp/batch_normalization_17/beta/Read/ReadVariableOp6batch_normalization_17/moving_mean/Read/ReadVariableOp:batch_normalization_17/moving_variance/Read/ReadVariableOpConst*B
Tin;
927*
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
GPU 2J 8? **
f%R#
!__inference__traced_save_22008922
?
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenameconv2d_55/kernelconv2d_55/biasconv2d_53/kernelconv2d_53/biasdense_3/kernelconv2d_54/kernelconv2d_54/biasconv2d_28/kernelconv2d_28/biasbatch_normalization_12/gammabatch_normalization_12/beta"batch_normalization_12/moving_mean&batch_normalization_12/moving_varianceconv2d_29/kernelconv2d_29/biasconv2d_30/kernelconv2d_30/biasbatch_normalization_13/gammabatch_normalization_13/beta"batch_normalization_13/moving_mean&batch_normalization_13/moving_varianceconv2d_31/kernelconv2d_31/biasconv2d_32/kernelconv2d_32/biasbatch_normalization_14/gammabatch_normalization_14/beta"batch_normalization_14/moving_mean&batch_normalization_14/moving_varianceconv2d_33/kernelconv2d_33/biasconv2d_34/kernelconv2d_34/biasbatch_normalization_15/gammabatch_normalization_15/beta"batch_normalization_15/moving_mean&batch_normalization_15/moving_varianceconv2d_35/kernelconv2d_35/biasconv2d_36/kernelconv2d_36/biasbatch_normalization_16/gammabatch_normalization_16/beta"batch_normalization_16/moving_mean&batch_normalization_16/moving_varianceconv2d_37/kernelconv2d_37/biasconv2d_38/kernelconv2d_38/biasbatch_normalization_17/gammabatch_normalization_17/beta"batch_normalization_17/moving_mean&batch_normalization_17/moving_variance*A
Tin:
826*
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
GPU 2J 8? *-
f(R&
$__inference__traced_restore_22009091??)
?
?
G__inference_conv2d_35_layer_call_and_return_conditional_losses_22003402

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
g
K__inference_activation_23_layer_call_and_return_conditional_losses_22007668

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
?	
?
9__inference_batch_normalization_15_layer_call_fn_22008169

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
GPU 2J 8? *]
fXRV
T__inference_batch_normalization_15_layer_call_and_return_conditional_losses_220027082
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
?	
?
9__inference_batch_normalization_13_layer_call_fn_22007742

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
GPU 2J 8? *]
fXRV
T__inference_batch_normalization_13_layer_call_and_return_conditional_losses_220025002
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
?
?
T__inference_batch_normalization_13_layer_call_and_return_conditional_losses_22007786

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
?
f
H__inference_dropout_35_layer_call_and_return_conditional_losses_22008563

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
g
K__inference_activation_30_layer_call_and_return_conditional_losses_22003444

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
?
f
H__inference_dropout_29_layer_call_and_return_conditional_losses_22007903

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
?
I
-__inference_dropout_33_layer_call_fn_22008338

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
GPU 2J 8? *Q
fLRJ
H__inference_dropout_33_layer_call_and_return_conditional_losses_220038572
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
G__inference_conv2d_37_layer_call_and_return_conditional_losses_22003490

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
d
H__inference_dropout_36_layer_call_and_return_conditional_losses_22008739

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
ٺ
?
E__inference_model_4_layer_call_and_return_conditional_losses_22004840
input_5-
conv2d_28_22004703:?!
conv2d_28_22004705:	?.
batch_normalization_12_22004708:	?.
batch_normalization_12_22004710:	?.
batch_normalization_12_22004712:	?.
batch_normalization_12_22004714:	?.
conv2d_29_22004719:??!
conv2d_29_22004721:	?.
conv2d_30_22004726:??!
conv2d_30_22004728:	?.
batch_normalization_13_22004733:	?.
batch_normalization_13_22004735:	?.
batch_normalization_13_22004737:	?.
batch_normalization_13_22004739:	?.
conv2d_31_22004743:??!
conv2d_31_22004745:	?.
conv2d_32_22004750:??!
conv2d_32_22004752:	?.
batch_normalization_14_22004757:	?.
batch_normalization_14_22004759:	?.
batch_normalization_14_22004761:	?.
batch_normalization_14_22004763:	?.
conv2d_33_22004767:??!
conv2d_33_22004769:	?.
conv2d_34_22004774:??!
conv2d_34_22004776:	?.
batch_normalization_15_22004781:	?.
batch_normalization_15_22004783:	?.
batch_normalization_15_22004785:	?.
batch_normalization_15_22004787:	?.
conv2d_35_22004791:??!
conv2d_35_22004793:	?.
conv2d_36_22004798:??!
conv2d_36_22004800:	?.
batch_normalization_16_22004805:	?.
batch_normalization_16_22004807:	?.
batch_normalization_16_22004809:	?.
batch_normalization_16_22004811:	?.
conv2d_37_22004815:??!
conv2d_37_22004817:	?.
conv2d_38_22004822:??!
conv2d_38_22004824:	?.
batch_normalization_17_22004829:	?.
batch_normalization_17_22004831:	?.
batch_normalization_17_22004833:	?.
batch_normalization_17_22004835:	?
identity??.batch_normalization_12/StatefulPartitionedCall?.batch_normalization_13/StatefulPartitionedCall?.batch_normalization_14/StatefulPartitionedCall?.batch_normalization_15/StatefulPartitionedCall?.batch_normalization_16/StatefulPartitionedCall?.batch_normalization_17/StatefulPartitionedCall?!conv2d_28/StatefulPartitionedCall?!conv2d_29/StatefulPartitionedCall?!conv2d_30/StatefulPartitionedCall?!conv2d_31/StatefulPartitionedCall?!conv2d_32/StatefulPartitionedCall?!conv2d_33/StatefulPartitionedCall?!conv2d_34/StatefulPartitionedCall?!conv2d_35/StatefulPartitionedCall?!conv2d_36/StatefulPartitionedCall?!conv2d_37/StatefulPartitionedCall?!conv2d_38/StatefulPartitionedCall?
!conv2d_28/StatefulPartitionedCallStatefulPartitionedCallinput_5conv2d_28_22004703conv2d_28_22004705*
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
GPU 2J 8? *P
fKRI
G__inference_conv2d_28_layer_call_and_return_conditional_losses_220030812#
!conv2d_28/StatefulPartitionedCall?
.batch_normalization_12/StatefulPartitionedCallStatefulPartitionedCall*conv2d_28/StatefulPartitionedCall:output:0batch_normalization_12_22004708batch_normalization_12_22004710batch_normalization_12_22004712batch_normalization_12_22004714*
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
GPU 2J 8? *]
fXRV
T__inference_batch_normalization_12_layer_call_and_return_conditional_losses_2200310420
.batch_normalization_12/StatefulPartitionedCall?
activation_22/PartitionedCallPartitionedCall7batch_normalization_12/StatefulPartitionedCall:output:0*
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
GPU 2J 8? *T
fORM
K__inference_activation_22_layer_call_and_return_conditional_losses_220031192
activation_22/PartitionedCall?
dropout_26/PartitionedCallPartitionedCall&activation_22/PartitionedCall:output:0*
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
GPU 2J 8? *Q
fLRJ
H__inference_dropout_26_layer_call_and_return_conditional_losses_220031262
dropout_26/PartitionedCall?
!conv2d_29/StatefulPartitionedCallStatefulPartitionedCall#dropout_26/PartitionedCall:output:0conv2d_29_22004719conv2d_29_22004721*
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
GPU 2J 8? *P
fKRI
G__inference_conv2d_29_layer_call_and_return_conditional_losses_220031382#
!conv2d_29/StatefulPartitionedCall?
activation_23/PartitionedCallPartitionedCall*conv2d_29/StatefulPartitionedCall:output:0*
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
GPU 2J 8? *T
fORM
K__inference_activation_23_layer_call_and_return_conditional_losses_220031492
activation_23/PartitionedCall?
dropout_27/PartitionedCallPartitionedCall&activation_23/PartitionedCall:output:0*
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
GPU 2J 8? *Q
fLRJ
H__inference_dropout_27_layer_call_and_return_conditional_losses_220031562
dropout_27/PartitionedCall?
!conv2d_30/StatefulPartitionedCallStatefulPartitionedCall#dropout_27/PartitionedCall:output:0conv2d_30_22004726conv2d_30_22004728*
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
GPU 2J 8? *P
fKRI
G__inference_conv2d_30_layer_call_and_return_conditional_losses_220031682#
!conv2d_30/StatefulPartitionedCall?
tf.__operators__.add_10/AddV2AddV2*conv2d_30/StatefulPartitionedCall:output:0#dropout_26/PartitionedCall:output:0*
T0*0
_output_shapes
:??????????2
tf.__operators__.add_10/AddV2?
activation_24/PartitionedCallPartitionedCall!tf.__operators__.add_10/AddV2:z:0*
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
GPU 2J 8? *T
fORM
K__inference_activation_24_layer_call_and_return_conditional_losses_220031802
activation_24/PartitionedCall?
.batch_normalization_13/StatefulPartitionedCallStatefulPartitionedCall&activation_24/PartitionedCall:output:0batch_normalization_13_22004733batch_normalization_13_22004735batch_normalization_13_22004737batch_normalization_13_22004739*
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
GPU 2J 8? *]
fXRV
T__inference_batch_normalization_13_layer_call_and_return_conditional_losses_2200319920
.batch_normalization_13/StatefulPartitionedCall?
dropout_28/PartitionedCallPartitionedCall7batch_normalization_13/StatefulPartitionedCall:output:0*
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
GPU 2J 8? *Q
fLRJ
H__inference_dropout_28_layer_call_and_return_conditional_losses_220032142
dropout_28/PartitionedCall?
!conv2d_31/StatefulPartitionedCallStatefulPartitionedCall#dropout_28/PartitionedCall:output:0conv2d_31_22004743conv2d_31_22004745*
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
GPU 2J 8? *P
fKRI
G__inference_conv2d_31_layer_call_and_return_conditional_losses_220032262#
!conv2d_31/StatefulPartitionedCall?
activation_25/PartitionedCallPartitionedCall*conv2d_31/StatefulPartitionedCall:output:0*
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
GPU 2J 8? *T
fORM
K__inference_activation_25_layer_call_and_return_conditional_losses_220032372
activation_25/PartitionedCall?
dropout_29/PartitionedCallPartitionedCall&activation_25/PartitionedCall:output:0*
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
GPU 2J 8? *Q
fLRJ
H__inference_dropout_29_layer_call_and_return_conditional_losses_220032442
dropout_29/PartitionedCall?
!conv2d_32/StatefulPartitionedCallStatefulPartitionedCall#dropout_29/PartitionedCall:output:0conv2d_32_22004750conv2d_32_22004752*
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
GPU 2J 8? *P
fKRI
G__inference_conv2d_32_layer_call_and_return_conditional_losses_220032562#
!conv2d_32/StatefulPartitionedCall?
tf.__operators__.add_11/AddV2AddV2*conv2d_32/StatefulPartitionedCall:output:0#dropout_28/PartitionedCall:output:0*
T0*0
_output_shapes
:??????????2
tf.__operators__.add_11/AddV2?
activation_26/PartitionedCallPartitionedCall!tf.__operators__.add_11/AddV2:z:0*
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
GPU 2J 8? *T
fORM
K__inference_activation_26_layer_call_and_return_conditional_losses_220032682
activation_26/PartitionedCall?
.batch_normalization_14/StatefulPartitionedCallStatefulPartitionedCall&activation_26/PartitionedCall:output:0batch_normalization_14_22004757batch_normalization_14_22004759batch_normalization_14_22004761batch_normalization_14_22004763*
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
GPU 2J 8? *]
fXRV
T__inference_batch_normalization_14_layer_call_and_return_conditional_losses_2200328720
.batch_normalization_14/StatefulPartitionedCall?
dropout_30/PartitionedCallPartitionedCall7batch_normalization_14/StatefulPartitionedCall:output:0*
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
GPU 2J 8? *Q
fLRJ
H__inference_dropout_30_layer_call_and_return_conditional_losses_220033022
dropout_30/PartitionedCall?
!conv2d_33/StatefulPartitionedCallStatefulPartitionedCall#dropout_30/PartitionedCall:output:0conv2d_33_22004767conv2d_33_22004769*
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
GPU 2J 8? *P
fKRI
G__inference_conv2d_33_layer_call_and_return_conditional_losses_220033142#
!conv2d_33/StatefulPartitionedCall?
activation_27/PartitionedCallPartitionedCall*conv2d_33/StatefulPartitionedCall:output:0*
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
GPU 2J 8? *T
fORM
K__inference_activation_27_layer_call_and_return_conditional_losses_220033252
activation_27/PartitionedCall?
dropout_31/PartitionedCallPartitionedCall&activation_27/PartitionedCall:output:0*
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
GPU 2J 8? *Q
fLRJ
H__inference_dropout_31_layer_call_and_return_conditional_losses_220033322
dropout_31/PartitionedCall?
!conv2d_34/StatefulPartitionedCallStatefulPartitionedCall#dropout_31/PartitionedCall:output:0conv2d_34_22004774conv2d_34_22004776*
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
GPU 2J 8? *P
fKRI
G__inference_conv2d_34_layer_call_and_return_conditional_losses_220033442#
!conv2d_34/StatefulPartitionedCall?
tf.__operators__.add_12/AddV2AddV2*conv2d_34/StatefulPartitionedCall:output:0#dropout_30/PartitionedCall:output:0*
T0*0
_output_shapes
:??????????2
tf.__operators__.add_12/AddV2?
activation_28/PartitionedCallPartitionedCall!tf.__operators__.add_12/AddV2:z:0*
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
GPU 2J 8? *T
fORM
K__inference_activation_28_layer_call_and_return_conditional_losses_220033562
activation_28/PartitionedCall?
.batch_normalization_15/StatefulPartitionedCallStatefulPartitionedCall&activation_28/PartitionedCall:output:0batch_normalization_15_22004781batch_normalization_15_22004783batch_normalization_15_22004785batch_normalization_15_22004787*
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
GPU 2J 8? *]
fXRV
T__inference_batch_normalization_15_layer_call_and_return_conditional_losses_2200337520
.batch_normalization_15/StatefulPartitionedCall?
dropout_32/PartitionedCallPartitionedCall7batch_normalization_15/StatefulPartitionedCall:output:0*
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
GPU 2J 8? *Q
fLRJ
H__inference_dropout_32_layer_call_and_return_conditional_losses_220033902
dropout_32/PartitionedCall?
!conv2d_35/StatefulPartitionedCallStatefulPartitionedCall#dropout_32/PartitionedCall:output:0conv2d_35_22004791conv2d_35_22004793*
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
GPU 2J 8? *P
fKRI
G__inference_conv2d_35_layer_call_and_return_conditional_losses_220034022#
!conv2d_35/StatefulPartitionedCall?
activation_29/PartitionedCallPartitionedCall*conv2d_35/StatefulPartitionedCall:output:0*
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
GPU 2J 8? *T
fORM
K__inference_activation_29_layer_call_and_return_conditional_losses_220034132
activation_29/PartitionedCall?
dropout_33/PartitionedCallPartitionedCall&activation_29/PartitionedCall:output:0*
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
GPU 2J 8? *Q
fLRJ
H__inference_dropout_33_layer_call_and_return_conditional_losses_220034202
dropout_33/PartitionedCall?
!conv2d_36/StatefulPartitionedCallStatefulPartitionedCall#dropout_33/PartitionedCall:output:0conv2d_36_22004798conv2d_36_22004800*
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
GPU 2J 8? *P
fKRI
G__inference_conv2d_36_layer_call_and_return_conditional_losses_220034322#
!conv2d_36/StatefulPartitionedCall?
tf.__operators__.add_13/AddV2AddV2*conv2d_36/StatefulPartitionedCall:output:0#dropout_32/PartitionedCall:output:0*
T0*0
_output_shapes
:??????????2
tf.__operators__.add_13/AddV2?
activation_30/PartitionedCallPartitionedCall!tf.__operators__.add_13/AddV2:z:0*
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
GPU 2J 8? *T
fORM
K__inference_activation_30_layer_call_and_return_conditional_losses_220034442
activation_30/PartitionedCall?
.batch_normalization_16/StatefulPartitionedCallStatefulPartitionedCall&activation_30/PartitionedCall:output:0batch_normalization_16_22004805batch_normalization_16_22004807batch_normalization_16_22004809batch_normalization_16_22004811*
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
GPU 2J 8? *]
fXRV
T__inference_batch_normalization_16_layer_call_and_return_conditional_losses_2200346320
.batch_normalization_16/StatefulPartitionedCall?
dropout_34/PartitionedCallPartitionedCall7batch_normalization_16/StatefulPartitionedCall:output:0*
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
GPU 2J 8? *Q
fLRJ
H__inference_dropout_34_layer_call_and_return_conditional_losses_220034782
dropout_34/PartitionedCall?
!conv2d_37/StatefulPartitionedCallStatefulPartitionedCall#dropout_34/PartitionedCall:output:0conv2d_37_22004815conv2d_37_22004817*
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
GPU 2J 8? *P
fKRI
G__inference_conv2d_37_layer_call_and_return_conditional_losses_220034902#
!conv2d_37/StatefulPartitionedCall?
activation_31/PartitionedCallPartitionedCall*conv2d_37/StatefulPartitionedCall:output:0*
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
GPU 2J 8? *T
fORM
K__inference_activation_31_layer_call_and_return_conditional_losses_220035012
activation_31/PartitionedCall?
dropout_35/PartitionedCallPartitionedCall&activation_31/PartitionedCall:output:0*
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
GPU 2J 8? *Q
fLRJ
H__inference_dropout_35_layer_call_and_return_conditional_losses_220035082
dropout_35/PartitionedCall?
!conv2d_38/StatefulPartitionedCallStatefulPartitionedCall#dropout_35/PartitionedCall:output:0conv2d_38_22004822conv2d_38_22004824*
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
GPU 2J 8? *P
fKRI
G__inference_conv2d_38_layer_call_and_return_conditional_losses_220035202#
!conv2d_38/StatefulPartitionedCall?
tf.__operators__.add_14/AddV2AddV2*conv2d_38/StatefulPartitionedCall:output:0#dropout_34/PartitionedCall:output:0*
T0*0
_output_shapes
:??????????2
tf.__operators__.add_14/AddV2?
activation_32/PartitionedCallPartitionedCall!tf.__operators__.add_14/AddV2:z:0*
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
GPU 2J 8? *T
fORM
K__inference_activation_32_layer_call_and_return_conditional_losses_220035322
activation_32/PartitionedCall?
.batch_normalization_17/StatefulPartitionedCallStatefulPartitionedCall&activation_32/PartitionedCall:output:0batch_normalization_17_22004829batch_normalization_17_22004831batch_normalization_17_22004833batch_normalization_17_22004835*
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
GPU 2J 8? *]
fXRV
T__inference_batch_normalization_17_layer_call_and_return_conditional_losses_2200355120
.batch_normalization_17/StatefulPartitionedCall?
dropout_36/PartitionedCallPartitionedCall7batch_normalization_17/StatefulPartitionedCall:output:0*
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
GPU 2J 8? *Q
fLRJ
H__inference_dropout_36_layer_call_and_return_conditional_losses_220035662
dropout_36/PartitionedCall?
IdentityIdentity#dropout_36/PartitionedCall:output:0^NoOp*
T0*0
_output_shapes
:??????????2

Identity?
NoOpNoOp/^batch_normalization_12/StatefulPartitionedCall/^batch_normalization_13/StatefulPartitionedCall/^batch_normalization_14/StatefulPartitionedCall/^batch_normalization_15/StatefulPartitionedCall/^batch_normalization_16/StatefulPartitionedCall/^batch_normalization_17/StatefulPartitionedCall"^conv2d_28/StatefulPartitionedCall"^conv2d_29/StatefulPartitionedCall"^conv2d_30/StatefulPartitionedCall"^conv2d_31/StatefulPartitionedCall"^conv2d_32/StatefulPartitionedCall"^conv2d_33/StatefulPartitionedCall"^conv2d_34/StatefulPartitionedCall"^conv2d_35/StatefulPartitionedCall"^conv2d_36/StatefulPartitionedCall"^conv2d_37/StatefulPartitionedCall"^conv2d_38/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapesy
w:?????????: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2`
.batch_normalization_12/StatefulPartitionedCall.batch_normalization_12/StatefulPartitionedCall2`
.batch_normalization_13/StatefulPartitionedCall.batch_normalization_13/StatefulPartitionedCall2`
.batch_normalization_14/StatefulPartitionedCall.batch_normalization_14/StatefulPartitionedCall2`
.batch_normalization_15/StatefulPartitionedCall.batch_normalization_15/StatefulPartitionedCall2`
.batch_normalization_16/StatefulPartitionedCall.batch_normalization_16/StatefulPartitionedCall2`
.batch_normalization_17/StatefulPartitionedCall.batch_normalization_17/StatefulPartitionedCall2F
!conv2d_28/StatefulPartitionedCall!conv2d_28/StatefulPartitionedCall2F
!conv2d_29/StatefulPartitionedCall!conv2d_29/StatefulPartitionedCall2F
!conv2d_30/StatefulPartitionedCall!conv2d_30/StatefulPartitionedCall2F
!conv2d_31/StatefulPartitionedCall!conv2d_31/StatefulPartitionedCall2F
!conv2d_32/StatefulPartitionedCall!conv2d_32/StatefulPartitionedCall2F
!conv2d_33/StatefulPartitionedCall!conv2d_33/StatefulPartitionedCall2F
!conv2d_34/StatefulPartitionedCall!conv2d_34/StatefulPartitionedCall2F
!conv2d_35/StatefulPartitionedCall!conv2d_35/StatefulPartitionedCall2F
!conv2d_36/StatefulPartitionedCall!conv2d_36/StatefulPartitionedCall2F
!conv2d_37/StatefulPartitionedCall!conv2d_37/StatefulPartitionedCall2F
!conv2d_38/StatefulPartitionedCall!conv2d_38/StatefulPartitionedCall:X T
/
_output_shapes
:?????????
!
_user_specified_name	input_5
?
?
,__inference_conv2d_35_layer_call_fn_22008308

inputs#
unknown:??
	unknown_0:	?
identity??StatefulPartitionedCall?
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
GPU 2J 8? *P
fKRI
G__inference_conv2d_35_layer_call_and_return_conditional_losses_220034022
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
d
H__inference_dropout_51_layer_call_and_return_conditional_losses_22005329

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
?
f
H__inference_dropout_35_layer_call_and_return_conditional_losses_22003508

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
?	
?
9__inference_batch_normalization_12_layer_call_fn_22007525

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
GPU 2J 8? *]
fXRV
T__inference_batch_normalization_12_layer_call_and_return_conditional_losses_220031042
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
?
d
H__inference_dropout_36_layer_call_and_return_conditional_losses_22003676

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
T__inference_batch_normalization_15_layer_call_and_return_conditional_losses_22008262

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
?
?
T__inference_batch_normalization_12_layer_call_and_return_conditional_losses_22002374

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
?
?
E__inference_dense_3_layer_call_and_return_conditional_losses_22007447

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
?
?
,__inference_conv2d_38_layer_call_fn_22008576

inputs#
unknown:??
	unknown_0:	?
identity??StatefulPartitionedCall?
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
GPU 2J 8? *P
fKRI
G__inference_conv2d_38_layer_call_and_return_conditional_losses_220035202
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
?
?
,__inference_conv2d_37_layer_call_fn_22008528

inputs#
unknown:??
	unknown_0:	?
identity??StatefulPartitionedCall?
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
GPU 2J 8? *P
fKRI
G__inference_conv2d_37_layer_call_and_return_conditional_losses_220034902
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
?
?
,__inference_conv2d_28_layer_call_fn_22007476

inputs"
unknown:?
	unknown_0:	?
identity??StatefulPartitionedCall?
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
GPU 2J 8? *P
fKRI
G__inference_conv2d_28_layer_call_and_return_conditional_losses_220030812
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
?
f
H__inference_dropout_36_layer_call_and_return_conditional_losses_22008735

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
f
H__inference_dropout_32_layer_call_and_return_conditional_losses_22003390

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
f
H__inference_dropout_27_layer_call_and_return_conditional_losses_22003156

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
?
?
T__inference_batch_normalization_16_layer_call_and_return_conditional_losses_22003463

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
T__inference_batch_normalization_14_layer_call_and_return_conditional_losses_22004030

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
?
d
H__inference_dropout_32_layer_call_and_return_conditional_losses_22003888

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
ʺ
?
E__inference_model_4_layer_call_and_return_conditional_losses_22004508

inputs-
conv2d_28_22004371:?!
conv2d_28_22004373:	?.
batch_normalization_12_22004376:	?.
batch_normalization_12_22004378:	?.
batch_normalization_12_22004380:	?.
batch_normalization_12_22004382:	?.
conv2d_29_22004387:??!
conv2d_29_22004389:	?.
conv2d_30_22004394:??!
conv2d_30_22004396:	?.
batch_normalization_13_22004401:	?.
batch_normalization_13_22004403:	?.
batch_normalization_13_22004405:	?.
batch_normalization_13_22004407:	?.
conv2d_31_22004411:??!
conv2d_31_22004413:	?.
conv2d_32_22004418:??!
conv2d_32_22004420:	?.
batch_normalization_14_22004425:	?.
batch_normalization_14_22004427:	?.
batch_normalization_14_22004429:	?.
batch_normalization_14_22004431:	?.
conv2d_33_22004435:??!
conv2d_33_22004437:	?.
conv2d_34_22004442:??!
conv2d_34_22004444:	?.
batch_normalization_15_22004449:	?.
batch_normalization_15_22004451:	?.
batch_normalization_15_22004453:	?.
batch_normalization_15_22004455:	?.
conv2d_35_22004459:??!
conv2d_35_22004461:	?.
conv2d_36_22004466:??!
conv2d_36_22004468:	?.
batch_normalization_16_22004473:	?.
batch_normalization_16_22004475:	?.
batch_normalization_16_22004477:	?.
batch_normalization_16_22004479:	?.
conv2d_37_22004483:??!
conv2d_37_22004485:	?.
conv2d_38_22004490:??!
conv2d_38_22004492:	?.
batch_normalization_17_22004497:	?.
batch_normalization_17_22004499:	?.
batch_normalization_17_22004501:	?.
batch_normalization_17_22004503:	?
identity??.batch_normalization_12/StatefulPartitionedCall?.batch_normalization_13/StatefulPartitionedCall?.batch_normalization_14/StatefulPartitionedCall?.batch_normalization_15/StatefulPartitionedCall?.batch_normalization_16/StatefulPartitionedCall?.batch_normalization_17/StatefulPartitionedCall?!conv2d_28/StatefulPartitionedCall?!conv2d_29/StatefulPartitionedCall?!conv2d_30/StatefulPartitionedCall?!conv2d_31/StatefulPartitionedCall?!conv2d_32/StatefulPartitionedCall?!conv2d_33/StatefulPartitionedCall?!conv2d_34/StatefulPartitionedCall?!conv2d_35/StatefulPartitionedCall?!conv2d_36/StatefulPartitionedCall?!conv2d_37/StatefulPartitionedCall?!conv2d_38/StatefulPartitionedCall?
!conv2d_28/StatefulPartitionedCallStatefulPartitionedCallinputsconv2d_28_22004371conv2d_28_22004373*
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
GPU 2J 8? *P
fKRI
G__inference_conv2d_28_layer_call_and_return_conditional_losses_220030812#
!conv2d_28/StatefulPartitionedCall?
.batch_normalization_12/StatefulPartitionedCallStatefulPartitionedCall*conv2d_28/StatefulPartitionedCall:output:0batch_normalization_12_22004376batch_normalization_12_22004378batch_normalization_12_22004380batch_normalization_12_22004382*
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
GPU 2J 8? *]
fXRV
T__inference_batch_normalization_12_layer_call_and_return_conditional_losses_2200424820
.batch_normalization_12/StatefulPartitionedCall?
activation_22/PartitionedCallPartitionedCall7batch_normalization_12/StatefulPartitionedCall:output:0*
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
GPU 2J 8? *T
fORM
K__inference_activation_22_layer_call_and_return_conditional_losses_220031192
activation_22/PartitionedCall?
dropout_26/PartitionedCallPartitionedCall&activation_22/PartitionedCall:output:0*
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
GPU 2J 8? *Q
fLRJ
H__inference_dropout_26_layer_call_and_return_conditional_losses_220042062
dropout_26/PartitionedCall?
!conv2d_29/StatefulPartitionedCallStatefulPartitionedCall#dropout_26/PartitionedCall:output:0conv2d_29_22004387conv2d_29_22004389*
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
GPU 2J 8? *P
fKRI
G__inference_conv2d_29_layer_call_and_return_conditional_losses_220031382#
!conv2d_29/StatefulPartitionedCall?
activation_23/PartitionedCallPartitionedCall*conv2d_29/StatefulPartitionedCall:output:0*
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
GPU 2J 8? *T
fORM
K__inference_activation_23_layer_call_and_return_conditional_losses_220031492
activation_23/PartitionedCall?
dropout_27/PartitionedCallPartitionedCall&activation_23/PartitionedCall:output:0*
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
GPU 2J 8? *Q
fLRJ
H__inference_dropout_27_layer_call_and_return_conditional_losses_220041752
dropout_27/PartitionedCall?
!conv2d_30/StatefulPartitionedCallStatefulPartitionedCall#dropout_27/PartitionedCall:output:0conv2d_30_22004394conv2d_30_22004396*
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
GPU 2J 8? *P
fKRI
G__inference_conv2d_30_layer_call_and_return_conditional_losses_220031682#
!conv2d_30/StatefulPartitionedCall?
tf.__operators__.add_10/AddV2AddV2*conv2d_30/StatefulPartitionedCall:output:0#dropout_26/PartitionedCall:output:0*
T0*0
_output_shapes
:??????????2
tf.__operators__.add_10/AddV2?
activation_24/PartitionedCallPartitionedCall!tf.__operators__.add_10/AddV2:z:0*
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
GPU 2J 8? *T
fORM
K__inference_activation_24_layer_call_and_return_conditional_losses_220031802
activation_24/PartitionedCall?
.batch_normalization_13/StatefulPartitionedCallStatefulPartitionedCall&activation_24/PartitionedCall:output:0batch_normalization_13_22004401batch_normalization_13_22004403batch_normalization_13_22004405batch_normalization_13_22004407*
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
GPU 2J 8? *]
fXRV
T__inference_batch_normalization_13_layer_call_and_return_conditional_losses_2200413620
.batch_normalization_13/StatefulPartitionedCall?
dropout_28/PartitionedCallPartitionedCall7batch_normalization_13/StatefulPartitionedCall:output:0*
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
GPU 2J 8? *Q
fLRJ
H__inference_dropout_28_layer_call_and_return_conditional_losses_220041002
dropout_28/PartitionedCall?
!conv2d_31/StatefulPartitionedCallStatefulPartitionedCall#dropout_28/PartitionedCall:output:0conv2d_31_22004411conv2d_31_22004413*
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
GPU 2J 8? *P
fKRI
G__inference_conv2d_31_layer_call_and_return_conditional_losses_220032262#
!conv2d_31/StatefulPartitionedCall?
activation_25/PartitionedCallPartitionedCall*conv2d_31/StatefulPartitionedCall:output:0*
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
GPU 2J 8? *T
fORM
K__inference_activation_25_layer_call_and_return_conditional_losses_220032372
activation_25/PartitionedCall?
dropout_29/PartitionedCallPartitionedCall&activation_25/PartitionedCall:output:0*
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
GPU 2J 8? *Q
fLRJ
H__inference_dropout_29_layer_call_and_return_conditional_losses_220040692
dropout_29/PartitionedCall?
!conv2d_32/StatefulPartitionedCallStatefulPartitionedCall#dropout_29/PartitionedCall:output:0conv2d_32_22004418conv2d_32_22004420*
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
GPU 2J 8? *P
fKRI
G__inference_conv2d_32_layer_call_and_return_conditional_losses_220032562#
!conv2d_32/StatefulPartitionedCall?
tf.__operators__.add_11/AddV2AddV2*conv2d_32/StatefulPartitionedCall:output:0#dropout_28/PartitionedCall:output:0*
T0*0
_output_shapes
:??????????2
tf.__operators__.add_11/AddV2?
activation_26/PartitionedCallPartitionedCall!tf.__operators__.add_11/AddV2:z:0*
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
GPU 2J 8? *T
fORM
K__inference_activation_26_layer_call_and_return_conditional_losses_220032682
activation_26/PartitionedCall?
.batch_normalization_14/StatefulPartitionedCallStatefulPartitionedCall&activation_26/PartitionedCall:output:0batch_normalization_14_22004425batch_normalization_14_22004427batch_normalization_14_22004429batch_normalization_14_22004431*
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
GPU 2J 8? *]
fXRV
T__inference_batch_normalization_14_layer_call_and_return_conditional_losses_2200403020
.batch_normalization_14/StatefulPartitionedCall?
dropout_30/PartitionedCallPartitionedCall7batch_normalization_14/StatefulPartitionedCall:output:0*
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
GPU 2J 8? *Q
fLRJ
H__inference_dropout_30_layer_call_and_return_conditional_losses_220039942
dropout_30/PartitionedCall?
!conv2d_33/StatefulPartitionedCallStatefulPartitionedCall#dropout_30/PartitionedCall:output:0conv2d_33_22004435conv2d_33_22004437*
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
GPU 2J 8? *P
fKRI
G__inference_conv2d_33_layer_call_and_return_conditional_losses_220033142#
!conv2d_33/StatefulPartitionedCall?
activation_27/PartitionedCallPartitionedCall*conv2d_33/StatefulPartitionedCall:output:0*
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
GPU 2J 8? *T
fORM
K__inference_activation_27_layer_call_and_return_conditional_losses_220033252
activation_27/PartitionedCall?
dropout_31/PartitionedCallPartitionedCall&activation_27/PartitionedCall:output:0*
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
GPU 2J 8? *Q
fLRJ
H__inference_dropout_31_layer_call_and_return_conditional_losses_220039632
dropout_31/PartitionedCall?
!conv2d_34/StatefulPartitionedCallStatefulPartitionedCall#dropout_31/PartitionedCall:output:0conv2d_34_22004442conv2d_34_22004444*
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
GPU 2J 8? *P
fKRI
G__inference_conv2d_34_layer_call_and_return_conditional_losses_220033442#
!conv2d_34/StatefulPartitionedCall?
tf.__operators__.add_12/AddV2AddV2*conv2d_34/StatefulPartitionedCall:output:0#dropout_30/PartitionedCall:output:0*
T0*0
_output_shapes
:??????????2
tf.__operators__.add_12/AddV2?
activation_28/PartitionedCallPartitionedCall!tf.__operators__.add_12/AddV2:z:0*
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
GPU 2J 8? *T
fORM
K__inference_activation_28_layer_call_and_return_conditional_losses_220033562
activation_28/PartitionedCall?
.batch_normalization_15/StatefulPartitionedCallStatefulPartitionedCall&activation_28/PartitionedCall:output:0batch_normalization_15_22004449batch_normalization_15_22004451batch_normalization_15_22004453batch_normalization_15_22004455*
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
GPU 2J 8? *]
fXRV
T__inference_batch_normalization_15_layer_call_and_return_conditional_losses_2200392420
.batch_normalization_15/StatefulPartitionedCall?
dropout_32/PartitionedCallPartitionedCall7batch_normalization_15/StatefulPartitionedCall:output:0*
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
GPU 2J 8? *Q
fLRJ
H__inference_dropout_32_layer_call_and_return_conditional_losses_220038882
dropout_32/PartitionedCall?
!conv2d_35/StatefulPartitionedCallStatefulPartitionedCall#dropout_32/PartitionedCall:output:0conv2d_35_22004459conv2d_35_22004461*
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
GPU 2J 8? *P
fKRI
G__inference_conv2d_35_layer_call_and_return_conditional_losses_220034022#
!conv2d_35/StatefulPartitionedCall?
activation_29/PartitionedCallPartitionedCall*conv2d_35/StatefulPartitionedCall:output:0*
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
GPU 2J 8? *T
fORM
K__inference_activation_29_layer_call_and_return_conditional_losses_220034132
activation_29/PartitionedCall?
dropout_33/PartitionedCallPartitionedCall&activation_29/PartitionedCall:output:0*
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
GPU 2J 8? *Q
fLRJ
H__inference_dropout_33_layer_call_and_return_conditional_losses_220038572
dropout_33/PartitionedCall?
!conv2d_36/StatefulPartitionedCallStatefulPartitionedCall#dropout_33/PartitionedCall:output:0conv2d_36_22004466conv2d_36_22004468*
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
GPU 2J 8? *P
fKRI
G__inference_conv2d_36_layer_call_and_return_conditional_losses_220034322#
!conv2d_36/StatefulPartitionedCall?
tf.__operators__.add_13/AddV2AddV2*conv2d_36/StatefulPartitionedCall:output:0#dropout_32/PartitionedCall:output:0*
T0*0
_output_shapes
:??????????2
tf.__operators__.add_13/AddV2?
activation_30/PartitionedCallPartitionedCall!tf.__operators__.add_13/AddV2:z:0*
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
GPU 2J 8? *T
fORM
K__inference_activation_30_layer_call_and_return_conditional_losses_220034442
activation_30/PartitionedCall?
.batch_normalization_16/StatefulPartitionedCallStatefulPartitionedCall&activation_30/PartitionedCall:output:0batch_normalization_16_22004473batch_normalization_16_22004475batch_normalization_16_22004477batch_normalization_16_22004479*
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
GPU 2J 8? *]
fXRV
T__inference_batch_normalization_16_layer_call_and_return_conditional_losses_2200381820
.batch_normalization_16/StatefulPartitionedCall?
dropout_34/PartitionedCallPartitionedCall7batch_normalization_16/StatefulPartitionedCall:output:0*
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
GPU 2J 8? *Q
fLRJ
H__inference_dropout_34_layer_call_and_return_conditional_losses_220037822
dropout_34/PartitionedCall?
!conv2d_37/StatefulPartitionedCallStatefulPartitionedCall#dropout_34/PartitionedCall:output:0conv2d_37_22004483conv2d_37_22004485*
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
GPU 2J 8? *P
fKRI
G__inference_conv2d_37_layer_call_and_return_conditional_losses_220034902#
!conv2d_37/StatefulPartitionedCall?
activation_31/PartitionedCallPartitionedCall*conv2d_37/StatefulPartitionedCall:output:0*
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
GPU 2J 8? *T
fORM
K__inference_activation_31_layer_call_and_return_conditional_losses_220035012
activation_31/PartitionedCall?
dropout_35/PartitionedCallPartitionedCall&activation_31/PartitionedCall:output:0*
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
GPU 2J 8? *Q
fLRJ
H__inference_dropout_35_layer_call_and_return_conditional_losses_220037512
dropout_35/PartitionedCall?
!conv2d_38/StatefulPartitionedCallStatefulPartitionedCall#dropout_35/PartitionedCall:output:0conv2d_38_22004490conv2d_38_22004492*
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
GPU 2J 8? *P
fKRI
G__inference_conv2d_38_layer_call_and_return_conditional_losses_220035202#
!conv2d_38/StatefulPartitionedCall?
tf.__operators__.add_14/AddV2AddV2*conv2d_38/StatefulPartitionedCall:output:0#dropout_34/PartitionedCall:output:0*
T0*0
_output_shapes
:??????????2
tf.__operators__.add_14/AddV2?
activation_32/PartitionedCallPartitionedCall!tf.__operators__.add_14/AddV2:z:0*
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
GPU 2J 8? *T
fORM
K__inference_activation_32_layer_call_and_return_conditional_losses_220035322
activation_32/PartitionedCall?
.batch_normalization_17/StatefulPartitionedCallStatefulPartitionedCall&activation_32/PartitionedCall:output:0batch_normalization_17_22004497batch_normalization_17_22004499batch_normalization_17_22004501batch_normalization_17_22004503*
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
GPU 2J 8? *]
fXRV
T__inference_batch_normalization_17_layer_call_and_return_conditional_losses_2200371220
.batch_normalization_17/StatefulPartitionedCall?
dropout_36/PartitionedCallPartitionedCall7batch_normalization_17/StatefulPartitionedCall:output:0*
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
GPU 2J 8? *Q
fLRJ
H__inference_dropout_36_layer_call_and_return_conditional_losses_220036762
dropout_36/PartitionedCall?
IdentityIdentity#dropout_36/PartitionedCall:output:0^NoOp*
T0*0
_output_shapes
:??????????2

Identity?
NoOpNoOp/^batch_normalization_12/StatefulPartitionedCall/^batch_normalization_13/StatefulPartitionedCall/^batch_normalization_14/StatefulPartitionedCall/^batch_normalization_15/StatefulPartitionedCall/^batch_normalization_16/StatefulPartitionedCall/^batch_normalization_17/StatefulPartitionedCall"^conv2d_28/StatefulPartitionedCall"^conv2d_29/StatefulPartitionedCall"^conv2d_30/StatefulPartitionedCall"^conv2d_31/StatefulPartitionedCall"^conv2d_32/StatefulPartitionedCall"^conv2d_33/StatefulPartitionedCall"^conv2d_34/StatefulPartitionedCall"^conv2d_35/StatefulPartitionedCall"^conv2d_36/StatefulPartitionedCall"^conv2d_37/StatefulPartitionedCall"^conv2d_38/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapesy
w:?????????: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2`
.batch_normalization_12/StatefulPartitionedCall.batch_normalization_12/StatefulPartitionedCall2`
.batch_normalization_13/StatefulPartitionedCall.batch_normalization_13/StatefulPartitionedCall2`
.batch_normalization_14/StatefulPartitionedCall.batch_normalization_14/StatefulPartitionedCall2`
.batch_normalization_15/StatefulPartitionedCall.batch_normalization_15/StatefulPartitionedCall2`
.batch_normalization_16/StatefulPartitionedCall.batch_normalization_16/StatefulPartitionedCall2`
.batch_normalization_17/StatefulPartitionedCall.batch_normalization_17/StatefulPartitionedCall2F
!conv2d_28/StatefulPartitionedCall!conv2d_28/StatefulPartitionedCall2F
!conv2d_29/StatefulPartitionedCall!conv2d_29/StatefulPartitionedCall2F
!conv2d_30/StatefulPartitionedCall!conv2d_30/StatefulPartitionedCall2F
!conv2d_31/StatefulPartitionedCall!conv2d_31/StatefulPartitionedCall2F
!conv2d_32/StatefulPartitionedCall!conv2d_32/StatefulPartitionedCall2F
!conv2d_33/StatefulPartitionedCall!conv2d_33/StatefulPartitionedCall2F
!conv2d_34/StatefulPartitionedCall!conv2d_34/StatefulPartitionedCall2F
!conv2d_35/StatefulPartitionedCall!conv2d_35/StatefulPartitionedCall2F
!conv2d_36/StatefulPartitionedCall!conv2d_36/StatefulPartitionedCall2F
!conv2d_37/StatefulPartitionedCall!conv2d_37/StatefulPartitionedCall2F
!conv2d_38/StatefulPartitionedCall!conv2d_38/StatefulPartitionedCall:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
T__inference_batch_normalization_13_layer_call_and_return_conditional_losses_22007822

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
?
G__inference_conv2d_54_layer_call_and_return_conditional_losses_22005147

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
?
?
T__inference_batch_normalization_14_layer_call_and_return_conditional_losses_22003287

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
d
H__inference_dropout_27_layer_call_and_return_conditional_losses_22007687

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
I
-__inference_dropout_26_layer_call_fn_22007625

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
GPU 2J 8? *Q
fLRJ
H__inference_dropout_26_layer_call_and_return_conditional_losses_220031262
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
?
d
H__inference_dropout_28_layer_call_and_return_conditional_losses_22004100

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
T__inference_batch_normalization_17_layer_call_and_return_conditional_losses_22002960

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
I
-__inference_dropout_33_layer_call_fn_22008333

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
GPU 2J 8? *Q
fLRJ
H__inference_dropout_33_layer_call_and_return_conditional_losses_220034202
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
?<
?
E__inference_model_7_layer_call_and_return_conditional_losses_22005167

inputs+
model_4_22004987:?
model_4_22004989:	?
model_4_22004991:	?
model_4_22004993:	?
model_4_22004995:	?
model_4_22004997:	?,
model_4_22004999:??
model_4_22005001:	?,
model_4_22005003:??
model_4_22005005:	?
model_4_22005007:	?
model_4_22005009:	?
model_4_22005011:	?
model_4_22005013:	?,
model_4_22005015:??
model_4_22005017:	?,
model_4_22005019:??
model_4_22005021:	?
model_4_22005023:	?
model_4_22005025:	?
model_4_22005027:	?
model_4_22005029:	?,
model_4_22005031:??
model_4_22005033:	?,
model_4_22005035:??
model_4_22005037:	?
model_4_22005039:	?
model_4_22005041:	?
model_4_22005043:	?
model_4_22005045:	?,
model_4_22005047:??
model_4_22005049:	?,
model_4_22005051:??
model_4_22005053:	?
model_4_22005055:	?
model_4_22005057:	?
model_4_22005059:	?
model_4_22005061:	?,
model_4_22005063:??
model_4_22005065:	?,
model_4_22005067:??
model_4_22005069:	?
model_4_22005071:	?
model_4_22005073:	?
model_4_22005075:	?
model_4_22005077:	?-
conv2d_55_22005092:? 
conv2d_55_22005094:.
conv2d_53_22005109:??!
conv2d_53_22005111:	?-
conv2d_54_22005148:?I 
conv2d_54_22005150:I"
dense_3_22005162:@
identity

identity_1??!conv2d_53/StatefulPartitionedCall?!conv2d_54/StatefulPartitionedCall?!conv2d_55/StatefulPartitionedCall?dense_3/StatefulPartitionedCall?model_4/StatefulPartitionedCall?

model_4/StatefulPartitionedCallStatefulPartitionedCallinputsmodel_4_22004987model_4_22004989model_4_22004991model_4_22004993model_4_22004995model_4_22004997model_4_22004999model_4_22005001model_4_22005003model_4_22005005model_4_22005007model_4_22005009model_4_22005011model_4_22005013model_4_22005015model_4_22005017model_4_22005019model_4_22005021model_4_22005023model_4_22005025model_4_22005027model_4_22005029model_4_22005031model_4_22005033model_4_22005035model_4_22005037model_4_22005039model_4_22005041model_4_22005043model_4_22005045model_4_22005047model_4_22005049model_4_22005051model_4_22005053model_4_22005055model_4_22005057model_4_22005059model_4_22005061model_4_22005063model_4_22005065model_4_22005067model_4_22005069model_4_22005071model_4_22005073model_4_22005075model_4_22005077*:
Tin3
12/*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????*P
_read_only_resource_inputs2
0.	
 !"#$%&'()*+,-.*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_model_4_layer_call_and_return_conditional_losses_220035692!
model_4/StatefulPartitionedCall?
!conv2d_55/StatefulPartitionedCallStatefulPartitionedCall(model_4/StatefulPartitionedCall:output:0conv2d_55_22005092conv2d_55_22005094*
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
GPU 2J 8? *P
fKRI
G__inference_conv2d_55_layer_call_and_return_conditional_losses_220050912#
!conv2d_55/StatefulPartitionedCall?
!conv2d_53/StatefulPartitionedCallStatefulPartitionedCall(model_4/StatefulPartitionedCall:output:0conv2d_53_22005109conv2d_53_22005111*
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
GPU 2J 8? *P
fKRI
G__inference_conv2d_53_layer_call_and_return_conditional_losses_220051082#
!conv2d_53/StatefulPartitionedCall?
dropout_51/PartitionedCallPartitionedCall*conv2d_55/StatefulPartitionedCall:output:0*
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
GPU 2J 8? *Q
fLRJ
H__inference_dropout_51_layer_call_and_return_conditional_losses_220051192
dropout_51/PartitionedCall?
dropout_50/PartitionedCallPartitionedCall*conv2d_53/StatefulPartitionedCall:output:0*
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
GPU 2J 8? *Q
fLRJ
H__inference_dropout_50_layer_call_and_return_conditional_losses_220051262
dropout_50/PartitionedCall?
flatten_3/PartitionedCallPartitionedCall#dropout_51/PartitionedCall:output:0*
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
GPU 2J 8? *P
fKRI
G__inference_flatten_3_layer_call_and_return_conditional_losses_220051342
flatten_3/PartitionedCall?
!conv2d_54/StatefulPartitionedCallStatefulPartitionedCall#dropout_50/PartitionedCall:output:0conv2d_54_22005148conv2d_54_22005150*
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
GPU 2J 8? *P
fKRI
G__inference_conv2d_54_layer_call_and_return_conditional_losses_220051472#
!conv2d_54/StatefulPartitionedCall?
dense_3/StatefulPartitionedCallStatefulPartitionedCall"flatten_3/PartitionedCall:output:0dense_3_22005162*
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
GPU 2J 8? *N
fIRG
E__inference_dense_3_layer_call_and_return_conditional_losses_220051612!
dense_3/StatefulPartitionedCall?
IdentityIdentity(dense_3/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????2

Identity?

Identity_1Identity*conv2d_54/StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:?????????I2

Identity_1?
NoOpNoOp"^conv2d_53/StatefulPartitionedCall"^conv2d_54/StatefulPartitionedCall"^conv2d_55/StatefulPartitionedCall ^dense_3/StatefulPartitionedCall ^model_4/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes?
?:?????????: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2F
!conv2d_53/StatefulPartitionedCall!conv2d_53/StatefulPartitionedCall2F
!conv2d_54/StatefulPartitionedCall!conv2d_54/StatefulPartitionedCall2F
!conv2d_55/StatefulPartitionedCall!conv2d_55/StatefulPartitionedCall2B
dense_3/StatefulPartitionedCalldense_3/StatefulPartitionedCall2B
model_4/StatefulPartitionedCallmodel_4/StatefulPartitionedCall:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
,__inference_conv2d_30_layer_call_fn_22007696

inputs#
unknown:??
	unknown_0:	?
identity??StatefulPartitionedCall?
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
GPU 2J 8? *P
fKRI
G__inference_conv2d_30_layer_call_and_return_conditional_losses_220031682
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
f
H__inference_dropout_26_layer_call_and_return_conditional_losses_22007635

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
?
I
-__inference_dropout_31_layer_call_fn_22008113

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
GPU 2J 8? *Q
fLRJ
H__inference_dropout_31_layer_call_and_return_conditional_losses_220033322
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
9__inference_batch_normalization_15_layer_call_fn_22008182

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
GPU 2J 8? *]
fXRV
T__inference_batch_normalization_15_layer_call_and_return_conditional_losses_220027522
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
g
K__inference_activation_27_layer_call_and_return_conditional_losses_22008108

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
,__inference_conv2d_54_layer_call_fn_22007456

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
GPU 2J 8? *P
fKRI
G__inference_conv2d_54_layer_call_and_return_conditional_losses_220051472
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
?
g
K__inference_activation_25_layer_call_and_return_conditional_losses_22007888

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
?
?
T__inference_batch_normalization_14_layer_call_and_return_conditional_losses_22008060

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
g
K__inference_activation_26_layer_call_and_return_conditional_losses_22003268

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
??
?3
E__inference_model_7_layer_call_and_return_conditional_losses_22006600

inputsK
0model_4_conv2d_28_conv2d_readvariableop_resource:?@
1model_4_conv2d_28_biasadd_readvariableop_resource:	?E
6model_4_batch_normalization_12_readvariableop_resource:	?G
8model_4_batch_normalization_12_readvariableop_1_resource:	?V
Gmodel_4_batch_normalization_12_fusedbatchnormv3_readvariableop_resource:	?X
Imodel_4_batch_normalization_12_fusedbatchnormv3_readvariableop_1_resource:	?L
0model_4_conv2d_29_conv2d_readvariableop_resource:??@
1model_4_conv2d_29_biasadd_readvariableop_resource:	?L
0model_4_conv2d_30_conv2d_readvariableop_resource:??@
1model_4_conv2d_30_biasadd_readvariableop_resource:	?E
6model_4_batch_normalization_13_readvariableop_resource:	?G
8model_4_batch_normalization_13_readvariableop_1_resource:	?V
Gmodel_4_batch_normalization_13_fusedbatchnormv3_readvariableop_resource:	?X
Imodel_4_batch_normalization_13_fusedbatchnormv3_readvariableop_1_resource:	?L
0model_4_conv2d_31_conv2d_readvariableop_resource:??@
1model_4_conv2d_31_biasadd_readvariableop_resource:	?L
0model_4_conv2d_32_conv2d_readvariableop_resource:??@
1model_4_conv2d_32_biasadd_readvariableop_resource:	?E
6model_4_batch_normalization_14_readvariableop_resource:	?G
8model_4_batch_normalization_14_readvariableop_1_resource:	?V
Gmodel_4_batch_normalization_14_fusedbatchnormv3_readvariableop_resource:	?X
Imodel_4_batch_normalization_14_fusedbatchnormv3_readvariableop_1_resource:	?L
0model_4_conv2d_33_conv2d_readvariableop_resource:??@
1model_4_conv2d_33_biasadd_readvariableop_resource:	?L
0model_4_conv2d_34_conv2d_readvariableop_resource:??@
1model_4_conv2d_34_biasadd_readvariableop_resource:	?E
6model_4_batch_normalization_15_readvariableop_resource:	?G
8model_4_batch_normalization_15_readvariableop_1_resource:	?V
Gmodel_4_batch_normalization_15_fusedbatchnormv3_readvariableop_resource:	?X
Imodel_4_batch_normalization_15_fusedbatchnormv3_readvariableop_1_resource:	?L
0model_4_conv2d_35_conv2d_readvariableop_resource:??@
1model_4_conv2d_35_biasadd_readvariableop_resource:	?L
0model_4_conv2d_36_conv2d_readvariableop_resource:??@
1model_4_conv2d_36_biasadd_readvariableop_resource:	?E
6model_4_batch_normalization_16_readvariableop_resource:	?G
8model_4_batch_normalization_16_readvariableop_1_resource:	?V
Gmodel_4_batch_normalization_16_fusedbatchnormv3_readvariableop_resource:	?X
Imodel_4_batch_normalization_16_fusedbatchnormv3_readvariableop_1_resource:	?L
0model_4_conv2d_37_conv2d_readvariableop_resource:??@
1model_4_conv2d_37_biasadd_readvariableop_resource:	?L
0model_4_conv2d_38_conv2d_readvariableop_resource:??@
1model_4_conv2d_38_biasadd_readvariableop_resource:	?E
6model_4_batch_normalization_17_readvariableop_resource:	?G
8model_4_batch_normalization_17_readvariableop_1_resource:	?V
Gmodel_4_batch_normalization_17_fusedbatchnormv3_readvariableop_resource:	?X
Imodel_4_batch_normalization_17_fusedbatchnormv3_readvariableop_1_resource:	?C
(conv2d_55_conv2d_readvariableop_resource:?7
)conv2d_55_biasadd_readvariableop_resource:D
(conv2d_53_conv2d_readvariableop_resource:??8
)conv2d_53_biasadd_readvariableop_resource:	?C
(conv2d_54_conv2d_readvariableop_resource:?I7
)conv2d_54_biasadd_readvariableop_resource:I8
&dense_3_matmul_readvariableop_resource:@
identity

identity_1?? conv2d_53/BiasAdd/ReadVariableOp?conv2d_53/Conv2D/ReadVariableOp? conv2d_54/BiasAdd/ReadVariableOp?conv2d_54/Conv2D/ReadVariableOp? conv2d_55/BiasAdd/ReadVariableOp?conv2d_55/Conv2D/ReadVariableOp?dense_3/MatMul/ReadVariableOp?>model_4/batch_normalization_12/FusedBatchNormV3/ReadVariableOp?@model_4/batch_normalization_12/FusedBatchNormV3/ReadVariableOp_1?-model_4/batch_normalization_12/ReadVariableOp?/model_4/batch_normalization_12/ReadVariableOp_1?>model_4/batch_normalization_13/FusedBatchNormV3/ReadVariableOp?@model_4/batch_normalization_13/FusedBatchNormV3/ReadVariableOp_1?-model_4/batch_normalization_13/ReadVariableOp?/model_4/batch_normalization_13/ReadVariableOp_1?>model_4/batch_normalization_14/FusedBatchNormV3/ReadVariableOp?@model_4/batch_normalization_14/FusedBatchNormV3/ReadVariableOp_1?-model_4/batch_normalization_14/ReadVariableOp?/model_4/batch_normalization_14/ReadVariableOp_1?>model_4/batch_normalization_15/FusedBatchNormV3/ReadVariableOp?@model_4/batch_normalization_15/FusedBatchNormV3/ReadVariableOp_1?-model_4/batch_normalization_15/ReadVariableOp?/model_4/batch_normalization_15/ReadVariableOp_1?>model_4/batch_normalization_16/FusedBatchNormV3/ReadVariableOp?@model_4/batch_normalization_16/FusedBatchNormV3/ReadVariableOp_1?-model_4/batch_normalization_16/ReadVariableOp?/model_4/batch_normalization_16/ReadVariableOp_1?>model_4/batch_normalization_17/FusedBatchNormV3/ReadVariableOp?@model_4/batch_normalization_17/FusedBatchNormV3/ReadVariableOp_1?-model_4/batch_normalization_17/ReadVariableOp?/model_4/batch_normalization_17/ReadVariableOp_1?(model_4/conv2d_28/BiasAdd/ReadVariableOp?'model_4/conv2d_28/Conv2D/ReadVariableOp?(model_4/conv2d_29/BiasAdd/ReadVariableOp?'model_4/conv2d_29/Conv2D/ReadVariableOp?(model_4/conv2d_30/BiasAdd/ReadVariableOp?'model_4/conv2d_30/Conv2D/ReadVariableOp?(model_4/conv2d_31/BiasAdd/ReadVariableOp?'model_4/conv2d_31/Conv2D/ReadVariableOp?(model_4/conv2d_32/BiasAdd/ReadVariableOp?'model_4/conv2d_32/Conv2D/ReadVariableOp?(model_4/conv2d_33/BiasAdd/ReadVariableOp?'model_4/conv2d_33/Conv2D/ReadVariableOp?(model_4/conv2d_34/BiasAdd/ReadVariableOp?'model_4/conv2d_34/Conv2D/ReadVariableOp?(model_4/conv2d_35/BiasAdd/ReadVariableOp?'model_4/conv2d_35/Conv2D/ReadVariableOp?(model_4/conv2d_36/BiasAdd/ReadVariableOp?'model_4/conv2d_36/Conv2D/ReadVariableOp?(model_4/conv2d_37/BiasAdd/ReadVariableOp?'model_4/conv2d_37/Conv2D/ReadVariableOp?(model_4/conv2d_38/BiasAdd/ReadVariableOp?'model_4/conv2d_38/Conv2D/ReadVariableOp?
'model_4/conv2d_28/Conv2D/ReadVariableOpReadVariableOp0model_4_conv2d_28_conv2d_readvariableop_resource*'
_output_shapes
:?*
dtype02)
'model_4/conv2d_28/Conv2D/ReadVariableOp?
model_4/conv2d_28/Conv2DConv2Dinputs/model_4/conv2d_28/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????*
paddingSAME*
strides
2
model_4/conv2d_28/Conv2D?
(model_4/conv2d_28/BiasAdd/ReadVariableOpReadVariableOp1model_4_conv2d_28_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02*
(model_4/conv2d_28/BiasAdd/ReadVariableOp?
model_4/conv2d_28/BiasAddBiasAdd!model_4/conv2d_28/Conv2D:output:00model_4/conv2d_28/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????2
model_4/conv2d_28/BiasAdd?
-model_4/batch_normalization_12/ReadVariableOpReadVariableOp6model_4_batch_normalization_12_readvariableop_resource*
_output_shapes	
:?*
dtype02/
-model_4/batch_normalization_12/ReadVariableOp?
/model_4/batch_normalization_12/ReadVariableOp_1ReadVariableOp8model_4_batch_normalization_12_readvariableop_1_resource*
_output_shapes	
:?*
dtype021
/model_4/batch_normalization_12/ReadVariableOp_1?
>model_4/batch_normalization_12/FusedBatchNormV3/ReadVariableOpReadVariableOpGmodel_4_batch_normalization_12_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:?*
dtype02@
>model_4/batch_normalization_12/FusedBatchNormV3/ReadVariableOp?
@model_4/batch_normalization_12/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpImodel_4_batch_normalization_12_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:?*
dtype02B
@model_4/batch_normalization_12/FusedBatchNormV3/ReadVariableOp_1?
/model_4/batch_normalization_12/FusedBatchNormV3FusedBatchNormV3"model_4/conv2d_28/BiasAdd:output:05model_4/batch_normalization_12/ReadVariableOp:value:07model_4/batch_normalization_12/ReadVariableOp_1:value:0Fmodel_4/batch_normalization_12/FusedBatchNormV3/ReadVariableOp:value:0Hmodel_4/batch_normalization_12/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:??????????:?:?:?:?:*
epsilon%o?:*
is_training( 21
/model_4/batch_normalization_12/FusedBatchNormV3?
model_4/activation_22/ReluRelu3model_4/batch_normalization_12/FusedBatchNormV3:y:0*
T0*0
_output_shapes
:??????????2
model_4/activation_22/Relu?
model_4/dropout_26/IdentityIdentity(model_4/activation_22/Relu:activations:0*
T0*0
_output_shapes
:??????????2
model_4/dropout_26/Identity?
'model_4/conv2d_29/Conv2D/ReadVariableOpReadVariableOp0model_4_conv2d_29_conv2d_readvariableop_resource*(
_output_shapes
:??*
dtype02)
'model_4/conv2d_29/Conv2D/ReadVariableOp?
model_4/conv2d_29/Conv2DConv2D$model_4/dropout_26/Identity:output:0/model_4/conv2d_29/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????*
paddingSAME*
strides
2
model_4/conv2d_29/Conv2D?
(model_4/conv2d_29/BiasAdd/ReadVariableOpReadVariableOp1model_4_conv2d_29_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02*
(model_4/conv2d_29/BiasAdd/ReadVariableOp?
model_4/conv2d_29/BiasAddBiasAdd!model_4/conv2d_29/Conv2D:output:00model_4/conv2d_29/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????2
model_4/conv2d_29/BiasAdd?
model_4/activation_23/ReluRelu"model_4/conv2d_29/BiasAdd:output:0*
T0*0
_output_shapes
:??????????2
model_4/activation_23/Relu?
model_4/dropout_27/IdentityIdentity(model_4/activation_23/Relu:activations:0*
T0*0
_output_shapes
:??????????2
model_4/dropout_27/Identity?
'model_4/conv2d_30/Conv2D/ReadVariableOpReadVariableOp0model_4_conv2d_30_conv2d_readvariableop_resource*(
_output_shapes
:??*
dtype02)
'model_4/conv2d_30/Conv2D/ReadVariableOp?
model_4/conv2d_30/Conv2DConv2D$model_4/dropout_27/Identity:output:0/model_4/conv2d_30/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????*
paddingSAME*
strides
2
model_4/conv2d_30/Conv2D?
(model_4/conv2d_30/BiasAdd/ReadVariableOpReadVariableOp1model_4_conv2d_30_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02*
(model_4/conv2d_30/BiasAdd/ReadVariableOp?
model_4/conv2d_30/BiasAddBiasAdd!model_4/conv2d_30/Conv2D:output:00model_4/conv2d_30/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????2
model_4/conv2d_30/BiasAdd?
%model_4/tf.__operators__.add_10/AddV2AddV2"model_4/conv2d_30/BiasAdd:output:0$model_4/dropout_26/Identity:output:0*
T0*0
_output_shapes
:??????????2'
%model_4/tf.__operators__.add_10/AddV2?
model_4/activation_24/ReluRelu)model_4/tf.__operators__.add_10/AddV2:z:0*
T0*0
_output_shapes
:??????????2
model_4/activation_24/Relu?
-model_4/batch_normalization_13/ReadVariableOpReadVariableOp6model_4_batch_normalization_13_readvariableop_resource*
_output_shapes	
:?*
dtype02/
-model_4/batch_normalization_13/ReadVariableOp?
/model_4/batch_normalization_13/ReadVariableOp_1ReadVariableOp8model_4_batch_normalization_13_readvariableop_1_resource*
_output_shapes	
:?*
dtype021
/model_4/batch_normalization_13/ReadVariableOp_1?
>model_4/batch_normalization_13/FusedBatchNormV3/ReadVariableOpReadVariableOpGmodel_4_batch_normalization_13_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:?*
dtype02@
>model_4/batch_normalization_13/FusedBatchNormV3/ReadVariableOp?
@model_4/batch_normalization_13/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpImodel_4_batch_normalization_13_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:?*
dtype02B
@model_4/batch_normalization_13/FusedBatchNormV3/ReadVariableOp_1?
/model_4/batch_normalization_13/FusedBatchNormV3FusedBatchNormV3(model_4/activation_24/Relu:activations:05model_4/batch_normalization_13/ReadVariableOp:value:07model_4/batch_normalization_13/ReadVariableOp_1:value:0Fmodel_4/batch_normalization_13/FusedBatchNormV3/ReadVariableOp:value:0Hmodel_4/batch_normalization_13/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:??????????:?:?:?:?:*
epsilon%o?:*
is_training( 21
/model_4/batch_normalization_13/FusedBatchNormV3?
model_4/dropout_28/IdentityIdentity3model_4/batch_normalization_13/FusedBatchNormV3:y:0*
T0*0
_output_shapes
:??????????2
model_4/dropout_28/Identity?
'model_4/conv2d_31/Conv2D/ReadVariableOpReadVariableOp0model_4_conv2d_31_conv2d_readvariableop_resource*(
_output_shapes
:??*
dtype02)
'model_4/conv2d_31/Conv2D/ReadVariableOp?
model_4/conv2d_31/Conv2DConv2D$model_4/dropout_28/Identity:output:0/model_4/conv2d_31/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????*
paddingSAME*
strides
2
model_4/conv2d_31/Conv2D?
(model_4/conv2d_31/BiasAdd/ReadVariableOpReadVariableOp1model_4_conv2d_31_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02*
(model_4/conv2d_31/BiasAdd/ReadVariableOp?
model_4/conv2d_31/BiasAddBiasAdd!model_4/conv2d_31/Conv2D:output:00model_4/conv2d_31/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????2
model_4/conv2d_31/BiasAdd?
model_4/activation_25/ReluRelu"model_4/conv2d_31/BiasAdd:output:0*
T0*0
_output_shapes
:??????????2
model_4/activation_25/Relu?
model_4/dropout_29/IdentityIdentity(model_4/activation_25/Relu:activations:0*
T0*0
_output_shapes
:??????????2
model_4/dropout_29/Identity?
'model_4/conv2d_32/Conv2D/ReadVariableOpReadVariableOp0model_4_conv2d_32_conv2d_readvariableop_resource*(
_output_shapes
:??*
dtype02)
'model_4/conv2d_32/Conv2D/ReadVariableOp?
model_4/conv2d_32/Conv2DConv2D$model_4/dropout_29/Identity:output:0/model_4/conv2d_32/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????*
paddingSAME*
strides
2
model_4/conv2d_32/Conv2D?
(model_4/conv2d_32/BiasAdd/ReadVariableOpReadVariableOp1model_4_conv2d_32_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02*
(model_4/conv2d_32/BiasAdd/ReadVariableOp?
model_4/conv2d_32/BiasAddBiasAdd!model_4/conv2d_32/Conv2D:output:00model_4/conv2d_32/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????2
model_4/conv2d_32/BiasAdd?
%model_4/tf.__operators__.add_11/AddV2AddV2"model_4/conv2d_32/BiasAdd:output:0$model_4/dropout_28/Identity:output:0*
T0*0
_output_shapes
:??????????2'
%model_4/tf.__operators__.add_11/AddV2?
model_4/activation_26/ReluRelu)model_4/tf.__operators__.add_11/AddV2:z:0*
T0*0
_output_shapes
:??????????2
model_4/activation_26/Relu?
-model_4/batch_normalization_14/ReadVariableOpReadVariableOp6model_4_batch_normalization_14_readvariableop_resource*
_output_shapes	
:?*
dtype02/
-model_4/batch_normalization_14/ReadVariableOp?
/model_4/batch_normalization_14/ReadVariableOp_1ReadVariableOp8model_4_batch_normalization_14_readvariableop_1_resource*
_output_shapes	
:?*
dtype021
/model_4/batch_normalization_14/ReadVariableOp_1?
>model_4/batch_normalization_14/FusedBatchNormV3/ReadVariableOpReadVariableOpGmodel_4_batch_normalization_14_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:?*
dtype02@
>model_4/batch_normalization_14/FusedBatchNormV3/ReadVariableOp?
@model_4/batch_normalization_14/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpImodel_4_batch_normalization_14_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:?*
dtype02B
@model_4/batch_normalization_14/FusedBatchNormV3/ReadVariableOp_1?
/model_4/batch_normalization_14/FusedBatchNormV3FusedBatchNormV3(model_4/activation_26/Relu:activations:05model_4/batch_normalization_14/ReadVariableOp:value:07model_4/batch_normalization_14/ReadVariableOp_1:value:0Fmodel_4/batch_normalization_14/FusedBatchNormV3/ReadVariableOp:value:0Hmodel_4/batch_normalization_14/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:??????????:?:?:?:?:*
epsilon%o?:*
is_training( 21
/model_4/batch_normalization_14/FusedBatchNormV3?
model_4/dropout_30/IdentityIdentity3model_4/batch_normalization_14/FusedBatchNormV3:y:0*
T0*0
_output_shapes
:??????????2
model_4/dropout_30/Identity?
'model_4/conv2d_33/Conv2D/ReadVariableOpReadVariableOp0model_4_conv2d_33_conv2d_readvariableop_resource*(
_output_shapes
:??*
dtype02)
'model_4/conv2d_33/Conv2D/ReadVariableOp?
model_4/conv2d_33/Conv2DConv2D$model_4/dropout_30/Identity:output:0/model_4/conv2d_33/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????*
paddingSAME*
strides
2
model_4/conv2d_33/Conv2D?
(model_4/conv2d_33/BiasAdd/ReadVariableOpReadVariableOp1model_4_conv2d_33_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02*
(model_4/conv2d_33/BiasAdd/ReadVariableOp?
model_4/conv2d_33/BiasAddBiasAdd!model_4/conv2d_33/Conv2D:output:00model_4/conv2d_33/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????2
model_4/conv2d_33/BiasAdd?
model_4/activation_27/ReluRelu"model_4/conv2d_33/BiasAdd:output:0*
T0*0
_output_shapes
:??????????2
model_4/activation_27/Relu?
model_4/dropout_31/IdentityIdentity(model_4/activation_27/Relu:activations:0*
T0*0
_output_shapes
:??????????2
model_4/dropout_31/Identity?
'model_4/conv2d_34/Conv2D/ReadVariableOpReadVariableOp0model_4_conv2d_34_conv2d_readvariableop_resource*(
_output_shapes
:??*
dtype02)
'model_4/conv2d_34/Conv2D/ReadVariableOp?
model_4/conv2d_34/Conv2DConv2D$model_4/dropout_31/Identity:output:0/model_4/conv2d_34/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????*
paddingSAME*
strides
2
model_4/conv2d_34/Conv2D?
(model_4/conv2d_34/BiasAdd/ReadVariableOpReadVariableOp1model_4_conv2d_34_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02*
(model_4/conv2d_34/BiasAdd/ReadVariableOp?
model_4/conv2d_34/BiasAddBiasAdd!model_4/conv2d_34/Conv2D:output:00model_4/conv2d_34/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????2
model_4/conv2d_34/BiasAdd?
%model_4/tf.__operators__.add_12/AddV2AddV2"model_4/conv2d_34/BiasAdd:output:0$model_4/dropout_30/Identity:output:0*
T0*0
_output_shapes
:??????????2'
%model_4/tf.__operators__.add_12/AddV2?
model_4/activation_28/ReluRelu)model_4/tf.__operators__.add_12/AddV2:z:0*
T0*0
_output_shapes
:??????????2
model_4/activation_28/Relu?
-model_4/batch_normalization_15/ReadVariableOpReadVariableOp6model_4_batch_normalization_15_readvariableop_resource*
_output_shapes	
:?*
dtype02/
-model_4/batch_normalization_15/ReadVariableOp?
/model_4/batch_normalization_15/ReadVariableOp_1ReadVariableOp8model_4_batch_normalization_15_readvariableop_1_resource*
_output_shapes	
:?*
dtype021
/model_4/batch_normalization_15/ReadVariableOp_1?
>model_4/batch_normalization_15/FusedBatchNormV3/ReadVariableOpReadVariableOpGmodel_4_batch_normalization_15_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:?*
dtype02@
>model_4/batch_normalization_15/FusedBatchNormV3/ReadVariableOp?
@model_4/batch_normalization_15/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpImodel_4_batch_normalization_15_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:?*
dtype02B
@model_4/batch_normalization_15/FusedBatchNormV3/ReadVariableOp_1?
/model_4/batch_normalization_15/FusedBatchNormV3FusedBatchNormV3(model_4/activation_28/Relu:activations:05model_4/batch_normalization_15/ReadVariableOp:value:07model_4/batch_normalization_15/ReadVariableOp_1:value:0Fmodel_4/batch_normalization_15/FusedBatchNormV3/ReadVariableOp:value:0Hmodel_4/batch_normalization_15/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:??????????:?:?:?:?:*
epsilon%o?:*
is_training( 21
/model_4/batch_normalization_15/FusedBatchNormV3?
model_4/dropout_32/IdentityIdentity3model_4/batch_normalization_15/FusedBatchNormV3:y:0*
T0*0
_output_shapes
:??????????2
model_4/dropout_32/Identity?
'model_4/conv2d_35/Conv2D/ReadVariableOpReadVariableOp0model_4_conv2d_35_conv2d_readvariableop_resource*(
_output_shapes
:??*
dtype02)
'model_4/conv2d_35/Conv2D/ReadVariableOp?
model_4/conv2d_35/Conv2DConv2D$model_4/dropout_32/Identity:output:0/model_4/conv2d_35/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????*
paddingSAME*
strides
2
model_4/conv2d_35/Conv2D?
(model_4/conv2d_35/BiasAdd/ReadVariableOpReadVariableOp1model_4_conv2d_35_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02*
(model_4/conv2d_35/BiasAdd/ReadVariableOp?
model_4/conv2d_35/BiasAddBiasAdd!model_4/conv2d_35/Conv2D:output:00model_4/conv2d_35/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????2
model_4/conv2d_35/BiasAdd?
model_4/activation_29/ReluRelu"model_4/conv2d_35/BiasAdd:output:0*
T0*0
_output_shapes
:??????????2
model_4/activation_29/Relu?
model_4/dropout_33/IdentityIdentity(model_4/activation_29/Relu:activations:0*
T0*0
_output_shapes
:??????????2
model_4/dropout_33/Identity?
'model_4/conv2d_36/Conv2D/ReadVariableOpReadVariableOp0model_4_conv2d_36_conv2d_readvariableop_resource*(
_output_shapes
:??*
dtype02)
'model_4/conv2d_36/Conv2D/ReadVariableOp?
model_4/conv2d_36/Conv2DConv2D$model_4/dropout_33/Identity:output:0/model_4/conv2d_36/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????*
paddingSAME*
strides
2
model_4/conv2d_36/Conv2D?
(model_4/conv2d_36/BiasAdd/ReadVariableOpReadVariableOp1model_4_conv2d_36_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02*
(model_4/conv2d_36/BiasAdd/ReadVariableOp?
model_4/conv2d_36/BiasAddBiasAdd!model_4/conv2d_36/Conv2D:output:00model_4/conv2d_36/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????2
model_4/conv2d_36/BiasAdd?
%model_4/tf.__operators__.add_13/AddV2AddV2"model_4/conv2d_36/BiasAdd:output:0$model_4/dropout_32/Identity:output:0*
T0*0
_output_shapes
:??????????2'
%model_4/tf.__operators__.add_13/AddV2?
model_4/activation_30/ReluRelu)model_4/tf.__operators__.add_13/AddV2:z:0*
T0*0
_output_shapes
:??????????2
model_4/activation_30/Relu?
-model_4/batch_normalization_16/ReadVariableOpReadVariableOp6model_4_batch_normalization_16_readvariableop_resource*
_output_shapes	
:?*
dtype02/
-model_4/batch_normalization_16/ReadVariableOp?
/model_4/batch_normalization_16/ReadVariableOp_1ReadVariableOp8model_4_batch_normalization_16_readvariableop_1_resource*
_output_shapes	
:?*
dtype021
/model_4/batch_normalization_16/ReadVariableOp_1?
>model_4/batch_normalization_16/FusedBatchNormV3/ReadVariableOpReadVariableOpGmodel_4_batch_normalization_16_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:?*
dtype02@
>model_4/batch_normalization_16/FusedBatchNormV3/ReadVariableOp?
@model_4/batch_normalization_16/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpImodel_4_batch_normalization_16_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:?*
dtype02B
@model_4/batch_normalization_16/FusedBatchNormV3/ReadVariableOp_1?
/model_4/batch_normalization_16/FusedBatchNormV3FusedBatchNormV3(model_4/activation_30/Relu:activations:05model_4/batch_normalization_16/ReadVariableOp:value:07model_4/batch_normalization_16/ReadVariableOp_1:value:0Fmodel_4/batch_normalization_16/FusedBatchNormV3/ReadVariableOp:value:0Hmodel_4/batch_normalization_16/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:??????????:?:?:?:?:*
epsilon%o?:*
is_training( 21
/model_4/batch_normalization_16/FusedBatchNormV3?
model_4/dropout_34/IdentityIdentity3model_4/batch_normalization_16/FusedBatchNormV3:y:0*
T0*0
_output_shapes
:??????????2
model_4/dropout_34/Identity?
'model_4/conv2d_37/Conv2D/ReadVariableOpReadVariableOp0model_4_conv2d_37_conv2d_readvariableop_resource*(
_output_shapes
:??*
dtype02)
'model_4/conv2d_37/Conv2D/ReadVariableOp?
model_4/conv2d_37/Conv2DConv2D$model_4/dropout_34/Identity:output:0/model_4/conv2d_37/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????*
paddingSAME*
strides
2
model_4/conv2d_37/Conv2D?
(model_4/conv2d_37/BiasAdd/ReadVariableOpReadVariableOp1model_4_conv2d_37_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02*
(model_4/conv2d_37/BiasAdd/ReadVariableOp?
model_4/conv2d_37/BiasAddBiasAdd!model_4/conv2d_37/Conv2D:output:00model_4/conv2d_37/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????2
model_4/conv2d_37/BiasAdd?
model_4/activation_31/ReluRelu"model_4/conv2d_37/BiasAdd:output:0*
T0*0
_output_shapes
:??????????2
model_4/activation_31/Relu?
model_4/dropout_35/IdentityIdentity(model_4/activation_31/Relu:activations:0*
T0*0
_output_shapes
:??????????2
model_4/dropout_35/Identity?
'model_4/conv2d_38/Conv2D/ReadVariableOpReadVariableOp0model_4_conv2d_38_conv2d_readvariableop_resource*(
_output_shapes
:??*
dtype02)
'model_4/conv2d_38/Conv2D/ReadVariableOp?
model_4/conv2d_38/Conv2DConv2D$model_4/dropout_35/Identity:output:0/model_4/conv2d_38/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????*
paddingSAME*
strides
2
model_4/conv2d_38/Conv2D?
(model_4/conv2d_38/BiasAdd/ReadVariableOpReadVariableOp1model_4_conv2d_38_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02*
(model_4/conv2d_38/BiasAdd/ReadVariableOp?
model_4/conv2d_38/BiasAddBiasAdd!model_4/conv2d_38/Conv2D:output:00model_4/conv2d_38/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????2
model_4/conv2d_38/BiasAdd?
%model_4/tf.__operators__.add_14/AddV2AddV2"model_4/conv2d_38/BiasAdd:output:0$model_4/dropout_34/Identity:output:0*
T0*0
_output_shapes
:??????????2'
%model_4/tf.__operators__.add_14/AddV2?
model_4/activation_32/ReluRelu)model_4/tf.__operators__.add_14/AddV2:z:0*
T0*0
_output_shapes
:??????????2
model_4/activation_32/Relu?
-model_4/batch_normalization_17/ReadVariableOpReadVariableOp6model_4_batch_normalization_17_readvariableop_resource*
_output_shapes	
:?*
dtype02/
-model_4/batch_normalization_17/ReadVariableOp?
/model_4/batch_normalization_17/ReadVariableOp_1ReadVariableOp8model_4_batch_normalization_17_readvariableop_1_resource*
_output_shapes	
:?*
dtype021
/model_4/batch_normalization_17/ReadVariableOp_1?
>model_4/batch_normalization_17/FusedBatchNormV3/ReadVariableOpReadVariableOpGmodel_4_batch_normalization_17_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:?*
dtype02@
>model_4/batch_normalization_17/FusedBatchNormV3/ReadVariableOp?
@model_4/batch_normalization_17/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpImodel_4_batch_normalization_17_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:?*
dtype02B
@model_4/batch_normalization_17/FusedBatchNormV3/ReadVariableOp_1?
/model_4/batch_normalization_17/FusedBatchNormV3FusedBatchNormV3(model_4/activation_32/Relu:activations:05model_4/batch_normalization_17/ReadVariableOp:value:07model_4/batch_normalization_17/ReadVariableOp_1:value:0Fmodel_4/batch_normalization_17/FusedBatchNormV3/ReadVariableOp:value:0Hmodel_4/batch_normalization_17/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:??????????:?:?:?:?:*
epsilon%o?:*
is_training( 21
/model_4/batch_normalization_17/FusedBatchNormV3?
model_4/dropout_36/IdentityIdentity3model_4/batch_normalization_17/FusedBatchNormV3:y:0*
T0*0
_output_shapes
:??????????2
model_4/dropout_36/Identity?
conv2d_55/Conv2D/ReadVariableOpReadVariableOp(conv2d_55_conv2d_readvariableop_resource*'
_output_shapes
:?*
dtype02!
conv2d_55/Conv2D/ReadVariableOp?
conv2d_55/Conv2DConv2D$model_4/dropout_36/Identity:output:0'conv2d_55/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????*
paddingVALID*
strides
2
conv2d_55/Conv2D?
 conv2d_55/BiasAdd/ReadVariableOpReadVariableOp)conv2d_55_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02"
 conv2d_55/BiasAdd/ReadVariableOp?
conv2d_55/BiasAddBiasAddconv2d_55/Conv2D:output:0(conv2d_55/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????2
conv2d_55/BiasAdd~
conv2d_55/ReluReluconv2d_55/BiasAdd:output:0*
T0*/
_output_shapes
:?????????2
conv2d_55/Relu?
conv2d_53/Conv2D/ReadVariableOpReadVariableOp(conv2d_53_conv2d_readvariableop_resource*(
_output_shapes
:??*
dtype02!
conv2d_53/Conv2D/ReadVariableOp?
conv2d_53/Conv2DConv2D$model_4/dropout_36/Identity:output:0'conv2d_53/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????*
paddingSAME*
strides
2
conv2d_53/Conv2D?
 conv2d_53/BiasAdd/ReadVariableOpReadVariableOp)conv2d_53_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02"
 conv2d_53/BiasAdd/ReadVariableOp?
conv2d_53/BiasAddBiasAddconv2d_53/Conv2D:output:0(conv2d_53/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????2
conv2d_53/BiasAdd
conv2d_53/ReluReluconv2d_53/BiasAdd:output:0*
T0*0
_output_shapes
:??????????2
conv2d_53/Relu?
dropout_51/IdentityIdentityconv2d_55/Relu:activations:0*
T0*/
_output_shapes
:?????????2
dropout_51/Identity?
dropout_50/IdentityIdentityconv2d_53/Relu:activations:0*
T0*0
_output_shapes
:??????????2
dropout_50/Identitys
flatten_3/ConstConst*
_output_shapes
:*
dtype0*
valueB"????@   2
flatten_3/Const?
flatten_3/ReshapeReshapedropout_51/Identity:output:0flatten_3/Const:output:0*
T0*'
_output_shapes
:?????????@2
flatten_3/Reshape?
conv2d_54/Conv2D/ReadVariableOpReadVariableOp(conv2d_54_conv2d_readvariableop_resource*'
_output_shapes
:?I*
dtype02!
conv2d_54/Conv2D/ReadVariableOp?
conv2d_54/Conv2DConv2Ddropout_50/Identity:output:0'conv2d_54/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????I*
paddingSAME*
strides
2
conv2d_54/Conv2D?
 conv2d_54/BiasAdd/ReadVariableOpReadVariableOp)conv2d_54_biasadd_readvariableop_resource*
_output_shapes
:I*
dtype02"
 conv2d_54/BiasAdd/ReadVariableOp?
conv2d_54/BiasAddBiasAddconv2d_54/Conv2D:output:0(conv2d_54/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????I2
conv2d_54/BiasAdd?
conv2d_54/SoftmaxSoftmaxconv2d_54/BiasAdd:output:0*
T0*/
_output_shapes
:?????????I2
conv2d_54/Softmax?
dense_3/MatMul/ReadVariableOpReadVariableOp&dense_3_matmul_readvariableop_resource*
_output_shapes

:@*
dtype02
dense_3/MatMul/ReadVariableOp?
dense_3/MatMulMatMulflatten_3/Reshape:output:0%dense_3/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_3/MatMulp
dense_3/TanhTanhdense_3/MatMul:product:0*
T0*'
_output_shapes
:?????????2
dense_3/Tanhk
IdentityIdentitydense_3/Tanh:y:0^NoOp*
T0*'
_output_shapes
:?????????2

Identity?

Identity_1Identityconv2d_54/Softmax:softmax:0^NoOp*
T0*/
_output_shapes
:?????????I2

Identity_1?
NoOpNoOp!^conv2d_53/BiasAdd/ReadVariableOp ^conv2d_53/Conv2D/ReadVariableOp!^conv2d_54/BiasAdd/ReadVariableOp ^conv2d_54/Conv2D/ReadVariableOp!^conv2d_55/BiasAdd/ReadVariableOp ^conv2d_55/Conv2D/ReadVariableOp^dense_3/MatMul/ReadVariableOp?^model_4/batch_normalization_12/FusedBatchNormV3/ReadVariableOpA^model_4/batch_normalization_12/FusedBatchNormV3/ReadVariableOp_1.^model_4/batch_normalization_12/ReadVariableOp0^model_4/batch_normalization_12/ReadVariableOp_1?^model_4/batch_normalization_13/FusedBatchNormV3/ReadVariableOpA^model_4/batch_normalization_13/FusedBatchNormV3/ReadVariableOp_1.^model_4/batch_normalization_13/ReadVariableOp0^model_4/batch_normalization_13/ReadVariableOp_1?^model_4/batch_normalization_14/FusedBatchNormV3/ReadVariableOpA^model_4/batch_normalization_14/FusedBatchNormV3/ReadVariableOp_1.^model_4/batch_normalization_14/ReadVariableOp0^model_4/batch_normalization_14/ReadVariableOp_1?^model_4/batch_normalization_15/FusedBatchNormV3/ReadVariableOpA^model_4/batch_normalization_15/FusedBatchNormV3/ReadVariableOp_1.^model_4/batch_normalization_15/ReadVariableOp0^model_4/batch_normalization_15/ReadVariableOp_1?^model_4/batch_normalization_16/FusedBatchNormV3/ReadVariableOpA^model_4/batch_normalization_16/FusedBatchNormV3/ReadVariableOp_1.^model_4/batch_normalization_16/ReadVariableOp0^model_4/batch_normalization_16/ReadVariableOp_1?^model_4/batch_normalization_17/FusedBatchNormV3/ReadVariableOpA^model_4/batch_normalization_17/FusedBatchNormV3/ReadVariableOp_1.^model_4/batch_normalization_17/ReadVariableOp0^model_4/batch_normalization_17/ReadVariableOp_1)^model_4/conv2d_28/BiasAdd/ReadVariableOp(^model_4/conv2d_28/Conv2D/ReadVariableOp)^model_4/conv2d_29/BiasAdd/ReadVariableOp(^model_4/conv2d_29/Conv2D/ReadVariableOp)^model_4/conv2d_30/BiasAdd/ReadVariableOp(^model_4/conv2d_30/Conv2D/ReadVariableOp)^model_4/conv2d_31/BiasAdd/ReadVariableOp(^model_4/conv2d_31/Conv2D/ReadVariableOp)^model_4/conv2d_32/BiasAdd/ReadVariableOp(^model_4/conv2d_32/Conv2D/ReadVariableOp)^model_4/conv2d_33/BiasAdd/ReadVariableOp(^model_4/conv2d_33/Conv2D/ReadVariableOp)^model_4/conv2d_34/BiasAdd/ReadVariableOp(^model_4/conv2d_34/Conv2D/ReadVariableOp)^model_4/conv2d_35/BiasAdd/ReadVariableOp(^model_4/conv2d_35/Conv2D/ReadVariableOp)^model_4/conv2d_36/BiasAdd/ReadVariableOp(^model_4/conv2d_36/Conv2D/ReadVariableOp)^model_4/conv2d_37/BiasAdd/ReadVariableOp(^model_4/conv2d_37/Conv2D/ReadVariableOp)^model_4/conv2d_38/BiasAdd/ReadVariableOp(^model_4/conv2d_38/Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes?
?:?????????: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2D
 conv2d_53/BiasAdd/ReadVariableOp conv2d_53/BiasAdd/ReadVariableOp2B
conv2d_53/Conv2D/ReadVariableOpconv2d_53/Conv2D/ReadVariableOp2D
 conv2d_54/BiasAdd/ReadVariableOp conv2d_54/BiasAdd/ReadVariableOp2B
conv2d_54/Conv2D/ReadVariableOpconv2d_54/Conv2D/ReadVariableOp2D
 conv2d_55/BiasAdd/ReadVariableOp conv2d_55/BiasAdd/ReadVariableOp2B
conv2d_55/Conv2D/ReadVariableOpconv2d_55/Conv2D/ReadVariableOp2>
dense_3/MatMul/ReadVariableOpdense_3/MatMul/ReadVariableOp2?
>model_4/batch_normalization_12/FusedBatchNormV3/ReadVariableOp>model_4/batch_normalization_12/FusedBatchNormV3/ReadVariableOp2?
@model_4/batch_normalization_12/FusedBatchNormV3/ReadVariableOp_1@model_4/batch_normalization_12/FusedBatchNormV3/ReadVariableOp_12^
-model_4/batch_normalization_12/ReadVariableOp-model_4/batch_normalization_12/ReadVariableOp2b
/model_4/batch_normalization_12/ReadVariableOp_1/model_4/batch_normalization_12/ReadVariableOp_12?
>model_4/batch_normalization_13/FusedBatchNormV3/ReadVariableOp>model_4/batch_normalization_13/FusedBatchNormV3/ReadVariableOp2?
@model_4/batch_normalization_13/FusedBatchNormV3/ReadVariableOp_1@model_4/batch_normalization_13/FusedBatchNormV3/ReadVariableOp_12^
-model_4/batch_normalization_13/ReadVariableOp-model_4/batch_normalization_13/ReadVariableOp2b
/model_4/batch_normalization_13/ReadVariableOp_1/model_4/batch_normalization_13/ReadVariableOp_12?
>model_4/batch_normalization_14/FusedBatchNormV3/ReadVariableOp>model_4/batch_normalization_14/FusedBatchNormV3/ReadVariableOp2?
@model_4/batch_normalization_14/FusedBatchNormV3/ReadVariableOp_1@model_4/batch_normalization_14/FusedBatchNormV3/ReadVariableOp_12^
-model_4/batch_normalization_14/ReadVariableOp-model_4/batch_normalization_14/ReadVariableOp2b
/model_4/batch_normalization_14/ReadVariableOp_1/model_4/batch_normalization_14/ReadVariableOp_12?
>model_4/batch_normalization_15/FusedBatchNormV3/ReadVariableOp>model_4/batch_normalization_15/FusedBatchNormV3/ReadVariableOp2?
@model_4/batch_normalization_15/FusedBatchNormV3/ReadVariableOp_1@model_4/batch_normalization_15/FusedBatchNormV3/ReadVariableOp_12^
-model_4/batch_normalization_15/ReadVariableOp-model_4/batch_normalization_15/ReadVariableOp2b
/model_4/batch_normalization_15/ReadVariableOp_1/model_4/batch_normalization_15/ReadVariableOp_12?
>model_4/batch_normalization_16/FusedBatchNormV3/ReadVariableOp>model_4/batch_normalization_16/FusedBatchNormV3/ReadVariableOp2?
@model_4/batch_normalization_16/FusedBatchNormV3/ReadVariableOp_1@model_4/batch_normalization_16/FusedBatchNormV3/ReadVariableOp_12^
-model_4/batch_normalization_16/ReadVariableOp-model_4/batch_normalization_16/ReadVariableOp2b
/model_4/batch_normalization_16/ReadVariableOp_1/model_4/batch_normalization_16/ReadVariableOp_12?
>model_4/batch_normalization_17/FusedBatchNormV3/ReadVariableOp>model_4/batch_normalization_17/FusedBatchNormV3/ReadVariableOp2?
@model_4/batch_normalization_17/FusedBatchNormV3/ReadVariableOp_1@model_4/batch_normalization_17/FusedBatchNormV3/ReadVariableOp_12^
-model_4/batch_normalization_17/ReadVariableOp-model_4/batch_normalization_17/ReadVariableOp2b
/model_4/batch_normalization_17/ReadVariableOp_1/model_4/batch_normalization_17/ReadVariableOp_12T
(model_4/conv2d_28/BiasAdd/ReadVariableOp(model_4/conv2d_28/BiasAdd/ReadVariableOp2R
'model_4/conv2d_28/Conv2D/ReadVariableOp'model_4/conv2d_28/Conv2D/ReadVariableOp2T
(model_4/conv2d_29/BiasAdd/ReadVariableOp(model_4/conv2d_29/BiasAdd/ReadVariableOp2R
'model_4/conv2d_29/Conv2D/ReadVariableOp'model_4/conv2d_29/Conv2D/ReadVariableOp2T
(model_4/conv2d_30/BiasAdd/ReadVariableOp(model_4/conv2d_30/BiasAdd/ReadVariableOp2R
'model_4/conv2d_30/Conv2D/ReadVariableOp'model_4/conv2d_30/Conv2D/ReadVariableOp2T
(model_4/conv2d_31/BiasAdd/ReadVariableOp(model_4/conv2d_31/BiasAdd/ReadVariableOp2R
'model_4/conv2d_31/Conv2D/ReadVariableOp'model_4/conv2d_31/Conv2D/ReadVariableOp2T
(model_4/conv2d_32/BiasAdd/ReadVariableOp(model_4/conv2d_32/BiasAdd/ReadVariableOp2R
'model_4/conv2d_32/Conv2D/ReadVariableOp'model_4/conv2d_32/Conv2D/ReadVariableOp2T
(model_4/conv2d_33/BiasAdd/ReadVariableOp(model_4/conv2d_33/BiasAdd/ReadVariableOp2R
'model_4/conv2d_33/Conv2D/ReadVariableOp'model_4/conv2d_33/Conv2D/ReadVariableOp2T
(model_4/conv2d_34/BiasAdd/ReadVariableOp(model_4/conv2d_34/BiasAdd/ReadVariableOp2R
'model_4/conv2d_34/Conv2D/ReadVariableOp'model_4/conv2d_34/Conv2D/ReadVariableOp2T
(model_4/conv2d_35/BiasAdd/ReadVariableOp(model_4/conv2d_35/BiasAdd/ReadVariableOp2R
'model_4/conv2d_35/Conv2D/ReadVariableOp'model_4/conv2d_35/Conv2D/ReadVariableOp2T
(model_4/conv2d_36/BiasAdd/ReadVariableOp(model_4/conv2d_36/BiasAdd/ReadVariableOp2R
'model_4/conv2d_36/Conv2D/ReadVariableOp'model_4/conv2d_36/Conv2D/ReadVariableOp2T
(model_4/conv2d_37/BiasAdd/ReadVariableOp(model_4/conv2d_37/BiasAdd/ReadVariableOp2R
'model_4/conv2d_37/Conv2D/ReadVariableOp'model_4/conv2d_37/Conv2D/ReadVariableOp2T
(model_4/conv2d_38/BiasAdd/ReadVariableOp(model_4/conv2d_38/BiasAdd/ReadVariableOp2R
'model_4/conv2d_38/Conv2D/ReadVariableOp'model_4/conv2d_38/Conv2D/ReadVariableOp:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?
I
-__inference_dropout_31_layer_call_fn_22008118

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
GPU 2J 8? *Q
fLRJ
H__inference_dropout_31_layer_call_and_return_conditional_losses_220039632
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
?
d
H__inference_dropout_26_layer_call_and_return_conditional_losses_22004206

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
?
?
T__inference_batch_normalization_12_layer_call_and_return_conditional_losses_22007574

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
G__inference_conv2d_30_layer_call_and_return_conditional_losses_22003168

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
?
I
-__inference_dropout_51_layer_call_fn_22007373

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
GPU 2J 8? *Q
fLRJ
H__inference_dropout_51_layer_call_and_return_conditional_losses_220053292
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
?
?
E__inference_dense_3_layer_call_and_return_conditional_losses_22005161

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
?
?
T__inference_batch_normalization_15_layer_call_and_return_conditional_losses_22008226

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
?
f
H__inference_dropout_28_layer_call_and_return_conditional_losses_22007855

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
d
H__inference_dropout_33_layer_call_and_return_conditional_losses_22003857

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
f
H__inference_dropout_36_layer_call_and_return_conditional_losses_22003566

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
?
?
*__inference_model_4_layer_call_fn_22003664
input_5"
unknown:?
	unknown_0:	?
	unknown_1:	?
	unknown_2:	?
	unknown_3:	?
	unknown_4:	?%
	unknown_5:??
	unknown_6:	?%
	unknown_7:??
	unknown_8:	?
	unknown_9:	?

unknown_10:	?

unknown_11:	?

unknown_12:	?&

unknown_13:??

unknown_14:	?&

unknown_15:??

unknown_16:	?

unknown_17:	?

unknown_18:	?

unknown_19:	?

unknown_20:	?&

unknown_21:??

unknown_22:	?&

unknown_23:??

unknown_24:	?

unknown_25:	?

unknown_26:	?

unknown_27:	?

unknown_28:	?&

unknown_29:??

unknown_30:	?&

unknown_31:??

unknown_32:	?

unknown_33:	?

unknown_34:	?

unknown_35:	?

unknown_36:	?&

unknown_37:??

unknown_38:	?&

unknown_39:??

unknown_40:	?

unknown_41:	?

unknown_42:	?

unknown_43:	?

unknown_44:	?
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_5unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
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
unknown_23
unknown_24
unknown_25
unknown_26
unknown_27
unknown_28
unknown_29
unknown_30
unknown_31
unknown_32
unknown_33
unknown_34
unknown_35
unknown_36
unknown_37
unknown_38
unknown_39
unknown_40
unknown_41
unknown_42
unknown_43
unknown_44*:
Tin3
12/*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????*P
_read_only_resource_inputs2
0.	
 !"#$%&'()*+,-.*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_model_4_layer_call_and_return_conditional_losses_220035692
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
_construction_contextkEagerRuntime*?
_input_shapesy
w:?????????: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
/
_output_shapes
:?????????
!
_user_specified_name	input_5
?
?
T__inference_batch_normalization_14_layer_call_and_return_conditional_losses_22002582

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
?
9__inference_batch_normalization_13_layer_call_fn_22007755

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
GPU 2J 8? *]
fXRV
T__inference_batch_normalization_13_layer_call_and_return_conditional_losses_220031992
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
G__inference_conv2d_37_layer_call_and_return_conditional_losses_22008538

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
d
H__inference_dropout_28_layer_call_and_return_conditional_losses_22007859

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
T__inference_batch_normalization_12_layer_call_and_return_conditional_losses_22003104

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
d
H__inference_dropout_35_layer_call_and_return_conditional_losses_22003751

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
?
?
*__inference_model_4_layer_call_fn_22006895

inputs"
unknown:?
	unknown_0:	?
	unknown_1:	?
	unknown_2:	?
	unknown_3:	?
	unknown_4:	?%
	unknown_5:??
	unknown_6:	?%
	unknown_7:??
	unknown_8:	?
	unknown_9:	?

unknown_10:	?

unknown_11:	?

unknown_12:	?&

unknown_13:??

unknown_14:	?&

unknown_15:??

unknown_16:	?

unknown_17:	?

unknown_18:	?

unknown_19:	?

unknown_20:	?&

unknown_21:??

unknown_22:	?&

unknown_23:??

unknown_24:	?

unknown_25:	?

unknown_26:	?

unknown_27:	?

unknown_28:	?&

unknown_29:??

unknown_30:	?&

unknown_31:??

unknown_32:	?

unknown_33:	?

unknown_34:	?

unknown_35:	?

unknown_36:	?&

unknown_37:??

unknown_38:	?&

unknown_39:??

unknown_40:	?

unknown_41:	?

unknown_42:	?

unknown_43:	?

unknown_44:	?
identity??StatefulPartitionedCall?
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
unknown_23
unknown_24
unknown_25
unknown_26
unknown_27
unknown_28
unknown_29
unknown_30
unknown_31
unknown_32
unknown_33
unknown_34
unknown_35
unknown_36
unknown_37
unknown_38
unknown_39
unknown_40
unknown_41
unknown_42
unknown_43
unknown_44*:
Tin3
12/*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????*P
_read_only_resource_inputs2
0.	
 !"#$%&'()*+,-.*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_model_4_layer_call_and_return_conditional_losses_220035692
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
_construction_contextkEagerRuntime*?
_input_shapesy
w:?????????: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?
c
G__inference_flatten_3_layer_call_and_return_conditional_losses_22005134

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
?
?
*__inference_model_7_layer_call_fn_22006389

inputs"
unknown:?
	unknown_0:	?
	unknown_1:	?
	unknown_2:	?
	unknown_3:	?
	unknown_4:	?%
	unknown_5:??
	unknown_6:	?%
	unknown_7:??
	unknown_8:	?
	unknown_9:	?

unknown_10:	?

unknown_11:	?

unknown_12:	?&

unknown_13:??

unknown_14:	?&

unknown_15:??

unknown_16:	?

unknown_17:	?

unknown_18:	?

unknown_19:	?

unknown_20:	?&

unknown_21:??

unknown_22:	?&

unknown_23:??

unknown_24:	?

unknown_25:	?

unknown_26:	?

unknown_27:	?

unknown_28:	?&

unknown_29:??

unknown_30:	?&

unknown_31:??

unknown_32:	?

unknown_33:	?

unknown_34:	?

unknown_35:	?

unknown_36:	?&

unknown_37:??

unknown_38:	?&

unknown_39:??

unknown_40:	?

unknown_41:	?

unknown_42:	?

unknown_43:	?

unknown_44:	?%

unknown_45:?

unknown_46:&

unknown_47:??

unknown_48:	?%

unknown_49:?I

unknown_50:I

unknown_51:@
identity

identity_1??StatefulPartitionedCall?
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
unknown_23
unknown_24
unknown_25
unknown_26
unknown_27
unknown_28
unknown_29
unknown_30
unknown_31
unknown_32
unknown_33
unknown_34
unknown_35
unknown_36
unknown_37
unknown_38
unknown_39
unknown_40
unknown_41
unknown_42
unknown_43
unknown_44
unknown_45
unknown_46
unknown_47
unknown_48
unknown_49
unknown_50
unknown_51*A
Tin:
826*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:?????????:?????????I*K
_read_only_resource_inputs-
+)	
 !"#$'()*+,/012345*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_model_7_layer_call_and_return_conditional_losses_220055862
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
_construction_contextkEagerRuntime*?
_input_shapes?
?:?????????: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?
I
-__inference_dropout_35_layer_call_fn_22008558

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
GPU 2J 8? *Q
fLRJ
H__inference_dropout_35_layer_call_and_return_conditional_losses_220037512
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
f
H__inference_dropout_30_layer_call_and_return_conditional_losses_22003302

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
?	
?
9__inference_batch_normalization_16_layer_call_fn_22008415

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
GPU 2J 8? *]
fXRV
T__inference_batch_normalization_16_layer_call_and_return_conditional_losses_220034632
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
?
g
K__inference_activation_32_layer_call_and_return_conditional_losses_22008596

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
?
g
K__inference_activation_28_layer_call_and_return_conditional_losses_22003356

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
?
?
G__inference_conv2d_32_layer_call_and_return_conditional_losses_22007926

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
9__inference_batch_normalization_16_layer_call_fn_22008389

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
GPU 2J 8? *]
fXRV
T__inference_batch_normalization_16_layer_call_and_return_conditional_losses_220028342
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
L
0__inference_activation_22_layer_call_fn_22007615

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
GPU 2J 8? *T
fORM
K__inference_activation_22_layer_call_and_return_conditional_losses_220031192
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
?<
?
E__inference_model_7_layer_call_and_return_conditional_losses_22006048
input_8+
model_4_22005932:?
model_4_22005934:	?
model_4_22005936:	?
model_4_22005938:	?
model_4_22005940:	?
model_4_22005942:	?,
model_4_22005944:??
model_4_22005946:	?,
model_4_22005948:??
model_4_22005950:	?
model_4_22005952:	?
model_4_22005954:	?
model_4_22005956:	?
model_4_22005958:	?,
model_4_22005960:??
model_4_22005962:	?,
model_4_22005964:??
model_4_22005966:	?
model_4_22005968:	?
model_4_22005970:	?
model_4_22005972:	?
model_4_22005974:	?,
model_4_22005976:??
model_4_22005978:	?,
model_4_22005980:??
model_4_22005982:	?
model_4_22005984:	?
model_4_22005986:	?
model_4_22005988:	?
model_4_22005990:	?,
model_4_22005992:??
model_4_22005994:	?,
model_4_22005996:??
model_4_22005998:	?
model_4_22006000:	?
model_4_22006002:	?
model_4_22006004:	?
model_4_22006006:	?,
model_4_22006008:??
model_4_22006010:	?,
model_4_22006012:??
model_4_22006014:	?
model_4_22006016:	?
model_4_22006018:	?
model_4_22006020:	?
model_4_22006022:	?-
conv2d_55_22006025:? 
conv2d_55_22006027:.
conv2d_53_22006030:??!
conv2d_53_22006032:	?-
conv2d_54_22006038:?I 
conv2d_54_22006040:I"
dense_3_22006043:@
identity

identity_1??!conv2d_53/StatefulPartitionedCall?!conv2d_54/StatefulPartitionedCall?!conv2d_55/StatefulPartitionedCall?dense_3/StatefulPartitionedCall?model_4/StatefulPartitionedCall?

model_4/StatefulPartitionedCallStatefulPartitionedCallinput_8model_4_22005932model_4_22005934model_4_22005936model_4_22005938model_4_22005940model_4_22005942model_4_22005944model_4_22005946model_4_22005948model_4_22005950model_4_22005952model_4_22005954model_4_22005956model_4_22005958model_4_22005960model_4_22005962model_4_22005964model_4_22005966model_4_22005968model_4_22005970model_4_22005972model_4_22005974model_4_22005976model_4_22005978model_4_22005980model_4_22005982model_4_22005984model_4_22005986model_4_22005988model_4_22005990model_4_22005992model_4_22005994model_4_22005996model_4_22005998model_4_22006000model_4_22006002model_4_22006004model_4_22006006model_4_22006008model_4_22006010model_4_22006012model_4_22006014model_4_22006016model_4_22006018model_4_22006020model_4_22006022*:
Tin3
12/*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????*D
_read_only_resource_inputs&
$"	
 !"#$'()*+,*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_model_4_layer_call_and_return_conditional_losses_220045082!
model_4/StatefulPartitionedCall?
!conv2d_55/StatefulPartitionedCallStatefulPartitionedCall(model_4/StatefulPartitionedCall:output:0conv2d_55_22006025conv2d_55_22006027*
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
GPU 2J 8? *P
fKRI
G__inference_conv2d_55_layer_call_and_return_conditional_losses_220050912#
!conv2d_55/StatefulPartitionedCall?
!conv2d_53/StatefulPartitionedCallStatefulPartitionedCall(model_4/StatefulPartitionedCall:output:0conv2d_53_22006030conv2d_53_22006032*
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
GPU 2J 8? *P
fKRI
G__inference_conv2d_53_layer_call_and_return_conditional_losses_220051082#
!conv2d_53/StatefulPartitionedCall?
dropout_51/PartitionedCallPartitionedCall*conv2d_55/StatefulPartitionedCall:output:0*
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
GPU 2J 8? *Q
fLRJ
H__inference_dropout_51_layer_call_and_return_conditional_losses_220053292
dropout_51/PartitionedCall?
dropout_50/PartitionedCallPartitionedCall*conv2d_53/StatefulPartitionedCall:output:0*
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
GPU 2J 8? *Q
fLRJ
H__inference_dropout_50_layer_call_and_return_conditional_losses_220053142
dropout_50/PartitionedCall?
flatten_3/PartitionedCallPartitionedCall#dropout_51/PartitionedCall:output:0*
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
GPU 2J 8? *P
fKRI
G__inference_flatten_3_layer_call_and_return_conditional_losses_220051342
flatten_3/PartitionedCall?
!conv2d_54/StatefulPartitionedCallStatefulPartitionedCall#dropout_50/PartitionedCall:output:0conv2d_54_22006038conv2d_54_22006040*
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
GPU 2J 8? *P
fKRI
G__inference_conv2d_54_layer_call_and_return_conditional_losses_220051472#
!conv2d_54/StatefulPartitionedCall?
dense_3/StatefulPartitionedCallStatefulPartitionedCall"flatten_3/PartitionedCall:output:0dense_3_22006043*
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
GPU 2J 8? *N
fIRG
E__inference_dense_3_layer_call_and_return_conditional_losses_220051612!
dense_3/StatefulPartitionedCall?
IdentityIdentity(dense_3/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????2

Identity?

Identity_1Identity*conv2d_54/StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:?????????I2

Identity_1?
NoOpNoOp"^conv2d_53/StatefulPartitionedCall"^conv2d_54/StatefulPartitionedCall"^conv2d_55/StatefulPartitionedCall ^dense_3/StatefulPartitionedCall ^model_4/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes?
?:?????????: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2F
!conv2d_53/StatefulPartitionedCall!conv2d_53/StatefulPartitionedCall2F
!conv2d_54/StatefulPartitionedCall!conv2d_54/StatefulPartitionedCall2F
!conv2d_55/StatefulPartitionedCall!conv2d_55/StatefulPartitionedCall2B
dense_3/StatefulPartitionedCalldense_3/StatefulPartitionedCall2B
model_4/StatefulPartitionedCallmodel_4/StatefulPartitionedCall:X T
/
_output_shapes
:?????????
!
_user_specified_name	input_8
?
I
-__inference_dropout_35_layer_call_fn_22008553

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
GPU 2J 8? *Q
fLRJ
H__inference_dropout_35_layer_call_and_return_conditional_losses_220035082
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
f
H__inference_dropout_50_layer_call_and_return_conditional_losses_22007428

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
?
f
H__inference_dropout_50_layer_call_and_return_conditional_losses_22005126

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
I
-__inference_dropout_30_layer_call_fn_22008070

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
GPU 2J 8? *Q
fLRJ
H__inference_dropout_30_layer_call_and_return_conditional_losses_220039942
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
g
K__inference_activation_31_layer_call_and_return_conditional_losses_22003501

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
?	
?
9__inference_batch_normalization_16_layer_call_fn_22008402

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
GPU 2J 8? *]
fXRV
T__inference_batch_normalization_16_layer_call_and_return_conditional_losses_220028782
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
f
H__inference_dropout_31_layer_call_and_return_conditional_losses_22003332

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
?<
?
E__inference_model_7_layer_call_and_return_conditional_losses_22005586

inputs+
model_4_22005470:?
model_4_22005472:	?
model_4_22005474:	?
model_4_22005476:	?
model_4_22005478:	?
model_4_22005480:	?,
model_4_22005482:??
model_4_22005484:	?,
model_4_22005486:??
model_4_22005488:	?
model_4_22005490:	?
model_4_22005492:	?
model_4_22005494:	?
model_4_22005496:	?,
model_4_22005498:??
model_4_22005500:	?,
model_4_22005502:??
model_4_22005504:	?
model_4_22005506:	?
model_4_22005508:	?
model_4_22005510:	?
model_4_22005512:	?,
model_4_22005514:??
model_4_22005516:	?,
model_4_22005518:??
model_4_22005520:	?
model_4_22005522:	?
model_4_22005524:	?
model_4_22005526:	?
model_4_22005528:	?,
model_4_22005530:??
model_4_22005532:	?,
model_4_22005534:??
model_4_22005536:	?
model_4_22005538:	?
model_4_22005540:	?
model_4_22005542:	?
model_4_22005544:	?,
model_4_22005546:??
model_4_22005548:	?,
model_4_22005550:??
model_4_22005552:	?
model_4_22005554:	?
model_4_22005556:	?
model_4_22005558:	?
model_4_22005560:	?-
conv2d_55_22005563:? 
conv2d_55_22005565:.
conv2d_53_22005568:??!
conv2d_53_22005570:	?-
conv2d_54_22005576:?I 
conv2d_54_22005578:I"
dense_3_22005581:@
identity

identity_1??!conv2d_53/StatefulPartitionedCall?!conv2d_54/StatefulPartitionedCall?!conv2d_55/StatefulPartitionedCall?dense_3/StatefulPartitionedCall?model_4/StatefulPartitionedCall?

model_4/StatefulPartitionedCallStatefulPartitionedCallinputsmodel_4_22005470model_4_22005472model_4_22005474model_4_22005476model_4_22005478model_4_22005480model_4_22005482model_4_22005484model_4_22005486model_4_22005488model_4_22005490model_4_22005492model_4_22005494model_4_22005496model_4_22005498model_4_22005500model_4_22005502model_4_22005504model_4_22005506model_4_22005508model_4_22005510model_4_22005512model_4_22005514model_4_22005516model_4_22005518model_4_22005520model_4_22005522model_4_22005524model_4_22005526model_4_22005528model_4_22005530model_4_22005532model_4_22005534model_4_22005536model_4_22005538model_4_22005540model_4_22005542model_4_22005544model_4_22005546model_4_22005548model_4_22005550model_4_22005552model_4_22005554model_4_22005556model_4_22005558model_4_22005560*:
Tin3
12/*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????*D
_read_only_resource_inputs&
$"	
 !"#$'()*+,*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_model_4_layer_call_and_return_conditional_losses_220045082!
model_4/StatefulPartitionedCall?
!conv2d_55/StatefulPartitionedCallStatefulPartitionedCall(model_4/StatefulPartitionedCall:output:0conv2d_55_22005563conv2d_55_22005565*
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
GPU 2J 8? *P
fKRI
G__inference_conv2d_55_layer_call_and_return_conditional_losses_220050912#
!conv2d_55/StatefulPartitionedCall?
!conv2d_53/StatefulPartitionedCallStatefulPartitionedCall(model_4/StatefulPartitionedCall:output:0conv2d_53_22005568conv2d_53_22005570*
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
GPU 2J 8? *P
fKRI
G__inference_conv2d_53_layer_call_and_return_conditional_losses_220051082#
!conv2d_53/StatefulPartitionedCall?
dropout_51/PartitionedCallPartitionedCall*conv2d_55/StatefulPartitionedCall:output:0*
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
GPU 2J 8? *Q
fLRJ
H__inference_dropout_51_layer_call_and_return_conditional_losses_220053292
dropout_51/PartitionedCall?
dropout_50/PartitionedCallPartitionedCall*conv2d_53/StatefulPartitionedCall:output:0*
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
GPU 2J 8? *Q
fLRJ
H__inference_dropout_50_layer_call_and_return_conditional_losses_220053142
dropout_50/PartitionedCall?
flatten_3/PartitionedCallPartitionedCall#dropout_51/PartitionedCall:output:0*
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
GPU 2J 8? *P
fKRI
G__inference_flatten_3_layer_call_and_return_conditional_losses_220051342
flatten_3/PartitionedCall?
!conv2d_54/StatefulPartitionedCallStatefulPartitionedCall#dropout_50/PartitionedCall:output:0conv2d_54_22005576conv2d_54_22005578*
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
GPU 2J 8? *P
fKRI
G__inference_conv2d_54_layer_call_and_return_conditional_losses_220051472#
!conv2d_54/StatefulPartitionedCall?
dense_3/StatefulPartitionedCallStatefulPartitionedCall"flatten_3/PartitionedCall:output:0dense_3_22005581*
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
GPU 2J 8? *N
fIRG
E__inference_dense_3_layer_call_and_return_conditional_losses_220051612!
dense_3/StatefulPartitionedCall?
IdentityIdentity(dense_3/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????2

Identity?

Identity_1Identity*conv2d_54/StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:?????????I2

Identity_1?
NoOpNoOp"^conv2d_53/StatefulPartitionedCall"^conv2d_54/StatefulPartitionedCall"^conv2d_55/StatefulPartitionedCall ^dense_3/StatefulPartitionedCall ^model_4/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes?
?:?????????: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2F
!conv2d_53/StatefulPartitionedCall!conv2d_53/StatefulPartitionedCall2F
!conv2d_54/StatefulPartitionedCall!conv2d_54/StatefulPartitionedCall2F
!conv2d_55/StatefulPartitionedCall!conv2d_55/StatefulPartitionedCall2B
dense_3/StatefulPartitionedCalldense_3/StatefulPartitionedCall2B
model_4/StatefulPartitionedCallmodel_4/StatefulPartitionedCall:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?	
?
9__inference_batch_normalization_16_layer_call_fn_22008428

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
GPU 2J 8? *]
fXRV
T__inference_batch_normalization_16_layer_call_and_return_conditional_losses_220038182
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
?
c
G__inference_flatten_3_layer_call_and_return_conditional_losses_22007413

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
T__inference_batch_normalization_13_layer_call_and_return_conditional_losses_22007840

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
?
?
G__inference_conv2d_33_layer_call_and_return_conditional_losses_22008098

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
d
H__inference_dropout_50_layer_call_and_return_conditional_losses_22007432

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
?
I
-__inference_dropout_27_layer_call_fn_22007678

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
GPU 2J 8? *Q
fLRJ
H__inference_dropout_27_layer_call_and_return_conditional_losses_220041752
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
T__inference_batch_normalization_16_layer_call_and_return_conditional_losses_22002878

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
?
d
H__inference_dropout_35_layer_call_and_return_conditional_losses_22008567

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
?
G__inference_conv2d_31_layer_call_and_return_conditional_losses_22007878

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
g
K__inference_activation_27_layer_call_and_return_conditional_losses_22003325

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
T__inference_batch_normalization_16_layer_call_and_return_conditional_losses_22008446

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
?
?
T__inference_batch_normalization_15_layer_call_and_return_conditional_losses_22002752

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
?
?
T__inference_batch_normalization_16_layer_call_and_return_conditional_losses_22008500

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
g
K__inference_activation_24_layer_call_and_return_conditional_losses_22003180

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
L
0__inference_activation_26_layer_call_fn_22007931

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
GPU 2J 8? *T
fORM
K__inference_activation_26_layer_call_and_return_conditional_losses_220032682
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
?
?
T__inference_batch_normalization_14_layer_call_and_return_conditional_losses_22008042

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
f
H__inference_dropout_33_layer_call_and_return_conditional_losses_22008343

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
ͺ
?
E__inference_model_4_layer_call_and_return_conditional_losses_22004980
input_5-
conv2d_28_22004843:?!
conv2d_28_22004845:	?.
batch_normalization_12_22004848:	?.
batch_normalization_12_22004850:	?.
batch_normalization_12_22004852:	?.
batch_normalization_12_22004854:	?.
conv2d_29_22004859:??!
conv2d_29_22004861:	?.
conv2d_30_22004866:??!
conv2d_30_22004868:	?.
batch_normalization_13_22004873:	?.
batch_normalization_13_22004875:	?.
batch_normalization_13_22004877:	?.
batch_normalization_13_22004879:	?.
conv2d_31_22004883:??!
conv2d_31_22004885:	?.
conv2d_32_22004890:??!
conv2d_32_22004892:	?.
batch_normalization_14_22004897:	?.
batch_normalization_14_22004899:	?.
batch_normalization_14_22004901:	?.
batch_normalization_14_22004903:	?.
conv2d_33_22004907:??!
conv2d_33_22004909:	?.
conv2d_34_22004914:??!
conv2d_34_22004916:	?.
batch_normalization_15_22004921:	?.
batch_normalization_15_22004923:	?.
batch_normalization_15_22004925:	?.
batch_normalization_15_22004927:	?.
conv2d_35_22004931:??!
conv2d_35_22004933:	?.
conv2d_36_22004938:??!
conv2d_36_22004940:	?.
batch_normalization_16_22004945:	?.
batch_normalization_16_22004947:	?.
batch_normalization_16_22004949:	?.
batch_normalization_16_22004951:	?.
conv2d_37_22004955:??!
conv2d_37_22004957:	?.
conv2d_38_22004962:??!
conv2d_38_22004964:	?.
batch_normalization_17_22004969:	?.
batch_normalization_17_22004971:	?.
batch_normalization_17_22004973:	?.
batch_normalization_17_22004975:	?
identity??.batch_normalization_12/StatefulPartitionedCall?.batch_normalization_13/StatefulPartitionedCall?.batch_normalization_14/StatefulPartitionedCall?.batch_normalization_15/StatefulPartitionedCall?.batch_normalization_16/StatefulPartitionedCall?.batch_normalization_17/StatefulPartitionedCall?!conv2d_28/StatefulPartitionedCall?!conv2d_29/StatefulPartitionedCall?!conv2d_30/StatefulPartitionedCall?!conv2d_31/StatefulPartitionedCall?!conv2d_32/StatefulPartitionedCall?!conv2d_33/StatefulPartitionedCall?!conv2d_34/StatefulPartitionedCall?!conv2d_35/StatefulPartitionedCall?!conv2d_36/StatefulPartitionedCall?!conv2d_37/StatefulPartitionedCall?!conv2d_38/StatefulPartitionedCall?
!conv2d_28/StatefulPartitionedCallStatefulPartitionedCallinput_5conv2d_28_22004843conv2d_28_22004845*
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
GPU 2J 8? *P
fKRI
G__inference_conv2d_28_layer_call_and_return_conditional_losses_220030812#
!conv2d_28/StatefulPartitionedCall?
.batch_normalization_12/StatefulPartitionedCallStatefulPartitionedCall*conv2d_28/StatefulPartitionedCall:output:0batch_normalization_12_22004848batch_normalization_12_22004850batch_normalization_12_22004852batch_normalization_12_22004854*
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
GPU 2J 8? *]
fXRV
T__inference_batch_normalization_12_layer_call_and_return_conditional_losses_2200424820
.batch_normalization_12/StatefulPartitionedCall?
activation_22/PartitionedCallPartitionedCall7batch_normalization_12/StatefulPartitionedCall:output:0*
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
GPU 2J 8? *T
fORM
K__inference_activation_22_layer_call_and_return_conditional_losses_220031192
activation_22/PartitionedCall?
dropout_26/PartitionedCallPartitionedCall&activation_22/PartitionedCall:output:0*
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
GPU 2J 8? *Q
fLRJ
H__inference_dropout_26_layer_call_and_return_conditional_losses_220042062
dropout_26/PartitionedCall?
!conv2d_29/StatefulPartitionedCallStatefulPartitionedCall#dropout_26/PartitionedCall:output:0conv2d_29_22004859conv2d_29_22004861*
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
GPU 2J 8? *P
fKRI
G__inference_conv2d_29_layer_call_and_return_conditional_losses_220031382#
!conv2d_29/StatefulPartitionedCall?
activation_23/PartitionedCallPartitionedCall*conv2d_29/StatefulPartitionedCall:output:0*
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
GPU 2J 8? *T
fORM
K__inference_activation_23_layer_call_and_return_conditional_losses_220031492
activation_23/PartitionedCall?
dropout_27/PartitionedCallPartitionedCall&activation_23/PartitionedCall:output:0*
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
GPU 2J 8? *Q
fLRJ
H__inference_dropout_27_layer_call_and_return_conditional_losses_220041752
dropout_27/PartitionedCall?
!conv2d_30/StatefulPartitionedCallStatefulPartitionedCall#dropout_27/PartitionedCall:output:0conv2d_30_22004866conv2d_30_22004868*
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
GPU 2J 8? *P
fKRI
G__inference_conv2d_30_layer_call_and_return_conditional_losses_220031682#
!conv2d_30/StatefulPartitionedCall?
tf.__operators__.add_10/AddV2AddV2*conv2d_30/StatefulPartitionedCall:output:0#dropout_26/PartitionedCall:output:0*
T0*0
_output_shapes
:??????????2
tf.__operators__.add_10/AddV2?
activation_24/PartitionedCallPartitionedCall!tf.__operators__.add_10/AddV2:z:0*
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
GPU 2J 8? *T
fORM
K__inference_activation_24_layer_call_and_return_conditional_losses_220031802
activation_24/PartitionedCall?
.batch_normalization_13/StatefulPartitionedCallStatefulPartitionedCall&activation_24/PartitionedCall:output:0batch_normalization_13_22004873batch_normalization_13_22004875batch_normalization_13_22004877batch_normalization_13_22004879*
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
GPU 2J 8? *]
fXRV
T__inference_batch_normalization_13_layer_call_and_return_conditional_losses_2200413620
.batch_normalization_13/StatefulPartitionedCall?
dropout_28/PartitionedCallPartitionedCall7batch_normalization_13/StatefulPartitionedCall:output:0*
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
GPU 2J 8? *Q
fLRJ
H__inference_dropout_28_layer_call_and_return_conditional_losses_220041002
dropout_28/PartitionedCall?
!conv2d_31/StatefulPartitionedCallStatefulPartitionedCall#dropout_28/PartitionedCall:output:0conv2d_31_22004883conv2d_31_22004885*
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
GPU 2J 8? *P
fKRI
G__inference_conv2d_31_layer_call_and_return_conditional_losses_220032262#
!conv2d_31/StatefulPartitionedCall?
activation_25/PartitionedCallPartitionedCall*conv2d_31/StatefulPartitionedCall:output:0*
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
GPU 2J 8? *T
fORM
K__inference_activation_25_layer_call_and_return_conditional_losses_220032372
activation_25/PartitionedCall?
dropout_29/PartitionedCallPartitionedCall&activation_25/PartitionedCall:output:0*
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
GPU 2J 8? *Q
fLRJ
H__inference_dropout_29_layer_call_and_return_conditional_losses_220040692
dropout_29/PartitionedCall?
!conv2d_32/StatefulPartitionedCallStatefulPartitionedCall#dropout_29/PartitionedCall:output:0conv2d_32_22004890conv2d_32_22004892*
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
GPU 2J 8? *P
fKRI
G__inference_conv2d_32_layer_call_and_return_conditional_losses_220032562#
!conv2d_32/StatefulPartitionedCall?
tf.__operators__.add_11/AddV2AddV2*conv2d_32/StatefulPartitionedCall:output:0#dropout_28/PartitionedCall:output:0*
T0*0
_output_shapes
:??????????2
tf.__operators__.add_11/AddV2?
activation_26/PartitionedCallPartitionedCall!tf.__operators__.add_11/AddV2:z:0*
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
GPU 2J 8? *T
fORM
K__inference_activation_26_layer_call_and_return_conditional_losses_220032682
activation_26/PartitionedCall?
.batch_normalization_14/StatefulPartitionedCallStatefulPartitionedCall&activation_26/PartitionedCall:output:0batch_normalization_14_22004897batch_normalization_14_22004899batch_normalization_14_22004901batch_normalization_14_22004903*
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
GPU 2J 8? *]
fXRV
T__inference_batch_normalization_14_layer_call_and_return_conditional_losses_2200403020
.batch_normalization_14/StatefulPartitionedCall?
dropout_30/PartitionedCallPartitionedCall7batch_normalization_14/StatefulPartitionedCall:output:0*
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
GPU 2J 8? *Q
fLRJ
H__inference_dropout_30_layer_call_and_return_conditional_losses_220039942
dropout_30/PartitionedCall?
!conv2d_33/StatefulPartitionedCallStatefulPartitionedCall#dropout_30/PartitionedCall:output:0conv2d_33_22004907conv2d_33_22004909*
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
GPU 2J 8? *P
fKRI
G__inference_conv2d_33_layer_call_and_return_conditional_losses_220033142#
!conv2d_33/StatefulPartitionedCall?
activation_27/PartitionedCallPartitionedCall*conv2d_33/StatefulPartitionedCall:output:0*
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
GPU 2J 8? *T
fORM
K__inference_activation_27_layer_call_and_return_conditional_losses_220033252
activation_27/PartitionedCall?
dropout_31/PartitionedCallPartitionedCall&activation_27/PartitionedCall:output:0*
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
GPU 2J 8? *Q
fLRJ
H__inference_dropout_31_layer_call_and_return_conditional_losses_220039632
dropout_31/PartitionedCall?
!conv2d_34/StatefulPartitionedCallStatefulPartitionedCall#dropout_31/PartitionedCall:output:0conv2d_34_22004914conv2d_34_22004916*
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
GPU 2J 8? *P
fKRI
G__inference_conv2d_34_layer_call_and_return_conditional_losses_220033442#
!conv2d_34/StatefulPartitionedCall?
tf.__operators__.add_12/AddV2AddV2*conv2d_34/StatefulPartitionedCall:output:0#dropout_30/PartitionedCall:output:0*
T0*0
_output_shapes
:??????????2
tf.__operators__.add_12/AddV2?
activation_28/PartitionedCallPartitionedCall!tf.__operators__.add_12/AddV2:z:0*
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
GPU 2J 8? *T
fORM
K__inference_activation_28_layer_call_and_return_conditional_losses_220033562
activation_28/PartitionedCall?
.batch_normalization_15/StatefulPartitionedCallStatefulPartitionedCall&activation_28/PartitionedCall:output:0batch_normalization_15_22004921batch_normalization_15_22004923batch_normalization_15_22004925batch_normalization_15_22004927*
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
GPU 2J 8? *]
fXRV
T__inference_batch_normalization_15_layer_call_and_return_conditional_losses_2200392420
.batch_normalization_15/StatefulPartitionedCall?
dropout_32/PartitionedCallPartitionedCall7batch_normalization_15/StatefulPartitionedCall:output:0*
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
GPU 2J 8? *Q
fLRJ
H__inference_dropout_32_layer_call_and_return_conditional_losses_220038882
dropout_32/PartitionedCall?
!conv2d_35/StatefulPartitionedCallStatefulPartitionedCall#dropout_32/PartitionedCall:output:0conv2d_35_22004931conv2d_35_22004933*
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
GPU 2J 8? *P
fKRI
G__inference_conv2d_35_layer_call_and_return_conditional_losses_220034022#
!conv2d_35/StatefulPartitionedCall?
activation_29/PartitionedCallPartitionedCall*conv2d_35/StatefulPartitionedCall:output:0*
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
GPU 2J 8? *T
fORM
K__inference_activation_29_layer_call_and_return_conditional_losses_220034132
activation_29/PartitionedCall?
dropout_33/PartitionedCallPartitionedCall&activation_29/PartitionedCall:output:0*
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
GPU 2J 8? *Q
fLRJ
H__inference_dropout_33_layer_call_and_return_conditional_losses_220038572
dropout_33/PartitionedCall?
!conv2d_36/StatefulPartitionedCallStatefulPartitionedCall#dropout_33/PartitionedCall:output:0conv2d_36_22004938conv2d_36_22004940*
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
GPU 2J 8? *P
fKRI
G__inference_conv2d_36_layer_call_and_return_conditional_losses_220034322#
!conv2d_36/StatefulPartitionedCall?
tf.__operators__.add_13/AddV2AddV2*conv2d_36/StatefulPartitionedCall:output:0#dropout_32/PartitionedCall:output:0*
T0*0
_output_shapes
:??????????2
tf.__operators__.add_13/AddV2?
activation_30/PartitionedCallPartitionedCall!tf.__operators__.add_13/AddV2:z:0*
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
GPU 2J 8? *T
fORM
K__inference_activation_30_layer_call_and_return_conditional_losses_220034442
activation_30/PartitionedCall?
.batch_normalization_16/StatefulPartitionedCallStatefulPartitionedCall&activation_30/PartitionedCall:output:0batch_normalization_16_22004945batch_normalization_16_22004947batch_normalization_16_22004949batch_normalization_16_22004951*
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
GPU 2J 8? *]
fXRV
T__inference_batch_normalization_16_layer_call_and_return_conditional_losses_2200381820
.batch_normalization_16/StatefulPartitionedCall?
dropout_34/PartitionedCallPartitionedCall7batch_normalization_16/StatefulPartitionedCall:output:0*
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
GPU 2J 8? *Q
fLRJ
H__inference_dropout_34_layer_call_and_return_conditional_losses_220037822
dropout_34/PartitionedCall?
!conv2d_37/StatefulPartitionedCallStatefulPartitionedCall#dropout_34/PartitionedCall:output:0conv2d_37_22004955conv2d_37_22004957*
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
GPU 2J 8? *P
fKRI
G__inference_conv2d_37_layer_call_and_return_conditional_losses_220034902#
!conv2d_37/StatefulPartitionedCall?
activation_31/PartitionedCallPartitionedCall*conv2d_37/StatefulPartitionedCall:output:0*
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
GPU 2J 8? *T
fORM
K__inference_activation_31_layer_call_and_return_conditional_losses_220035012
activation_31/PartitionedCall?
dropout_35/PartitionedCallPartitionedCall&activation_31/PartitionedCall:output:0*
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
GPU 2J 8? *Q
fLRJ
H__inference_dropout_35_layer_call_and_return_conditional_losses_220037512
dropout_35/PartitionedCall?
!conv2d_38/StatefulPartitionedCallStatefulPartitionedCall#dropout_35/PartitionedCall:output:0conv2d_38_22004962conv2d_38_22004964*
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
GPU 2J 8? *P
fKRI
G__inference_conv2d_38_layer_call_and_return_conditional_losses_220035202#
!conv2d_38/StatefulPartitionedCall?
tf.__operators__.add_14/AddV2AddV2*conv2d_38/StatefulPartitionedCall:output:0#dropout_34/PartitionedCall:output:0*
T0*0
_output_shapes
:??????????2
tf.__operators__.add_14/AddV2?
activation_32/PartitionedCallPartitionedCall!tf.__operators__.add_14/AddV2:z:0*
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
GPU 2J 8? *T
fORM
K__inference_activation_32_layer_call_and_return_conditional_losses_220035322
activation_32/PartitionedCall?
.batch_normalization_17/StatefulPartitionedCallStatefulPartitionedCall&activation_32/PartitionedCall:output:0batch_normalization_17_22004969batch_normalization_17_22004971batch_normalization_17_22004973batch_normalization_17_22004975*
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
GPU 2J 8? *]
fXRV
T__inference_batch_normalization_17_layer_call_and_return_conditional_losses_2200371220
.batch_normalization_17/StatefulPartitionedCall?
dropout_36/PartitionedCallPartitionedCall7batch_normalization_17/StatefulPartitionedCall:output:0*
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
GPU 2J 8? *Q
fLRJ
H__inference_dropout_36_layer_call_and_return_conditional_losses_220036762
dropout_36/PartitionedCall?
IdentityIdentity#dropout_36/PartitionedCall:output:0^NoOp*
T0*0
_output_shapes
:??????????2

Identity?
NoOpNoOp/^batch_normalization_12/StatefulPartitionedCall/^batch_normalization_13/StatefulPartitionedCall/^batch_normalization_14/StatefulPartitionedCall/^batch_normalization_15/StatefulPartitionedCall/^batch_normalization_16/StatefulPartitionedCall/^batch_normalization_17/StatefulPartitionedCall"^conv2d_28/StatefulPartitionedCall"^conv2d_29/StatefulPartitionedCall"^conv2d_30/StatefulPartitionedCall"^conv2d_31/StatefulPartitionedCall"^conv2d_32/StatefulPartitionedCall"^conv2d_33/StatefulPartitionedCall"^conv2d_34/StatefulPartitionedCall"^conv2d_35/StatefulPartitionedCall"^conv2d_36/StatefulPartitionedCall"^conv2d_37/StatefulPartitionedCall"^conv2d_38/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapesy
w:?????????: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2`
.batch_normalization_12/StatefulPartitionedCall.batch_normalization_12/StatefulPartitionedCall2`
.batch_normalization_13/StatefulPartitionedCall.batch_normalization_13/StatefulPartitionedCall2`
.batch_normalization_14/StatefulPartitionedCall.batch_normalization_14/StatefulPartitionedCall2`
.batch_normalization_15/StatefulPartitionedCall.batch_normalization_15/StatefulPartitionedCall2`
.batch_normalization_16/StatefulPartitionedCall.batch_normalization_16/StatefulPartitionedCall2`
.batch_normalization_17/StatefulPartitionedCall.batch_normalization_17/StatefulPartitionedCall2F
!conv2d_28/StatefulPartitionedCall!conv2d_28/StatefulPartitionedCall2F
!conv2d_29/StatefulPartitionedCall!conv2d_29/StatefulPartitionedCall2F
!conv2d_30/StatefulPartitionedCall!conv2d_30/StatefulPartitionedCall2F
!conv2d_31/StatefulPartitionedCall!conv2d_31/StatefulPartitionedCall2F
!conv2d_32/StatefulPartitionedCall!conv2d_32/StatefulPartitionedCall2F
!conv2d_33/StatefulPartitionedCall!conv2d_33/StatefulPartitionedCall2F
!conv2d_34/StatefulPartitionedCall!conv2d_34/StatefulPartitionedCall2F
!conv2d_35/StatefulPartitionedCall!conv2d_35/StatefulPartitionedCall2F
!conv2d_36/StatefulPartitionedCall!conv2d_36/StatefulPartitionedCall2F
!conv2d_37/StatefulPartitionedCall!conv2d_37/StatefulPartitionedCall2F
!conv2d_38/StatefulPartitionedCall!conv2d_38/StatefulPartitionedCall:X T
/
_output_shapes
:?????????
!
_user_specified_name	input_5
?
H
,__inference_flatten_3_layer_call_fn_22007407

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
GPU 2J 8? *P
fKRI
G__inference_flatten_3_layer_call_and_return_conditional_losses_220051342
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
?
?
T__inference_batch_normalization_15_layer_call_and_return_conditional_losses_22008280

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
L
0__inference_activation_32_layer_call_fn_22008591

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
GPU 2J 8? *T
fORM
K__inference_activation_32_layer_call_and_return_conditional_losses_220035322
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
f
H__inference_dropout_51_layer_call_and_return_conditional_losses_22005119

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
L
0__inference_activation_30_layer_call_fn_22008371

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
GPU 2J 8? *T
fORM
K__inference_activation_30_layer_call_and_return_conditional_losses_220034442
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
9__inference_batch_normalization_14_layer_call_fn_22007962

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
GPU 2J 8? *]
fXRV
T__inference_batch_normalization_14_layer_call_and_return_conditional_losses_220026262
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
?
d
H__inference_dropout_31_layer_call_and_return_conditional_losses_22008127

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
9__inference_batch_normalization_17_layer_call_fn_22008635

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
GPU 2J 8? *]
fXRV
T__inference_batch_normalization_17_layer_call_and_return_conditional_losses_220035512
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
?f
?
!__inference__traced_save_22008922
file_prefix/
+savev2_conv2d_55_kernel_read_readvariableop-
)savev2_conv2d_55_bias_read_readvariableop/
+savev2_conv2d_53_kernel_read_readvariableop-
)savev2_conv2d_53_bias_read_readvariableop-
)savev2_dense_3_kernel_read_readvariableop/
+savev2_conv2d_54_kernel_read_readvariableop-
)savev2_conv2d_54_bias_read_readvariableop/
+savev2_conv2d_28_kernel_read_readvariableop-
)savev2_conv2d_28_bias_read_readvariableop;
7savev2_batch_normalization_12_gamma_read_readvariableop:
6savev2_batch_normalization_12_beta_read_readvariableopA
=savev2_batch_normalization_12_moving_mean_read_readvariableopE
Asavev2_batch_normalization_12_moving_variance_read_readvariableop/
+savev2_conv2d_29_kernel_read_readvariableop-
)savev2_conv2d_29_bias_read_readvariableop/
+savev2_conv2d_30_kernel_read_readvariableop-
)savev2_conv2d_30_bias_read_readvariableop;
7savev2_batch_normalization_13_gamma_read_readvariableop:
6savev2_batch_normalization_13_beta_read_readvariableopA
=savev2_batch_normalization_13_moving_mean_read_readvariableopE
Asavev2_batch_normalization_13_moving_variance_read_readvariableop/
+savev2_conv2d_31_kernel_read_readvariableop-
)savev2_conv2d_31_bias_read_readvariableop/
+savev2_conv2d_32_kernel_read_readvariableop-
)savev2_conv2d_32_bias_read_readvariableop;
7savev2_batch_normalization_14_gamma_read_readvariableop:
6savev2_batch_normalization_14_beta_read_readvariableopA
=savev2_batch_normalization_14_moving_mean_read_readvariableopE
Asavev2_batch_normalization_14_moving_variance_read_readvariableop/
+savev2_conv2d_33_kernel_read_readvariableop-
)savev2_conv2d_33_bias_read_readvariableop/
+savev2_conv2d_34_kernel_read_readvariableop-
)savev2_conv2d_34_bias_read_readvariableop;
7savev2_batch_normalization_15_gamma_read_readvariableop:
6savev2_batch_normalization_15_beta_read_readvariableopA
=savev2_batch_normalization_15_moving_mean_read_readvariableopE
Asavev2_batch_normalization_15_moving_variance_read_readvariableop/
+savev2_conv2d_35_kernel_read_readvariableop-
)savev2_conv2d_35_bias_read_readvariableop/
+savev2_conv2d_36_kernel_read_readvariableop-
)savev2_conv2d_36_bias_read_readvariableop;
7savev2_batch_normalization_16_gamma_read_readvariableop:
6savev2_batch_normalization_16_beta_read_readvariableopA
=savev2_batch_normalization_16_moving_mean_read_readvariableopE
Asavev2_batch_normalization_16_moving_variance_read_readvariableop/
+savev2_conv2d_37_kernel_read_readvariableop-
)savev2_conv2d_37_bias_read_readvariableop/
+savev2_conv2d_38_kernel_read_readvariableop-
)savev2_conv2d_38_bias_read_readvariableop;
7savev2_batch_normalization_17_gamma_read_readvariableop:
6savev2_batch_normalization_17_beta_read_readvariableopA
=savev2_batch_normalization_17_moving_mean_read_readvariableopE
Asavev2_batch_normalization_17_moving_variance_read_readvariableop
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
ShardedFilename?
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:6*
dtype0*?
value?B?6B6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB&variables/0/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB&variables/3/.ATTRIBUTES/VARIABLE_VALUEB&variables/4/.ATTRIBUTES/VARIABLE_VALUEB&variables/5/.ATTRIBUTES/VARIABLE_VALUEB&variables/6/.ATTRIBUTES/VARIABLE_VALUEB&variables/7/.ATTRIBUTES/VARIABLE_VALUEB&variables/8/.ATTRIBUTES/VARIABLE_VALUEB&variables/9/.ATTRIBUTES/VARIABLE_VALUEB'variables/10/.ATTRIBUTES/VARIABLE_VALUEB'variables/11/.ATTRIBUTES/VARIABLE_VALUEB'variables/12/.ATTRIBUTES/VARIABLE_VALUEB'variables/13/.ATTRIBUTES/VARIABLE_VALUEB'variables/14/.ATTRIBUTES/VARIABLE_VALUEB'variables/15/.ATTRIBUTES/VARIABLE_VALUEB'variables/16/.ATTRIBUTES/VARIABLE_VALUEB'variables/17/.ATTRIBUTES/VARIABLE_VALUEB'variables/18/.ATTRIBUTES/VARIABLE_VALUEB'variables/19/.ATTRIBUTES/VARIABLE_VALUEB'variables/20/.ATTRIBUTES/VARIABLE_VALUEB'variables/21/.ATTRIBUTES/VARIABLE_VALUEB'variables/22/.ATTRIBUTES/VARIABLE_VALUEB'variables/23/.ATTRIBUTES/VARIABLE_VALUEB'variables/24/.ATTRIBUTES/VARIABLE_VALUEB'variables/25/.ATTRIBUTES/VARIABLE_VALUEB'variables/26/.ATTRIBUTES/VARIABLE_VALUEB'variables/27/.ATTRIBUTES/VARIABLE_VALUEB'variables/28/.ATTRIBUTES/VARIABLE_VALUEB'variables/29/.ATTRIBUTES/VARIABLE_VALUEB'variables/30/.ATTRIBUTES/VARIABLE_VALUEB'variables/31/.ATTRIBUTES/VARIABLE_VALUEB'variables/32/.ATTRIBUTES/VARIABLE_VALUEB'variables/33/.ATTRIBUTES/VARIABLE_VALUEB'variables/34/.ATTRIBUTES/VARIABLE_VALUEB'variables/35/.ATTRIBUTES/VARIABLE_VALUEB'variables/36/.ATTRIBUTES/VARIABLE_VALUEB'variables/37/.ATTRIBUTES/VARIABLE_VALUEB'variables/38/.ATTRIBUTES/VARIABLE_VALUEB'variables/39/.ATTRIBUTES/VARIABLE_VALUEB'variables/40/.ATTRIBUTES/VARIABLE_VALUEB'variables/41/.ATTRIBUTES/VARIABLE_VALUEB'variables/42/.ATTRIBUTES/VARIABLE_VALUEB'variables/43/.ATTRIBUTES/VARIABLE_VALUEB'variables/44/.ATTRIBUTES/VARIABLE_VALUEB'variables/45/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2/tensor_names?
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:6*
dtype0*
valuevBt6B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
SaveV2/shape_and_slices?
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0+savev2_conv2d_55_kernel_read_readvariableop)savev2_conv2d_55_bias_read_readvariableop+savev2_conv2d_53_kernel_read_readvariableop)savev2_conv2d_53_bias_read_readvariableop)savev2_dense_3_kernel_read_readvariableop+savev2_conv2d_54_kernel_read_readvariableop)savev2_conv2d_54_bias_read_readvariableop+savev2_conv2d_28_kernel_read_readvariableop)savev2_conv2d_28_bias_read_readvariableop7savev2_batch_normalization_12_gamma_read_readvariableop6savev2_batch_normalization_12_beta_read_readvariableop=savev2_batch_normalization_12_moving_mean_read_readvariableopAsavev2_batch_normalization_12_moving_variance_read_readvariableop+savev2_conv2d_29_kernel_read_readvariableop)savev2_conv2d_29_bias_read_readvariableop+savev2_conv2d_30_kernel_read_readvariableop)savev2_conv2d_30_bias_read_readvariableop7savev2_batch_normalization_13_gamma_read_readvariableop6savev2_batch_normalization_13_beta_read_readvariableop=savev2_batch_normalization_13_moving_mean_read_readvariableopAsavev2_batch_normalization_13_moving_variance_read_readvariableop+savev2_conv2d_31_kernel_read_readvariableop)savev2_conv2d_31_bias_read_readvariableop+savev2_conv2d_32_kernel_read_readvariableop)savev2_conv2d_32_bias_read_readvariableop7savev2_batch_normalization_14_gamma_read_readvariableop6savev2_batch_normalization_14_beta_read_readvariableop=savev2_batch_normalization_14_moving_mean_read_readvariableopAsavev2_batch_normalization_14_moving_variance_read_readvariableop+savev2_conv2d_33_kernel_read_readvariableop)savev2_conv2d_33_bias_read_readvariableop+savev2_conv2d_34_kernel_read_readvariableop)savev2_conv2d_34_bias_read_readvariableop7savev2_batch_normalization_15_gamma_read_readvariableop6savev2_batch_normalization_15_beta_read_readvariableop=savev2_batch_normalization_15_moving_mean_read_readvariableopAsavev2_batch_normalization_15_moving_variance_read_readvariableop+savev2_conv2d_35_kernel_read_readvariableop)savev2_conv2d_35_bias_read_readvariableop+savev2_conv2d_36_kernel_read_readvariableop)savev2_conv2d_36_bias_read_readvariableop7savev2_batch_normalization_16_gamma_read_readvariableop6savev2_batch_normalization_16_beta_read_readvariableop=savev2_batch_normalization_16_moving_mean_read_readvariableopAsavev2_batch_normalization_16_moving_variance_read_readvariableop+savev2_conv2d_37_kernel_read_readvariableop)savev2_conv2d_37_bias_read_readvariableop+savev2_conv2d_38_kernel_read_readvariableop)savev2_conv2d_38_bias_read_readvariableop7savev2_batch_normalization_17_gamma_read_readvariableop6savev2_batch_normalization_17_beta_read_readvariableop=savev2_batch_normalization_17_moving_mean_read_readvariableopAsavev2_batch_normalization_17_moving_variance_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *D
dtypes:
8262
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

identity_1Identity_1:output:0*?
_input_shapes?
?: :?::??:?:@:?I:I:?:?:?:?:?:?:??:?:??:?:?:?:?:?:??:?:??:?:?:?:?:?:??:?:??:?:?:?:?:?:??:?:??:?:?:?:?:?:??:?:??:?:?:?:?:?: 2(
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
:?:.*
(
_output_shapes
:??:!

_output_shapes	
:?:!

_output_shapes	
:?:!

_output_shapes	
:?:!

_output_shapes	
:?:!

_output_shapes	
:?:.*
(
_output_shapes
:??:!

_output_shapes	
:?:.*
(
_output_shapes
:??:!

_output_shapes	
:?:!

_output_shapes	
:?:!

_output_shapes	
:?:!

_output_shapes	
:?:!

_output_shapes	
:?:.*
(
_output_shapes
:??:!

_output_shapes	
:?:. *
(
_output_shapes
:??:!!

_output_shapes	
:?:!"

_output_shapes	
:?:!#

_output_shapes	
:?:!$

_output_shapes	
:?:!%

_output_shapes	
:?:.&*
(
_output_shapes
:??:!'

_output_shapes	
:?:.(*
(
_output_shapes
:??:!)

_output_shapes	
:?:!*

_output_shapes	
:?:!+

_output_shapes	
:?:!,

_output_shapes	
:?:!-

_output_shapes	
:?:..*
(
_output_shapes
:??:!/

_output_shapes	
:?:.0*
(
_output_shapes
:??:!1

_output_shapes	
:?:!2

_output_shapes	
:?:!3

_output_shapes	
:?:!4

_output_shapes	
:?:!5

_output_shapes	
:?:6

_output_shapes
: 
?	
?
9__inference_batch_normalization_15_layer_call_fn_22008208

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
GPU 2J 8? *]
fXRV
T__inference_batch_normalization_15_layer_call_and_return_conditional_losses_220039242
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
?
?
,__inference_conv2d_36_layer_call_fn_22008356

inputs#
unknown:??
	unknown_0:	?
identity??StatefulPartitionedCall?
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
GPU 2J 8? *P
fKRI
G__inference_conv2d_36_layer_call_and_return_conditional_losses_220034322
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
?
?
T__inference_batch_normalization_16_layer_call_and_return_conditional_losses_22002834

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
?
9__inference_batch_normalization_17_layer_call_fn_22008648

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
GPU 2J 8? *]
fXRV
T__inference_batch_normalization_17_layer_call_and_return_conditional_losses_220037122
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
?
?
,__inference_conv2d_55_layer_call_fn_22007352

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
GPU 2J 8? *P
fKRI
G__inference_conv2d_55_layer_call_and_return_conditional_losses_220050912
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
?
g
K__inference_activation_29_layer_call_and_return_conditional_losses_22003413

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
T__inference_batch_normalization_13_layer_call_and_return_conditional_losses_22002456

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
?
9__inference_batch_normalization_15_layer_call_fn_22008195

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
GPU 2J 8? *]
fXRV
T__inference_batch_normalization_15_layer_call_and_return_conditional_losses_220033752
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
G__inference_conv2d_54_layer_call_and_return_conditional_losses_22007467

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
?
L
0__inference_activation_28_layer_call_fn_22008151

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
GPU 2J 8? *T
fORM
K__inference_activation_28_layer_call_and_return_conditional_losses_220033562
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
g
K__inference_activation_22_layer_call_and_return_conditional_losses_22007620

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
?
d
H__inference_dropout_34_layer_call_and_return_conditional_losses_22003782

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
?
G__inference_conv2d_38_layer_call_and_return_conditional_losses_22008586

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
g
K__inference_activation_22_layer_call_and_return_conditional_losses_22003119

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
?
g
K__inference_activation_29_layer_call_and_return_conditional_losses_22008328

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
?
d
H__inference_dropout_27_layer_call_and_return_conditional_losses_22004175

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
f
H__inference_dropout_31_layer_call_and_return_conditional_losses_22008123

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
?
?
T__inference_batch_normalization_17_layer_call_and_return_conditional_losses_22008720

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
I
-__inference_dropout_29_layer_call_fn_22007898

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
GPU 2J 8? *Q
fLRJ
H__inference_dropout_29_layer_call_and_return_conditional_losses_220040692
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
?
?
&__inference_signature_wrapper_22006163
input_8"
unknown:?
	unknown_0:	?
	unknown_1:	?
	unknown_2:	?
	unknown_3:	?
	unknown_4:	?%
	unknown_5:??
	unknown_6:	?%
	unknown_7:??
	unknown_8:	?
	unknown_9:	?

unknown_10:	?

unknown_11:	?

unknown_12:	?&

unknown_13:??

unknown_14:	?&

unknown_15:??

unknown_16:	?

unknown_17:	?

unknown_18:	?

unknown_19:	?

unknown_20:	?&

unknown_21:??

unknown_22:	?&

unknown_23:??

unknown_24:	?

unknown_25:	?

unknown_26:	?

unknown_27:	?

unknown_28:	?&

unknown_29:??

unknown_30:	?&

unknown_31:??

unknown_32:	?

unknown_33:	?

unknown_34:	?

unknown_35:	?

unknown_36:	?&

unknown_37:??

unknown_38:	?&

unknown_39:??

unknown_40:	?

unknown_41:	?

unknown_42:	?

unknown_43:	?

unknown_44:	?%

unknown_45:?

unknown_46:&

unknown_47:??

unknown_48:	?%

unknown_49:?I

unknown_50:I

unknown_51:@
identity

identity_1??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_8unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
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
unknown_23
unknown_24
unknown_25
unknown_26
unknown_27
unknown_28
unknown_29
unknown_30
unknown_31
unknown_32
unknown_33
unknown_34
unknown_35
unknown_36
unknown_37
unknown_38
unknown_39
unknown_40
unknown_41
unknown_42
unknown_43
unknown_44
unknown_45
unknown_46
unknown_47
unknown_48
unknown_49
unknown_50
unknown_51*A
Tin:
826*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:?????????I:?????????*W
_read_only_resource_inputs9
75	
 !"#$%&'()*+,-./012345*-
config_proto

CPU

GPU 2J 8? *,
f'R%
#__inference__wrapped_model_220023082
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
_construction_contextkEagerRuntime*?
_input_shapes?
?:?????????: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
/
_output_shapes
:?????????
!
_user_specified_name	input_8
?
?
G__inference_conv2d_34_layer_call_and_return_conditional_losses_22003344

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
?
?
T__inference_batch_normalization_13_layer_call_and_return_conditional_losses_22007804

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
?
?
T__inference_batch_normalization_15_layer_call_and_return_conditional_losses_22003924

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
?
?
T__inference_batch_normalization_14_layer_call_and_return_conditional_losses_22002626

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
G__inference_conv2d_31_layer_call_and_return_conditional_losses_22003226

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
??
?8
E__inference_model_7_layer_call_and_return_conditional_losses_22006798

inputsK
0model_4_conv2d_28_conv2d_readvariableop_resource:?@
1model_4_conv2d_28_biasadd_readvariableop_resource:	?E
6model_4_batch_normalization_12_readvariableop_resource:	?G
8model_4_batch_normalization_12_readvariableop_1_resource:	?V
Gmodel_4_batch_normalization_12_fusedbatchnormv3_readvariableop_resource:	?X
Imodel_4_batch_normalization_12_fusedbatchnormv3_readvariableop_1_resource:	?L
0model_4_conv2d_29_conv2d_readvariableop_resource:??@
1model_4_conv2d_29_biasadd_readvariableop_resource:	?L
0model_4_conv2d_30_conv2d_readvariableop_resource:??@
1model_4_conv2d_30_biasadd_readvariableop_resource:	?E
6model_4_batch_normalization_13_readvariableop_resource:	?G
8model_4_batch_normalization_13_readvariableop_1_resource:	?V
Gmodel_4_batch_normalization_13_fusedbatchnormv3_readvariableop_resource:	?X
Imodel_4_batch_normalization_13_fusedbatchnormv3_readvariableop_1_resource:	?L
0model_4_conv2d_31_conv2d_readvariableop_resource:??@
1model_4_conv2d_31_biasadd_readvariableop_resource:	?L
0model_4_conv2d_32_conv2d_readvariableop_resource:??@
1model_4_conv2d_32_biasadd_readvariableop_resource:	?E
6model_4_batch_normalization_14_readvariableop_resource:	?G
8model_4_batch_normalization_14_readvariableop_1_resource:	?V
Gmodel_4_batch_normalization_14_fusedbatchnormv3_readvariableop_resource:	?X
Imodel_4_batch_normalization_14_fusedbatchnormv3_readvariableop_1_resource:	?L
0model_4_conv2d_33_conv2d_readvariableop_resource:??@
1model_4_conv2d_33_biasadd_readvariableop_resource:	?L
0model_4_conv2d_34_conv2d_readvariableop_resource:??@
1model_4_conv2d_34_biasadd_readvariableop_resource:	?E
6model_4_batch_normalization_15_readvariableop_resource:	?G
8model_4_batch_normalization_15_readvariableop_1_resource:	?V
Gmodel_4_batch_normalization_15_fusedbatchnormv3_readvariableop_resource:	?X
Imodel_4_batch_normalization_15_fusedbatchnormv3_readvariableop_1_resource:	?L
0model_4_conv2d_35_conv2d_readvariableop_resource:??@
1model_4_conv2d_35_biasadd_readvariableop_resource:	?L
0model_4_conv2d_36_conv2d_readvariableop_resource:??@
1model_4_conv2d_36_biasadd_readvariableop_resource:	?E
6model_4_batch_normalization_16_readvariableop_resource:	?G
8model_4_batch_normalization_16_readvariableop_1_resource:	?V
Gmodel_4_batch_normalization_16_fusedbatchnormv3_readvariableop_resource:	?X
Imodel_4_batch_normalization_16_fusedbatchnormv3_readvariableop_1_resource:	?L
0model_4_conv2d_37_conv2d_readvariableop_resource:??@
1model_4_conv2d_37_biasadd_readvariableop_resource:	?L
0model_4_conv2d_38_conv2d_readvariableop_resource:??@
1model_4_conv2d_38_biasadd_readvariableop_resource:	?E
6model_4_batch_normalization_17_readvariableop_resource:	?G
8model_4_batch_normalization_17_readvariableop_1_resource:	?V
Gmodel_4_batch_normalization_17_fusedbatchnormv3_readvariableop_resource:	?X
Imodel_4_batch_normalization_17_fusedbatchnormv3_readvariableop_1_resource:	?C
(conv2d_55_conv2d_readvariableop_resource:?7
)conv2d_55_biasadd_readvariableop_resource:D
(conv2d_53_conv2d_readvariableop_resource:??8
)conv2d_53_biasadd_readvariableop_resource:	?C
(conv2d_54_conv2d_readvariableop_resource:?I7
)conv2d_54_biasadd_readvariableop_resource:I8
&dense_3_matmul_readvariableop_resource:@
identity

identity_1?? conv2d_53/BiasAdd/ReadVariableOp?conv2d_53/Conv2D/ReadVariableOp? conv2d_54/BiasAdd/ReadVariableOp?conv2d_54/Conv2D/ReadVariableOp? conv2d_55/BiasAdd/ReadVariableOp?conv2d_55/Conv2D/ReadVariableOp?dense_3/MatMul/ReadVariableOp?-model_4/batch_normalization_12/AssignNewValue?/model_4/batch_normalization_12/AssignNewValue_1?>model_4/batch_normalization_12/FusedBatchNormV3/ReadVariableOp?@model_4/batch_normalization_12/FusedBatchNormV3/ReadVariableOp_1?-model_4/batch_normalization_12/ReadVariableOp?/model_4/batch_normalization_12/ReadVariableOp_1?-model_4/batch_normalization_13/AssignNewValue?/model_4/batch_normalization_13/AssignNewValue_1?>model_4/batch_normalization_13/FusedBatchNormV3/ReadVariableOp?@model_4/batch_normalization_13/FusedBatchNormV3/ReadVariableOp_1?-model_4/batch_normalization_13/ReadVariableOp?/model_4/batch_normalization_13/ReadVariableOp_1?-model_4/batch_normalization_14/AssignNewValue?/model_4/batch_normalization_14/AssignNewValue_1?>model_4/batch_normalization_14/FusedBatchNormV3/ReadVariableOp?@model_4/batch_normalization_14/FusedBatchNormV3/ReadVariableOp_1?-model_4/batch_normalization_14/ReadVariableOp?/model_4/batch_normalization_14/ReadVariableOp_1?-model_4/batch_normalization_15/AssignNewValue?/model_4/batch_normalization_15/AssignNewValue_1?>model_4/batch_normalization_15/FusedBatchNormV3/ReadVariableOp?@model_4/batch_normalization_15/FusedBatchNormV3/ReadVariableOp_1?-model_4/batch_normalization_15/ReadVariableOp?/model_4/batch_normalization_15/ReadVariableOp_1?-model_4/batch_normalization_16/AssignNewValue?/model_4/batch_normalization_16/AssignNewValue_1?>model_4/batch_normalization_16/FusedBatchNormV3/ReadVariableOp?@model_4/batch_normalization_16/FusedBatchNormV3/ReadVariableOp_1?-model_4/batch_normalization_16/ReadVariableOp?/model_4/batch_normalization_16/ReadVariableOp_1?-model_4/batch_normalization_17/AssignNewValue?/model_4/batch_normalization_17/AssignNewValue_1?>model_4/batch_normalization_17/FusedBatchNormV3/ReadVariableOp?@model_4/batch_normalization_17/FusedBatchNormV3/ReadVariableOp_1?-model_4/batch_normalization_17/ReadVariableOp?/model_4/batch_normalization_17/ReadVariableOp_1?(model_4/conv2d_28/BiasAdd/ReadVariableOp?'model_4/conv2d_28/Conv2D/ReadVariableOp?(model_4/conv2d_29/BiasAdd/ReadVariableOp?'model_4/conv2d_29/Conv2D/ReadVariableOp?(model_4/conv2d_30/BiasAdd/ReadVariableOp?'model_4/conv2d_30/Conv2D/ReadVariableOp?(model_4/conv2d_31/BiasAdd/ReadVariableOp?'model_4/conv2d_31/Conv2D/ReadVariableOp?(model_4/conv2d_32/BiasAdd/ReadVariableOp?'model_4/conv2d_32/Conv2D/ReadVariableOp?(model_4/conv2d_33/BiasAdd/ReadVariableOp?'model_4/conv2d_33/Conv2D/ReadVariableOp?(model_4/conv2d_34/BiasAdd/ReadVariableOp?'model_4/conv2d_34/Conv2D/ReadVariableOp?(model_4/conv2d_35/BiasAdd/ReadVariableOp?'model_4/conv2d_35/Conv2D/ReadVariableOp?(model_4/conv2d_36/BiasAdd/ReadVariableOp?'model_4/conv2d_36/Conv2D/ReadVariableOp?(model_4/conv2d_37/BiasAdd/ReadVariableOp?'model_4/conv2d_37/Conv2D/ReadVariableOp?(model_4/conv2d_38/BiasAdd/ReadVariableOp?'model_4/conv2d_38/Conv2D/ReadVariableOp?
'model_4/conv2d_28/Conv2D/ReadVariableOpReadVariableOp0model_4_conv2d_28_conv2d_readvariableop_resource*'
_output_shapes
:?*
dtype02)
'model_4/conv2d_28/Conv2D/ReadVariableOp?
model_4/conv2d_28/Conv2DConv2Dinputs/model_4/conv2d_28/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????*
paddingSAME*
strides
2
model_4/conv2d_28/Conv2D?
(model_4/conv2d_28/BiasAdd/ReadVariableOpReadVariableOp1model_4_conv2d_28_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02*
(model_4/conv2d_28/BiasAdd/ReadVariableOp?
model_4/conv2d_28/BiasAddBiasAdd!model_4/conv2d_28/Conv2D:output:00model_4/conv2d_28/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????2
model_4/conv2d_28/BiasAdd?
-model_4/batch_normalization_12/ReadVariableOpReadVariableOp6model_4_batch_normalization_12_readvariableop_resource*
_output_shapes	
:?*
dtype02/
-model_4/batch_normalization_12/ReadVariableOp?
/model_4/batch_normalization_12/ReadVariableOp_1ReadVariableOp8model_4_batch_normalization_12_readvariableop_1_resource*
_output_shapes	
:?*
dtype021
/model_4/batch_normalization_12/ReadVariableOp_1?
>model_4/batch_normalization_12/FusedBatchNormV3/ReadVariableOpReadVariableOpGmodel_4_batch_normalization_12_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:?*
dtype02@
>model_4/batch_normalization_12/FusedBatchNormV3/ReadVariableOp?
@model_4/batch_normalization_12/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpImodel_4_batch_normalization_12_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:?*
dtype02B
@model_4/batch_normalization_12/FusedBatchNormV3/ReadVariableOp_1?
/model_4/batch_normalization_12/FusedBatchNormV3FusedBatchNormV3"model_4/conv2d_28/BiasAdd:output:05model_4/batch_normalization_12/ReadVariableOp:value:07model_4/batch_normalization_12/ReadVariableOp_1:value:0Fmodel_4/batch_normalization_12/FusedBatchNormV3/ReadVariableOp:value:0Hmodel_4/batch_normalization_12/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:??????????:?:?:?:?:*
epsilon%o?:*
exponential_avg_factor%
?#<21
/model_4/batch_normalization_12/FusedBatchNormV3?
-model_4/batch_normalization_12/AssignNewValueAssignVariableOpGmodel_4_batch_normalization_12_fusedbatchnormv3_readvariableop_resource<model_4/batch_normalization_12/FusedBatchNormV3:batch_mean:0?^model_4/batch_normalization_12/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype02/
-model_4/batch_normalization_12/AssignNewValue?
/model_4/batch_normalization_12/AssignNewValue_1AssignVariableOpImodel_4_batch_normalization_12_fusedbatchnormv3_readvariableop_1_resource@model_4/batch_normalization_12/FusedBatchNormV3:batch_variance:0A^model_4/batch_normalization_12/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype021
/model_4/batch_normalization_12/AssignNewValue_1?
model_4/activation_22/ReluRelu3model_4/batch_normalization_12/FusedBatchNormV3:y:0*
T0*0
_output_shapes
:??????????2
model_4/activation_22/Relu?
'model_4/conv2d_29/Conv2D/ReadVariableOpReadVariableOp0model_4_conv2d_29_conv2d_readvariableop_resource*(
_output_shapes
:??*
dtype02)
'model_4/conv2d_29/Conv2D/ReadVariableOp?
model_4/conv2d_29/Conv2DConv2D(model_4/activation_22/Relu:activations:0/model_4/conv2d_29/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????*
paddingSAME*
strides
2
model_4/conv2d_29/Conv2D?
(model_4/conv2d_29/BiasAdd/ReadVariableOpReadVariableOp1model_4_conv2d_29_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02*
(model_4/conv2d_29/BiasAdd/ReadVariableOp?
model_4/conv2d_29/BiasAddBiasAdd!model_4/conv2d_29/Conv2D:output:00model_4/conv2d_29/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????2
model_4/conv2d_29/BiasAdd?
model_4/activation_23/ReluRelu"model_4/conv2d_29/BiasAdd:output:0*
T0*0
_output_shapes
:??????????2
model_4/activation_23/Relu?
'model_4/conv2d_30/Conv2D/ReadVariableOpReadVariableOp0model_4_conv2d_30_conv2d_readvariableop_resource*(
_output_shapes
:??*
dtype02)
'model_4/conv2d_30/Conv2D/ReadVariableOp?
model_4/conv2d_30/Conv2DConv2D(model_4/activation_23/Relu:activations:0/model_4/conv2d_30/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????*
paddingSAME*
strides
2
model_4/conv2d_30/Conv2D?
(model_4/conv2d_30/BiasAdd/ReadVariableOpReadVariableOp1model_4_conv2d_30_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02*
(model_4/conv2d_30/BiasAdd/ReadVariableOp?
model_4/conv2d_30/BiasAddBiasAdd!model_4/conv2d_30/Conv2D:output:00model_4/conv2d_30/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????2
model_4/conv2d_30/BiasAdd?
%model_4/tf.__operators__.add_10/AddV2AddV2"model_4/conv2d_30/BiasAdd:output:0(model_4/activation_22/Relu:activations:0*
T0*0
_output_shapes
:??????????2'
%model_4/tf.__operators__.add_10/AddV2?
model_4/activation_24/ReluRelu)model_4/tf.__operators__.add_10/AddV2:z:0*
T0*0
_output_shapes
:??????????2
model_4/activation_24/Relu?
-model_4/batch_normalization_13/ReadVariableOpReadVariableOp6model_4_batch_normalization_13_readvariableop_resource*
_output_shapes	
:?*
dtype02/
-model_4/batch_normalization_13/ReadVariableOp?
/model_4/batch_normalization_13/ReadVariableOp_1ReadVariableOp8model_4_batch_normalization_13_readvariableop_1_resource*
_output_shapes	
:?*
dtype021
/model_4/batch_normalization_13/ReadVariableOp_1?
>model_4/batch_normalization_13/FusedBatchNormV3/ReadVariableOpReadVariableOpGmodel_4_batch_normalization_13_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:?*
dtype02@
>model_4/batch_normalization_13/FusedBatchNormV3/ReadVariableOp?
@model_4/batch_normalization_13/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpImodel_4_batch_normalization_13_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:?*
dtype02B
@model_4/batch_normalization_13/FusedBatchNormV3/ReadVariableOp_1?
/model_4/batch_normalization_13/FusedBatchNormV3FusedBatchNormV3(model_4/activation_24/Relu:activations:05model_4/batch_normalization_13/ReadVariableOp:value:07model_4/batch_normalization_13/ReadVariableOp_1:value:0Fmodel_4/batch_normalization_13/FusedBatchNormV3/ReadVariableOp:value:0Hmodel_4/batch_normalization_13/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:??????????:?:?:?:?:*
epsilon%o?:*
exponential_avg_factor%
?#<21
/model_4/batch_normalization_13/FusedBatchNormV3?
-model_4/batch_normalization_13/AssignNewValueAssignVariableOpGmodel_4_batch_normalization_13_fusedbatchnormv3_readvariableop_resource<model_4/batch_normalization_13/FusedBatchNormV3:batch_mean:0?^model_4/batch_normalization_13/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype02/
-model_4/batch_normalization_13/AssignNewValue?
/model_4/batch_normalization_13/AssignNewValue_1AssignVariableOpImodel_4_batch_normalization_13_fusedbatchnormv3_readvariableop_1_resource@model_4/batch_normalization_13/FusedBatchNormV3:batch_variance:0A^model_4/batch_normalization_13/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype021
/model_4/batch_normalization_13/AssignNewValue_1?
'model_4/conv2d_31/Conv2D/ReadVariableOpReadVariableOp0model_4_conv2d_31_conv2d_readvariableop_resource*(
_output_shapes
:??*
dtype02)
'model_4/conv2d_31/Conv2D/ReadVariableOp?
model_4/conv2d_31/Conv2DConv2D3model_4/batch_normalization_13/FusedBatchNormV3:y:0/model_4/conv2d_31/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????*
paddingSAME*
strides
2
model_4/conv2d_31/Conv2D?
(model_4/conv2d_31/BiasAdd/ReadVariableOpReadVariableOp1model_4_conv2d_31_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02*
(model_4/conv2d_31/BiasAdd/ReadVariableOp?
model_4/conv2d_31/BiasAddBiasAdd!model_4/conv2d_31/Conv2D:output:00model_4/conv2d_31/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????2
model_4/conv2d_31/BiasAdd?
model_4/activation_25/ReluRelu"model_4/conv2d_31/BiasAdd:output:0*
T0*0
_output_shapes
:??????????2
model_4/activation_25/Relu?
'model_4/conv2d_32/Conv2D/ReadVariableOpReadVariableOp0model_4_conv2d_32_conv2d_readvariableop_resource*(
_output_shapes
:??*
dtype02)
'model_4/conv2d_32/Conv2D/ReadVariableOp?
model_4/conv2d_32/Conv2DConv2D(model_4/activation_25/Relu:activations:0/model_4/conv2d_32/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????*
paddingSAME*
strides
2
model_4/conv2d_32/Conv2D?
(model_4/conv2d_32/BiasAdd/ReadVariableOpReadVariableOp1model_4_conv2d_32_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02*
(model_4/conv2d_32/BiasAdd/ReadVariableOp?
model_4/conv2d_32/BiasAddBiasAdd!model_4/conv2d_32/Conv2D:output:00model_4/conv2d_32/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????2
model_4/conv2d_32/BiasAdd?
%model_4/tf.__operators__.add_11/AddV2AddV2"model_4/conv2d_32/BiasAdd:output:03model_4/batch_normalization_13/FusedBatchNormV3:y:0*
T0*0
_output_shapes
:??????????2'
%model_4/tf.__operators__.add_11/AddV2?
model_4/activation_26/ReluRelu)model_4/tf.__operators__.add_11/AddV2:z:0*
T0*0
_output_shapes
:??????????2
model_4/activation_26/Relu?
-model_4/batch_normalization_14/ReadVariableOpReadVariableOp6model_4_batch_normalization_14_readvariableop_resource*
_output_shapes	
:?*
dtype02/
-model_4/batch_normalization_14/ReadVariableOp?
/model_4/batch_normalization_14/ReadVariableOp_1ReadVariableOp8model_4_batch_normalization_14_readvariableop_1_resource*
_output_shapes	
:?*
dtype021
/model_4/batch_normalization_14/ReadVariableOp_1?
>model_4/batch_normalization_14/FusedBatchNormV3/ReadVariableOpReadVariableOpGmodel_4_batch_normalization_14_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:?*
dtype02@
>model_4/batch_normalization_14/FusedBatchNormV3/ReadVariableOp?
@model_4/batch_normalization_14/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpImodel_4_batch_normalization_14_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:?*
dtype02B
@model_4/batch_normalization_14/FusedBatchNormV3/ReadVariableOp_1?
/model_4/batch_normalization_14/FusedBatchNormV3FusedBatchNormV3(model_4/activation_26/Relu:activations:05model_4/batch_normalization_14/ReadVariableOp:value:07model_4/batch_normalization_14/ReadVariableOp_1:value:0Fmodel_4/batch_normalization_14/FusedBatchNormV3/ReadVariableOp:value:0Hmodel_4/batch_normalization_14/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:??????????:?:?:?:?:*
epsilon%o?:*
exponential_avg_factor%
?#<21
/model_4/batch_normalization_14/FusedBatchNormV3?
-model_4/batch_normalization_14/AssignNewValueAssignVariableOpGmodel_4_batch_normalization_14_fusedbatchnormv3_readvariableop_resource<model_4/batch_normalization_14/FusedBatchNormV3:batch_mean:0?^model_4/batch_normalization_14/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype02/
-model_4/batch_normalization_14/AssignNewValue?
/model_4/batch_normalization_14/AssignNewValue_1AssignVariableOpImodel_4_batch_normalization_14_fusedbatchnormv3_readvariableop_1_resource@model_4/batch_normalization_14/FusedBatchNormV3:batch_variance:0A^model_4/batch_normalization_14/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype021
/model_4/batch_normalization_14/AssignNewValue_1?
'model_4/conv2d_33/Conv2D/ReadVariableOpReadVariableOp0model_4_conv2d_33_conv2d_readvariableop_resource*(
_output_shapes
:??*
dtype02)
'model_4/conv2d_33/Conv2D/ReadVariableOp?
model_4/conv2d_33/Conv2DConv2D3model_4/batch_normalization_14/FusedBatchNormV3:y:0/model_4/conv2d_33/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????*
paddingSAME*
strides
2
model_4/conv2d_33/Conv2D?
(model_4/conv2d_33/BiasAdd/ReadVariableOpReadVariableOp1model_4_conv2d_33_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02*
(model_4/conv2d_33/BiasAdd/ReadVariableOp?
model_4/conv2d_33/BiasAddBiasAdd!model_4/conv2d_33/Conv2D:output:00model_4/conv2d_33/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????2
model_4/conv2d_33/BiasAdd?
model_4/activation_27/ReluRelu"model_4/conv2d_33/BiasAdd:output:0*
T0*0
_output_shapes
:??????????2
model_4/activation_27/Relu?
'model_4/conv2d_34/Conv2D/ReadVariableOpReadVariableOp0model_4_conv2d_34_conv2d_readvariableop_resource*(
_output_shapes
:??*
dtype02)
'model_4/conv2d_34/Conv2D/ReadVariableOp?
model_4/conv2d_34/Conv2DConv2D(model_4/activation_27/Relu:activations:0/model_4/conv2d_34/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????*
paddingSAME*
strides
2
model_4/conv2d_34/Conv2D?
(model_4/conv2d_34/BiasAdd/ReadVariableOpReadVariableOp1model_4_conv2d_34_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02*
(model_4/conv2d_34/BiasAdd/ReadVariableOp?
model_4/conv2d_34/BiasAddBiasAdd!model_4/conv2d_34/Conv2D:output:00model_4/conv2d_34/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????2
model_4/conv2d_34/BiasAdd?
%model_4/tf.__operators__.add_12/AddV2AddV2"model_4/conv2d_34/BiasAdd:output:03model_4/batch_normalization_14/FusedBatchNormV3:y:0*
T0*0
_output_shapes
:??????????2'
%model_4/tf.__operators__.add_12/AddV2?
model_4/activation_28/ReluRelu)model_4/tf.__operators__.add_12/AddV2:z:0*
T0*0
_output_shapes
:??????????2
model_4/activation_28/Relu?
-model_4/batch_normalization_15/ReadVariableOpReadVariableOp6model_4_batch_normalization_15_readvariableop_resource*
_output_shapes	
:?*
dtype02/
-model_4/batch_normalization_15/ReadVariableOp?
/model_4/batch_normalization_15/ReadVariableOp_1ReadVariableOp8model_4_batch_normalization_15_readvariableop_1_resource*
_output_shapes	
:?*
dtype021
/model_4/batch_normalization_15/ReadVariableOp_1?
>model_4/batch_normalization_15/FusedBatchNormV3/ReadVariableOpReadVariableOpGmodel_4_batch_normalization_15_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:?*
dtype02@
>model_4/batch_normalization_15/FusedBatchNormV3/ReadVariableOp?
@model_4/batch_normalization_15/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpImodel_4_batch_normalization_15_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:?*
dtype02B
@model_4/batch_normalization_15/FusedBatchNormV3/ReadVariableOp_1?
/model_4/batch_normalization_15/FusedBatchNormV3FusedBatchNormV3(model_4/activation_28/Relu:activations:05model_4/batch_normalization_15/ReadVariableOp:value:07model_4/batch_normalization_15/ReadVariableOp_1:value:0Fmodel_4/batch_normalization_15/FusedBatchNormV3/ReadVariableOp:value:0Hmodel_4/batch_normalization_15/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:??????????:?:?:?:?:*
epsilon%o?:*
exponential_avg_factor%
?#<21
/model_4/batch_normalization_15/FusedBatchNormV3?
-model_4/batch_normalization_15/AssignNewValueAssignVariableOpGmodel_4_batch_normalization_15_fusedbatchnormv3_readvariableop_resource<model_4/batch_normalization_15/FusedBatchNormV3:batch_mean:0?^model_4/batch_normalization_15/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype02/
-model_4/batch_normalization_15/AssignNewValue?
/model_4/batch_normalization_15/AssignNewValue_1AssignVariableOpImodel_4_batch_normalization_15_fusedbatchnormv3_readvariableop_1_resource@model_4/batch_normalization_15/FusedBatchNormV3:batch_variance:0A^model_4/batch_normalization_15/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype021
/model_4/batch_normalization_15/AssignNewValue_1?
'model_4/conv2d_35/Conv2D/ReadVariableOpReadVariableOp0model_4_conv2d_35_conv2d_readvariableop_resource*(
_output_shapes
:??*
dtype02)
'model_4/conv2d_35/Conv2D/ReadVariableOp?
model_4/conv2d_35/Conv2DConv2D3model_4/batch_normalization_15/FusedBatchNormV3:y:0/model_4/conv2d_35/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????*
paddingSAME*
strides
2
model_4/conv2d_35/Conv2D?
(model_4/conv2d_35/BiasAdd/ReadVariableOpReadVariableOp1model_4_conv2d_35_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02*
(model_4/conv2d_35/BiasAdd/ReadVariableOp?
model_4/conv2d_35/BiasAddBiasAdd!model_4/conv2d_35/Conv2D:output:00model_4/conv2d_35/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????2
model_4/conv2d_35/BiasAdd?
model_4/activation_29/ReluRelu"model_4/conv2d_35/BiasAdd:output:0*
T0*0
_output_shapes
:??????????2
model_4/activation_29/Relu?
'model_4/conv2d_36/Conv2D/ReadVariableOpReadVariableOp0model_4_conv2d_36_conv2d_readvariableop_resource*(
_output_shapes
:??*
dtype02)
'model_4/conv2d_36/Conv2D/ReadVariableOp?
model_4/conv2d_36/Conv2DConv2D(model_4/activation_29/Relu:activations:0/model_4/conv2d_36/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????*
paddingSAME*
strides
2
model_4/conv2d_36/Conv2D?
(model_4/conv2d_36/BiasAdd/ReadVariableOpReadVariableOp1model_4_conv2d_36_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02*
(model_4/conv2d_36/BiasAdd/ReadVariableOp?
model_4/conv2d_36/BiasAddBiasAdd!model_4/conv2d_36/Conv2D:output:00model_4/conv2d_36/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????2
model_4/conv2d_36/BiasAdd?
%model_4/tf.__operators__.add_13/AddV2AddV2"model_4/conv2d_36/BiasAdd:output:03model_4/batch_normalization_15/FusedBatchNormV3:y:0*
T0*0
_output_shapes
:??????????2'
%model_4/tf.__operators__.add_13/AddV2?
model_4/activation_30/ReluRelu)model_4/tf.__operators__.add_13/AddV2:z:0*
T0*0
_output_shapes
:??????????2
model_4/activation_30/Relu?
-model_4/batch_normalization_16/ReadVariableOpReadVariableOp6model_4_batch_normalization_16_readvariableop_resource*
_output_shapes	
:?*
dtype02/
-model_4/batch_normalization_16/ReadVariableOp?
/model_4/batch_normalization_16/ReadVariableOp_1ReadVariableOp8model_4_batch_normalization_16_readvariableop_1_resource*
_output_shapes	
:?*
dtype021
/model_4/batch_normalization_16/ReadVariableOp_1?
>model_4/batch_normalization_16/FusedBatchNormV3/ReadVariableOpReadVariableOpGmodel_4_batch_normalization_16_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:?*
dtype02@
>model_4/batch_normalization_16/FusedBatchNormV3/ReadVariableOp?
@model_4/batch_normalization_16/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpImodel_4_batch_normalization_16_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:?*
dtype02B
@model_4/batch_normalization_16/FusedBatchNormV3/ReadVariableOp_1?
/model_4/batch_normalization_16/FusedBatchNormV3FusedBatchNormV3(model_4/activation_30/Relu:activations:05model_4/batch_normalization_16/ReadVariableOp:value:07model_4/batch_normalization_16/ReadVariableOp_1:value:0Fmodel_4/batch_normalization_16/FusedBatchNormV3/ReadVariableOp:value:0Hmodel_4/batch_normalization_16/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:??????????:?:?:?:?:*
epsilon%o?:*
exponential_avg_factor%
?#<21
/model_4/batch_normalization_16/FusedBatchNormV3?
-model_4/batch_normalization_16/AssignNewValueAssignVariableOpGmodel_4_batch_normalization_16_fusedbatchnormv3_readvariableop_resource<model_4/batch_normalization_16/FusedBatchNormV3:batch_mean:0?^model_4/batch_normalization_16/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype02/
-model_4/batch_normalization_16/AssignNewValue?
/model_4/batch_normalization_16/AssignNewValue_1AssignVariableOpImodel_4_batch_normalization_16_fusedbatchnormv3_readvariableop_1_resource@model_4/batch_normalization_16/FusedBatchNormV3:batch_variance:0A^model_4/batch_normalization_16/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype021
/model_4/batch_normalization_16/AssignNewValue_1?
'model_4/conv2d_37/Conv2D/ReadVariableOpReadVariableOp0model_4_conv2d_37_conv2d_readvariableop_resource*(
_output_shapes
:??*
dtype02)
'model_4/conv2d_37/Conv2D/ReadVariableOp?
model_4/conv2d_37/Conv2DConv2D3model_4/batch_normalization_16/FusedBatchNormV3:y:0/model_4/conv2d_37/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????*
paddingSAME*
strides
2
model_4/conv2d_37/Conv2D?
(model_4/conv2d_37/BiasAdd/ReadVariableOpReadVariableOp1model_4_conv2d_37_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02*
(model_4/conv2d_37/BiasAdd/ReadVariableOp?
model_4/conv2d_37/BiasAddBiasAdd!model_4/conv2d_37/Conv2D:output:00model_4/conv2d_37/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????2
model_4/conv2d_37/BiasAdd?
model_4/activation_31/ReluRelu"model_4/conv2d_37/BiasAdd:output:0*
T0*0
_output_shapes
:??????????2
model_4/activation_31/Relu?
'model_4/conv2d_38/Conv2D/ReadVariableOpReadVariableOp0model_4_conv2d_38_conv2d_readvariableop_resource*(
_output_shapes
:??*
dtype02)
'model_4/conv2d_38/Conv2D/ReadVariableOp?
model_4/conv2d_38/Conv2DConv2D(model_4/activation_31/Relu:activations:0/model_4/conv2d_38/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????*
paddingSAME*
strides
2
model_4/conv2d_38/Conv2D?
(model_4/conv2d_38/BiasAdd/ReadVariableOpReadVariableOp1model_4_conv2d_38_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02*
(model_4/conv2d_38/BiasAdd/ReadVariableOp?
model_4/conv2d_38/BiasAddBiasAdd!model_4/conv2d_38/Conv2D:output:00model_4/conv2d_38/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????2
model_4/conv2d_38/BiasAdd?
%model_4/tf.__operators__.add_14/AddV2AddV2"model_4/conv2d_38/BiasAdd:output:03model_4/batch_normalization_16/FusedBatchNormV3:y:0*
T0*0
_output_shapes
:??????????2'
%model_4/tf.__operators__.add_14/AddV2?
model_4/activation_32/ReluRelu)model_4/tf.__operators__.add_14/AddV2:z:0*
T0*0
_output_shapes
:??????????2
model_4/activation_32/Relu?
-model_4/batch_normalization_17/ReadVariableOpReadVariableOp6model_4_batch_normalization_17_readvariableop_resource*
_output_shapes	
:?*
dtype02/
-model_4/batch_normalization_17/ReadVariableOp?
/model_4/batch_normalization_17/ReadVariableOp_1ReadVariableOp8model_4_batch_normalization_17_readvariableop_1_resource*
_output_shapes	
:?*
dtype021
/model_4/batch_normalization_17/ReadVariableOp_1?
>model_4/batch_normalization_17/FusedBatchNormV3/ReadVariableOpReadVariableOpGmodel_4_batch_normalization_17_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:?*
dtype02@
>model_4/batch_normalization_17/FusedBatchNormV3/ReadVariableOp?
@model_4/batch_normalization_17/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpImodel_4_batch_normalization_17_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:?*
dtype02B
@model_4/batch_normalization_17/FusedBatchNormV3/ReadVariableOp_1?
/model_4/batch_normalization_17/FusedBatchNormV3FusedBatchNormV3(model_4/activation_32/Relu:activations:05model_4/batch_normalization_17/ReadVariableOp:value:07model_4/batch_normalization_17/ReadVariableOp_1:value:0Fmodel_4/batch_normalization_17/FusedBatchNormV3/ReadVariableOp:value:0Hmodel_4/batch_normalization_17/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:??????????:?:?:?:?:*
epsilon%o?:*
exponential_avg_factor%
?#<21
/model_4/batch_normalization_17/FusedBatchNormV3?
-model_4/batch_normalization_17/AssignNewValueAssignVariableOpGmodel_4_batch_normalization_17_fusedbatchnormv3_readvariableop_resource<model_4/batch_normalization_17/FusedBatchNormV3:batch_mean:0?^model_4/batch_normalization_17/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype02/
-model_4/batch_normalization_17/AssignNewValue?
/model_4/batch_normalization_17/AssignNewValue_1AssignVariableOpImodel_4_batch_normalization_17_fusedbatchnormv3_readvariableop_1_resource@model_4/batch_normalization_17/FusedBatchNormV3:batch_variance:0A^model_4/batch_normalization_17/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype021
/model_4/batch_normalization_17/AssignNewValue_1?
conv2d_55/Conv2D/ReadVariableOpReadVariableOp(conv2d_55_conv2d_readvariableop_resource*'
_output_shapes
:?*
dtype02!
conv2d_55/Conv2D/ReadVariableOp?
conv2d_55/Conv2DConv2D3model_4/batch_normalization_17/FusedBatchNormV3:y:0'conv2d_55/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????*
paddingVALID*
strides
2
conv2d_55/Conv2D?
 conv2d_55/BiasAdd/ReadVariableOpReadVariableOp)conv2d_55_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02"
 conv2d_55/BiasAdd/ReadVariableOp?
conv2d_55/BiasAddBiasAddconv2d_55/Conv2D:output:0(conv2d_55/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????2
conv2d_55/BiasAdd~
conv2d_55/ReluReluconv2d_55/BiasAdd:output:0*
T0*/
_output_shapes
:?????????2
conv2d_55/Relu?
conv2d_53/Conv2D/ReadVariableOpReadVariableOp(conv2d_53_conv2d_readvariableop_resource*(
_output_shapes
:??*
dtype02!
conv2d_53/Conv2D/ReadVariableOp?
conv2d_53/Conv2DConv2D3model_4/batch_normalization_17/FusedBatchNormV3:y:0'conv2d_53/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????*
paddingSAME*
strides
2
conv2d_53/Conv2D?
 conv2d_53/BiasAdd/ReadVariableOpReadVariableOp)conv2d_53_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02"
 conv2d_53/BiasAdd/ReadVariableOp?
conv2d_53/BiasAddBiasAddconv2d_53/Conv2D:output:0(conv2d_53/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????2
conv2d_53/BiasAdd
conv2d_53/ReluReluconv2d_53/BiasAdd:output:0*
T0*0
_output_shapes
:??????????2
conv2d_53/Relus
flatten_3/ConstConst*
_output_shapes
:*
dtype0*
valueB"????@   2
flatten_3/Const?
flatten_3/ReshapeReshapeconv2d_55/Relu:activations:0flatten_3/Const:output:0*
T0*'
_output_shapes
:?????????@2
flatten_3/Reshape?
conv2d_54/Conv2D/ReadVariableOpReadVariableOp(conv2d_54_conv2d_readvariableop_resource*'
_output_shapes
:?I*
dtype02!
conv2d_54/Conv2D/ReadVariableOp?
conv2d_54/Conv2DConv2Dconv2d_53/Relu:activations:0'conv2d_54/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????I*
paddingSAME*
strides
2
conv2d_54/Conv2D?
 conv2d_54/BiasAdd/ReadVariableOpReadVariableOp)conv2d_54_biasadd_readvariableop_resource*
_output_shapes
:I*
dtype02"
 conv2d_54/BiasAdd/ReadVariableOp?
conv2d_54/BiasAddBiasAddconv2d_54/Conv2D:output:0(conv2d_54/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????I2
conv2d_54/BiasAdd?
conv2d_54/SoftmaxSoftmaxconv2d_54/BiasAdd:output:0*
T0*/
_output_shapes
:?????????I2
conv2d_54/Softmax?
dense_3/MatMul/ReadVariableOpReadVariableOp&dense_3_matmul_readvariableop_resource*
_output_shapes

:@*
dtype02
dense_3/MatMul/ReadVariableOp?
dense_3/MatMulMatMulflatten_3/Reshape:output:0%dense_3/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_3/MatMulp
dense_3/TanhTanhdense_3/MatMul:product:0*
T0*'
_output_shapes
:?????????2
dense_3/Tanhk
IdentityIdentitydense_3/Tanh:y:0^NoOp*
T0*'
_output_shapes
:?????????2

Identity?

Identity_1Identityconv2d_54/Softmax:softmax:0^NoOp*
T0*/
_output_shapes
:?????????I2

Identity_1?
NoOpNoOp!^conv2d_53/BiasAdd/ReadVariableOp ^conv2d_53/Conv2D/ReadVariableOp!^conv2d_54/BiasAdd/ReadVariableOp ^conv2d_54/Conv2D/ReadVariableOp!^conv2d_55/BiasAdd/ReadVariableOp ^conv2d_55/Conv2D/ReadVariableOp^dense_3/MatMul/ReadVariableOp.^model_4/batch_normalization_12/AssignNewValue0^model_4/batch_normalization_12/AssignNewValue_1?^model_4/batch_normalization_12/FusedBatchNormV3/ReadVariableOpA^model_4/batch_normalization_12/FusedBatchNormV3/ReadVariableOp_1.^model_4/batch_normalization_12/ReadVariableOp0^model_4/batch_normalization_12/ReadVariableOp_1.^model_4/batch_normalization_13/AssignNewValue0^model_4/batch_normalization_13/AssignNewValue_1?^model_4/batch_normalization_13/FusedBatchNormV3/ReadVariableOpA^model_4/batch_normalization_13/FusedBatchNormV3/ReadVariableOp_1.^model_4/batch_normalization_13/ReadVariableOp0^model_4/batch_normalization_13/ReadVariableOp_1.^model_4/batch_normalization_14/AssignNewValue0^model_4/batch_normalization_14/AssignNewValue_1?^model_4/batch_normalization_14/FusedBatchNormV3/ReadVariableOpA^model_4/batch_normalization_14/FusedBatchNormV3/ReadVariableOp_1.^model_4/batch_normalization_14/ReadVariableOp0^model_4/batch_normalization_14/ReadVariableOp_1.^model_4/batch_normalization_15/AssignNewValue0^model_4/batch_normalization_15/AssignNewValue_1?^model_4/batch_normalization_15/FusedBatchNormV3/ReadVariableOpA^model_4/batch_normalization_15/FusedBatchNormV3/ReadVariableOp_1.^model_4/batch_normalization_15/ReadVariableOp0^model_4/batch_normalization_15/ReadVariableOp_1.^model_4/batch_normalization_16/AssignNewValue0^model_4/batch_normalization_16/AssignNewValue_1?^model_4/batch_normalization_16/FusedBatchNormV3/ReadVariableOpA^model_4/batch_normalization_16/FusedBatchNormV3/ReadVariableOp_1.^model_4/batch_normalization_16/ReadVariableOp0^model_4/batch_normalization_16/ReadVariableOp_1.^model_4/batch_normalization_17/AssignNewValue0^model_4/batch_normalization_17/AssignNewValue_1?^model_4/batch_normalization_17/FusedBatchNormV3/ReadVariableOpA^model_4/batch_normalization_17/FusedBatchNormV3/ReadVariableOp_1.^model_4/batch_normalization_17/ReadVariableOp0^model_4/batch_normalization_17/ReadVariableOp_1)^model_4/conv2d_28/BiasAdd/ReadVariableOp(^model_4/conv2d_28/Conv2D/ReadVariableOp)^model_4/conv2d_29/BiasAdd/ReadVariableOp(^model_4/conv2d_29/Conv2D/ReadVariableOp)^model_4/conv2d_30/BiasAdd/ReadVariableOp(^model_4/conv2d_30/Conv2D/ReadVariableOp)^model_4/conv2d_31/BiasAdd/ReadVariableOp(^model_4/conv2d_31/Conv2D/ReadVariableOp)^model_4/conv2d_32/BiasAdd/ReadVariableOp(^model_4/conv2d_32/Conv2D/ReadVariableOp)^model_4/conv2d_33/BiasAdd/ReadVariableOp(^model_4/conv2d_33/Conv2D/ReadVariableOp)^model_4/conv2d_34/BiasAdd/ReadVariableOp(^model_4/conv2d_34/Conv2D/ReadVariableOp)^model_4/conv2d_35/BiasAdd/ReadVariableOp(^model_4/conv2d_35/Conv2D/ReadVariableOp)^model_4/conv2d_36/BiasAdd/ReadVariableOp(^model_4/conv2d_36/Conv2D/ReadVariableOp)^model_4/conv2d_37/BiasAdd/ReadVariableOp(^model_4/conv2d_37/Conv2D/ReadVariableOp)^model_4/conv2d_38/BiasAdd/ReadVariableOp(^model_4/conv2d_38/Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes?
?:?????????: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2D
 conv2d_53/BiasAdd/ReadVariableOp conv2d_53/BiasAdd/ReadVariableOp2B
conv2d_53/Conv2D/ReadVariableOpconv2d_53/Conv2D/ReadVariableOp2D
 conv2d_54/BiasAdd/ReadVariableOp conv2d_54/BiasAdd/ReadVariableOp2B
conv2d_54/Conv2D/ReadVariableOpconv2d_54/Conv2D/ReadVariableOp2D
 conv2d_55/BiasAdd/ReadVariableOp conv2d_55/BiasAdd/ReadVariableOp2B
conv2d_55/Conv2D/ReadVariableOpconv2d_55/Conv2D/ReadVariableOp2>
dense_3/MatMul/ReadVariableOpdense_3/MatMul/ReadVariableOp2^
-model_4/batch_normalization_12/AssignNewValue-model_4/batch_normalization_12/AssignNewValue2b
/model_4/batch_normalization_12/AssignNewValue_1/model_4/batch_normalization_12/AssignNewValue_12?
>model_4/batch_normalization_12/FusedBatchNormV3/ReadVariableOp>model_4/batch_normalization_12/FusedBatchNormV3/ReadVariableOp2?
@model_4/batch_normalization_12/FusedBatchNormV3/ReadVariableOp_1@model_4/batch_normalization_12/FusedBatchNormV3/ReadVariableOp_12^
-model_4/batch_normalization_12/ReadVariableOp-model_4/batch_normalization_12/ReadVariableOp2b
/model_4/batch_normalization_12/ReadVariableOp_1/model_4/batch_normalization_12/ReadVariableOp_12^
-model_4/batch_normalization_13/AssignNewValue-model_4/batch_normalization_13/AssignNewValue2b
/model_4/batch_normalization_13/AssignNewValue_1/model_4/batch_normalization_13/AssignNewValue_12?
>model_4/batch_normalization_13/FusedBatchNormV3/ReadVariableOp>model_4/batch_normalization_13/FusedBatchNormV3/ReadVariableOp2?
@model_4/batch_normalization_13/FusedBatchNormV3/ReadVariableOp_1@model_4/batch_normalization_13/FusedBatchNormV3/ReadVariableOp_12^
-model_4/batch_normalization_13/ReadVariableOp-model_4/batch_normalization_13/ReadVariableOp2b
/model_4/batch_normalization_13/ReadVariableOp_1/model_4/batch_normalization_13/ReadVariableOp_12^
-model_4/batch_normalization_14/AssignNewValue-model_4/batch_normalization_14/AssignNewValue2b
/model_4/batch_normalization_14/AssignNewValue_1/model_4/batch_normalization_14/AssignNewValue_12?
>model_4/batch_normalization_14/FusedBatchNormV3/ReadVariableOp>model_4/batch_normalization_14/FusedBatchNormV3/ReadVariableOp2?
@model_4/batch_normalization_14/FusedBatchNormV3/ReadVariableOp_1@model_4/batch_normalization_14/FusedBatchNormV3/ReadVariableOp_12^
-model_4/batch_normalization_14/ReadVariableOp-model_4/batch_normalization_14/ReadVariableOp2b
/model_4/batch_normalization_14/ReadVariableOp_1/model_4/batch_normalization_14/ReadVariableOp_12^
-model_4/batch_normalization_15/AssignNewValue-model_4/batch_normalization_15/AssignNewValue2b
/model_4/batch_normalization_15/AssignNewValue_1/model_4/batch_normalization_15/AssignNewValue_12?
>model_4/batch_normalization_15/FusedBatchNormV3/ReadVariableOp>model_4/batch_normalization_15/FusedBatchNormV3/ReadVariableOp2?
@model_4/batch_normalization_15/FusedBatchNormV3/ReadVariableOp_1@model_4/batch_normalization_15/FusedBatchNormV3/ReadVariableOp_12^
-model_4/batch_normalization_15/ReadVariableOp-model_4/batch_normalization_15/ReadVariableOp2b
/model_4/batch_normalization_15/ReadVariableOp_1/model_4/batch_normalization_15/ReadVariableOp_12^
-model_4/batch_normalization_16/AssignNewValue-model_4/batch_normalization_16/AssignNewValue2b
/model_4/batch_normalization_16/AssignNewValue_1/model_4/batch_normalization_16/AssignNewValue_12?
>model_4/batch_normalization_16/FusedBatchNormV3/ReadVariableOp>model_4/batch_normalization_16/FusedBatchNormV3/ReadVariableOp2?
@model_4/batch_normalization_16/FusedBatchNormV3/ReadVariableOp_1@model_4/batch_normalization_16/FusedBatchNormV3/ReadVariableOp_12^
-model_4/batch_normalization_16/ReadVariableOp-model_4/batch_normalization_16/ReadVariableOp2b
/model_4/batch_normalization_16/ReadVariableOp_1/model_4/batch_normalization_16/ReadVariableOp_12^
-model_4/batch_normalization_17/AssignNewValue-model_4/batch_normalization_17/AssignNewValue2b
/model_4/batch_normalization_17/AssignNewValue_1/model_4/batch_normalization_17/AssignNewValue_12?
>model_4/batch_normalization_17/FusedBatchNormV3/ReadVariableOp>model_4/batch_normalization_17/FusedBatchNormV3/ReadVariableOp2?
@model_4/batch_normalization_17/FusedBatchNormV3/ReadVariableOp_1@model_4/batch_normalization_17/FusedBatchNormV3/ReadVariableOp_12^
-model_4/batch_normalization_17/ReadVariableOp-model_4/batch_normalization_17/ReadVariableOp2b
/model_4/batch_normalization_17/ReadVariableOp_1/model_4/batch_normalization_17/ReadVariableOp_12T
(model_4/conv2d_28/BiasAdd/ReadVariableOp(model_4/conv2d_28/BiasAdd/ReadVariableOp2R
'model_4/conv2d_28/Conv2D/ReadVariableOp'model_4/conv2d_28/Conv2D/ReadVariableOp2T
(model_4/conv2d_29/BiasAdd/ReadVariableOp(model_4/conv2d_29/BiasAdd/ReadVariableOp2R
'model_4/conv2d_29/Conv2D/ReadVariableOp'model_4/conv2d_29/Conv2D/ReadVariableOp2T
(model_4/conv2d_30/BiasAdd/ReadVariableOp(model_4/conv2d_30/BiasAdd/ReadVariableOp2R
'model_4/conv2d_30/Conv2D/ReadVariableOp'model_4/conv2d_30/Conv2D/ReadVariableOp2T
(model_4/conv2d_31/BiasAdd/ReadVariableOp(model_4/conv2d_31/BiasAdd/ReadVariableOp2R
'model_4/conv2d_31/Conv2D/ReadVariableOp'model_4/conv2d_31/Conv2D/ReadVariableOp2T
(model_4/conv2d_32/BiasAdd/ReadVariableOp(model_4/conv2d_32/BiasAdd/ReadVariableOp2R
'model_4/conv2d_32/Conv2D/ReadVariableOp'model_4/conv2d_32/Conv2D/ReadVariableOp2T
(model_4/conv2d_33/BiasAdd/ReadVariableOp(model_4/conv2d_33/BiasAdd/ReadVariableOp2R
'model_4/conv2d_33/Conv2D/ReadVariableOp'model_4/conv2d_33/Conv2D/ReadVariableOp2T
(model_4/conv2d_34/BiasAdd/ReadVariableOp(model_4/conv2d_34/BiasAdd/ReadVariableOp2R
'model_4/conv2d_34/Conv2D/ReadVariableOp'model_4/conv2d_34/Conv2D/ReadVariableOp2T
(model_4/conv2d_35/BiasAdd/ReadVariableOp(model_4/conv2d_35/BiasAdd/ReadVariableOp2R
'model_4/conv2d_35/Conv2D/ReadVariableOp'model_4/conv2d_35/Conv2D/ReadVariableOp2T
(model_4/conv2d_36/BiasAdd/ReadVariableOp(model_4/conv2d_36/BiasAdd/ReadVariableOp2R
'model_4/conv2d_36/Conv2D/ReadVariableOp'model_4/conv2d_36/Conv2D/ReadVariableOp2T
(model_4/conv2d_37/BiasAdd/ReadVariableOp(model_4/conv2d_37/BiasAdd/ReadVariableOp2R
'model_4/conv2d_37/Conv2D/ReadVariableOp'model_4/conv2d_37/Conv2D/ReadVariableOp2T
(model_4/conv2d_38/BiasAdd/ReadVariableOp(model_4/conv2d_38/BiasAdd/ReadVariableOp2R
'model_4/conv2d_38/Conv2D/ReadVariableOp'model_4/conv2d_38/Conv2D/ReadVariableOp:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
*__inference_model_7_layer_call_fn_22005278
input_8"
unknown:?
	unknown_0:	?
	unknown_1:	?
	unknown_2:	?
	unknown_3:	?
	unknown_4:	?%
	unknown_5:??
	unknown_6:	?%
	unknown_7:??
	unknown_8:	?
	unknown_9:	?

unknown_10:	?

unknown_11:	?

unknown_12:	?&

unknown_13:??

unknown_14:	?&

unknown_15:??

unknown_16:	?

unknown_17:	?

unknown_18:	?

unknown_19:	?

unknown_20:	?&

unknown_21:??

unknown_22:	?&

unknown_23:??

unknown_24:	?

unknown_25:	?

unknown_26:	?

unknown_27:	?

unknown_28:	?&

unknown_29:??

unknown_30:	?&

unknown_31:??

unknown_32:	?

unknown_33:	?

unknown_34:	?

unknown_35:	?

unknown_36:	?&

unknown_37:??

unknown_38:	?&

unknown_39:??

unknown_40:	?

unknown_41:	?

unknown_42:	?

unknown_43:	?

unknown_44:	?%

unknown_45:?

unknown_46:&

unknown_47:??

unknown_48:	?%

unknown_49:?I

unknown_50:I

unknown_51:@
identity

identity_1??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_8unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
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
unknown_23
unknown_24
unknown_25
unknown_26
unknown_27
unknown_28
unknown_29
unknown_30
unknown_31
unknown_32
unknown_33
unknown_34
unknown_35
unknown_36
unknown_37
unknown_38
unknown_39
unknown_40
unknown_41
unknown_42
unknown_43
unknown_44
unknown_45
unknown_46
unknown_47
unknown_48
unknown_49
unknown_50
unknown_51*A
Tin:
826*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:?????????:?????????I*W
_read_only_resource_inputs9
75	
 !"#$%&'()*+,-./012345*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_model_7_layer_call_and_return_conditional_losses_220051672
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
_construction_contextkEagerRuntime*?
_input_shapes?
?:?????????: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
/
_output_shapes
:?????????
!
_user_specified_name	input_8
?
I
-__inference_dropout_34_layer_call_fn_22008505

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
GPU 2J 8? *Q
fLRJ
H__inference_dropout_34_layer_call_and_return_conditional_losses_220034782
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
L
0__inference_activation_31_layer_call_fn_22008543

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
GPU 2J 8? *T
fORM
K__inference_activation_31_layer_call_and_return_conditional_losses_220035012
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
?
?
T__inference_batch_normalization_13_layer_call_and_return_conditional_losses_22004136

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
?
d
H__inference_dropout_32_layer_call_and_return_conditional_losses_22008299

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
g
K__inference_activation_31_layer_call_and_return_conditional_losses_22008548

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
?	
?
9__inference_batch_normalization_13_layer_call_fn_22007729

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
GPU 2J 8? *]
fXRV
T__inference_batch_normalization_13_layer_call_and_return_conditional_losses_220024562
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
?
d
H__inference_dropout_30_layer_call_and_return_conditional_losses_22008079

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
f
H__inference_dropout_29_layer_call_and_return_conditional_losses_22003244

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
d
H__inference_dropout_26_layer_call_and_return_conditional_losses_22007639

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
f
H__inference_dropout_27_layer_call_and_return_conditional_losses_22007683

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
?
?
G__inference_conv2d_28_layer_call_and_return_conditional_losses_22007486

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
T__inference_batch_normalization_17_layer_call_and_return_conditional_losses_22008666

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
I
-__inference_dropout_32_layer_call_fn_22008285

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
GPU 2J 8? *Q
fLRJ
H__inference_dropout_32_layer_call_and_return_conditional_losses_220033902
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
I
-__inference_dropout_50_layer_call_fn_22007418

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
GPU 2J 8? *Q
fLRJ
H__inference_dropout_50_layer_call_and_return_conditional_losses_220051262
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
?
I
-__inference_dropout_27_layer_call_fn_22007673

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
GPU 2J 8? *Q
fLRJ
H__inference_dropout_27_layer_call_and_return_conditional_losses_220031562
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
??
?:
#__inference__wrapped_model_22002308
input_8S
8model_7_model_4_conv2d_28_conv2d_readvariableop_resource:?H
9model_7_model_4_conv2d_28_biasadd_readvariableop_resource:	?M
>model_7_model_4_batch_normalization_12_readvariableop_resource:	?O
@model_7_model_4_batch_normalization_12_readvariableop_1_resource:	?^
Omodel_7_model_4_batch_normalization_12_fusedbatchnormv3_readvariableop_resource:	?`
Qmodel_7_model_4_batch_normalization_12_fusedbatchnormv3_readvariableop_1_resource:	?T
8model_7_model_4_conv2d_29_conv2d_readvariableop_resource:??H
9model_7_model_4_conv2d_29_biasadd_readvariableop_resource:	?T
8model_7_model_4_conv2d_30_conv2d_readvariableop_resource:??H
9model_7_model_4_conv2d_30_biasadd_readvariableop_resource:	?M
>model_7_model_4_batch_normalization_13_readvariableop_resource:	?O
@model_7_model_4_batch_normalization_13_readvariableop_1_resource:	?^
Omodel_7_model_4_batch_normalization_13_fusedbatchnormv3_readvariableop_resource:	?`
Qmodel_7_model_4_batch_normalization_13_fusedbatchnormv3_readvariableop_1_resource:	?T
8model_7_model_4_conv2d_31_conv2d_readvariableop_resource:??H
9model_7_model_4_conv2d_31_biasadd_readvariableop_resource:	?T
8model_7_model_4_conv2d_32_conv2d_readvariableop_resource:??H
9model_7_model_4_conv2d_32_biasadd_readvariableop_resource:	?M
>model_7_model_4_batch_normalization_14_readvariableop_resource:	?O
@model_7_model_4_batch_normalization_14_readvariableop_1_resource:	?^
Omodel_7_model_4_batch_normalization_14_fusedbatchnormv3_readvariableop_resource:	?`
Qmodel_7_model_4_batch_normalization_14_fusedbatchnormv3_readvariableop_1_resource:	?T
8model_7_model_4_conv2d_33_conv2d_readvariableop_resource:??H
9model_7_model_4_conv2d_33_biasadd_readvariableop_resource:	?T
8model_7_model_4_conv2d_34_conv2d_readvariableop_resource:??H
9model_7_model_4_conv2d_34_biasadd_readvariableop_resource:	?M
>model_7_model_4_batch_normalization_15_readvariableop_resource:	?O
@model_7_model_4_batch_normalization_15_readvariableop_1_resource:	?^
Omodel_7_model_4_batch_normalization_15_fusedbatchnormv3_readvariableop_resource:	?`
Qmodel_7_model_4_batch_normalization_15_fusedbatchnormv3_readvariableop_1_resource:	?T
8model_7_model_4_conv2d_35_conv2d_readvariableop_resource:??H
9model_7_model_4_conv2d_35_biasadd_readvariableop_resource:	?T
8model_7_model_4_conv2d_36_conv2d_readvariableop_resource:??H
9model_7_model_4_conv2d_36_biasadd_readvariableop_resource:	?M
>model_7_model_4_batch_normalization_16_readvariableop_resource:	?O
@model_7_model_4_batch_normalization_16_readvariableop_1_resource:	?^
Omodel_7_model_4_batch_normalization_16_fusedbatchnormv3_readvariableop_resource:	?`
Qmodel_7_model_4_batch_normalization_16_fusedbatchnormv3_readvariableop_1_resource:	?T
8model_7_model_4_conv2d_37_conv2d_readvariableop_resource:??H
9model_7_model_4_conv2d_37_biasadd_readvariableop_resource:	?T
8model_7_model_4_conv2d_38_conv2d_readvariableop_resource:??H
9model_7_model_4_conv2d_38_biasadd_readvariableop_resource:	?M
>model_7_model_4_batch_normalization_17_readvariableop_resource:	?O
@model_7_model_4_batch_normalization_17_readvariableop_1_resource:	?^
Omodel_7_model_4_batch_normalization_17_fusedbatchnormv3_readvariableop_resource:	?`
Qmodel_7_model_4_batch_normalization_17_fusedbatchnormv3_readvariableop_1_resource:	?K
0model_7_conv2d_55_conv2d_readvariableop_resource:??
1model_7_conv2d_55_biasadd_readvariableop_resource:L
0model_7_conv2d_53_conv2d_readvariableop_resource:??@
1model_7_conv2d_53_biasadd_readvariableop_resource:	?K
0model_7_conv2d_54_conv2d_readvariableop_resource:?I?
1model_7_conv2d_54_biasadd_readvariableop_resource:I@
.model_7_dense_3_matmul_readvariableop_resource:@
identity

identity_1??(model_7/conv2d_53/BiasAdd/ReadVariableOp?'model_7/conv2d_53/Conv2D/ReadVariableOp?(model_7/conv2d_54/BiasAdd/ReadVariableOp?'model_7/conv2d_54/Conv2D/ReadVariableOp?(model_7/conv2d_55/BiasAdd/ReadVariableOp?'model_7/conv2d_55/Conv2D/ReadVariableOp?%model_7/dense_3/MatMul/ReadVariableOp?Fmodel_7/model_4/batch_normalization_12/FusedBatchNormV3/ReadVariableOp?Hmodel_7/model_4/batch_normalization_12/FusedBatchNormV3/ReadVariableOp_1?5model_7/model_4/batch_normalization_12/ReadVariableOp?7model_7/model_4/batch_normalization_12/ReadVariableOp_1?Fmodel_7/model_4/batch_normalization_13/FusedBatchNormV3/ReadVariableOp?Hmodel_7/model_4/batch_normalization_13/FusedBatchNormV3/ReadVariableOp_1?5model_7/model_4/batch_normalization_13/ReadVariableOp?7model_7/model_4/batch_normalization_13/ReadVariableOp_1?Fmodel_7/model_4/batch_normalization_14/FusedBatchNormV3/ReadVariableOp?Hmodel_7/model_4/batch_normalization_14/FusedBatchNormV3/ReadVariableOp_1?5model_7/model_4/batch_normalization_14/ReadVariableOp?7model_7/model_4/batch_normalization_14/ReadVariableOp_1?Fmodel_7/model_4/batch_normalization_15/FusedBatchNormV3/ReadVariableOp?Hmodel_7/model_4/batch_normalization_15/FusedBatchNormV3/ReadVariableOp_1?5model_7/model_4/batch_normalization_15/ReadVariableOp?7model_7/model_4/batch_normalization_15/ReadVariableOp_1?Fmodel_7/model_4/batch_normalization_16/FusedBatchNormV3/ReadVariableOp?Hmodel_7/model_4/batch_normalization_16/FusedBatchNormV3/ReadVariableOp_1?5model_7/model_4/batch_normalization_16/ReadVariableOp?7model_7/model_4/batch_normalization_16/ReadVariableOp_1?Fmodel_7/model_4/batch_normalization_17/FusedBatchNormV3/ReadVariableOp?Hmodel_7/model_4/batch_normalization_17/FusedBatchNormV3/ReadVariableOp_1?5model_7/model_4/batch_normalization_17/ReadVariableOp?7model_7/model_4/batch_normalization_17/ReadVariableOp_1?0model_7/model_4/conv2d_28/BiasAdd/ReadVariableOp?/model_7/model_4/conv2d_28/Conv2D/ReadVariableOp?0model_7/model_4/conv2d_29/BiasAdd/ReadVariableOp?/model_7/model_4/conv2d_29/Conv2D/ReadVariableOp?0model_7/model_4/conv2d_30/BiasAdd/ReadVariableOp?/model_7/model_4/conv2d_30/Conv2D/ReadVariableOp?0model_7/model_4/conv2d_31/BiasAdd/ReadVariableOp?/model_7/model_4/conv2d_31/Conv2D/ReadVariableOp?0model_7/model_4/conv2d_32/BiasAdd/ReadVariableOp?/model_7/model_4/conv2d_32/Conv2D/ReadVariableOp?0model_7/model_4/conv2d_33/BiasAdd/ReadVariableOp?/model_7/model_4/conv2d_33/Conv2D/ReadVariableOp?0model_7/model_4/conv2d_34/BiasAdd/ReadVariableOp?/model_7/model_4/conv2d_34/Conv2D/ReadVariableOp?0model_7/model_4/conv2d_35/BiasAdd/ReadVariableOp?/model_7/model_4/conv2d_35/Conv2D/ReadVariableOp?0model_7/model_4/conv2d_36/BiasAdd/ReadVariableOp?/model_7/model_4/conv2d_36/Conv2D/ReadVariableOp?0model_7/model_4/conv2d_37/BiasAdd/ReadVariableOp?/model_7/model_4/conv2d_37/Conv2D/ReadVariableOp?0model_7/model_4/conv2d_38/BiasAdd/ReadVariableOp?/model_7/model_4/conv2d_38/Conv2D/ReadVariableOp?
/model_7/model_4/conv2d_28/Conv2D/ReadVariableOpReadVariableOp8model_7_model_4_conv2d_28_conv2d_readvariableop_resource*'
_output_shapes
:?*
dtype021
/model_7/model_4/conv2d_28/Conv2D/ReadVariableOp?
 model_7/model_4/conv2d_28/Conv2DConv2Dinput_87model_7/model_4/conv2d_28/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????*
paddingSAME*
strides
2"
 model_7/model_4/conv2d_28/Conv2D?
0model_7/model_4/conv2d_28/BiasAdd/ReadVariableOpReadVariableOp9model_7_model_4_conv2d_28_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype022
0model_7/model_4/conv2d_28/BiasAdd/ReadVariableOp?
!model_7/model_4/conv2d_28/BiasAddBiasAdd)model_7/model_4/conv2d_28/Conv2D:output:08model_7/model_4/conv2d_28/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????2#
!model_7/model_4/conv2d_28/BiasAdd?
5model_7/model_4/batch_normalization_12/ReadVariableOpReadVariableOp>model_7_model_4_batch_normalization_12_readvariableop_resource*
_output_shapes	
:?*
dtype027
5model_7/model_4/batch_normalization_12/ReadVariableOp?
7model_7/model_4/batch_normalization_12/ReadVariableOp_1ReadVariableOp@model_7_model_4_batch_normalization_12_readvariableop_1_resource*
_output_shapes	
:?*
dtype029
7model_7/model_4/batch_normalization_12/ReadVariableOp_1?
Fmodel_7/model_4/batch_normalization_12/FusedBatchNormV3/ReadVariableOpReadVariableOpOmodel_7_model_4_batch_normalization_12_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:?*
dtype02H
Fmodel_7/model_4/batch_normalization_12/FusedBatchNormV3/ReadVariableOp?
Hmodel_7/model_4/batch_normalization_12/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpQmodel_7_model_4_batch_normalization_12_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:?*
dtype02J
Hmodel_7/model_4/batch_normalization_12/FusedBatchNormV3/ReadVariableOp_1?
7model_7/model_4/batch_normalization_12/FusedBatchNormV3FusedBatchNormV3*model_7/model_4/conv2d_28/BiasAdd:output:0=model_7/model_4/batch_normalization_12/ReadVariableOp:value:0?model_7/model_4/batch_normalization_12/ReadVariableOp_1:value:0Nmodel_7/model_4/batch_normalization_12/FusedBatchNormV3/ReadVariableOp:value:0Pmodel_7/model_4/batch_normalization_12/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:??????????:?:?:?:?:*
epsilon%o?:*
is_training( 29
7model_7/model_4/batch_normalization_12/FusedBatchNormV3?
"model_7/model_4/activation_22/ReluRelu;model_7/model_4/batch_normalization_12/FusedBatchNormV3:y:0*
T0*0
_output_shapes
:??????????2$
"model_7/model_4/activation_22/Relu?
#model_7/model_4/dropout_26/IdentityIdentity0model_7/model_4/activation_22/Relu:activations:0*
T0*0
_output_shapes
:??????????2%
#model_7/model_4/dropout_26/Identity?
/model_7/model_4/conv2d_29/Conv2D/ReadVariableOpReadVariableOp8model_7_model_4_conv2d_29_conv2d_readvariableop_resource*(
_output_shapes
:??*
dtype021
/model_7/model_4/conv2d_29/Conv2D/ReadVariableOp?
 model_7/model_4/conv2d_29/Conv2DConv2D,model_7/model_4/dropout_26/Identity:output:07model_7/model_4/conv2d_29/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????*
paddingSAME*
strides
2"
 model_7/model_4/conv2d_29/Conv2D?
0model_7/model_4/conv2d_29/BiasAdd/ReadVariableOpReadVariableOp9model_7_model_4_conv2d_29_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype022
0model_7/model_4/conv2d_29/BiasAdd/ReadVariableOp?
!model_7/model_4/conv2d_29/BiasAddBiasAdd)model_7/model_4/conv2d_29/Conv2D:output:08model_7/model_4/conv2d_29/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????2#
!model_7/model_4/conv2d_29/BiasAdd?
"model_7/model_4/activation_23/ReluRelu*model_7/model_4/conv2d_29/BiasAdd:output:0*
T0*0
_output_shapes
:??????????2$
"model_7/model_4/activation_23/Relu?
#model_7/model_4/dropout_27/IdentityIdentity0model_7/model_4/activation_23/Relu:activations:0*
T0*0
_output_shapes
:??????????2%
#model_7/model_4/dropout_27/Identity?
/model_7/model_4/conv2d_30/Conv2D/ReadVariableOpReadVariableOp8model_7_model_4_conv2d_30_conv2d_readvariableop_resource*(
_output_shapes
:??*
dtype021
/model_7/model_4/conv2d_30/Conv2D/ReadVariableOp?
 model_7/model_4/conv2d_30/Conv2DConv2D,model_7/model_4/dropout_27/Identity:output:07model_7/model_4/conv2d_30/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????*
paddingSAME*
strides
2"
 model_7/model_4/conv2d_30/Conv2D?
0model_7/model_4/conv2d_30/BiasAdd/ReadVariableOpReadVariableOp9model_7_model_4_conv2d_30_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype022
0model_7/model_4/conv2d_30/BiasAdd/ReadVariableOp?
!model_7/model_4/conv2d_30/BiasAddBiasAdd)model_7/model_4/conv2d_30/Conv2D:output:08model_7/model_4/conv2d_30/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????2#
!model_7/model_4/conv2d_30/BiasAdd?
-model_7/model_4/tf.__operators__.add_10/AddV2AddV2*model_7/model_4/conv2d_30/BiasAdd:output:0,model_7/model_4/dropout_26/Identity:output:0*
T0*0
_output_shapes
:??????????2/
-model_7/model_4/tf.__operators__.add_10/AddV2?
"model_7/model_4/activation_24/ReluRelu1model_7/model_4/tf.__operators__.add_10/AddV2:z:0*
T0*0
_output_shapes
:??????????2$
"model_7/model_4/activation_24/Relu?
5model_7/model_4/batch_normalization_13/ReadVariableOpReadVariableOp>model_7_model_4_batch_normalization_13_readvariableop_resource*
_output_shapes	
:?*
dtype027
5model_7/model_4/batch_normalization_13/ReadVariableOp?
7model_7/model_4/batch_normalization_13/ReadVariableOp_1ReadVariableOp@model_7_model_4_batch_normalization_13_readvariableop_1_resource*
_output_shapes	
:?*
dtype029
7model_7/model_4/batch_normalization_13/ReadVariableOp_1?
Fmodel_7/model_4/batch_normalization_13/FusedBatchNormV3/ReadVariableOpReadVariableOpOmodel_7_model_4_batch_normalization_13_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:?*
dtype02H
Fmodel_7/model_4/batch_normalization_13/FusedBatchNormV3/ReadVariableOp?
Hmodel_7/model_4/batch_normalization_13/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpQmodel_7_model_4_batch_normalization_13_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:?*
dtype02J
Hmodel_7/model_4/batch_normalization_13/FusedBatchNormV3/ReadVariableOp_1?
7model_7/model_4/batch_normalization_13/FusedBatchNormV3FusedBatchNormV30model_7/model_4/activation_24/Relu:activations:0=model_7/model_4/batch_normalization_13/ReadVariableOp:value:0?model_7/model_4/batch_normalization_13/ReadVariableOp_1:value:0Nmodel_7/model_4/batch_normalization_13/FusedBatchNormV3/ReadVariableOp:value:0Pmodel_7/model_4/batch_normalization_13/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:??????????:?:?:?:?:*
epsilon%o?:*
is_training( 29
7model_7/model_4/batch_normalization_13/FusedBatchNormV3?
#model_7/model_4/dropout_28/IdentityIdentity;model_7/model_4/batch_normalization_13/FusedBatchNormV3:y:0*
T0*0
_output_shapes
:??????????2%
#model_7/model_4/dropout_28/Identity?
/model_7/model_4/conv2d_31/Conv2D/ReadVariableOpReadVariableOp8model_7_model_4_conv2d_31_conv2d_readvariableop_resource*(
_output_shapes
:??*
dtype021
/model_7/model_4/conv2d_31/Conv2D/ReadVariableOp?
 model_7/model_4/conv2d_31/Conv2DConv2D,model_7/model_4/dropout_28/Identity:output:07model_7/model_4/conv2d_31/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????*
paddingSAME*
strides
2"
 model_7/model_4/conv2d_31/Conv2D?
0model_7/model_4/conv2d_31/BiasAdd/ReadVariableOpReadVariableOp9model_7_model_4_conv2d_31_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype022
0model_7/model_4/conv2d_31/BiasAdd/ReadVariableOp?
!model_7/model_4/conv2d_31/BiasAddBiasAdd)model_7/model_4/conv2d_31/Conv2D:output:08model_7/model_4/conv2d_31/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????2#
!model_7/model_4/conv2d_31/BiasAdd?
"model_7/model_4/activation_25/ReluRelu*model_7/model_4/conv2d_31/BiasAdd:output:0*
T0*0
_output_shapes
:??????????2$
"model_7/model_4/activation_25/Relu?
#model_7/model_4/dropout_29/IdentityIdentity0model_7/model_4/activation_25/Relu:activations:0*
T0*0
_output_shapes
:??????????2%
#model_7/model_4/dropout_29/Identity?
/model_7/model_4/conv2d_32/Conv2D/ReadVariableOpReadVariableOp8model_7_model_4_conv2d_32_conv2d_readvariableop_resource*(
_output_shapes
:??*
dtype021
/model_7/model_4/conv2d_32/Conv2D/ReadVariableOp?
 model_7/model_4/conv2d_32/Conv2DConv2D,model_7/model_4/dropout_29/Identity:output:07model_7/model_4/conv2d_32/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????*
paddingSAME*
strides
2"
 model_7/model_4/conv2d_32/Conv2D?
0model_7/model_4/conv2d_32/BiasAdd/ReadVariableOpReadVariableOp9model_7_model_4_conv2d_32_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype022
0model_7/model_4/conv2d_32/BiasAdd/ReadVariableOp?
!model_7/model_4/conv2d_32/BiasAddBiasAdd)model_7/model_4/conv2d_32/Conv2D:output:08model_7/model_4/conv2d_32/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????2#
!model_7/model_4/conv2d_32/BiasAdd?
-model_7/model_4/tf.__operators__.add_11/AddV2AddV2*model_7/model_4/conv2d_32/BiasAdd:output:0,model_7/model_4/dropout_28/Identity:output:0*
T0*0
_output_shapes
:??????????2/
-model_7/model_4/tf.__operators__.add_11/AddV2?
"model_7/model_4/activation_26/ReluRelu1model_7/model_4/tf.__operators__.add_11/AddV2:z:0*
T0*0
_output_shapes
:??????????2$
"model_7/model_4/activation_26/Relu?
5model_7/model_4/batch_normalization_14/ReadVariableOpReadVariableOp>model_7_model_4_batch_normalization_14_readvariableop_resource*
_output_shapes	
:?*
dtype027
5model_7/model_4/batch_normalization_14/ReadVariableOp?
7model_7/model_4/batch_normalization_14/ReadVariableOp_1ReadVariableOp@model_7_model_4_batch_normalization_14_readvariableop_1_resource*
_output_shapes	
:?*
dtype029
7model_7/model_4/batch_normalization_14/ReadVariableOp_1?
Fmodel_7/model_4/batch_normalization_14/FusedBatchNormV3/ReadVariableOpReadVariableOpOmodel_7_model_4_batch_normalization_14_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:?*
dtype02H
Fmodel_7/model_4/batch_normalization_14/FusedBatchNormV3/ReadVariableOp?
Hmodel_7/model_4/batch_normalization_14/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpQmodel_7_model_4_batch_normalization_14_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:?*
dtype02J
Hmodel_7/model_4/batch_normalization_14/FusedBatchNormV3/ReadVariableOp_1?
7model_7/model_4/batch_normalization_14/FusedBatchNormV3FusedBatchNormV30model_7/model_4/activation_26/Relu:activations:0=model_7/model_4/batch_normalization_14/ReadVariableOp:value:0?model_7/model_4/batch_normalization_14/ReadVariableOp_1:value:0Nmodel_7/model_4/batch_normalization_14/FusedBatchNormV3/ReadVariableOp:value:0Pmodel_7/model_4/batch_normalization_14/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:??????????:?:?:?:?:*
epsilon%o?:*
is_training( 29
7model_7/model_4/batch_normalization_14/FusedBatchNormV3?
#model_7/model_4/dropout_30/IdentityIdentity;model_7/model_4/batch_normalization_14/FusedBatchNormV3:y:0*
T0*0
_output_shapes
:??????????2%
#model_7/model_4/dropout_30/Identity?
/model_7/model_4/conv2d_33/Conv2D/ReadVariableOpReadVariableOp8model_7_model_4_conv2d_33_conv2d_readvariableop_resource*(
_output_shapes
:??*
dtype021
/model_7/model_4/conv2d_33/Conv2D/ReadVariableOp?
 model_7/model_4/conv2d_33/Conv2DConv2D,model_7/model_4/dropout_30/Identity:output:07model_7/model_4/conv2d_33/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????*
paddingSAME*
strides
2"
 model_7/model_4/conv2d_33/Conv2D?
0model_7/model_4/conv2d_33/BiasAdd/ReadVariableOpReadVariableOp9model_7_model_4_conv2d_33_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype022
0model_7/model_4/conv2d_33/BiasAdd/ReadVariableOp?
!model_7/model_4/conv2d_33/BiasAddBiasAdd)model_7/model_4/conv2d_33/Conv2D:output:08model_7/model_4/conv2d_33/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????2#
!model_7/model_4/conv2d_33/BiasAdd?
"model_7/model_4/activation_27/ReluRelu*model_7/model_4/conv2d_33/BiasAdd:output:0*
T0*0
_output_shapes
:??????????2$
"model_7/model_4/activation_27/Relu?
#model_7/model_4/dropout_31/IdentityIdentity0model_7/model_4/activation_27/Relu:activations:0*
T0*0
_output_shapes
:??????????2%
#model_7/model_4/dropout_31/Identity?
/model_7/model_4/conv2d_34/Conv2D/ReadVariableOpReadVariableOp8model_7_model_4_conv2d_34_conv2d_readvariableop_resource*(
_output_shapes
:??*
dtype021
/model_7/model_4/conv2d_34/Conv2D/ReadVariableOp?
 model_7/model_4/conv2d_34/Conv2DConv2D,model_7/model_4/dropout_31/Identity:output:07model_7/model_4/conv2d_34/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????*
paddingSAME*
strides
2"
 model_7/model_4/conv2d_34/Conv2D?
0model_7/model_4/conv2d_34/BiasAdd/ReadVariableOpReadVariableOp9model_7_model_4_conv2d_34_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype022
0model_7/model_4/conv2d_34/BiasAdd/ReadVariableOp?
!model_7/model_4/conv2d_34/BiasAddBiasAdd)model_7/model_4/conv2d_34/Conv2D:output:08model_7/model_4/conv2d_34/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????2#
!model_7/model_4/conv2d_34/BiasAdd?
-model_7/model_4/tf.__operators__.add_12/AddV2AddV2*model_7/model_4/conv2d_34/BiasAdd:output:0,model_7/model_4/dropout_30/Identity:output:0*
T0*0
_output_shapes
:??????????2/
-model_7/model_4/tf.__operators__.add_12/AddV2?
"model_7/model_4/activation_28/ReluRelu1model_7/model_4/tf.__operators__.add_12/AddV2:z:0*
T0*0
_output_shapes
:??????????2$
"model_7/model_4/activation_28/Relu?
5model_7/model_4/batch_normalization_15/ReadVariableOpReadVariableOp>model_7_model_4_batch_normalization_15_readvariableop_resource*
_output_shapes	
:?*
dtype027
5model_7/model_4/batch_normalization_15/ReadVariableOp?
7model_7/model_4/batch_normalization_15/ReadVariableOp_1ReadVariableOp@model_7_model_4_batch_normalization_15_readvariableop_1_resource*
_output_shapes	
:?*
dtype029
7model_7/model_4/batch_normalization_15/ReadVariableOp_1?
Fmodel_7/model_4/batch_normalization_15/FusedBatchNormV3/ReadVariableOpReadVariableOpOmodel_7_model_4_batch_normalization_15_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:?*
dtype02H
Fmodel_7/model_4/batch_normalization_15/FusedBatchNormV3/ReadVariableOp?
Hmodel_7/model_4/batch_normalization_15/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpQmodel_7_model_4_batch_normalization_15_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:?*
dtype02J
Hmodel_7/model_4/batch_normalization_15/FusedBatchNormV3/ReadVariableOp_1?
7model_7/model_4/batch_normalization_15/FusedBatchNormV3FusedBatchNormV30model_7/model_4/activation_28/Relu:activations:0=model_7/model_4/batch_normalization_15/ReadVariableOp:value:0?model_7/model_4/batch_normalization_15/ReadVariableOp_1:value:0Nmodel_7/model_4/batch_normalization_15/FusedBatchNormV3/ReadVariableOp:value:0Pmodel_7/model_4/batch_normalization_15/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:??????????:?:?:?:?:*
epsilon%o?:*
is_training( 29
7model_7/model_4/batch_normalization_15/FusedBatchNormV3?
#model_7/model_4/dropout_32/IdentityIdentity;model_7/model_4/batch_normalization_15/FusedBatchNormV3:y:0*
T0*0
_output_shapes
:??????????2%
#model_7/model_4/dropout_32/Identity?
/model_7/model_4/conv2d_35/Conv2D/ReadVariableOpReadVariableOp8model_7_model_4_conv2d_35_conv2d_readvariableop_resource*(
_output_shapes
:??*
dtype021
/model_7/model_4/conv2d_35/Conv2D/ReadVariableOp?
 model_7/model_4/conv2d_35/Conv2DConv2D,model_7/model_4/dropout_32/Identity:output:07model_7/model_4/conv2d_35/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????*
paddingSAME*
strides
2"
 model_7/model_4/conv2d_35/Conv2D?
0model_7/model_4/conv2d_35/BiasAdd/ReadVariableOpReadVariableOp9model_7_model_4_conv2d_35_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype022
0model_7/model_4/conv2d_35/BiasAdd/ReadVariableOp?
!model_7/model_4/conv2d_35/BiasAddBiasAdd)model_7/model_4/conv2d_35/Conv2D:output:08model_7/model_4/conv2d_35/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????2#
!model_7/model_4/conv2d_35/BiasAdd?
"model_7/model_4/activation_29/ReluRelu*model_7/model_4/conv2d_35/BiasAdd:output:0*
T0*0
_output_shapes
:??????????2$
"model_7/model_4/activation_29/Relu?
#model_7/model_4/dropout_33/IdentityIdentity0model_7/model_4/activation_29/Relu:activations:0*
T0*0
_output_shapes
:??????????2%
#model_7/model_4/dropout_33/Identity?
/model_7/model_4/conv2d_36/Conv2D/ReadVariableOpReadVariableOp8model_7_model_4_conv2d_36_conv2d_readvariableop_resource*(
_output_shapes
:??*
dtype021
/model_7/model_4/conv2d_36/Conv2D/ReadVariableOp?
 model_7/model_4/conv2d_36/Conv2DConv2D,model_7/model_4/dropout_33/Identity:output:07model_7/model_4/conv2d_36/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????*
paddingSAME*
strides
2"
 model_7/model_4/conv2d_36/Conv2D?
0model_7/model_4/conv2d_36/BiasAdd/ReadVariableOpReadVariableOp9model_7_model_4_conv2d_36_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype022
0model_7/model_4/conv2d_36/BiasAdd/ReadVariableOp?
!model_7/model_4/conv2d_36/BiasAddBiasAdd)model_7/model_4/conv2d_36/Conv2D:output:08model_7/model_4/conv2d_36/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????2#
!model_7/model_4/conv2d_36/BiasAdd?
-model_7/model_4/tf.__operators__.add_13/AddV2AddV2*model_7/model_4/conv2d_36/BiasAdd:output:0,model_7/model_4/dropout_32/Identity:output:0*
T0*0
_output_shapes
:??????????2/
-model_7/model_4/tf.__operators__.add_13/AddV2?
"model_7/model_4/activation_30/ReluRelu1model_7/model_4/tf.__operators__.add_13/AddV2:z:0*
T0*0
_output_shapes
:??????????2$
"model_7/model_4/activation_30/Relu?
5model_7/model_4/batch_normalization_16/ReadVariableOpReadVariableOp>model_7_model_4_batch_normalization_16_readvariableop_resource*
_output_shapes	
:?*
dtype027
5model_7/model_4/batch_normalization_16/ReadVariableOp?
7model_7/model_4/batch_normalization_16/ReadVariableOp_1ReadVariableOp@model_7_model_4_batch_normalization_16_readvariableop_1_resource*
_output_shapes	
:?*
dtype029
7model_7/model_4/batch_normalization_16/ReadVariableOp_1?
Fmodel_7/model_4/batch_normalization_16/FusedBatchNormV3/ReadVariableOpReadVariableOpOmodel_7_model_4_batch_normalization_16_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:?*
dtype02H
Fmodel_7/model_4/batch_normalization_16/FusedBatchNormV3/ReadVariableOp?
Hmodel_7/model_4/batch_normalization_16/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpQmodel_7_model_4_batch_normalization_16_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:?*
dtype02J
Hmodel_7/model_4/batch_normalization_16/FusedBatchNormV3/ReadVariableOp_1?
7model_7/model_4/batch_normalization_16/FusedBatchNormV3FusedBatchNormV30model_7/model_4/activation_30/Relu:activations:0=model_7/model_4/batch_normalization_16/ReadVariableOp:value:0?model_7/model_4/batch_normalization_16/ReadVariableOp_1:value:0Nmodel_7/model_4/batch_normalization_16/FusedBatchNormV3/ReadVariableOp:value:0Pmodel_7/model_4/batch_normalization_16/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:??????????:?:?:?:?:*
epsilon%o?:*
is_training( 29
7model_7/model_4/batch_normalization_16/FusedBatchNormV3?
#model_7/model_4/dropout_34/IdentityIdentity;model_7/model_4/batch_normalization_16/FusedBatchNormV3:y:0*
T0*0
_output_shapes
:??????????2%
#model_7/model_4/dropout_34/Identity?
/model_7/model_4/conv2d_37/Conv2D/ReadVariableOpReadVariableOp8model_7_model_4_conv2d_37_conv2d_readvariableop_resource*(
_output_shapes
:??*
dtype021
/model_7/model_4/conv2d_37/Conv2D/ReadVariableOp?
 model_7/model_4/conv2d_37/Conv2DConv2D,model_7/model_4/dropout_34/Identity:output:07model_7/model_4/conv2d_37/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????*
paddingSAME*
strides
2"
 model_7/model_4/conv2d_37/Conv2D?
0model_7/model_4/conv2d_37/BiasAdd/ReadVariableOpReadVariableOp9model_7_model_4_conv2d_37_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype022
0model_7/model_4/conv2d_37/BiasAdd/ReadVariableOp?
!model_7/model_4/conv2d_37/BiasAddBiasAdd)model_7/model_4/conv2d_37/Conv2D:output:08model_7/model_4/conv2d_37/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????2#
!model_7/model_4/conv2d_37/BiasAdd?
"model_7/model_4/activation_31/ReluRelu*model_7/model_4/conv2d_37/BiasAdd:output:0*
T0*0
_output_shapes
:??????????2$
"model_7/model_4/activation_31/Relu?
#model_7/model_4/dropout_35/IdentityIdentity0model_7/model_4/activation_31/Relu:activations:0*
T0*0
_output_shapes
:??????????2%
#model_7/model_4/dropout_35/Identity?
/model_7/model_4/conv2d_38/Conv2D/ReadVariableOpReadVariableOp8model_7_model_4_conv2d_38_conv2d_readvariableop_resource*(
_output_shapes
:??*
dtype021
/model_7/model_4/conv2d_38/Conv2D/ReadVariableOp?
 model_7/model_4/conv2d_38/Conv2DConv2D,model_7/model_4/dropout_35/Identity:output:07model_7/model_4/conv2d_38/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????*
paddingSAME*
strides
2"
 model_7/model_4/conv2d_38/Conv2D?
0model_7/model_4/conv2d_38/BiasAdd/ReadVariableOpReadVariableOp9model_7_model_4_conv2d_38_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype022
0model_7/model_4/conv2d_38/BiasAdd/ReadVariableOp?
!model_7/model_4/conv2d_38/BiasAddBiasAdd)model_7/model_4/conv2d_38/Conv2D:output:08model_7/model_4/conv2d_38/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????2#
!model_7/model_4/conv2d_38/BiasAdd?
-model_7/model_4/tf.__operators__.add_14/AddV2AddV2*model_7/model_4/conv2d_38/BiasAdd:output:0,model_7/model_4/dropout_34/Identity:output:0*
T0*0
_output_shapes
:??????????2/
-model_7/model_4/tf.__operators__.add_14/AddV2?
"model_7/model_4/activation_32/ReluRelu1model_7/model_4/tf.__operators__.add_14/AddV2:z:0*
T0*0
_output_shapes
:??????????2$
"model_7/model_4/activation_32/Relu?
5model_7/model_4/batch_normalization_17/ReadVariableOpReadVariableOp>model_7_model_4_batch_normalization_17_readvariableop_resource*
_output_shapes	
:?*
dtype027
5model_7/model_4/batch_normalization_17/ReadVariableOp?
7model_7/model_4/batch_normalization_17/ReadVariableOp_1ReadVariableOp@model_7_model_4_batch_normalization_17_readvariableop_1_resource*
_output_shapes	
:?*
dtype029
7model_7/model_4/batch_normalization_17/ReadVariableOp_1?
Fmodel_7/model_4/batch_normalization_17/FusedBatchNormV3/ReadVariableOpReadVariableOpOmodel_7_model_4_batch_normalization_17_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:?*
dtype02H
Fmodel_7/model_4/batch_normalization_17/FusedBatchNormV3/ReadVariableOp?
Hmodel_7/model_4/batch_normalization_17/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpQmodel_7_model_4_batch_normalization_17_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:?*
dtype02J
Hmodel_7/model_4/batch_normalization_17/FusedBatchNormV3/ReadVariableOp_1?
7model_7/model_4/batch_normalization_17/FusedBatchNormV3FusedBatchNormV30model_7/model_4/activation_32/Relu:activations:0=model_7/model_4/batch_normalization_17/ReadVariableOp:value:0?model_7/model_4/batch_normalization_17/ReadVariableOp_1:value:0Nmodel_7/model_4/batch_normalization_17/FusedBatchNormV3/ReadVariableOp:value:0Pmodel_7/model_4/batch_normalization_17/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:??????????:?:?:?:?:*
epsilon%o?:*
is_training( 29
7model_7/model_4/batch_normalization_17/FusedBatchNormV3?
#model_7/model_4/dropout_36/IdentityIdentity;model_7/model_4/batch_normalization_17/FusedBatchNormV3:y:0*
T0*0
_output_shapes
:??????????2%
#model_7/model_4/dropout_36/Identity?
'model_7/conv2d_55/Conv2D/ReadVariableOpReadVariableOp0model_7_conv2d_55_conv2d_readvariableop_resource*'
_output_shapes
:?*
dtype02)
'model_7/conv2d_55/Conv2D/ReadVariableOp?
model_7/conv2d_55/Conv2DConv2D,model_7/model_4/dropout_36/Identity:output:0/model_7/conv2d_55/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????*
paddingVALID*
strides
2
model_7/conv2d_55/Conv2D?
(model_7/conv2d_55/BiasAdd/ReadVariableOpReadVariableOp1model_7_conv2d_55_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02*
(model_7/conv2d_55/BiasAdd/ReadVariableOp?
model_7/conv2d_55/BiasAddBiasAdd!model_7/conv2d_55/Conv2D:output:00model_7/conv2d_55/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????2
model_7/conv2d_55/BiasAdd?
model_7/conv2d_55/ReluRelu"model_7/conv2d_55/BiasAdd:output:0*
T0*/
_output_shapes
:?????????2
model_7/conv2d_55/Relu?
'model_7/conv2d_53/Conv2D/ReadVariableOpReadVariableOp0model_7_conv2d_53_conv2d_readvariableop_resource*(
_output_shapes
:??*
dtype02)
'model_7/conv2d_53/Conv2D/ReadVariableOp?
model_7/conv2d_53/Conv2DConv2D,model_7/model_4/dropout_36/Identity:output:0/model_7/conv2d_53/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????*
paddingSAME*
strides
2
model_7/conv2d_53/Conv2D?
(model_7/conv2d_53/BiasAdd/ReadVariableOpReadVariableOp1model_7_conv2d_53_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02*
(model_7/conv2d_53/BiasAdd/ReadVariableOp?
model_7/conv2d_53/BiasAddBiasAdd!model_7/conv2d_53/Conv2D:output:00model_7/conv2d_53/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????2
model_7/conv2d_53/BiasAdd?
model_7/conv2d_53/ReluRelu"model_7/conv2d_53/BiasAdd:output:0*
T0*0
_output_shapes
:??????????2
model_7/conv2d_53/Relu?
model_7/dropout_51/IdentityIdentity$model_7/conv2d_55/Relu:activations:0*
T0*/
_output_shapes
:?????????2
model_7/dropout_51/Identity?
model_7/dropout_50/IdentityIdentity$model_7/conv2d_53/Relu:activations:0*
T0*0
_output_shapes
:??????????2
model_7/dropout_50/Identity?
model_7/flatten_3/ConstConst*
_output_shapes
:*
dtype0*
valueB"????@   2
model_7/flatten_3/Const?
model_7/flatten_3/ReshapeReshape$model_7/dropout_51/Identity:output:0 model_7/flatten_3/Const:output:0*
T0*'
_output_shapes
:?????????@2
model_7/flatten_3/Reshape?
'model_7/conv2d_54/Conv2D/ReadVariableOpReadVariableOp0model_7_conv2d_54_conv2d_readvariableop_resource*'
_output_shapes
:?I*
dtype02)
'model_7/conv2d_54/Conv2D/ReadVariableOp?
model_7/conv2d_54/Conv2DConv2D$model_7/dropout_50/Identity:output:0/model_7/conv2d_54/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????I*
paddingSAME*
strides
2
model_7/conv2d_54/Conv2D?
(model_7/conv2d_54/BiasAdd/ReadVariableOpReadVariableOp1model_7_conv2d_54_biasadd_readvariableop_resource*
_output_shapes
:I*
dtype02*
(model_7/conv2d_54/BiasAdd/ReadVariableOp?
model_7/conv2d_54/BiasAddBiasAdd!model_7/conv2d_54/Conv2D:output:00model_7/conv2d_54/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????I2
model_7/conv2d_54/BiasAdd?
model_7/conv2d_54/SoftmaxSoftmax"model_7/conv2d_54/BiasAdd:output:0*
T0*/
_output_shapes
:?????????I2
model_7/conv2d_54/Softmax?
%model_7/dense_3/MatMul/ReadVariableOpReadVariableOp.model_7_dense_3_matmul_readvariableop_resource*
_output_shapes

:@*
dtype02'
%model_7/dense_3/MatMul/ReadVariableOp?
model_7/dense_3/MatMulMatMul"model_7/flatten_3/Reshape:output:0-model_7/dense_3/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
model_7/dense_3/MatMul?
model_7/dense_3/TanhTanh model_7/dense_3/MatMul:product:0*
T0*'
_output_shapes
:?????????2
model_7/dense_3/Tanh?
IdentityIdentity#model_7/conv2d_54/Softmax:softmax:0^NoOp*
T0*/
_output_shapes
:?????????I2

Identityw

Identity_1Identitymodel_7/dense_3/Tanh:y:0^NoOp*
T0*'
_output_shapes
:?????????2

Identity_1?
NoOpNoOp)^model_7/conv2d_53/BiasAdd/ReadVariableOp(^model_7/conv2d_53/Conv2D/ReadVariableOp)^model_7/conv2d_54/BiasAdd/ReadVariableOp(^model_7/conv2d_54/Conv2D/ReadVariableOp)^model_7/conv2d_55/BiasAdd/ReadVariableOp(^model_7/conv2d_55/Conv2D/ReadVariableOp&^model_7/dense_3/MatMul/ReadVariableOpG^model_7/model_4/batch_normalization_12/FusedBatchNormV3/ReadVariableOpI^model_7/model_4/batch_normalization_12/FusedBatchNormV3/ReadVariableOp_16^model_7/model_4/batch_normalization_12/ReadVariableOp8^model_7/model_4/batch_normalization_12/ReadVariableOp_1G^model_7/model_4/batch_normalization_13/FusedBatchNormV3/ReadVariableOpI^model_7/model_4/batch_normalization_13/FusedBatchNormV3/ReadVariableOp_16^model_7/model_4/batch_normalization_13/ReadVariableOp8^model_7/model_4/batch_normalization_13/ReadVariableOp_1G^model_7/model_4/batch_normalization_14/FusedBatchNormV3/ReadVariableOpI^model_7/model_4/batch_normalization_14/FusedBatchNormV3/ReadVariableOp_16^model_7/model_4/batch_normalization_14/ReadVariableOp8^model_7/model_4/batch_normalization_14/ReadVariableOp_1G^model_7/model_4/batch_normalization_15/FusedBatchNormV3/ReadVariableOpI^model_7/model_4/batch_normalization_15/FusedBatchNormV3/ReadVariableOp_16^model_7/model_4/batch_normalization_15/ReadVariableOp8^model_7/model_4/batch_normalization_15/ReadVariableOp_1G^model_7/model_4/batch_normalization_16/FusedBatchNormV3/ReadVariableOpI^model_7/model_4/batch_normalization_16/FusedBatchNormV3/ReadVariableOp_16^model_7/model_4/batch_normalization_16/ReadVariableOp8^model_7/model_4/batch_normalization_16/ReadVariableOp_1G^model_7/model_4/batch_normalization_17/FusedBatchNormV3/ReadVariableOpI^model_7/model_4/batch_normalization_17/FusedBatchNormV3/ReadVariableOp_16^model_7/model_4/batch_normalization_17/ReadVariableOp8^model_7/model_4/batch_normalization_17/ReadVariableOp_11^model_7/model_4/conv2d_28/BiasAdd/ReadVariableOp0^model_7/model_4/conv2d_28/Conv2D/ReadVariableOp1^model_7/model_4/conv2d_29/BiasAdd/ReadVariableOp0^model_7/model_4/conv2d_29/Conv2D/ReadVariableOp1^model_7/model_4/conv2d_30/BiasAdd/ReadVariableOp0^model_7/model_4/conv2d_30/Conv2D/ReadVariableOp1^model_7/model_4/conv2d_31/BiasAdd/ReadVariableOp0^model_7/model_4/conv2d_31/Conv2D/ReadVariableOp1^model_7/model_4/conv2d_32/BiasAdd/ReadVariableOp0^model_7/model_4/conv2d_32/Conv2D/ReadVariableOp1^model_7/model_4/conv2d_33/BiasAdd/ReadVariableOp0^model_7/model_4/conv2d_33/Conv2D/ReadVariableOp1^model_7/model_4/conv2d_34/BiasAdd/ReadVariableOp0^model_7/model_4/conv2d_34/Conv2D/ReadVariableOp1^model_7/model_4/conv2d_35/BiasAdd/ReadVariableOp0^model_7/model_4/conv2d_35/Conv2D/ReadVariableOp1^model_7/model_4/conv2d_36/BiasAdd/ReadVariableOp0^model_7/model_4/conv2d_36/Conv2D/ReadVariableOp1^model_7/model_4/conv2d_37/BiasAdd/ReadVariableOp0^model_7/model_4/conv2d_37/Conv2D/ReadVariableOp1^model_7/model_4/conv2d_38/BiasAdd/ReadVariableOp0^model_7/model_4/conv2d_38/Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes?
?:?????????: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2T
(model_7/conv2d_53/BiasAdd/ReadVariableOp(model_7/conv2d_53/BiasAdd/ReadVariableOp2R
'model_7/conv2d_53/Conv2D/ReadVariableOp'model_7/conv2d_53/Conv2D/ReadVariableOp2T
(model_7/conv2d_54/BiasAdd/ReadVariableOp(model_7/conv2d_54/BiasAdd/ReadVariableOp2R
'model_7/conv2d_54/Conv2D/ReadVariableOp'model_7/conv2d_54/Conv2D/ReadVariableOp2T
(model_7/conv2d_55/BiasAdd/ReadVariableOp(model_7/conv2d_55/BiasAdd/ReadVariableOp2R
'model_7/conv2d_55/Conv2D/ReadVariableOp'model_7/conv2d_55/Conv2D/ReadVariableOp2N
%model_7/dense_3/MatMul/ReadVariableOp%model_7/dense_3/MatMul/ReadVariableOp2?
Fmodel_7/model_4/batch_normalization_12/FusedBatchNormV3/ReadVariableOpFmodel_7/model_4/batch_normalization_12/FusedBatchNormV3/ReadVariableOp2?
Hmodel_7/model_4/batch_normalization_12/FusedBatchNormV3/ReadVariableOp_1Hmodel_7/model_4/batch_normalization_12/FusedBatchNormV3/ReadVariableOp_12n
5model_7/model_4/batch_normalization_12/ReadVariableOp5model_7/model_4/batch_normalization_12/ReadVariableOp2r
7model_7/model_4/batch_normalization_12/ReadVariableOp_17model_7/model_4/batch_normalization_12/ReadVariableOp_12?
Fmodel_7/model_4/batch_normalization_13/FusedBatchNormV3/ReadVariableOpFmodel_7/model_4/batch_normalization_13/FusedBatchNormV3/ReadVariableOp2?
Hmodel_7/model_4/batch_normalization_13/FusedBatchNormV3/ReadVariableOp_1Hmodel_7/model_4/batch_normalization_13/FusedBatchNormV3/ReadVariableOp_12n
5model_7/model_4/batch_normalization_13/ReadVariableOp5model_7/model_4/batch_normalization_13/ReadVariableOp2r
7model_7/model_4/batch_normalization_13/ReadVariableOp_17model_7/model_4/batch_normalization_13/ReadVariableOp_12?
Fmodel_7/model_4/batch_normalization_14/FusedBatchNormV3/ReadVariableOpFmodel_7/model_4/batch_normalization_14/FusedBatchNormV3/ReadVariableOp2?
Hmodel_7/model_4/batch_normalization_14/FusedBatchNormV3/ReadVariableOp_1Hmodel_7/model_4/batch_normalization_14/FusedBatchNormV3/ReadVariableOp_12n
5model_7/model_4/batch_normalization_14/ReadVariableOp5model_7/model_4/batch_normalization_14/ReadVariableOp2r
7model_7/model_4/batch_normalization_14/ReadVariableOp_17model_7/model_4/batch_normalization_14/ReadVariableOp_12?
Fmodel_7/model_4/batch_normalization_15/FusedBatchNormV3/ReadVariableOpFmodel_7/model_4/batch_normalization_15/FusedBatchNormV3/ReadVariableOp2?
Hmodel_7/model_4/batch_normalization_15/FusedBatchNormV3/ReadVariableOp_1Hmodel_7/model_4/batch_normalization_15/FusedBatchNormV3/ReadVariableOp_12n
5model_7/model_4/batch_normalization_15/ReadVariableOp5model_7/model_4/batch_normalization_15/ReadVariableOp2r
7model_7/model_4/batch_normalization_15/ReadVariableOp_17model_7/model_4/batch_normalization_15/ReadVariableOp_12?
Fmodel_7/model_4/batch_normalization_16/FusedBatchNormV3/ReadVariableOpFmodel_7/model_4/batch_normalization_16/FusedBatchNormV3/ReadVariableOp2?
Hmodel_7/model_4/batch_normalization_16/FusedBatchNormV3/ReadVariableOp_1Hmodel_7/model_4/batch_normalization_16/FusedBatchNormV3/ReadVariableOp_12n
5model_7/model_4/batch_normalization_16/ReadVariableOp5model_7/model_4/batch_normalization_16/ReadVariableOp2r
7model_7/model_4/batch_normalization_16/ReadVariableOp_17model_7/model_4/batch_normalization_16/ReadVariableOp_12?
Fmodel_7/model_4/batch_normalization_17/FusedBatchNormV3/ReadVariableOpFmodel_7/model_4/batch_normalization_17/FusedBatchNormV3/ReadVariableOp2?
Hmodel_7/model_4/batch_normalization_17/FusedBatchNormV3/ReadVariableOp_1Hmodel_7/model_4/batch_normalization_17/FusedBatchNormV3/ReadVariableOp_12n
5model_7/model_4/batch_normalization_17/ReadVariableOp5model_7/model_4/batch_normalization_17/ReadVariableOp2r
7model_7/model_4/batch_normalization_17/ReadVariableOp_17model_7/model_4/batch_normalization_17/ReadVariableOp_12d
0model_7/model_4/conv2d_28/BiasAdd/ReadVariableOp0model_7/model_4/conv2d_28/BiasAdd/ReadVariableOp2b
/model_7/model_4/conv2d_28/Conv2D/ReadVariableOp/model_7/model_4/conv2d_28/Conv2D/ReadVariableOp2d
0model_7/model_4/conv2d_29/BiasAdd/ReadVariableOp0model_7/model_4/conv2d_29/BiasAdd/ReadVariableOp2b
/model_7/model_4/conv2d_29/Conv2D/ReadVariableOp/model_7/model_4/conv2d_29/Conv2D/ReadVariableOp2d
0model_7/model_4/conv2d_30/BiasAdd/ReadVariableOp0model_7/model_4/conv2d_30/BiasAdd/ReadVariableOp2b
/model_7/model_4/conv2d_30/Conv2D/ReadVariableOp/model_7/model_4/conv2d_30/Conv2D/ReadVariableOp2d
0model_7/model_4/conv2d_31/BiasAdd/ReadVariableOp0model_7/model_4/conv2d_31/BiasAdd/ReadVariableOp2b
/model_7/model_4/conv2d_31/Conv2D/ReadVariableOp/model_7/model_4/conv2d_31/Conv2D/ReadVariableOp2d
0model_7/model_4/conv2d_32/BiasAdd/ReadVariableOp0model_7/model_4/conv2d_32/BiasAdd/ReadVariableOp2b
/model_7/model_4/conv2d_32/Conv2D/ReadVariableOp/model_7/model_4/conv2d_32/Conv2D/ReadVariableOp2d
0model_7/model_4/conv2d_33/BiasAdd/ReadVariableOp0model_7/model_4/conv2d_33/BiasAdd/ReadVariableOp2b
/model_7/model_4/conv2d_33/Conv2D/ReadVariableOp/model_7/model_4/conv2d_33/Conv2D/ReadVariableOp2d
0model_7/model_4/conv2d_34/BiasAdd/ReadVariableOp0model_7/model_4/conv2d_34/BiasAdd/ReadVariableOp2b
/model_7/model_4/conv2d_34/Conv2D/ReadVariableOp/model_7/model_4/conv2d_34/Conv2D/ReadVariableOp2d
0model_7/model_4/conv2d_35/BiasAdd/ReadVariableOp0model_7/model_4/conv2d_35/BiasAdd/ReadVariableOp2b
/model_7/model_4/conv2d_35/Conv2D/ReadVariableOp/model_7/model_4/conv2d_35/Conv2D/ReadVariableOp2d
0model_7/model_4/conv2d_36/BiasAdd/ReadVariableOp0model_7/model_4/conv2d_36/BiasAdd/ReadVariableOp2b
/model_7/model_4/conv2d_36/Conv2D/ReadVariableOp/model_7/model_4/conv2d_36/Conv2D/ReadVariableOp2d
0model_7/model_4/conv2d_37/BiasAdd/ReadVariableOp0model_7/model_4/conv2d_37/BiasAdd/ReadVariableOp2b
/model_7/model_4/conv2d_37/Conv2D/ReadVariableOp/model_7/model_4/conv2d_37/Conv2D/ReadVariableOp2d
0model_7/model_4/conv2d_38/BiasAdd/ReadVariableOp0model_7/model_4/conv2d_38/BiasAdd/ReadVariableOp2b
/model_7/model_4/conv2d_38/Conv2D/ReadVariableOp/model_7/model_4/conv2d_38/Conv2D/ReadVariableOp:X T
/
_output_shapes
:?????????
!
_user_specified_name	input_8
?
f
H__inference_dropout_30_layer_call_and_return_conditional_losses_22008075

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
?
?
T__inference_batch_normalization_13_layer_call_and_return_conditional_losses_22003199

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
f
H__inference_dropout_51_layer_call_and_return_conditional_losses_22007378

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
T__inference_batch_normalization_15_layer_call_and_return_conditional_losses_22003375

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
g
K__inference_activation_28_layer_call_and_return_conditional_losses_22008156

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
?
d
H__inference_dropout_33_layer_call_and_return_conditional_losses_22008347

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
?
?
,__inference_conv2d_34_layer_call_fn_22008136

inputs#
unknown:??
	unknown_0:	?
identity??StatefulPartitionedCall?
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
GPU 2J 8? *P
fKRI
G__inference_conv2d_34_layer_call_and_return_conditional_losses_220033442
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
?
?
,__inference_conv2d_53_layer_call_fn_22007391

inputs#
unknown:??
	unknown_0:	?
identity??StatefulPartitionedCall?
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
GPU 2J 8? *P
fKRI
G__inference_conv2d_53_layer_call_and_return_conditional_losses_220051082
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
?
?
,__inference_conv2d_31_layer_call_fn_22007868

inputs#
unknown:??
	unknown_0:	?
identity??StatefulPartitionedCall?
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
GPU 2J 8? *P
fKRI
G__inference_conv2d_31_layer_call_and_return_conditional_losses_220032262
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
?	
?
9__inference_batch_normalization_14_layer_call_fn_22007988

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
GPU 2J 8? *]
fXRV
T__inference_batch_normalization_14_layer_call_and_return_conditional_losses_220040302
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
?
g
K__inference_activation_30_layer_call_and_return_conditional_losses_22008376

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
,__inference_conv2d_32_layer_call_fn_22007916

inputs#
unknown:??
	unknown_0:	?
identity??StatefulPartitionedCall?
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
GPU 2J 8? *P
fKRI
G__inference_conv2d_32_layer_call_and_return_conditional_losses_220032562
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
?
?
T__inference_batch_normalization_16_layer_call_and_return_conditional_losses_22003818

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
ֺ
?
E__inference_model_4_layer_call_and_return_conditional_losses_22003569

inputs-
conv2d_28_22003082:?!
conv2d_28_22003084:	?.
batch_normalization_12_22003105:	?.
batch_normalization_12_22003107:	?.
batch_normalization_12_22003109:	?.
batch_normalization_12_22003111:	?.
conv2d_29_22003139:??!
conv2d_29_22003141:	?.
conv2d_30_22003169:??!
conv2d_30_22003171:	?.
batch_normalization_13_22003200:	?.
batch_normalization_13_22003202:	?.
batch_normalization_13_22003204:	?.
batch_normalization_13_22003206:	?.
conv2d_31_22003227:??!
conv2d_31_22003229:	?.
conv2d_32_22003257:??!
conv2d_32_22003259:	?.
batch_normalization_14_22003288:	?.
batch_normalization_14_22003290:	?.
batch_normalization_14_22003292:	?.
batch_normalization_14_22003294:	?.
conv2d_33_22003315:??!
conv2d_33_22003317:	?.
conv2d_34_22003345:??!
conv2d_34_22003347:	?.
batch_normalization_15_22003376:	?.
batch_normalization_15_22003378:	?.
batch_normalization_15_22003380:	?.
batch_normalization_15_22003382:	?.
conv2d_35_22003403:??!
conv2d_35_22003405:	?.
conv2d_36_22003433:??!
conv2d_36_22003435:	?.
batch_normalization_16_22003464:	?.
batch_normalization_16_22003466:	?.
batch_normalization_16_22003468:	?.
batch_normalization_16_22003470:	?.
conv2d_37_22003491:??!
conv2d_37_22003493:	?.
conv2d_38_22003521:??!
conv2d_38_22003523:	?.
batch_normalization_17_22003552:	?.
batch_normalization_17_22003554:	?.
batch_normalization_17_22003556:	?.
batch_normalization_17_22003558:	?
identity??.batch_normalization_12/StatefulPartitionedCall?.batch_normalization_13/StatefulPartitionedCall?.batch_normalization_14/StatefulPartitionedCall?.batch_normalization_15/StatefulPartitionedCall?.batch_normalization_16/StatefulPartitionedCall?.batch_normalization_17/StatefulPartitionedCall?!conv2d_28/StatefulPartitionedCall?!conv2d_29/StatefulPartitionedCall?!conv2d_30/StatefulPartitionedCall?!conv2d_31/StatefulPartitionedCall?!conv2d_32/StatefulPartitionedCall?!conv2d_33/StatefulPartitionedCall?!conv2d_34/StatefulPartitionedCall?!conv2d_35/StatefulPartitionedCall?!conv2d_36/StatefulPartitionedCall?!conv2d_37/StatefulPartitionedCall?!conv2d_38/StatefulPartitionedCall?
!conv2d_28/StatefulPartitionedCallStatefulPartitionedCallinputsconv2d_28_22003082conv2d_28_22003084*
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
GPU 2J 8? *P
fKRI
G__inference_conv2d_28_layer_call_and_return_conditional_losses_220030812#
!conv2d_28/StatefulPartitionedCall?
.batch_normalization_12/StatefulPartitionedCallStatefulPartitionedCall*conv2d_28/StatefulPartitionedCall:output:0batch_normalization_12_22003105batch_normalization_12_22003107batch_normalization_12_22003109batch_normalization_12_22003111*
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
GPU 2J 8? *]
fXRV
T__inference_batch_normalization_12_layer_call_and_return_conditional_losses_2200310420
.batch_normalization_12/StatefulPartitionedCall?
activation_22/PartitionedCallPartitionedCall7batch_normalization_12/StatefulPartitionedCall:output:0*
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
GPU 2J 8? *T
fORM
K__inference_activation_22_layer_call_and_return_conditional_losses_220031192
activation_22/PartitionedCall?
dropout_26/PartitionedCallPartitionedCall&activation_22/PartitionedCall:output:0*
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
GPU 2J 8? *Q
fLRJ
H__inference_dropout_26_layer_call_and_return_conditional_losses_220031262
dropout_26/PartitionedCall?
!conv2d_29/StatefulPartitionedCallStatefulPartitionedCall#dropout_26/PartitionedCall:output:0conv2d_29_22003139conv2d_29_22003141*
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
GPU 2J 8? *P
fKRI
G__inference_conv2d_29_layer_call_and_return_conditional_losses_220031382#
!conv2d_29/StatefulPartitionedCall?
activation_23/PartitionedCallPartitionedCall*conv2d_29/StatefulPartitionedCall:output:0*
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
GPU 2J 8? *T
fORM
K__inference_activation_23_layer_call_and_return_conditional_losses_220031492
activation_23/PartitionedCall?
dropout_27/PartitionedCallPartitionedCall&activation_23/PartitionedCall:output:0*
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
GPU 2J 8? *Q
fLRJ
H__inference_dropout_27_layer_call_and_return_conditional_losses_220031562
dropout_27/PartitionedCall?
!conv2d_30/StatefulPartitionedCallStatefulPartitionedCall#dropout_27/PartitionedCall:output:0conv2d_30_22003169conv2d_30_22003171*
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
GPU 2J 8? *P
fKRI
G__inference_conv2d_30_layer_call_and_return_conditional_losses_220031682#
!conv2d_30/StatefulPartitionedCall?
tf.__operators__.add_10/AddV2AddV2*conv2d_30/StatefulPartitionedCall:output:0#dropout_26/PartitionedCall:output:0*
T0*0
_output_shapes
:??????????2
tf.__operators__.add_10/AddV2?
activation_24/PartitionedCallPartitionedCall!tf.__operators__.add_10/AddV2:z:0*
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
GPU 2J 8? *T
fORM
K__inference_activation_24_layer_call_and_return_conditional_losses_220031802
activation_24/PartitionedCall?
.batch_normalization_13/StatefulPartitionedCallStatefulPartitionedCall&activation_24/PartitionedCall:output:0batch_normalization_13_22003200batch_normalization_13_22003202batch_normalization_13_22003204batch_normalization_13_22003206*
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
GPU 2J 8? *]
fXRV
T__inference_batch_normalization_13_layer_call_and_return_conditional_losses_2200319920
.batch_normalization_13/StatefulPartitionedCall?
dropout_28/PartitionedCallPartitionedCall7batch_normalization_13/StatefulPartitionedCall:output:0*
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
GPU 2J 8? *Q
fLRJ
H__inference_dropout_28_layer_call_and_return_conditional_losses_220032142
dropout_28/PartitionedCall?
!conv2d_31/StatefulPartitionedCallStatefulPartitionedCall#dropout_28/PartitionedCall:output:0conv2d_31_22003227conv2d_31_22003229*
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
GPU 2J 8? *P
fKRI
G__inference_conv2d_31_layer_call_and_return_conditional_losses_220032262#
!conv2d_31/StatefulPartitionedCall?
activation_25/PartitionedCallPartitionedCall*conv2d_31/StatefulPartitionedCall:output:0*
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
GPU 2J 8? *T
fORM
K__inference_activation_25_layer_call_and_return_conditional_losses_220032372
activation_25/PartitionedCall?
dropout_29/PartitionedCallPartitionedCall&activation_25/PartitionedCall:output:0*
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
GPU 2J 8? *Q
fLRJ
H__inference_dropout_29_layer_call_and_return_conditional_losses_220032442
dropout_29/PartitionedCall?
!conv2d_32/StatefulPartitionedCallStatefulPartitionedCall#dropout_29/PartitionedCall:output:0conv2d_32_22003257conv2d_32_22003259*
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
GPU 2J 8? *P
fKRI
G__inference_conv2d_32_layer_call_and_return_conditional_losses_220032562#
!conv2d_32/StatefulPartitionedCall?
tf.__operators__.add_11/AddV2AddV2*conv2d_32/StatefulPartitionedCall:output:0#dropout_28/PartitionedCall:output:0*
T0*0
_output_shapes
:??????????2
tf.__operators__.add_11/AddV2?
activation_26/PartitionedCallPartitionedCall!tf.__operators__.add_11/AddV2:z:0*
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
GPU 2J 8? *T
fORM
K__inference_activation_26_layer_call_and_return_conditional_losses_220032682
activation_26/PartitionedCall?
.batch_normalization_14/StatefulPartitionedCallStatefulPartitionedCall&activation_26/PartitionedCall:output:0batch_normalization_14_22003288batch_normalization_14_22003290batch_normalization_14_22003292batch_normalization_14_22003294*
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
GPU 2J 8? *]
fXRV
T__inference_batch_normalization_14_layer_call_and_return_conditional_losses_2200328720
.batch_normalization_14/StatefulPartitionedCall?
dropout_30/PartitionedCallPartitionedCall7batch_normalization_14/StatefulPartitionedCall:output:0*
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
GPU 2J 8? *Q
fLRJ
H__inference_dropout_30_layer_call_and_return_conditional_losses_220033022
dropout_30/PartitionedCall?
!conv2d_33/StatefulPartitionedCallStatefulPartitionedCall#dropout_30/PartitionedCall:output:0conv2d_33_22003315conv2d_33_22003317*
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
GPU 2J 8? *P
fKRI
G__inference_conv2d_33_layer_call_and_return_conditional_losses_220033142#
!conv2d_33/StatefulPartitionedCall?
activation_27/PartitionedCallPartitionedCall*conv2d_33/StatefulPartitionedCall:output:0*
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
GPU 2J 8? *T
fORM
K__inference_activation_27_layer_call_and_return_conditional_losses_220033252
activation_27/PartitionedCall?
dropout_31/PartitionedCallPartitionedCall&activation_27/PartitionedCall:output:0*
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
GPU 2J 8? *Q
fLRJ
H__inference_dropout_31_layer_call_and_return_conditional_losses_220033322
dropout_31/PartitionedCall?
!conv2d_34/StatefulPartitionedCallStatefulPartitionedCall#dropout_31/PartitionedCall:output:0conv2d_34_22003345conv2d_34_22003347*
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
GPU 2J 8? *P
fKRI
G__inference_conv2d_34_layer_call_and_return_conditional_losses_220033442#
!conv2d_34/StatefulPartitionedCall?
tf.__operators__.add_12/AddV2AddV2*conv2d_34/StatefulPartitionedCall:output:0#dropout_30/PartitionedCall:output:0*
T0*0
_output_shapes
:??????????2
tf.__operators__.add_12/AddV2?
activation_28/PartitionedCallPartitionedCall!tf.__operators__.add_12/AddV2:z:0*
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
GPU 2J 8? *T
fORM
K__inference_activation_28_layer_call_and_return_conditional_losses_220033562
activation_28/PartitionedCall?
.batch_normalization_15/StatefulPartitionedCallStatefulPartitionedCall&activation_28/PartitionedCall:output:0batch_normalization_15_22003376batch_normalization_15_22003378batch_normalization_15_22003380batch_normalization_15_22003382*
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
GPU 2J 8? *]
fXRV
T__inference_batch_normalization_15_layer_call_and_return_conditional_losses_2200337520
.batch_normalization_15/StatefulPartitionedCall?
dropout_32/PartitionedCallPartitionedCall7batch_normalization_15/StatefulPartitionedCall:output:0*
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
GPU 2J 8? *Q
fLRJ
H__inference_dropout_32_layer_call_and_return_conditional_losses_220033902
dropout_32/PartitionedCall?
!conv2d_35/StatefulPartitionedCallStatefulPartitionedCall#dropout_32/PartitionedCall:output:0conv2d_35_22003403conv2d_35_22003405*
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
GPU 2J 8? *P
fKRI
G__inference_conv2d_35_layer_call_and_return_conditional_losses_220034022#
!conv2d_35/StatefulPartitionedCall?
activation_29/PartitionedCallPartitionedCall*conv2d_35/StatefulPartitionedCall:output:0*
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
GPU 2J 8? *T
fORM
K__inference_activation_29_layer_call_and_return_conditional_losses_220034132
activation_29/PartitionedCall?
dropout_33/PartitionedCallPartitionedCall&activation_29/PartitionedCall:output:0*
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
GPU 2J 8? *Q
fLRJ
H__inference_dropout_33_layer_call_and_return_conditional_losses_220034202
dropout_33/PartitionedCall?
!conv2d_36/StatefulPartitionedCallStatefulPartitionedCall#dropout_33/PartitionedCall:output:0conv2d_36_22003433conv2d_36_22003435*
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
GPU 2J 8? *P
fKRI
G__inference_conv2d_36_layer_call_and_return_conditional_losses_220034322#
!conv2d_36/StatefulPartitionedCall?
tf.__operators__.add_13/AddV2AddV2*conv2d_36/StatefulPartitionedCall:output:0#dropout_32/PartitionedCall:output:0*
T0*0
_output_shapes
:??????????2
tf.__operators__.add_13/AddV2?
activation_30/PartitionedCallPartitionedCall!tf.__operators__.add_13/AddV2:z:0*
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
GPU 2J 8? *T
fORM
K__inference_activation_30_layer_call_and_return_conditional_losses_220034442
activation_30/PartitionedCall?
.batch_normalization_16/StatefulPartitionedCallStatefulPartitionedCall&activation_30/PartitionedCall:output:0batch_normalization_16_22003464batch_normalization_16_22003466batch_normalization_16_22003468batch_normalization_16_22003470*
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
GPU 2J 8? *]
fXRV
T__inference_batch_normalization_16_layer_call_and_return_conditional_losses_2200346320
.batch_normalization_16/StatefulPartitionedCall?
dropout_34/PartitionedCallPartitionedCall7batch_normalization_16/StatefulPartitionedCall:output:0*
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
GPU 2J 8? *Q
fLRJ
H__inference_dropout_34_layer_call_and_return_conditional_losses_220034782
dropout_34/PartitionedCall?
!conv2d_37/StatefulPartitionedCallStatefulPartitionedCall#dropout_34/PartitionedCall:output:0conv2d_37_22003491conv2d_37_22003493*
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
GPU 2J 8? *P
fKRI
G__inference_conv2d_37_layer_call_and_return_conditional_losses_220034902#
!conv2d_37/StatefulPartitionedCall?
activation_31/PartitionedCallPartitionedCall*conv2d_37/StatefulPartitionedCall:output:0*
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
GPU 2J 8? *T
fORM
K__inference_activation_31_layer_call_and_return_conditional_losses_220035012
activation_31/PartitionedCall?
dropout_35/PartitionedCallPartitionedCall&activation_31/PartitionedCall:output:0*
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
GPU 2J 8? *Q
fLRJ
H__inference_dropout_35_layer_call_and_return_conditional_losses_220035082
dropout_35/PartitionedCall?
!conv2d_38/StatefulPartitionedCallStatefulPartitionedCall#dropout_35/PartitionedCall:output:0conv2d_38_22003521conv2d_38_22003523*
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
GPU 2J 8? *P
fKRI
G__inference_conv2d_38_layer_call_and_return_conditional_losses_220035202#
!conv2d_38/StatefulPartitionedCall?
tf.__operators__.add_14/AddV2AddV2*conv2d_38/StatefulPartitionedCall:output:0#dropout_34/PartitionedCall:output:0*
T0*0
_output_shapes
:??????????2
tf.__operators__.add_14/AddV2?
activation_32/PartitionedCallPartitionedCall!tf.__operators__.add_14/AddV2:z:0*
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
GPU 2J 8? *T
fORM
K__inference_activation_32_layer_call_and_return_conditional_losses_220035322
activation_32/PartitionedCall?
.batch_normalization_17/StatefulPartitionedCallStatefulPartitionedCall&activation_32/PartitionedCall:output:0batch_normalization_17_22003552batch_normalization_17_22003554batch_normalization_17_22003556batch_normalization_17_22003558*
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
GPU 2J 8? *]
fXRV
T__inference_batch_normalization_17_layer_call_and_return_conditional_losses_2200355120
.batch_normalization_17/StatefulPartitionedCall?
dropout_36/PartitionedCallPartitionedCall7batch_normalization_17/StatefulPartitionedCall:output:0*
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
GPU 2J 8? *Q
fLRJ
H__inference_dropout_36_layer_call_and_return_conditional_losses_220035662
dropout_36/PartitionedCall?
IdentityIdentity#dropout_36/PartitionedCall:output:0^NoOp*
T0*0
_output_shapes
:??????????2

Identity?
NoOpNoOp/^batch_normalization_12/StatefulPartitionedCall/^batch_normalization_13/StatefulPartitionedCall/^batch_normalization_14/StatefulPartitionedCall/^batch_normalization_15/StatefulPartitionedCall/^batch_normalization_16/StatefulPartitionedCall/^batch_normalization_17/StatefulPartitionedCall"^conv2d_28/StatefulPartitionedCall"^conv2d_29/StatefulPartitionedCall"^conv2d_30/StatefulPartitionedCall"^conv2d_31/StatefulPartitionedCall"^conv2d_32/StatefulPartitionedCall"^conv2d_33/StatefulPartitionedCall"^conv2d_34/StatefulPartitionedCall"^conv2d_35/StatefulPartitionedCall"^conv2d_36/StatefulPartitionedCall"^conv2d_37/StatefulPartitionedCall"^conv2d_38/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapesy
w:?????????: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2`
.batch_normalization_12/StatefulPartitionedCall.batch_normalization_12/StatefulPartitionedCall2`
.batch_normalization_13/StatefulPartitionedCall.batch_normalization_13/StatefulPartitionedCall2`
.batch_normalization_14/StatefulPartitionedCall.batch_normalization_14/StatefulPartitionedCall2`
.batch_normalization_15/StatefulPartitionedCall.batch_normalization_15/StatefulPartitionedCall2`
.batch_normalization_16/StatefulPartitionedCall.batch_normalization_16/StatefulPartitionedCall2`
.batch_normalization_17/StatefulPartitionedCall.batch_normalization_17/StatefulPartitionedCall2F
!conv2d_28/StatefulPartitionedCall!conv2d_28/StatefulPartitionedCall2F
!conv2d_29/StatefulPartitionedCall!conv2d_29/StatefulPartitionedCall2F
!conv2d_30/StatefulPartitionedCall!conv2d_30/StatefulPartitionedCall2F
!conv2d_31/StatefulPartitionedCall!conv2d_31/StatefulPartitionedCall2F
!conv2d_32/StatefulPartitionedCall!conv2d_32/StatefulPartitionedCall2F
!conv2d_33/StatefulPartitionedCall!conv2d_33/StatefulPartitionedCall2F
!conv2d_34/StatefulPartitionedCall!conv2d_34/StatefulPartitionedCall2F
!conv2d_35/StatefulPartitionedCall!conv2d_35/StatefulPartitionedCall2F
!conv2d_36/StatefulPartitionedCall!conv2d_36/StatefulPartitionedCall2F
!conv2d_37/StatefulPartitionedCall!conv2d_37/StatefulPartitionedCall2F
!conv2d_38/StatefulPartitionedCall!conv2d_38/StatefulPartitionedCall:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?
f
H__inference_dropout_34_layer_call_and_return_conditional_losses_22003478

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
g
K__inference_activation_24_layer_call_and_return_conditional_losses_22007716

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
?
?
G__inference_conv2d_29_layer_call_and_return_conditional_losses_22003138

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
?
I
-__inference_dropout_28_layer_call_fn_22007850

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
GPU 2J 8? *Q
fLRJ
H__inference_dropout_28_layer_call_and_return_conditional_losses_220041002
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
L
0__inference_activation_27_layer_call_fn_22008103

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
GPU 2J 8? *T
fORM
K__inference_activation_27_layer_call_and_return_conditional_losses_220033252
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
?
d
H__inference_dropout_30_layer_call_and_return_conditional_losses_22003994

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
T__inference_batch_normalization_12_layer_call_and_return_conditional_losses_22007556

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
G__inference_conv2d_36_layer_call_and_return_conditional_losses_22008366

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
?
?
T__inference_batch_normalization_17_layer_call_and_return_conditional_losses_22008684

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
T__inference_batch_normalization_15_layer_call_and_return_conditional_losses_22002708

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
?
?
T__inference_batch_normalization_15_layer_call_and_return_conditional_losses_22008244

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
?
?
T__inference_batch_normalization_17_layer_call_and_return_conditional_losses_22003712

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
g
K__inference_activation_26_layer_call_and_return_conditional_losses_22007936

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
?
?
G__inference_conv2d_55_layer_call_and_return_conditional_losses_22005091

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
?
I
-__inference_dropout_36_layer_call_fn_22008730

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
GPU 2J 8? *Q
fLRJ
H__inference_dropout_36_layer_call_and_return_conditional_losses_220036762
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
G__inference_conv2d_32_layer_call_and_return_conditional_losses_22003256

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
9__inference_batch_normalization_12_layer_call_fn_22007512

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
GPU 2J 8? *]
fXRV
T__inference_batch_normalization_12_layer_call_and_return_conditional_losses_220023742
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
f
H__inference_dropout_33_layer_call_and_return_conditional_losses_22003420

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
d
H__inference_dropout_31_layer_call_and_return_conditional_losses_22003963

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
g
K__inference_activation_25_layer_call_and_return_conditional_losses_22003237

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
T__inference_batch_normalization_17_layer_call_and_return_conditional_losses_22003551

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
d
H__inference_dropout_51_layer_call_and_return_conditional_losses_22007382

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
??
?(
E__inference_model_4_layer_call_and_return_conditional_losses_22007173

inputsC
(conv2d_28_conv2d_readvariableop_resource:?8
)conv2d_28_biasadd_readvariableop_resource:	?=
.batch_normalization_12_readvariableop_resource:	??
0batch_normalization_12_readvariableop_1_resource:	?N
?batch_normalization_12_fusedbatchnormv3_readvariableop_resource:	?P
Abatch_normalization_12_fusedbatchnormv3_readvariableop_1_resource:	?D
(conv2d_29_conv2d_readvariableop_resource:??8
)conv2d_29_biasadd_readvariableop_resource:	?D
(conv2d_30_conv2d_readvariableop_resource:??8
)conv2d_30_biasadd_readvariableop_resource:	?=
.batch_normalization_13_readvariableop_resource:	??
0batch_normalization_13_readvariableop_1_resource:	?N
?batch_normalization_13_fusedbatchnormv3_readvariableop_resource:	?P
Abatch_normalization_13_fusedbatchnormv3_readvariableop_1_resource:	?D
(conv2d_31_conv2d_readvariableop_resource:??8
)conv2d_31_biasadd_readvariableop_resource:	?D
(conv2d_32_conv2d_readvariableop_resource:??8
)conv2d_32_biasadd_readvariableop_resource:	?=
.batch_normalization_14_readvariableop_resource:	??
0batch_normalization_14_readvariableop_1_resource:	?N
?batch_normalization_14_fusedbatchnormv3_readvariableop_resource:	?P
Abatch_normalization_14_fusedbatchnormv3_readvariableop_1_resource:	?D
(conv2d_33_conv2d_readvariableop_resource:??8
)conv2d_33_biasadd_readvariableop_resource:	?D
(conv2d_34_conv2d_readvariableop_resource:??8
)conv2d_34_biasadd_readvariableop_resource:	?=
.batch_normalization_15_readvariableop_resource:	??
0batch_normalization_15_readvariableop_1_resource:	?N
?batch_normalization_15_fusedbatchnormv3_readvariableop_resource:	?P
Abatch_normalization_15_fusedbatchnormv3_readvariableop_1_resource:	?D
(conv2d_35_conv2d_readvariableop_resource:??8
)conv2d_35_biasadd_readvariableop_resource:	?D
(conv2d_36_conv2d_readvariableop_resource:??8
)conv2d_36_biasadd_readvariableop_resource:	?=
.batch_normalization_16_readvariableop_resource:	??
0batch_normalization_16_readvariableop_1_resource:	?N
?batch_normalization_16_fusedbatchnormv3_readvariableop_resource:	?P
Abatch_normalization_16_fusedbatchnormv3_readvariableop_1_resource:	?D
(conv2d_37_conv2d_readvariableop_resource:??8
)conv2d_37_biasadd_readvariableop_resource:	?D
(conv2d_38_conv2d_readvariableop_resource:??8
)conv2d_38_biasadd_readvariableop_resource:	?=
.batch_normalization_17_readvariableop_resource:	??
0batch_normalization_17_readvariableop_1_resource:	?N
?batch_normalization_17_fusedbatchnormv3_readvariableop_resource:	?P
Abatch_normalization_17_fusedbatchnormv3_readvariableop_1_resource:	?
identity??6batch_normalization_12/FusedBatchNormV3/ReadVariableOp?8batch_normalization_12/FusedBatchNormV3/ReadVariableOp_1?%batch_normalization_12/ReadVariableOp?'batch_normalization_12/ReadVariableOp_1?6batch_normalization_13/FusedBatchNormV3/ReadVariableOp?8batch_normalization_13/FusedBatchNormV3/ReadVariableOp_1?%batch_normalization_13/ReadVariableOp?'batch_normalization_13/ReadVariableOp_1?6batch_normalization_14/FusedBatchNormV3/ReadVariableOp?8batch_normalization_14/FusedBatchNormV3/ReadVariableOp_1?%batch_normalization_14/ReadVariableOp?'batch_normalization_14/ReadVariableOp_1?6batch_normalization_15/FusedBatchNormV3/ReadVariableOp?8batch_normalization_15/FusedBatchNormV3/ReadVariableOp_1?%batch_normalization_15/ReadVariableOp?'batch_normalization_15/ReadVariableOp_1?6batch_normalization_16/FusedBatchNormV3/ReadVariableOp?8batch_normalization_16/FusedBatchNormV3/ReadVariableOp_1?%batch_normalization_16/ReadVariableOp?'batch_normalization_16/ReadVariableOp_1?6batch_normalization_17/FusedBatchNormV3/ReadVariableOp?8batch_normalization_17/FusedBatchNormV3/ReadVariableOp_1?%batch_normalization_17/ReadVariableOp?'batch_normalization_17/ReadVariableOp_1? conv2d_28/BiasAdd/ReadVariableOp?conv2d_28/Conv2D/ReadVariableOp? conv2d_29/BiasAdd/ReadVariableOp?conv2d_29/Conv2D/ReadVariableOp? conv2d_30/BiasAdd/ReadVariableOp?conv2d_30/Conv2D/ReadVariableOp? conv2d_31/BiasAdd/ReadVariableOp?conv2d_31/Conv2D/ReadVariableOp? conv2d_32/BiasAdd/ReadVariableOp?conv2d_32/Conv2D/ReadVariableOp? conv2d_33/BiasAdd/ReadVariableOp?conv2d_33/Conv2D/ReadVariableOp? conv2d_34/BiasAdd/ReadVariableOp?conv2d_34/Conv2D/ReadVariableOp? conv2d_35/BiasAdd/ReadVariableOp?conv2d_35/Conv2D/ReadVariableOp? conv2d_36/BiasAdd/ReadVariableOp?conv2d_36/Conv2D/ReadVariableOp? conv2d_37/BiasAdd/ReadVariableOp?conv2d_37/Conv2D/ReadVariableOp? conv2d_38/BiasAdd/ReadVariableOp?conv2d_38/Conv2D/ReadVariableOp?
conv2d_28/Conv2D/ReadVariableOpReadVariableOp(conv2d_28_conv2d_readvariableop_resource*'
_output_shapes
:?*
dtype02!
conv2d_28/Conv2D/ReadVariableOp?
conv2d_28/Conv2DConv2Dinputs'conv2d_28/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????*
paddingSAME*
strides
2
conv2d_28/Conv2D?
 conv2d_28/BiasAdd/ReadVariableOpReadVariableOp)conv2d_28_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02"
 conv2d_28/BiasAdd/ReadVariableOp?
conv2d_28/BiasAddBiasAddconv2d_28/Conv2D:output:0(conv2d_28/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????2
conv2d_28/BiasAdd?
%batch_normalization_12/ReadVariableOpReadVariableOp.batch_normalization_12_readvariableop_resource*
_output_shapes	
:?*
dtype02'
%batch_normalization_12/ReadVariableOp?
'batch_normalization_12/ReadVariableOp_1ReadVariableOp0batch_normalization_12_readvariableop_1_resource*
_output_shapes	
:?*
dtype02)
'batch_normalization_12/ReadVariableOp_1?
6batch_normalization_12/FusedBatchNormV3/ReadVariableOpReadVariableOp?batch_normalization_12_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:?*
dtype028
6batch_normalization_12/FusedBatchNormV3/ReadVariableOp?
8batch_normalization_12/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpAbatch_normalization_12_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:?*
dtype02:
8batch_normalization_12/FusedBatchNormV3/ReadVariableOp_1?
'batch_normalization_12/FusedBatchNormV3FusedBatchNormV3conv2d_28/BiasAdd:output:0-batch_normalization_12/ReadVariableOp:value:0/batch_normalization_12/ReadVariableOp_1:value:0>batch_normalization_12/FusedBatchNormV3/ReadVariableOp:value:0@batch_normalization_12/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:??????????:?:?:?:?:*
epsilon%o?:*
is_training( 2)
'batch_normalization_12/FusedBatchNormV3?
activation_22/ReluRelu+batch_normalization_12/FusedBatchNormV3:y:0*
T0*0
_output_shapes
:??????????2
activation_22/Relu?
dropout_26/IdentityIdentity activation_22/Relu:activations:0*
T0*0
_output_shapes
:??????????2
dropout_26/Identity?
conv2d_29/Conv2D/ReadVariableOpReadVariableOp(conv2d_29_conv2d_readvariableop_resource*(
_output_shapes
:??*
dtype02!
conv2d_29/Conv2D/ReadVariableOp?
conv2d_29/Conv2DConv2Ddropout_26/Identity:output:0'conv2d_29/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????*
paddingSAME*
strides
2
conv2d_29/Conv2D?
 conv2d_29/BiasAdd/ReadVariableOpReadVariableOp)conv2d_29_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02"
 conv2d_29/BiasAdd/ReadVariableOp?
conv2d_29/BiasAddBiasAddconv2d_29/Conv2D:output:0(conv2d_29/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????2
conv2d_29/BiasAdd?
activation_23/ReluReluconv2d_29/BiasAdd:output:0*
T0*0
_output_shapes
:??????????2
activation_23/Relu?
dropout_27/IdentityIdentity activation_23/Relu:activations:0*
T0*0
_output_shapes
:??????????2
dropout_27/Identity?
conv2d_30/Conv2D/ReadVariableOpReadVariableOp(conv2d_30_conv2d_readvariableop_resource*(
_output_shapes
:??*
dtype02!
conv2d_30/Conv2D/ReadVariableOp?
conv2d_30/Conv2DConv2Ddropout_27/Identity:output:0'conv2d_30/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????*
paddingSAME*
strides
2
conv2d_30/Conv2D?
 conv2d_30/BiasAdd/ReadVariableOpReadVariableOp)conv2d_30_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02"
 conv2d_30/BiasAdd/ReadVariableOp?
conv2d_30/BiasAddBiasAddconv2d_30/Conv2D:output:0(conv2d_30/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????2
conv2d_30/BiasAdd?
tf.__operators__.add_10/AddV2AddV2conv2d_30/BiasAdd:output:0dropout_26/Identity:output:0*
T0*0
_output_shapes
:??????????2
tf.__operators__.add_10/AddV2?
activation_24/ReluRelu!tf.__operators__.add_10/AddV2:z:0*
T0*0
_output_shapes
:??????????2
activation_24/Relu?
%batch_normalization_13/ReadVariableOpReadVariableOp.batch_normalization_13_readvariableop_resource*
_output_shapes	
:?*
dtype02'
%batch_normalization_13/ReadVariableOp?
'batch_normalization_13/ReadVariableOp_1ReadVariableOp0batch_normalization_13_readvariableop_1_resource*
_output_shapes	
:?*
dtype02)
'batch_normalization_13/ReadVariableOp_1?
6batch_normalization_13/FusedBatchNormV3/ReadVariableOpReadVariableOp?batch_normalization_13_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:?*
dtype028
6batch_normalization_13/FusedBatchNormV3/ReadVariableOp?
8batch_normalization_13/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpAbatch_normalization_13_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:?*
dtype02:
8batch_normalization_13/FusedBatchNormV3/ReadVariableOp_1?
'batch_normalization_13/FusedBatchNormV3FusedBatchNormV3 activation_24/Relu:activations:0-batch_normalization_13/ReadVariableOp:value:0/batch_normalization_13/ReadVariableOp_1:value:0>batch_normalization_13/FusedBatchNormV3/ReadVariableOp:value:0@batch_normalization_13/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:??????????:?:?:?:?:*
epsilon%o?:*
is_training( 2)
'batch_normalization_13/FusedBatchNormV3?
dropout_28/IdentityIdentity+batch_normalization_13/FusedBatchNormV3:y:0*
T0*0
_output_shapes
:??????????2
dropout_28/Identity?
conv2d_31/Conv2D/ReadVariableOpReadVariableOp(conv2d_31_conv2d_readvariableop_resource*(
_output_shapes
:??*
dtype02!
conv2d_31/Conv2D/ReadVariableOp?
conv2d_31/Conv2DConv2Ddropout_28/Identity:output:0'conv2d_31/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????*
paddingSAME*
strides
2
conv2d_31/Conv2D?
 conv2d_31/BiasAdd/ReadVariableOpReadVariableOp)conv2d_31_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02"
 conv2d_31/BiasAdd/ReadVariableOp?
conv2d_31/BiasAddBiasAddconv2d_31/Conv2D:output:0(conv2d_31/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????2
conv2d_31/BiasAdd?
activation_25/ReluReluconv2d_31/BiasAdd:output:0*
T0*0
_output_shapes
:??????????2
activation_25/Relu?
dropout_29/IdentityIdentity activation_25/Relu:activations:0*
T0*0
_output_shapes
:??????????2
dropout_29/Identity?
conv2d_32/Conv2D/ReadVariableOpReadVariableOp(conv2d_32_conv2d_readvariableop_resource*(
_output_shapes
:??*
dtype02!
conv2d_32/Conv2D/ReadVariableOp?
conv2d_32/Conv2DConv2Ddropout_29/Identity:output:0'conv2d_32/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????*
paddingSAME*
strides
2
conv2d_32/Conv2D?
 conv2d_32/BiasAdd/ReadVariableOpReadVariableOp)conv2d_32_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02"
 conv2d_32/BiasAdd/ReadVariableOp?
conv2d_32/BiasAddBiasAddconv2d_32/Conv2D:output:0(conv2d_32/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????2
conv2d_32/BiasAdd?
tf.__operators__.add_11/AddV2AddV2conv2d_32/BiasAdd:output:0dropout_28/Identity:output:0*
T0*0
_output_shapes
:??????????2
tf.__operators__.add_11/AddV2?
activation_26/ReluRelu!tf.__operators__.add_11/AddV2:z:0*
T0*0
_output_shapes
:??????????2
activation_26/Relu?
%batch_normalization_14/ReadVariableOpReadVariableOp.batch_normalization_14_readvariableop_resource*
_output_shapes	
:?*
dtype02'
%batch_normalization_14/ReadVariableOp?
'batch_normalization_14/ReadVariableOp_1ReadVariableOp0batch_normalization_14_readvariableop_1_resource*
_output_shapes	
:?*
dtype02)
'batch_normalization_14/ReadVariableOp_1?
6batch_normalization_14/FusedBatchNormV3/ReadVariableOpReadVariableOp?batch_normalization_14_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:?*
dtype028
6batch_normalization_14/FusedBatchNormV3/ReadVariableOp?
8batch_normalization_14/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpAbatch_normalization_14_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:?*
dtype02:
8batch_normalization_14/FusedBatchNormV3/ReadVariableOp_1?
'batch_normalization_14/FusedBatchNormV3FusedBatchNormV3 activation_26/Relu:activations:0-batch_normalization_14/ReadVariableOp:value:0/batch_normalization_14/ReadVariableOp_1:value:0>batch_normalization_14/FusedBatchNormV3/ReadVariableOp:value:0@batch_normalization_14/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:??????????:?:?:?:?:*
epsilon%o?:*
is_training( 2)
'batch_normalization_14/FusedBatchNormV3?
dropout_30/IdentityIdentity+batch_normalization_14/FusedBatchNormV3:y:0*
T0*0
_output_shapes
:??????????2
dropout_30/Identity?
conv2d_33/Conv2D/ReadVariableOpReadVariableOp(conv2d_33_conv2d_readvariableop_resource*(
_output_shapes
:??*
dtype02!
conv2d_33/Conv2D/ReadVariableOp?
conv2d_33/Conv2DConv2Ddropout_30/Identity:output:0'conv2d_33/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????*
paddingSAME*
strides
2
conv2d_33/Conv2D?
 conv2d_33/BiasAdd/ReadVariableOpReadVariableOp)conv2d_33_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02"
 conv2d_33/BiasAdd/ReadVariableOp?
conv2d_33/BiasAddBiasAddconv2d_33/Conv2D:output:0(conv2d_33/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????2
conv2d_33/BiasAdd?
activation_27/ReluReluconv2d_33/BiasAdd:output:0*
T0*0
_output_shapes
:??????????2
activation_27/Relu?
dropout_31/IdentityIdentity activation_27/Relu:activations:0*
T0*0
_output_shapes
:??????????2
dropout_31/Identity?
conv2d_34/Conv2D/ReadVariableOpReadVariableOp(conv2d_34_conv2d_readvariableop_resource*(
_output_shapes
:??*
dtype02!
conv2d_34/Conv2D/ReadVariableOp?
conv2d_34/Conv2DConv2Ddropout_31/Identity:output:0'conv2d_34/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????*
paddingSAME*
strides
2
conv2d_34/Conv2D?
 conv2d_34/BiasAdd/ReadVariableOpReadVariableOp)conv2d_34_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02"
 conv2d_34/BiasAdd/ReadVariableOp?
conv2d_34/BiasAddBiasAddconv2d_34/Conv2D:output:0(conv2d_34/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????2
conv2d_34/BiasAdd?
tf.__operators__.add_12/AddV2AddV2conv2d_34/BiasAdd:output:0dropout_30/Identity:output:0*
T0*0
_output_shapes
:??????????2
tf.__operators__.add_12/AddV2?
activation_28/ReluRelu!tf.__operators__.add_12/AddV2:z:0*
T0*0
_output_shapes
:??????????2
activation_28/Relu?
%batch_normalization_15/ReadVariableOpReadVariableOp.batch_normalization_15_readvariableop_resource*
_output_shapes	
:?*
dtype02'
%batch_normalization_15/ReadVariableOp?
'batch_normalization_15/ReadVariableOp_1ReadVariableOp0batch_normalization_15_readvariableop_1_resource*
_output_shapes	
:?*
dtype02)
'batch_normalization_15/ReadVariableOp_1?
6batch_normalization_15/FusedBatchNormV3/ReadVariableOpReadVariableOp?batch_normalization_15_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:?*
dtype028
6batch_normalization_15/FusedBatchNormV3/ReadVariableOp?
8batch_normalization_15/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpAbatch_normalization_15_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:?*
dtype02:
8batch_normalization_15/FusedBatchNormV3/ReadVariableOp_1?
'batch_normalization_15/FusedBatchNormV3FusedBatchNormV3 activation_28/Relu:activations:0-batch_normalization_15/ReadVariableOp:value:0/batch_normalization_15/ReadVariableOp_1:value:0>batch_normalization_15/FusedBatchNormV3/ReadVariableOp:value:0@batch_normalization_15/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:??????????:?:?:?:?:*
epsilon%o?:*
is_training( 2)
'batch_normalization_15/FusedBatchNormV3?
dropout_32/IdentityIdentity+batch_normalization_15/FusedBatchNormV3:y:0*
T0*0
_output_shapes
:??????????2
dropout_32/Identity?
conv2d_35/Conv2D/ReadVariableOpReadVariableOp(conv2d_35_conv2d_readvariableop_resource*(
_output_shapes
:??*
dtype02!
conv2d_35/Conv2D/ReadVariableOp?
conv2d_35/Conv2DConv2Ddropout_32/Identity:output:0'conv2d_35/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????*
paddingSAME*
strides
2
conv2d_35/Conv2D?
 conv2d_35/BiasAdd/ReadVariableOpReadVariableOp)conv2d_35_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02"
 conv2d_35/BiasAdd/ReadVariableOp?
conv2d_35/BiasAddBiasAddconv2d_35/Conv2D:output:0(conv2d_35/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????2
conv2d_35/BiasAdd?
activation_29/ReluReluconv2d_35/BiasAdd:output:0*
T0*0
_output_shapes
:??????????2
activation_29/Relu?
dropout_33/IdentityIdentity activation_29/Relu:activations:0*
T0*0
_output_shapes
:??????????2
dropout_33/Identity?
conv2d_36/Conv2D/ReadVariableOpReadVariableOp(conv2d_36_conv2d_readvariableop_resource*(
_output_shapes
:??*
dtype02!
conv2d_36/Conv2D/ReadVariableOp?
conv2d_36/Conv2DConv2Ddropout_33/Identity:output:0'conv2d_36/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????*
paddingSAME*
strides
2
conv2d_36/Conv2D?
 conv2d_36/BiasAdd/ReadVariableOpReadVariableOp)conv2d_36_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02"
 conv2d_36/BiasAdd/ReadVariableOp?
conv2d_36/BiasAddBiasAddconv2d_36/Conv2D:output:0(conv2d_36/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????2
conv2d_36/BiasAdd?
tf.__operators__.add_13/AddV2AddV2conv2d_36/BiasAdd:output:0dropout_32/Identity:output:0*
T0*0
_output_shapes
:??????????2
tf.__operators__.add_13/AddV2?
activation_30/ReluRelu!tf.__operators__.add_13/AddV2:z:0*
T0*0
_output_shapes
:??????????2
activation_30/Relu?
%batch_normalization_16/ReadVariableOpReadVariableOp.batch_normalization_16_readvariableop_resource*
_output_shapes	
:?*
dtype02'
%batch_normalization_16/ReadVariableOp?
'batch_normalization_16/ReadVariableOp_1ReadVariableOp0batch_normalization_16_readvariableop_1_resource*
_output_shapes	
:?*
dtype02)
'batch_normalization_16/ReadVariableOp_1?
6batch_normalization_16/FusedBatchNormV3/ReadVariableOpReadVariableOp?batch_normalization_16_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:?*
dtype028
6batch_normalization_16/FusedBatchNormV3/ReadVariableOp?
8batch_normalization_16/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpAbatch_normalization_16_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:?*
dtype02:
8batch_normalization_16/FusedBatchNormV3/ReadVariableOp_1?
'batch_normalization_16/FusedBatchNormV3FusedBatchNormV3 activation_30/Relu:activations:0-batch_normalization_16/ReadVariableOp:value:0/batch_normalization_16/ReadVariableOp_1:value:0>batch_normalization_16/FusedBatchNormV3/ReadVariableOp:value:0@batch_normalization_16/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:??????????:?:?:?:?:*
epsilon%o?:*
is_training( 2)
'batch_normalization_16/FusedBatchNormV3?
dropout_34/IdentityIdentity+batch_normalization_16/FusedBatchNormV3:y:0*
T0*0
_output_shapes
:??????????2
dropout_34/Identity?
conv2d_37/Conv2D/ReadVariableOpReadVariableOp(conv2d_37_conv2d_readvariableop_resource*(
_output_shapes
:??*
dtype02!
conv2d_37/Conv2D/ReadVariableOp?
conv2d_37/Conv2DConv2Ddropout_34/Identity:output:0'conv2d_37/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????*
paddingSAME*
strides
2
conv2d_37/Conv2D?
 conv2d_37/BiasAdd/ReadVariableOpReadVariableOp)conv2d_37_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02"
 conv2d_37/BiasAdd/ReadVariableOp?
conv2d_37/BiasAddBiasAddconv2d_37/Conv2D:output:0(conv2d_37/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????2
conv2d_37/BiasAdd?
activation_31/ReluReluconv2d_37/BiasAdd:output:0*
T0*0
_output_shapes
:??????????2
activation_31/Relu?
dropout_35/IdentityIdentity activation_31/Relu:activations:0*
T0*0
_output_shapes
:??????????2
dropout_35/Identity?
conv2d_38/Conv2D/ReadVariableOpReadVariableOp(conv2d_38_conv2d_readvariableop_resource*(
_output_shapes
:??*
dtype02!
conv2d_38/Conv2D/ReadVariableOp?
conv2d_38/Conv2DConv2Ddropout_35/Identity:output:0'conv2d_38/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????*
paddingSAME*
strides
2
conv2d_38/Conv2D?
 conv2d_38/BiasAdd/ReadVariableOpReadVariableOp)conv2d_38_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02"
 conv2d_38/BiasAdd/ReadVariableOp?
conv2d_38/BiasAddBiasAddconv2d_38/Conv2D:output:0(conv2d_38/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????2
conv2d_38/BiasAdd?
tf.__operators__.add_14/AddV2AddV2conv2d_38/BiasAdd:output:0dropout_34/Identity:output:0*
T0*0
_output_shapes
:??????????2
tf.__operators__.add_14/AddV2?
activation_32/ReluRelu!tf.__operators__.add_14/AddV2:z:0*
T0*0
_output_shapes
:??????????2
activation_32/Relu?
%batch_normalization_17/ReadVariableOpReadVariableOp.batch_normalization_17_readvariableop_resource*
_output_shapes	
:?*
dtype02'
%batch_normalization_17/ReadVariableOp?
'batch_normalization_17/ReadVariableOp_1ReadVariableOp0batch_normalization_17_readvariableop_1_resource*
_output_shapes	
:?*
dtype02)
'batch_normalization_17/ReadVariableOp_1?
6batch_normalization_17/FusedBatchNormV3/ReadVariableOpReadVariableOp?batch_normalization_17_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:?*
dtype028
6batch_normalization_17/FusedBatchNormV3/ReadVariableOp?
8batch_normalization_17/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpAbatch_normalization_17_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:?*
dtype02:
8batch_normalization_17/FusedBatchNormV3/ReadVariableOp_1?
'batch_normalization_17/FusedBatchNormV3FusedBatchNormV3 activation_32/Relu:activations:0-batch_normalization_17/ReadVariableOp:value:0/batch_normalization_17/ReadVariableOp_1:value:0>batch_normalization_17/FusedBatchNormV3/ReadVariableOp:value:0@batch_normalization_17/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:??????????:?:?:?:?:*
epsilon%o?:*
is_training( 2)
'batch_normalization_17/FusedBatchNormV3?
dropout_36/IdentityIdentity+batch_normalization_17/FusedBatchNormV3:y:0*
T0*0
_output_shapes
:??????????2
dropout_36/Identity?
IdentityIdentitydropout_36/Identity:output:0^NoOp*
T0*0
_output_shapes
:??????????2

Identity?
NoOpNoOp7^batch_normalization_12/FusedBatchNormV3/ReadVariableOp9^batch_normalization_12/FusedBatchNormV3/ReadVariableOp_1&^batch_normalization_12/ReadVariableOp(^batch_normalization_12/ReadVariableOp_17^batch_normalization_13/FusedBatchNormV3/ReadVariableOp9^batch_normalization_13/FusedBatchNormV3/ReadVariableOp_1&^batch_normalization_13/ReadVariableOp(^batch_normalization_13/ReadVariableOp_17^batch_normalization_14/FusedBatchNormV3/ReadVariableOp9^batch_normalization_14/FusedBatchNormV3/ReadVariableOp_1&^batch_normalization_14/ReadVariableOp(^batch_normalization_14/ReadVariableOp_17^batch_normalization_15/FusedBatchNormV3/ReadVariableOp9^batch_normalization_15/FusedBatchNormV3/ReadVariableOp_1&^batch_normalization_15/ReadVariableOp(^batch_normalization_15/ReadVariableOp_17^batch_normalization_16/FusedBatchNormV3/ReadVariableOp9^batch_normalization_16/FusedBatchNormV3/ReadVariableOp_1&^batch_normalization_16/ReadVariableOp(^batch_normalization_16/ReadVariableOp_17^batch_normalization_17/FusedBatchNormV3/ReadVariableOp9^batch_normalization_17/FusedBatchNormV3/ReadVariableOp_1&^batch_normalization_17/ReadVariableOp(^batch_normalization_17/ReadVariableOp_1!^conv2d_28/BiasAdd/ReadVariableOp ^conv2d_28/Conv2D/ReadVariableOp!^conv2d_29/BiasAdd/ReadVariableOp ^conv2d_29/Conv2D/ReadVariableOp!^conv2d_30/BiasAdd/ReadVariableOp ^conv2d_30/Conv2D/ReadVariableOp!^conv2d_31/BiasAdd/ReadVariableOp ^conv2d_31/Conv2D/ReadVariableOp!^conv2d_32/BiasAdd/ReadVariableOp ^conv2d_32/Conv2D/ReadVariableOp!^conv2d_33/BiasAdd/ReadVariableOp ^conv2d_33/Conv2D/ReadVariableOp!^conv2d_34/BiasAdd/ReadVariableOp ^conv2d_34/Conv2D/ReadVariableOp!^conv2d_35/BiasAdd/ReadVariableOp ^conv2d_35/Conv2D/ReadVariableOp!^conv2d_36/BiasAdd/ReadVariableOp ^conv2d_36/Conv2D/ReadVariableOp!^conv2d_37/BiasAdd/ReadVariableOp ^conv2d_37/Conv2D/ReadVariableOp!^conv2d_38/BiasAdd/ReadVariableOp ^conv2d_38/Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapesy
w:?????????: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2p
6batch_normalization_12/FusedBatchNormV3/ReadVariableOp6batch_normalization_12/FusedBatchNormV3/ReadVariableOp2t
8batch_normalization_12/FusedBatchNormV3/ReadVariableOp_18batch_normalization_12/FusedBatchNormV3/ReadVariableOp_12N
%batch_normalization_12/ReadVariableOp%batch_normalization_12/ReadVariableOp2R
'batch_normalization_12/ReadVariableOp_1'batch_normalization_12/ReadVariableOp_12p
6batch_normalization_13/FusedBatchNormV3/ReadVariableOp6batch_normalization_13/FusedBatchNormV3/ReadVariableOp2t
8batch_normalization_13/FusedBatchNormV3/ReadVariableOp_18batch_normalization_13/FusedBatchNormV3/ReadVariableOp_12N
%batch_normalization_13/ReadVariableOp%batch_normalization_13/ReadVariableOp2R
'batch_normalization_13/ReadVariableOp_1'batch_normalization_13/ReadVariableOp_12p
6batch_normalization_14/FusedBatchNormV3/ReadVariableOp6batch_normalization_14/FusedBatchNormV3/ReadVariableOp2t
8batch_normalization_14/FusedBatchNormV3/ReadVariableOp_18batch_normalization_14/FusedBatchNormV3/ReadVariableOp_12N
%batch_normalization_14/ReadVariableOp%batch_normalization_14/ReadVariableOp2R
'batch_normalization_14/ReadVariableOp_1'batch_normalization_14/ReadVariableOp_12p
6batch_normalization_15/FusedBatchNormV3/ReadVariableOp6batch_normalization_15/FusedBatchNormV3/ReadVariableOp2t
8batch_normalization_15/FusedBatchNormV3/ReadVariableOp_18batch_normalization_15/FusedBatchNormV3/ReadVariableOp_12N
%batch_normalization_15/ReadVariableOp%batch_normalization_15/ReadVariableOp2R
'batch_normalization_15/ReadVariableOp_1'batch_normalization_15/ReadVariableOp_12p
6batch_normalization_16/FusedBatchNormV3/ReadVariableOp6batch_normalization_16/FusedBatchNormV3/ReadVariableOp2t
8batch_normalization_16/FusedBatchNormV3/ReadVariableOp_18batch_normalization_16/FusedBatchNormV3/ReadVariableOp_12N
%batch_normalization_16/ReadVariableOp%batch_normalization_16/ReadVariableOp2R
'batch_normalization_16/ReadVariableOp_1'batch_normalization_16/ReadVariableOp_12p
6batch_normalization_17/FusedBatchNormV3/ReadVariableOp6batch_normalization_17/FusedBatchNormV3/ReadVariableOp2t
8batch_normalization_17/FusedBatchNormV3/ReadVariableOp_18batch_normalization_17/FusedBatchNormV3/ReadVariableOp_12N
%batch_normalization_17/ReadVariableOp%batch_normalization_17/ReadVariableOp2R
'batch_normalization_17/ReadVariableOp_1'batch_normalization_17/ReadVariableOp_12D
 conv2d_28/BiasAdd/ReadVariableOp conv2d_28/BiasAdd/ReadVariableOp2B
conv2d_28/Conv2D/ReadVariableOpconv2d_28/Conv2D/ReadVariableOp2D
 conv2d_29/BiasAdd/ReadVariableOp conv2d_29/BiasAdd/ReadVariableOp2B
conv2d_29/Conv2D/ReadVariableOpconv2d_29/Conv2D/ReadVariableOp2D
 conv2d_30/BiasAdd/ReadVariableOp conv2d_30/BiasAdd/ReadVariableOp2B
conv2d_30/Conv2D/ReadVariableOpconv2d_30/Conv2D/ReadVariableOp2D
 conv2d_31/BiasAdd/ReadVariableOp conv2d_31/BiasAdd/ReadVariableOp2B
conv2d_31/Conv2D/ReadVariableOpconv2d_31/Conv2D/ReadVariableOp2D
 conv2d_32/BiasAdd/ReadVariableOp conv2d_32/BiasAdd/ReadVariableOp2B
conv2d_32/Conv2D/ReadVariableOpconv2d_32/Conv2D/ReadVariableOp2D
 conv2d_33/BiasAdd/ReadVariableOp conv2d_33/BiasAdd/ReadVariableOp2B
conv2d_33/Conv2D/ReadVariableOpconv2d_33/Conv2D/ReadVariableOp2D
 conv2d_34/BiasAdd/ReadVariableOp conv2d_34/BiasAdd/ReadVariableOp2B
conv2d_34/Conv2D/ReadVariableOpconv2d_34/Conv2D/ReadVariableOp2D
 conv2d_35/BiasAdd/ReadVariableOp conv2d_35/BiasAdd/ReadVariableOp2B
conv2d_35/Conv2D/ReadVariableOpconv2d_35/Conv2D/ReadVariableOp2D
 conv2d_36/BiasAdd/ReadVariableOp conv2d_36/BiasAdd/ReadVariableOp2B
conv2d_36/Conv2D/ReadVariableOpconv2d_36/Conv2D/ReadVariableOp2D
 conv2d_37/BiasAdd/ReadVariableOp conv2d_37/BiasAdd/ReadVariableOp2B
conv2d_37/Conv2D/ReadVariableOpconv2d_37/Conv2D/ReadVariableOp2D
 conv2d_38/BiasAdd/ReadVariableOp conv2d_38/BiasAdd/ReadVariableOp2B
conv2d_38/Conv2D/ReadVariableOpconv2d_38/Conv2D/ReadVariableOp:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?	
?
9__inference_batch_normalization_13_layer_call_fn_22007768

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
GPU 2J 8? *]
fXRV
T__inference_batch_normalization_13_layer_call_and_return_conditional_losses_220041362
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
?
?
T__inference_batch_normalization_16_layer_call_and_return_conditional_losses_22008464

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
?
~
*__inference_dense_3_layer_call_fn_22007439

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
GPU 2J 8? *N
fIRG
E__inference_dense_3_layer_call_and_return_conditional_losses_220051612
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
?
I
-__inference_dropout_50_layer_call_fn_22007423

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
GPU 2J 8? *Q
fLRJ
H__inference_dropout_50_layer_call_and_return_conditional_losses_220053142
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
?
d
H__inference_dropout_34_layer_call_and_return_conditional_losses_22008519

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
?
?
*__inference_model_4_layer_call_fn_22004700
input_5"
unknown:?
	unknown_0:	?
	unknown_1:	?
	unknown_2:	?
	unknown_3:	?
	unknown_4:	?%
	unknown_5:??
	unknown_6:	?%
	unknown_7:??
	unknown_8:	?
	unknown_9:	?

unknown_10:	?

unknown_11:	?

unknown_12:	?&

unknown_13:??

unknown_14:	?&

unknown_15:??

unknown_16:	?

unknown_17:	?

unknown_18:	?

unknown_19:	?

unknown_20:	?&

unknown_21:??

unknown_22:	?&

unknown_23:??

unknown_24:	?

unknown_25:	?

unknown_26:	?

unknown_27:	?

unknown_28:	?&

unknown_29:??

unknown_30:	?&

unknown_31:??

unknown_32:	?

unknown_33:	?

unknown_34:	?

unknown_35:	?

unknown_36:	?&

unknown_37:??

unknown_38:	?&

unknown_39:??

unknown_40:	?

unknown_41:	?

unknown_42:	?

unknown_43:	?

unknown_44:	?
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_5unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
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
unknown_23
unknown_24
unknown_25
unknown_26
unknown_27
unknown_28
unknown_29
unknown_30
unknown_31
unknown_32
unknown_33
unknown_34
unknown_35
unknown_36
unknown_37
unknown_38
unknown_39
unknown_40
unknown_41
unknown_42
unknown_43
unknown_44*:
Tin3
12/*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????*D
_read_only_resource_inputs&
$"	
 !"#$'()*+,*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_model_4_layer_call_and_return_conditional_losses_220045082
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
_construction_contextkEagerRuntime*?
_input_shapesy
w:?????????: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
/
_output_shapes
:?????????
!
_user_specified_name	input_5
??
?,
E__inference_model_4_layer_call_and_return_conditional_losses_22007343

inputsC
(conv2d_28_conv2d_readvariableop_resource:?8
)conv2d_28_biasadd_readvariableop_resource:	?=
.batch_normalization_12_readvariableop_resource:	??
0batch_normalization_12_readvariableop_1_resource:	?N
?batch_normalization_12_fusedbatchnormv3_readvariableop_resource:	?P
Abatch_normalization_12_fusedbatchnormv3_readvariableop_1_resource:	?D
(conv2d_29_conv2d_readvariableop_resource:??8
)conv2d_29_biasadd_readvariableop_resource:	?D
(conv2d_30_conv2d_readvariableop_resource:??8
)conv2d_30_biasadd_readvariableop_resource:	?=
.batch_normalization_13_readvariableop_resource:	??
0batch_normalization_13_readvariableop_1_resource:	?N
?batch_normalization_13_fusedbatchnormv3_readvariableop_resource:	?P
Abatch_normalization_13_fusedbatchnormv3_readvariableop_1_resource:	?D
(conv2d_31_conv2d_readvariableop_resource:??8
)conv2d_31_biasadd_readvariableop_resource:	?D
(conv2d_32_conv2d_readvariableop_resource:??8
)conv2d_32_biasadd_readvariableop_resource:	?=
.batch_normalization_14_readvariableop_resource:	??
0batch_normalization_14_readvariableop_1_resource:	?N
?batch_normalization_14_fusedbatchnormv3_readvariableop_resource:	?P
Abatch_normalization_14_fusedbatchnormv3_readvariableop_1_resource:	?D
(conv2d_33_conv2d_readvariableop_resource:??8
)conv2d_33_biasadd_readvariableop_resource:	?D
(conv2d_34_conv2d_readvariableop_resource:??8
)conv2d_34_biasadd_readvariableop_resource:	?=
.batch_normalization_15_readvariableop_resource:	??
0batch_normalization_15_readvariableop_1_resource:	?N
?batch_normalization_15_fusedbatchnormv3_readvariableop_resource:	?P
Abatch_normalization_15_fusedbatchnormv3_readvariableop_1_resource:	?D
(conv2d_35_conv2d_readvariableop_resource:??8
)conv2d_35_biasadd_readvariableop_resource:	?D
(conv2d_36_conv2d_readvariableop_resource:??8
)conv2d_36_biasadd_readvariableop_resource:	?=
.batch_normalization_16_readvariableop_resource:	??
0batch_normalization_16_readvariableop_1_resource:	?N
?batch_normalization_16_fusedbatchnormv3_readvariableop_resource:	?P
Abatch_normalization_16_fusedbatchnormv3_readvariableop_1_resource:	?D
(conv2d_37_conv2d_readvariableop_resource:??8
)conv2d_37_biasadd_readvariableop_resource:	?D
(conv2d_38_conv2d_readvariableop_resource:??8
)conv2d_38_biasadd_readvariableop_resource:	?=
.batch_normalization_17_readvariableop_resource:	??
0batch_normalization_17_readvariableop_1_resource:	?N
?batch_normalization_17_fusedbatchnormv3_readvariableop_resource:	?P
Abatch_normalization_17_fusedbatchnormv3_readvariableop_1_resource:	?
identity??%batch_normalization_12/AssignNewValue?'batch_normalization_12/AssignNewValue_1?6batch_normalization_12/FusedBatchNormV3/ReadVariableOp?8batch_normalization_12/FusedBatchNormV3/ReadVariableOp_1?%batch_normalization_12/ReadVariableOp?'batch_normalization_12/ReadVariableOp_1?%batch_normalization_13/AssignNewValue?'batch_normalization_13/AssignNewValue_1?6batch_normalization_13/FusedBatchNormV3/ReadVariableOp?8batch_normalization_13/FusedBatchNormV3/ReadVariableOp_1?%batch_normalization_13/ReadVariableOp?'batch_normalization_13/ReadVariableOp_1?%batch_normalization_14/AssignNewValue?'batch_normalization_14/AssignNewValue_1?6batch_normalization_14/FusedBatchNormV3/ReadVariableOp?8batch_normalization_14/FusedBatchNormV3/ReadVariableOp_1?%batch_normalization_14/ReadVariableOp?'batch_normalization_14/ReadVariableOp_1?%batch_normalization_15/AssignNewValue?'batch_normalization_15/AssignNewValue_1?6batch_normalization_15/FusedBatchNormV3/ReadVariableOp?8batch_normalization_15/FusedBatchNormV3/ReadVariableOp_1?%batch_normalization_15/ReadVariableOp?'batch_normalization_15/ReadVariableOp_1?%batch_normalization_16/AssignNewValue?'batch_normalization_16/AssignNewValue_1?6batch_normalization_16/FusedBatchNormV3/ReadVariableOp?8batch_normalization_16/FusedBatchNormV3/ReadVariableOp_1?%batch_normalization_16/ReadVariableOp?'batch_normalization_16/ReadVariableOp_1?%batch_normalization_17/AssignNewValue?'batch_normalization_17/AssignNewValue_1?6batch_normalization_17/FusedBatchNormV3/ReadVariableOp?8batch_normalization_17/FusedBatchNormV3/ReadVariableOp_1?%batch_normalization_17/ReadVariableOp?'batch_normalization_17/ReadVariableOp_1? conv2d_28/BiasAdd/ReadVariableOp?conv2d_28/Conv2D/ReadVariableOp? conv2d_29/BiasAdd/ReadVariableOp?conv2d_29/Conv2D/ReadVariableOp? conv2d_30/BiasAdd/ReadVariableOp?conv2d_30/Conv2D/ReadVariableOp? conv2d_31/BiasAdd/ReadVariableOp?conv2d_31/Conv2D/ReadVariableOp? conv2d_32/BiasAdd/ReadVariableOp?conv2d_32/Conv2D/ReadVariableOp? conv2d_33/BiasAdd/ReadVariableOp?conv2d_33/Conv2D/ReadVariableOp? conv2d_34/BiasAdd/ReadVariableOp?conv2d_34/Conv2D/ReadVariableOp? conv2d_35/BiasAdd/ReadVariableOp?conv2d_35/Conv2D/ReadVariableOp? conv2d_36/BiasAdd/ReadVariableOp?conv2d_36/Conv2D/ReadVariableOp? conv2d_37/BiasAdd/ReadVariableOp?conv2d_37/Conv2D/ReadVariableOp? conv2d_38/BiasAdd/ReadVariableOp?conv2d_38/Conv2D/ReadVariableOp?
conv2d_28/Conv2D/ReadVariableOpReadVariableOp(conv2d_28_conv2d_readvariableop_resource*'
_output_shapes
:?*
dtype02!
conv2d_28/Conv2D/ReadVariableOp?
conv2d_28/Conv2DConv2Dinputs'conv2d_28/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????*
paddingSAME*
strides
2
conv2d_28/Conv2D?
 conv2d_28/BiasAdd/ReadVariableOpReadVariableOp)conv2d_28_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02"
 conv2d_28/BiasAdd/ReadVariableOp?
conv2d_28/BiasAddBiasAddconv2d_28/Conv2D:output:0(conv2d_28/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????2
conv2d_28/BiasAdd?
%batch_normalization_12/ReadVariableOpReadVariableOp.batch_normalization_12_readvariableop_resource*
_output_shapes	
:?*
dtype02'
%batch_normalization_12/ReadVariableOp?
'batch_normalization_12/ReadVariableOp_1ReadVariableOp0batch_normalization_12_readvariableop_1_resource*
_output_shapes	
:?*
dtype02)
'batch_normalization_12/ReadVariableOp_1?
6batch_normalization_12/FusedBatchNormV3/ReadVariableOpReadVariableOp?batch_normalization_12_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:?*
dtype028
6batch_normalization_12/FusedBatchNormV3/ReadVariableOp?
8batch_normalization_12/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpAbatch_normalization_12_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:?*
dtype02:
8batch_normalization_12/FusedBatchNormV3/ReadVariableOp_1?
'batch_normalization_12/FusedBatchNormV3FusedBatchNormV3conv2d_28/BiasAdd:output:0-batch_normalization_12/ReadVariableOp:value:0/batch_normalization_12/ReadVariableOp_1:value:0>batch_normalization_12/FusedBatchNormV3/ReadVariableOp:value:0@batch_normalization_12/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:??????????:?:?:?:?:*
epsilon%o?:*
exponential_avg_factor%
?#<2)
'batch_normalization_12/FusedBatchNormV3?
%batch_normalization_12/AssignNewValueAssignVariableOp?batch_normalization_12_fusedbatchnormv3_readvariableop_resource4batch_normalization_12/FusedBatchNormV3:batch_mean:07^batch_normalization_12/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype02'
%batch_normalization_12/AssignNewValue?
'batch_normalization_12/AssignNewValue_1AssignVariableOpAbatch_normalization_12_fusedbatchnormv3_readvariableop_1_resource8batch_normalization_12/FusedBatchNormV3:batch_variance:09^batch_normalization_12/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype02)
'batch_normalization_12/AssignNewValue_1?
activation_22/ReluRelu+batch_normalization_12/FusedBatchNormV3:y:0*
T0*0
_output_shapes
:??????????2
activation_22/Relu?
conv2d_29/Conv2D/ReadVariableOpReadVariableOp(conv2d_29_conv2d_readvariableop_resource*(
_output_shapes
:??*
dtype02!
conv2d_29/Conv2D/ReadVariableOp?
conv2d_29/Conv2DConv2D activation_22/Relu:activations:0'conv2d_29/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????*
paddingSAME*
strides
2
conv2d_29/Conv2D?
 conv2d_29/BiasAdd/ReadVariableOpReadVariableOp)conv2d_29_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02"
 conv2d_29/BiasAdd/ReadVariableOp?
conv2d_29/BiasAddBiasAddconv2d_29/Conv2D:output:0(conv2d_29/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????2
conv2d_29/BiasAdd?
activation_23/ReluReluconv2d_29/BiasAdd:output:0*
T0*0
_output_shapes
:??????????2
activation_23/Relu?
conv2d_30/Conv2D/ReadVariableOpReadVariableOp(conv2d_30_conv2d_readvariableop_resource*(
_output_shapes
:??*
dtype02!
conv2d_30/Conv2D/ReadVariableOp?
conv2d_30/Conv2DConv2D activation_23/Relu:activations:0'conv2d_30/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????*
paddingSAME*
strides
2
conv2d_30/Conv2D?
 conv2d_30/BiasAdd/ReadVariableOpReadVariableOp)conv2d_30_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02"
 conv2d_30/BiasAdd/ReadVariableOp?
conv2d_30/BiasAddBiasAddconv2d_30/Conv2D:output:0(conv2d_30/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????2
conv2d_30/BiasAdd?
tf.__operators__.add_10/AddV2AddV2conv2d_30/BiasAdd:output:0 activation_22/Relu:activations:0*
T0*0
_output_shapes
:??????????2
tf.__operators__.add_10/AddV2?
activation_24/ReluRelu!tf.__operators__.add_10/AddV2:z:0*
T0*0
_output_shapes
:??????????2
activation_24/Relu?
%batch_normalization_13/ReadVariableOpReadVariableOp.batch_normalization_13_readvariableop_resource*
_output_shapes	
:?*
dtype02'
%batch_normalization_13/ReadVariableOp?
'batch_normalization_13/ReadVariableOp_1ReadVariableOp0batch_normalization_13_readvariableop_1_resource*
_output_shapes	
:?*
dtype02)
'batch_normalization_13/ReadVariableOp_1?
6batch_normalization_13/FusedBatchNormV3/ReadVariableOpReadVariableOp?batch_normalization_13_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:?*
dtype028
6batch_normalization_13/FusedBatchNormV3/ReadVariableOp?
8batch_normalization_13/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpAbatch_normalization_13_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:?*
dtype02:
8batch_normalization_13/FusedBatchNormV3/ReadVariableOp_1?
'batch_normalization_13/FusedBatchNormV3FusedBatchNormV3 activation_24/Relu:activations:0-batch_normalization_13/ReadVariableOp:value:0/batch_normalization_13/ReadVariableOp_1:value:0>batch_normalization_13/FusedBatchNormV3/ReadVariableOp:value:0@batch_normalization_13/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:??????????:?:?:?:?:*
epsilon%o?:*
exponential_avg_factor%
?#<2)
'batch_normalization_13/FusedBatchNormV3?
%batch_normalization_13/AssignNewValueAssignVariableOp?batch_normalization_13_fusedbatchnormv3_readvariableop_resource4batch_normalization_13/FusedBatchNormV3:batch_mean:07^batch_normalization_13/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype02'
%batch_normalization_13/AssignNewValue?
'batch_normalization_13/AssignNewValue_1AssignVariableOpAbatch_normalization_13_fusedbatchnormv3_readvariableop_1_resource8batch_normalization_13/FusedBatchNormV3:batch_variance:09^batch_normalization_13/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype02)
'batch_normalization_13/AssignNewValue_1?
conv2d_31/Conv2D/ReadVariableOpReadVariableOp(conv2d_31_conv2d_readvariableop_resource*(
_output_shapes
:??*
dtype02!
conv2d_31/Conv2D/ReadVariableOp?
conv2d_31/Conv2DConv2D+batch_normalization_13/FusedBatchNormV3:y:0'conv2d_31/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????*
paddingSAME*
strides
2
conv2d_31/Conv2D?
 conv2d_31/BiasAdd/ReadVariableOpReadVariableOp)conv2d_31_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02"
 conv2d_31/BiasAdd/ReadVariableOp?
conv2d_31/BiasAddBiasAddconv2d_31/Conv2D:output:0(conv2d_31/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????2
conv2d_31/BiasAdd?
activation_25/ReluReluconv2d_31/BiasAdd:output:0*
T0*0
_output_shapes
:??????????2
activation_25/Relu?
conv2d_32/Conv2D/ReadVariableOpReadVariableOp(conv2d_32_conv2d_readvariableop_resource*(
_output_shapes
:??*
dtype02!
conv2d_32/Conv2D/ReadVariableOp?
conv2d_32/Conv2DConv2D activation_25/Relu:activations:0'conv2d_32/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????*
paddingSAME*
strides
2
conv2d_32/Conv2D?
 conv2d_32/BiasAdd/ReadVariableOpReadVariableOp)conv2d_32_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02"
 conv2d_32/BiasAdd/ReadVariableOp?
conv2d_32/BiasAddBiasAddconv2d_32/Conv2D:output:0(conv2d_32/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????2
conv2d_32/BiasAdd?
tf.__operators__.add_11/AddV2AddV2conv2d_32/BiasAdd:output:0+batch_normalization_13/FusedBatchNormV3:y:0*
T0*0
_output_shapes
:??????????2
tf.__operators__.add_11/AddV2?
activation_26/ReluRelu!tf.__operators__.add_11/AddV2:z:0*
T0*0
_output_shapes
:??????????2
activation_26/Relu?
%batch_normalization_14/ReadVariableOpReadVariableOp.batch_normalization_14_readvariableop_resource*
_output_shapes	
:?*
dtype02'
%batch_normalization_14/ReadVariableOp?
'batch_normalization_14/ReadVariableOp_1ReadVariableOp0batch_normalization_14_readvariableop_1_resource*
_output_shapes	
:?*
dtype02)
'batch_normalization_14/ReadVariableOp_1?
6batch_normalization_14/FusedBatchNormV3/ReadVariableOpReadVariableOp?batch_normalization_14_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:?*
dtype028
6batch_normalization_14/FusedBatchNormV3/ReadVariableOp?
8batch_normalization_14/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpAbatch_normalization_14_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:?*
dtype02:
8batch_normalization_14/FusedBatchNormV3/ReadVariableOp_1?
'batch_normalization_14/FusedBatchNormV3FusedBatchNormV3 activation_26/Relu:activations:0-batch_normalization_14/ReadVariableOp:value:0/batch_normalization_14/ReadVariableOp_1:value:0>batch_normalization_14/FusedBatchNormV3/ReadVariableOp:value:0@batch_normalization_14/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:??????????:?:?:?:?:*
epsilon%o?:*
exponential_avg_factor%
?#<2)
'batch_normalization_14/FusedBatchNormV3?
%batch_normalization_14/AssignNewValueAssignVariableOp?batch_normalization_14_fusedbatchnormv3_readvariableop_resource4batch_normalization_14/FusedBatchNormV3:batch_mean:07^batch_normalization_14/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype02'
%batch_normalization_14/AssignNewValue?
'batch_normalization_14/AssignNewValue_1AssignVariableOpAbatch_normalization_14_fusedbatchnormv3_readvariableop_1_resource8batch_normalization_14/FusedBatchNormV3:batch_variance:09^batch_normalization_14/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype02)
'batch_normalization_14/AssignNewValue_1?
conv2d_33/Conv2D/ReadVariableOpReadVariableOp(conv2d_33_conv2d_readvariableop_resource*(
_output_shapes
:??*
dtype02!
conv2d_33/Conv2D/ReadVariableOp?
conv2d_33/Conv2DConv2D+batch_normalization_14/FusedBatchNormV3:y:0'conv2d_33/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????*
paddingSAME*
strides
2
conv2d_33/Conv2D?
 conv2d_33/BiasAdd/ReadVariableOpReadVariableOp)conv2d_33_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02"
 conv2d_33/BiasAdd/ReadVariableOp?
conv2d_33/BiasAddBiasAddconv2d_33/Conv2D:output:0(conv2d_33/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????2
conv2d_33/BiasAdd?
activation_27/ReluReluconv2d_33/BiasAdd:output:0*
T0*0
_output_shapes
:??????????2
activation_27/Relu?
conv2d_34/Conv2D/ReadVariableOpReadVariableOp(conv2d_34_conv2d_readvariableop_resource*(
_output_shapes
:??*
dtype02!
conv2d_34/Conv2D/ReadVariableOp?
conv2d_34/Conv2DConv2D activation_27/Relu:activations:0'conv2d_34/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????*
paddingSAME*
strides
2
conv2d_34/Conv2D?
 conv2d_34/BiasAdd/ReadVariableOpReadVariableOp)conv2d_34_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02"
 conv2d_34/BiasAdd/ReadVariableOp?
conv2d_34/BiasAddBiasAddconv2d_34/Conv2D:output:0(conv2d_34/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????2
conv2d_34/BiasAdd?
tf.__operators__.add_12/AddV2AddV2conv2d_34/BiasAdd:output:0+batch_normalization_14/FusedBatchNormV3:y:0*
T0*0
_output_shapes
:??????????2
tf.__operators__.add_12/AddV2?
activation_28/ReluRelu!tf.__operators__.add_12/AddV2:z:0*
T0*0
_output_shapes
:??????????2
activation_28/Relu?
%batch_normalization_15/ReadVariableOpReadVariableOp.batch_normalization_15_readvariableop_resource*
_output_shapes	
:?*
dtype02'
%batch_normalization_15/ReadVariableOp?
'batch_normalization_15/ReadVariableOp_1ReadVariableOp0batch_normalization_15_readvariableop_1_resource*
_output_shapes	
:?*
dtype02)
'batch_normalization_15/ReadVariableOp_1?
6batch_normalization_15/FusedBatchNormV3/ReadVariableOpReadVariableOp?batch_normalization_15_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:?*
dtype028
6batch_normalization_15/FusedBatchNormV3/ReadVariableOp?
8batch_normalization_15/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpAbatch_normalization_15_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:?*
dtype02:
8batch_normalization_15/FusedBatchNormV3/ReadVariableOp_1?
'batch_normalization_15/FusedBatchNormV3FusedBatchNormV3 activation_28/Relu:activations:0-batch_normalization_15/ReadVariableOp:value:0/batch_normalization_15/ReadVariableOp_1:value:0>batch_normalization_15/FusedBatchNormV3/ReadVariableOp:value:0@batch_normalization_15/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:??????????:?:?:?:?:*
epsilon%o?:*
exponential_avg_factor%
?#<2)
'batch_normalization_15/FusedBatchNormV3?
%batch_normalization_15/AssignNewValueAssignVariableOp?batch_normalization_15_fusedbatchnormv3_readvariableop_resource4batch_normalization_15/FusedBatchNormV3:batch_mean:07^batch_normalization_15/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype02'
%batch_normalization_15/AssignNewValue?
'batch_normalization_15/AssignNewValue_1AssignVariableOpAbatch_normalization_15_fusedbatchnormv3_readvariableop_1_resource8batch_normalization_15/FusedBatchNormV3:batch_variance:09^batch_normalization_15/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype02)
'batch_normalization_15/AssignNewValue_1?
conv2d_35/Conv2D/ReadVariableOpReadVariableOp(conv2d_35_conv2d_readvariableop_resource*(
_output_shapes
:??*
dtype02!
conv2d_35/Conv2D/ReadVariableOp?
conv2d_35/Conv2DConv2D+batch_normalization_15/FusedBatchNormV3:y:0'conv2d_35/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????*
paddingSAME*
strides
2
conv2d_35/Conv2D?
 conv2d_35/BiasAdd/ReadVariableOpReadVariableOp)conv2d_35_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02"
 conv2d_35/BiasAdd/ReadVariableOp?
conv2d_35/BiasAddBiasAddconv2d_35/Conv2D:output:0(conv2d_35/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????2
conv2d_35/BiasAdd?
activation_29/ReluReluconv2d_35/BiasAdd:output:0*
T0*0
_output_shapes
:??????????2
activation_29/Relu?
conv2d_36/Conv2D/ReadVariableOpReadVariableOp(conv2d_36_conv2d_readvariableop_resource*(
_output_shapes
:??*
dtype02!
conv2d_36/Conv2D/ReadVariableOp?
conv2d_36/Conv2DConv2D activation_29/Relu:activations:0'conv2d_36/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????*
paddingSAME*
strides
2
conv2d_36/Conv2D?
 conv2d_36/BiasAdd/ReadVariableOpReadVariableOp)conv2d_36_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02"
 conv2d_36/BiasAdd/ReadVariableOp?
conv2d_36/BiasAddBiasAddconv2d_36/Conv2D:output:0(conv2d_36/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????2
conv2d_36/BiasAdd?
tf.__operators__.add_13/AddV2AddV2conv2d_36/BiasAdd:output:0+batch_normalization_15/FusedBatchNormV3:y:0*
T0*0
_output_shapes
:??????????2
tf.__operators__.add_13/AddV2?
activation_30/ReluRelu!tf.__operators__.add_13/AddV2:z:0*
T0*0
_output_shapes
:??????????2
activation_30/Relu?
%batch_normalization_16/ReadVariableOpReadVariableOp.batch_normalization_16_readvariableop_resource*
_output_shapes	
:?*
dtype02'
%batch_normalization_16/ReadVariableOp?
'batch_normalization_16/ReadVariableOp_1ReadVariableOp0batch_normalization_16_readvariableop_1_resource*
_output_shapes	
:?*
dtype02)
'batch_normalization_16/ReadVariableOp_1?
6batch_normalization_16/FusedBatchNormV3/ReadVariableOpReadVariableOp?batch_normalization_16_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:?*
dtype028
6batch_normalization_16/FusedBatchNormV3/ReadVariableOp?
8batch_normalization_16/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpAbatch_normalization_16_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:?*
dtype02:
8batch_normalization_16/FusedBatchNormV3/ReadVariableOp_1?
'batch_normalization_16/FusedBatchNormV3FusedBatchNormV3 activation_30/Relu:activations:0-batch_normalization_16/ReadVariableOp:value:0/batch_normalization_16/ReadVariableOp_1:value:0>batch_normalization_16/FusedBatchNormV3/ReadVariableOp:value:0@batch_normalization_16/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:??????????:?:?:?:?:*
epsilon%o?:*
exponential_avg_factor%
?#<2)
'batch_normalization_16/FusedBatchNormV3?
%batch_normalization_16/AssignNewValueAssignVariableOp?batch_normalization_16_fusedbatchnormv3_readvariableop_resource4batch_normalization_16/FusedBatchNormV3:batch_mean:07^batch_normalization_16/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype02'
%batch_normalization_16/AssignNewValue?
'batch_normalization_16/AssignNewValue_1AssignVariableOpAbatch_normalization_16_fusedbatchnormv3_readvariableop_1_resource8batch_normalization_16/FusedBatchNormV3:batch_variance:09^batch_normalization_16/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype02)
'batch_normalization_16/AssignNewValue_1?
conv2d_37/Conv2D/ReadVariableOpReadVariableOp(conv2d_37_conv2d_readvariableop_resource*(
_output_shapes
:??*
dtype02!
conv2d_37/Conv2D/ReadVariableOp?
conv2d_37/Conv2DConv2D+batch_normalization_16/FusedBatchNormV3:y:0'conv2d_37/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????*
paddingSAME*
strides
2
conv2d_37/Conv2D?
 conv2d_37/BiasAdd/ReadVariableOpReadVariableOp)conv2d_37_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02"
 conv2d_37/BiasAdd/ReadVariableOp?
conv2d_37/BiasAddBiasAddconv2d_37/Conv2D:output:0(conv2d_37/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????2
conv2d_37/BiasAdd?
activation_31/ReluReluconv2d_37/BiasAdd:output:0*
T0*0
_output_shapes
:??????????2
activation_31/Relu?
conv2d_38/Conv2D/ReadVariableOpReadVariableOp(conv2d_38_conv2d_readvariableop_resource*(
_output_shapes
:??*
dtype02!
conv2d_38/Conv2D/ReadVariableOp?
conv2d_38/Conv2DConv2D activation_31/Relu:activations:0'conv2d_38/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????*
paddingSAME*
strides
2
conv2d_38/Conv2D?
 conv2d_38/BiasAdd/ReadVariableOpReadVariableOp)conv2d_38_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02"
 conv2d_38/BiasAdd/ReadVariableOp?
conv2d_38/BiasAddBiasAddconv2d_38/Conv2D:output:0(conv2d_38/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????2
conv2d_38/BiasAdd?
tf.__operators__.add_14/AddV2AddV2conv2d_38/BiasAdd:output:0+batch_normalization_16/FusedBatchNormV3:y:0*
T0*0
_output_shapes
:??????????2
tf.__operators__.add_14/AddV2?
activation_32/ReluRelu!tf.__operators__.add_14/AddV2:z:0*
T0*0
_output_shapes
:??????????2
activation_32/Relu?
%batch_normalization_17/ReadVariableOpReadVariableOp.batch_normalization_17_readvariableop_resource*
_output_shapes	
:?*
dtype02'
%batch_normalization_17/ReadVariableOp?
'batch_normalization_17/ReadVariableOp_1ReadVariableOp0batch_normalization_17_readvariableop_1_resource*
_output_shapes	
:?*
dtype02)
'batch_normalization_17/ReadVariableOp_1?
6batch_normalization_17/FusedBatchNormV3/ReadVariableOpReadVariableOp?batch_normalization_17_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:?*
dtype028
6batch_normalization_17/FusedBatchNormV3/ReadVariableOp?
8batch_normalization_17/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpAbatch_normalization_17_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:?*
dtype02:
8batch_normalization_17/FusedBatchNormV3/ReadVariableOp_1?
'batch_normalization_17/FusedBatchNormV3FusedBatchNormV3 activation_32/Relu:activations:0-batch_normalization_17/ReadVariableOp:value:0/batch_normalization_17/ReadVariableOp_1:value:0>batch_normalization_17/FusedBatchNormV3/ReadVariableOp:value:0@batch_normalization_17/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:??????????:?:?:?:?:*
epsilon%o?:*
exponential_avg_factor%
?#<2)
'batch_normalization_17/FusedBatchNormV3?
%batch_normalization_17/AssignNewValueAssignVariableOp?batch_normalization_17_fusedbatchnormv3_readvariableop_resource4batch_normalization_17/FusedBatchNormV3:batch_mean:07^batch_normalization_17/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype02'
%batch_normalization_17/AssignNewValue?
'batch_normalization_17/AssignNewValue_1AssignVariableOpAbatch_normalization_17_fusedbatchnormv3_readvariableop_1_resource8batch_normalization_17/FusedBatchNormV3:batch_variance:09^batch_normalization_17/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype02)
'batch_normalization_17/AssignNewValue_1?
IdentityIdentity+batch_normalization_17/FusedBatchNormV3:y:0^NoOp*
T0*0
_output_shapes
:??????????2

Identity?
NoOpNoOp&^batch_normalization_12/AssignNewValue(^batch_normalization_12/AssignNewValue_17^batch_normalization_12/FusedBatchNormV3/ReadVariableOp9^batch_normalization_12/FusedBatchNormV3/ReadVariableOp_1&^batch_normalization_12/ReadVariableOp(^batch_normalization_12/ReadVariableOp_1&^batch_normalization_13/AssignNewValue(^batch_normalization_13/AssignNewValue_17^batch_normalization_13/FusedBatchNormV3/ReadVariableOp9^batch_normalization_13/FusedBatchNormV3/ReadVariableOp_1&^batch_normalization_13/ReadVariableOp(^batch_normalization_13/ReadVariableOp_1&^batch_normalization_14/AssignNewValue(^batch_normalization_14/AssignNewValue_17^batch_normalization_14/FusedBatchNormV3/ReadVariableOp9^batch_normalization_14/FusedBatchNormV3/ReadVariableOp_1&^batch_normalization_14/ReadVariableOp(^batch_normalization_14/ReadVariableOp_1&^batch_normalization_15/AssignNewValue(^batch_normalization_15/AssignNewValue_17^batch_normalization_15/FusedBatchNormV3/ReadVariableOp9^batch_normalization_15/FusedBatchNormV3/ReadVariableOp_1&^batch_normalization_15/ReadVariableOp(^batch_normalization_15/ReadVariableOp_1&^batch_normalization_16/AssignNewValue(^batch_normalization_16/AssignNewValue_17^batch_normalization_16/FusedBatchNormV3/ReadVariableOp9^batch_normalization_16/FusedBatchNormV3/ReadVariableOp_1&^batch_normalization_16/ReadVariableOp(^batch_normalization_16/ReadVariableOp_1&^batch_normalization_17/AssignNewValue(^batch_normalization_17/AssignNewValue_17^batch_normalization_17/FusedBatchNormV3/ReadVariableOp9^batch_normalization_17/FusedBatchNormV3/ReadVariableOp_1&^batch_normalization_17/ReadVariableOp(^batch_normalization_17/ReadVariableOp_1!^conv2d_28/BiasAdd/ReadVariableOp ^conv2d_28/Conv2D/ReadVariableOp!^conv2d_29/BiasAdd/ReadVariableOp ^conv2d_29/Conv2D/ReadVariableOp!^conv2d_30/BiasAdd/ReadVariableOp ^conv2d_30/Conv2D/ReadVariableOp!^conv2d_31/BiasAdd/ReadVariableOp ^conv2d_31/Conv2D/ReadVariableOp!^conv2d_32/BiasAdd/ReadVariableOp ^conv2d_32/Conv2D/ReadVariableOp!^conv2d_33/BiasAdd/ReadVariableOp ^conv2d_33/Conv2D/ReadVariableOp!^conv2d_34/BiasAdd/ReadVariableOp ^conv2d_34/Conv2D/ReadVariableOp!^conv2d_35/BiasAdd/ReadVariableOp ^conv2d_35/Conv2D/ReadVariableOp!^conv2d_36/BiasAdd/ReadVariableOp ^conv2d_36/Conv2D/ReadVariableOp!^conv2d_37/BiasAdd/ReadVariableOp ^conv2d_37/Conv2D/ReadVariableOp!^conv2d_38/BiasAdd/ReadVariableOp ^conv2d_38/Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapesy
w:?????????: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2N
%batch_normalization_12/AssignNewValue%batch_normalization_12/AssignNewValue2R
'batch_normalization_12/AssignNewValue_1'batch_normalization_12/AssignNewValue_12p
6batch_normalization_12/FusedBatchNormV3/ReadVariableOp6batch_normalization_12/FusedBatchNormV3/ReadVariableOp2t
8batch_normalization_12/FusedBatchNormV3/ReadVariableOp_18batch_normalization_12/FusedBatchNormV3/ReadVariableOp_12N
%batch_normalization_12/ReadVariableOp%batch_normalization_12/ReadVariableOp2R
'batch_normalization_12/ReadVariableOp_1'batch_normalization_12/ReadVariableOp_12N
%batch_normalization_13/AssignNewValue%batch_normalization_13/AssignNewValue2R
'batch_normalization_13/AssignNewValue_1'batch_normalization_13/AssignNewValue_12p
6batch_normalization_13/FusedBatchNormV3/ReadVariableOp6batch_normalization_13/FusedBatchNormV3/ReadVariableOp2t
8batch_normalization_13/FusedBatchNormV3/ReadVariableOp_18batch_normalization_13/FusedBatchNormV3/ReadVariableOp_12N
%batch_normalization_13/ReadVariableOp%batch_normalization_13/ReadVariableOp2R
'batch_normalization_13/ReadVariableOp_1'batch_normalization_13/ReadVariableOp_12N
%batch_normalization_14/AssignNewValue%batch_normalization_14/AssignNewValue2R
'batch_normalization_14/AssignNewValue_1'batch_normalization_14/AssignNewValue_12p
6batch_normalization_14/FusedBatchNormV3/ReadVariableOp6batch_normalization_14/FusedBatchNormV3/ReadVariableOp2t
8batch_normalization_14/FusedBatchNormV3/ReadVariableOp_18batch_normalization_14/FusedBatchNormV3/ReadVariableOp_12N
%batch_normalization_14/ReadVariableOp%batch_normalization_14/ReadVariableOp2R
'batch_normalization_14/ReadVariableOp_1'batch_normalization_14/ReadVariableOp_12N
%batch_normalization_15/AssignNewValue%batch_normalization_15/AssignNewValue2R
'batch_normalization_15/AssignNewValue_1'batch_normalization_15/AssignNewValue_12p
6batch_normalization_15/FusedBatchNormV3/ReadVariableOp6batch_normalization_15/FusedBatchNormV3/ReadVariableOp2t
8batch_normalization_15/FusedBatchNormV3/ReadVariableOp_18batch_normalization_15/FusedBatchNormV3/ReadVariableOp_12N
%batch_normalization_15/ReadVariableOp%batch_normalization_15/ReadVariableOp2R
'batch_normalization_15/ReadVariableOp_1'batch_normalization_15/ReadVariableOp_12N
%batch_normalization_16/AssignNewValue%batch_normalization_16/AssignNewValue2R
'batch_normalization_16/AssignNewValue_1'batch_normalization_16/AssignNewValue_12p
6batch_normalization_16/FusedBatchNormV3/ReadVariableOp6batch_normalization_16/FusedBatchNormV3/ReadVariableOp2t
8batch_normalization_16/FusedBatchNormV3/ReadVariableOp_18batch_normalization_16/FusedBatchNormV3/ReadVariableOp_12N
%batch_normalization_16/ReadVariableOp%batch_normalization_16/ReadVariableOp2R
'batch_normalization_16/ReadVariableOp_1'batch_normalization_16/ReadVariableOp_12N
%batch_normalization_17/AssignNewValue%batch_normalization_17/AssignNewValue2R
'batch_normalization_17/AssignNewValue_1'batch_normalization_17/AssignNewValue_12p
6batch_normalization_17/FusedBatchNormV3/ReadVariableOp6batch_normalization_17/FusedBatchNormV3/ReadVariableOp2t
8batch_normalization_17/FusedBatchNormV3/ReadVariableOp_18batch_normalization_17/FusedBatchNormV3/ReadVariableOp_12N
%batch_normalization_17/ReadVariableOp%batch_normalization_17/ReadVariableOp2R
'batch_normalization_17/ReadVariableOp_1'batch_normalization_17/ReadVariableOp_12D
 conv2d_28/BiasAdd/ReadVariableOp conv2d_28/BiasAdd/ReadVariableOp2B
conv2d_28/Conv2D/ReadVariableOpconv2d_28/Conv2D/ReadVariableOp2D
 conv2d_29/BiasAdd/ReadVariableOp conv2d_29/BiasAdd/ReadVariableOp2B
conv2d_29/Conv2D/ReadVariableOpconv2d_29/Conv2D/ReadVariableOp2D
 conv2d_30/BiasAdd/ReadVariableOp conv2d_30/BiasAdd/ReadVariableOp2B
conv2d_30/Conv2D/ReadVariableOpconv2d_30/Conv2D/ReadVariableOp2D
 conv2d_31/BiasAdd/ReadVariableOp conv2d_31/BiasAdd/ReadVariableOp2B
conv2d_31/Conv2D/ReadVariableOpconv2d_31/Conv2D/ReadVariableOp2D
 conv2d_32/BiasAdd/ReadVariableOp conv2d_32/BiasAdd/ReadVariableOp2B
conv2d_32/Conv2D/ReadVariableOpconv2d_32/Conv2D/ReadVariableOp2D
 conv2d_33/BiasAdd/ReadVariableOp conv2d_33/BiasAdd/ReadVariableOp2B
conv2d_33/Conv2D/ReadVariableOpconv2d_33/Conv2D/ReadVariableOp2D
 conv2d_34/BiasAdd/ReadVariableOp conv2d_34/BiasAdd/ReadVariableOp2B
conv2d_34/Conv2D/ReadVariableOpconv2d_34/Conv2D/ReadVariableOp2D
 conv2d_35/BiasAdd/ReadVariableOp conv2d_35/BiasAdd/ReadVariableOp2B
conv2d_35/Conv2D/ReadVariableOpconv2d_35/Conv2D/ReadVariableOp2D
 conv2d_36/BiasAdd/ReadVariableOp conv2d_36/BiasAdd/ReadVariableOp2B
conv2d_36/Conv2D/ReadVariableOpconv2d_36/Conv2D/ReadVariableOp2D
 conv2d_37/BiasAdd/ReadVariableOp conv2d_37/BiasAdd/ReadVariableOp2B
conv2d_37/Conv2D/ReadVariableOpconv2d_37/Conv2D/ReadVariableOp2D
 conv2d_38/BiasAdd/ReadVariableOp conv2d_38/BiasAdd/ReadVariableOp2B
conv2d_38/Conv2D/ReadVariableOpconv2d_38/Conv2D/ReadVariableOp:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
*__inference_model_7_layer_call_fn_22006276

inputs"
unknown:?
	unknown_0:	?
	unknown_1:	?
	unknown_2:	?
	unknown_3:	?
	unknown_4:	?%
	unknown_5:??
	unknown_6:	?%
	unknown_7:??
	unknown_8:	?
	unknown_9:	?

unknown_10:	?

unknown_11:	?

unknown_12:	?&

unknown_13:??

unknown_14:	?&

unknown_15:??

unknown_16:	?

unknown_17:	?

unknown_18:	?

unknown_19:	?

unknown_20:	?&

unknown_21:??

unknown_22:	?&

unknown_23:??

unknown_24:	?

unknown_25:	?

unknown_26:	?

unknown_27:	?

unknown_28:	?&

unknown_29:??

unknown_30:	?&

unknown_31:??

unknown_32:	?

unknown_33:	?

unknown_34:	?

unknown_35:	?

unknown_36:	?&

unknown_37:??

unknown_38:	?&

unknown_39:??

unknown_40:	?

unknown_41:	?

unknown_42:	?

unknown_43:	?

unknown_44:	?%

unknown_45:?

unknown_46:&

unknown_47:??

unknown_48:	?%

unknown_49:?I

unknown_50:I

unknown_51:@
identity

identity_1??StatefulPartitionedCall?
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
unknown_23
unknown_24
unknown_25
unknown_26
unknown_27
unknown_28
unknown_29
unknown_30
unknown_31
unknown_32
unknown_33
unknown_34
unknown_35
unknown_36
unknown_37
unknown_38
unknown_39
unknown_40
unknown_41
unknown_42
unknown_43
unknown_44
unknown_45
unknown_46
unknown_47
unknown_48
unknown_49
unknown_50
unknown_51*A
Tin:
826*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:?????????:?????????I*W
_read_only_resource_inputs9
75	
 !"#$%&'()*+,-./012345*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_model_7_layer_call_and_return_conditional_losses_220051672
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
_construction_contextkEagerRuntime*?
_input_shapes?
?:?????????: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
T__inference_batch_normalization_12_layer_call_and_return_conditional_losses_22002330

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
I
-__inference_dropout_51_layer_call_fn_22007368

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
GPU 2J 8? *Q
fLRJ
H__inference_dropout_51_layer_call_and_return_conditional_losses_220051192
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
?
f
H__inference_dropout_32_layer_call_and_return_conditional_losses_22008295

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
d
H__inference_dropout_50_layer_call_and_return_conditional_losses_22005314

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
?
G__inference_conv2d_36_layer_call_and_return_conditional_losses_22003432

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
f
H__inference_dropout_28_layer_call_and_return_conditional_losses_22003214

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
f
H__inference_dropout_34_layer_call_and_return_conditional_losses_22008515

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
?	
?
9__inference_batch_normalization_17_layer_call_fn_22008622

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
GPU 2J 8? *]
fXRV
T__inference_batch_normalization_17_layer_call_and_return_conditional_losses_220030042
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
?	
?
9__inference_batch_normalization_17_layer_call_fn_22008609

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
GPU 2J 8? *]
fXRV
T__inference_batch_normalization_17_layer_call_and_return_conditional_losses_220029602
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
?
?
G__inference_conv2d_30_layer_call_and_return_conditional_losses_22007706

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
?
I
-__inference_dropout_34_layer_call_fn_22008510

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
GPU 2J 8? *Q
fLRJ
H__inference_dropout_34_layer_call_and_return_conditional_losses_220037822
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
I
-__inference_dropout_26_layer_call_fn_22007630

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
GPU 2J 8? *Q
fLRJ
H__inference_dropout_26_layer_call_and_return_conditional_losses_220042062
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
I
-__inference_dropout_29_layer_call_fn_22007893

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
GPU 2J 8? *Q
fLRJ
H__inference_dropout_29_layer_call_and_return_conditional_losses_220032442
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
9__inference_batch_normalization_14_layer_call_fn_22007975

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
GPU 2J 8? *]
fXRV
T__inference_batch_normalization_14_layer_call_and_return_conditional_losses_220032872
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
9__inference_batch_normalization_14_layer_call_fn_22007949

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
GPU 2J 8? *]
fXRV
T__inference_batch_normalization_14_layer_call_and_return_conditional_losses_220025822
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
?
?
G__inference_conv2d_38_layer_call_and_return_conditional_losses_22003520

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
?
I
-__inference_dropout_36_layer_call_fn_22008725

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
GPU 2J 8? *Q
fLRJ
H__inference_dropout_36_layer_call_and_return_conditional_losses_220035662
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
?
?
T__inference_batch_normalization_12_layer_call_and_return_conditional_losses_22004248

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
?
?
*__inference_model_7_layer_call_fn_22005810
input_8"
unknown:?
	unknown_0:	?
	unknown_1:	?
	unknown_2:	?
	unknown_3:	?
	unknown_4:	?%
	unknown_5:??
	unknown_6:	?%
	unknown_7:??
	unknown_8:	?
	unknown_9:	?

unknown_10:	?

unknown_11:	?

unknown_12:	?&

unknown_13:??

unknown_14:	?&

unknown_15:??

unknown_16:	?

unknown_17:	?

unknown_18:	?

unknown_19:	?

unknown_20:	?&

unknown_21:??

unknown_22:	?&

unknown_23:??

unknown_24:	?

unknown_25:	?

unknown_26:	?

unknown_27:	?

unknown_28:	?&

unknown_29:??

unknown_30:	?&

unknown_31:??

unknown_32:	?

unknown_33:	?

unknown_34:	?

unknown_35:	?

unknown_36:	?&

unknown_37:??

unknown_38:	?&

unknown_39:??

unknown_40:	?

unknown_41:	?

unknown_42:	?

unknown_43:	?

unknown_44:	?%

unknown_45:?

unknown_46:&

unknown_47:??

unknown_48:	?%

unknown_49:?I

unknown_50:I

unknown_51:@
identity

identity_1??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_8unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
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
unknown_23
unknown_24
unknown_25
unknown_26
unknown_27
unknown_28
unknown_29
unknown_30
unknown_31
unknown_32
unknown_33
unknown_34
unknown_35
unknown_36
unknown_37
unknown_38
unknown_39
unknown_40
unknown_41
unknown_42
unknown_43
unknown_44
unknown_45
unknown_46
unknown_47
unknown_48
unknown_49
unknown_50
unknown_51*A
Tin:
826*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:?????????:?????????I*K
_read_only_resource_inputs-
+)	
 !"#$'()*+,/012345*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_model_7_layer_call_and_return_conditional_losses_220055862
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
_construction_contextkEagerRuntime*?
_input_shapes?
?:?????????: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
/
_output_shapes
:?????????
!
_user_specified_name	input_8
?
?
,__inference_conv2d_33_layer_call_fn_22008088

inputs#
unknown:??
	unknown_0:	?
identity??StatefulPartitionedCall?
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
GPU 2J 8? *P
fKRI
G__inference_conv2d_33_layer_call_and_return_conditional_losses_220033142
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
?<
?
E__inference_model_7_layer_call_and_return_conditional_losses_22005929
input_8+
model_4_22005813:?
model_4_22005815:	?
model_4_22005817:	?
model_4_22005819:	?
model_4_22005821:	?
model_4_22005823:	?,
model_4_22005825:??
model_4_22005827:	?,
model_4_22005829:??
model_4_22005831:	?
model_4_22005833:	?
model_4_22005835:	?
model_4_22005837:	?
model_4_22005839:	?,
model_4_22005841:??
model_4_22005843:	?,
model_4_22005845:??
model_4_22005847:	?
model_4_22005849:	?
model_4_22005851:	?
model_4_22005853:	?
model_4_22005855:	?,
model_4_22005857:??
model_4_22005859:	?,
model_4_22005861:??
model_4_22005863:	?
model_4_22005865:	?
model_4_22005867:	?
model_4_22005869:	?
model_4_22005871:	?,
model_4_22005873:??
model_4_22005875:	?,
model_4_22005877:??
model_4_22005879:	?
model_4_22005881:	?
model_4_22005883:	?
model_4_22005885:	?
model_4_22005887:	?,
model_4_22005889:??
model_4_22005891:	?,
model_4_22005893:??
model_4_22005895:	?
model_4_22005897:	?
model_4_22005899:	?
model_4_22005901:	?
model_4_22005903:	?-
conv2d_55_22005906:? 
conv2d_55_22005908:.
conv2d_53_22005911:??!
conv2d_53_22005913:	?-
conv2d_54_22005919:?I 
conv2d_54_22005921:I"
dense_3_22005924:@
identity

identity_1??!conv2d_53/StatefulPartitionedCall?!conv2d_54/StatefulPartitionedCall?!conv2d_55/StatefulPartitionedCall?dense_3/StatefulPartitionedCall?model_4/StatefulPartitionedCall?

model_4/StatefulPartitionedCallStatefulPartitionedCallinput_8model_4_22005813model_4_22005815model_4_22005817model_4_22005819model_4_22005821model_4_22005823model_4_22005825model_4_22005827model_4_22005829model_4_22005831model_4_22005833model_4_22005835model_4_22005837model_4_22005839model_4_22005841model_4_22005843model_4_22005845model_4_22005847model_4_22005849model_4_22005851model_4_22005853model_4_22005855model_4_22005857model_4_22005859model_4_22005861model_4_22005863model_4_22005865model_4_22005867model_4_22005869model_4_22005871model_4_22005873model_4_22005875model_4_22005877model_4_22005879model_4_22005881model_4_22005883model_4_22005885model_4_22005887model_4_22005889model_4_22005891model_4_22005893model_4_22005895model_4_22005897model_4_22005899model_4_22005901model_4_22005903*:
Tin3
12/*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????*P
_read_only_resource_inputs2
0.	
 !"#$%&'()*+,-.*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_model_4_layer_call_and_return_conditional_losses_220035692!
model_4/StatefulPartitionedCall?
!conv2d_55/StatefulPartitionedCallStatefulPartitionedCall(model_4/StatefulPartitionedCall:output:0conv2d_55_22005906conv2d_55_22005908*
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
GPU 2J 8? *P
fKRI
G__inference_conv2d_55_layer_call_and_return_conditional_losses_220050912#
!conv2d_55/StatefulPartitionedCall?
!conv2d_53/StatefulPartitionedCallStatefulPartitionedCall(model_4/StatefulPartitionedCall:output:0conv2d_53_22005911conv2d_53_22005913*
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
GPU 2J 8? *P
fKRI
G__inference_conv2d_53_layer_call_and_return_conditional_losses_220051082#
!conv2d_53/StatefulPartitionedCall?
dropout_51/PartitionedCallPartitionedCall*conv2d_55/StatefulPartitionedCall:output:0*
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
GPU 2J 8? *Q
fLRJ
H__inference_dropout_51_layer_call_and_return_conditional_losses_220051192
dropout_51/PartitionedCall?
dropout_50/PartitionedCallPartitionedCall*conv2d_53/StatefulPartitionedCall:output:0*
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
GPU 2J 8? *Q
fLRJ
H__inference_dropout_50_layer_call_and_return_conditional_losses_220051262
dropout_50/PartitionedCall?
flatten_3/PartitionedCallPartitionedCall#dropout_51/PartitionedCall:output:0*
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
GPU 2J 8? *P
fKRI
G__inference_flatten_3_layer_call_and_return_conditional_losses_220051342
flatten_3/PartitionedCall?
!conv2d_54/StatefulPartitionedCallStatefulPartitionedCall#dropout_50/PartitionedCall:output:0conv2d_54_22005919conv2d_54_22005921*
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
GPU 2J 8? *P
fKRI
G__inference_conv2d_54_layer_call_and_return_conditional_losses_220051472#
!conv2d_54/StatefulPartitionedCall?
dense_3/StatefulPartitionedCallStatefulPartitionedCall"flatten_3/PartitionedCall:output:0dense_3_22005924*
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
GPU 2J 8? *N
fIRG
E__inference_dense_3_layer_call_and_return_conditional_losses_220051612!
dense_3/StatefulPartitionedCall?
IdentityIdentity(dense_3/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????2

Identity?

Identity_1Identity*conv2d_54/StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:?????????I2

Identity_1?
NoOpNoOp"^conv2d_53/StatefulPartitionedCall"^conv2d_54/StatefulPartitionedCall"^conv2d_55/StatefulPartitionedCall ^dense_3/StatefulPartitionedCall ^model_4/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes?
?:?????????: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2F
!conv2d_53/StatefulPartitionedCall!conv2d_53/StatefulPartitionedCall2F
!conv2d_54/StatefulPartitionedCall!conv2d_54/StatefulPartitionedCall2F
!conv2d_55/StatefulPartitionedCall!conv2d_55/StatefulPartitionedCall2B
dense_3/StatefulPartitionedCalldense_3/StatefulPartitionedCall2B
model_4/StatefulPartitionedCallmodel_4/StatefulPartitionedCall:X T
/
_output_shapes
:?????????
!
_user_specified_name	input_8
?
g
K__inference_activation_32_layer_call_and_return_conditional_losses_22003532

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
?
?
G__inference_conv2d_34_layer_call_and_return_conditional_losses_22008146

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
?
?
T__inference_batch_normalization_16_layer_call_and_return_conditional_losses_22008482

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
?
?
,__inference_conv2d_29_layer_call_fn_22007648

inputs#
unknown:??
	unknown_0:	?
identity??StatefulPartitionedCall?
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
GPU 2J 8? *P
fKRI
G__inference_conv2d_29_layer_call_and_return_conditional_losses_220031382
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
?
L
0__inference_activation_29_layer_call_fn_22008323

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
GPU 2J 8? *T
fORM
K__inference_activation_29_layer_call_and_return_conditional_losses_220034132
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
G__inference_conv2d_53_layer_call_and_return_conditional_losses_22005108

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
?
d
H__inference_dropout_29_layer_call_and_return_conditional_losses_22004069

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
I
-__inference_dropout_30_layer_call_fn_22008065

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
GPU 2J 8? *Q
fLRJ
H__inference_dropout_30_layer_call_and_return_conditional_losses_220033022
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
?
?
*__inference_model_4_layer_call_fn_22006992

inputs"
unknown:?
	unknown_0:	?
	unknown_1:	?
	unknown_2:	?
	unknown_3:	?
	unknown_4:	?%
	unknown_5:??
	unknown_6:	?%
	unknown_7:??
	unknown_8:	?
	unknown_9:	?

unknown_10:	?

unknown_11:	?

unknown_12:	?&

unknown_13:??

unknown_14:	?&

unknown_15:??

unknown_16:	?

unknown_17:	?

unknown_18:	?

unknown_19:	?

unknown_20:	?&

unknown_21:??

unknown_22:	?&

unknown_23:??

unknown_24:	?

unknown_25:	?

unknown_26:	?

unknown_27:	?

unknown_28:	?&

unknown_29:??

unknown_30:	?&

unknown_31:??

unknown_32:	?

unknown_33:	?

unknown_34:	?

unknown_35:	?

unknown_36:	?&

unknown_37:??

unknown_38:	?&

unknown_39:??

unknown_40:	?

unknown_41:	?

unknown_42:	?

unknown_43:	?

unknown_44:	?
identity??StatefulPartitionedCall?
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
unknown_23
unknown_24
unknown_25
unknown_26
unknown_27
unknown_28
unknown_29
unknown_30
unknown_31
unknown_32
unknown_33
unknown_34
unknown_35
unknown_36
unknown_37
unknown_38
unknown_39
unknown_40
unknown_41
unknown_42
unknown_43
unknown_44*:
Tin3
12/*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????*D
_read_only_resource_inputs&
$"	
 !"#$'()*+,*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_model_4_layer_call_and_return_conditional_losses_220045082
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
_construction_contextkEagerRuntime*?
_input_shapesy
w:?????????: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
T__inference_batch_normalization_17_layer_call_and_return_conditional_losses_22003004

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
9__inference_batch_normalization_12_layer_call_fn_22007538

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
GPU 2J 8? *]
fXRV
T__inference_batch_normalization_12_layer_call_and_return_conditional_losses_220042482
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
?
?
T__inference_batch_normalization_17_layer_call_and_return_conditional_losses_22008702

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
??
?#
$__inference__traced_restore_22009091
file_prefix<
!assignvariableop_conv2d_55_kernel:?/
!assignvariableop_1_conv2d_55_bias:?
#assignvariableop_2_conv2d_53_kernel:??0
!assignvariableop_3_conv2d_53_bias:	?3
!assignvariableop_4_dense_3_kernel:@>
#assignvariableop_5_conv2d_54_kernel:?I/
!assignvariableop_6_conv2d_54_bias:I>
#assignvariableop_7_conv2d_28_kernel:?0
!assignvariableop_8_conv2d_28_bias:	?>
/assignvariableop_9_batch_normalization_12_gamma:	?>
/assignvariableop_10_batch_normalization_12_beta:	?E
6assignvariableop_11_batch_normalization_12_moving_mean:	?I
:assignvariableop_12_batch_normalization_12_moving_variance:	?@
$assignvariableop_13_conv2d_29_kernel:??1
"assignvariableop_14_conv2d_29_bias:	?@
$assignvariableop_15_conv2d_30_kernel:??1
"assignvariableop_16_conv2d_30_bias:	??
0assignvariableop_17_batch_normalization_13_gamma:	?>
/assignvariableop_18_batch_normalization_13_beta:	?E
6assignvariableop_19_batch_normalization_13_moving_mean:	?I
:assignvariableop_20_batch_normalization_13_moving_variance:	?@
$assignvariableop_21_conv2d_31_kernel:??1
"assignvariableop_22_conv2d_31_bias:	?@
$assignvariableop_23_conv2d_32_kernel:??1
"assignvariableop_24_conv2d_32_bias:	??
0assignvariableop_25_batch_normalization_14_gamma:	?>
/assignvariableop_26_batch_normalization_14_beta:	?E
6assignvariableop_27_batch_normalization_14_moving_mean:	?I
:assignvariableop_28_batch_normalization_14_moving_variance:	?@
$assignvariableop_29_conv2d_33_kernel:??1
"assignvariableop_30_conv2d_33_bias:	?@
$assignvariableop_31_conv2d_34_kernel:??1
"assignvariableop_32_conv2d_34_bias:	??
0assignvariableop_33_batch_normalization_15_gamma:	?>
/assignvariableop_34_batch_normalization_15_beta:	?E
6assignvariableop_35_batch_normalization_15_moving_mean:	?I
:assignvariableop_36_batch_normalization_15_moving_variance:	?@
$assignvariableop_37_conv2d_35_kernel:??1
"assignvariableop_38_conv2d_35_bias:	?@
$assignvariableop_39_conv2d_36_kernel:??1
"assignvariableop_40_conv2d_36_bias:	??
0assignvariableop_41_batch_normalization_16_gamma:	?>
/assignvariableop_42_batch_normalization_16_beta:	?E
6assignvariableop_43_batch_normalization_16_moving_mean:	?I
:assignvariableop_44_batch_normalization_16_moving_variance:	?@
$assignvariableop_45_conv2d_37_kernel:??1
"assignvariableop_46_conv2d_37_bias:	?@
$assignvariableop_47_conv2d_38_kernel:??1
"assignvariableop_48_conv2d_38_bias:	??
0assignvariableop_49_batch_normalization_17_gamma:	?>
/assignvariableop_50_batch_normalization_17_beta:	?E
6assignvariableop_51_batch_normalization_17_moving_mean:	?I
:assignvariableop_52_batch_normalization_17_moving_variance:	?
identity_54??AssignVariableOp?AssignVariableOp_1?AssignVariableOp_10?AssignVariableOp_11?AssignVariableOp_12?AssignVariableOp_13?AssignVariableOp_14?AssignVariableOp_15?AssignVariableOp_16?AssignVariableOp_17?AssignVariableOp_18?AssignVariableOp_19?AssignVariableOp_2?AssignVariableOp_20?AssignVariableOp_21?AssignVariableOp_22?AssignVariableOp_23?AssignVariableOp_24?AssignVariableOp_25?AssignVariableOp_26?AssignVariableOp_27?AssignVariableOp_28?AssignVariableOp_29?AssignVariableOp_3?AssignVariableOp_30?AssignVariableOp_31?AssignVariableOp_32?AssignVariableOp_33?AssignVariableOp_34?AssignVariableOp_35?AssignVariableOp_36?AssignVariableOp_37?AssignVariableOp_38?AssignVariableOp_39?AssignVariableOp_4?AssignVariableOp_40?AssignVariableOp_41?AssignVariableOp_42?AssignVariableOp_43?AssignVariableOp_44?AssignVariableOp_45?AssignVariableOp_46?AssignVariableOp_47?AssignVariableOp_48?AssignVariableOp_49?AssignVariableOp_5?AssignVariableOp_50?AssignVariableOp_51?AssignVariableOp_52?AssignVariableOp_6?AssignVariableOp_7?AssignVariableOp_8?AssignVariableOp_9?
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:6*
dtype0*?
value?B?6B6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB&variables/0/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB&variables/3/.ATTRIBUTES/VARIABLE_VALUEB&variables/4/.ATTRIBUTES/VARIABLE_VALUEB&variables/5/.ATTRIBUTES/VARIABLE_VALUEB&variables/6/.ATTRIBUTES/VARIABLE_VALUEB&variables/7/.ATTRIBUTES/VARIABLE_VALUEB&variables/8/.ATTRIBUTES/VARIABLE_VALUEB&variables/9/.ATTRIBUTES/VARIABLE_VALUEB'variables/10/.ATTRIBUTES/VARIABLE_VALUEB'variables/11/.ATTRIBUTES/VARIABLE_VALUEB'variables/12/.ATTRIBUTES/VARIABLE_VALUEB'variables/13/.ATTRIBUTES/VARIABLE_VALUEB'variables/14/.ATTRIBUTES/VARIABLE_VALUEB'variables/15/.ATTRIBUTES/VARIABLE_VALUEB'variables/16/.ATTRIBUTES/VARIABLE_VALUEB'variables/17/.ATTRIBUTES/VARIABLE_VALUEB'variables/18/.ATTRIBUTES/VARIABLE_VALUEB'variables/19/.ATTRIBUTES/VARIABLE_VALUEB'variables/20/.ATTRIBUTES/VARIABLE_VALUEB'variables/21/.ATTRIBUTES/VARIABLE_VALUEB'variables/22/.ATTRIBUTES/VARIABLE_VALUEB'variables/23/.ATTRIBUTES/VARIABLE_VALUEB'variables/24/.ATTRIBUTES/VARIABLE_VALUEB'variables/25/.ATTRIBUTES/VARIABLE_VALUEB'variables/26/.ATTRIBUTES/VARIABLE_VALUEB'variables/27/.ATTRIBUTES/VARIABLE_VALUEB'variables/28/.ATTRIBUTES/VARIABLE_VALUEB'variables/29/.ATTRIBUTES/VARIABLE_VALUEB'variables/30/.ATTRIBUTES/VARIABLE_VALUEB'variables/31/.ATTRIBUTES/VARIABLE_VALUEB'variables/32/.ATTRIBUTES/VARIABLE_VALUEB'variables/33/.ATTRIBUTES/VARIABLE_VALUEB'variables/34/.ATTRIBUTES/VARIABLE_VALUEB'variables/35/.ATTRIBUTES/VARIABLE_VALUEB'variables/36/.ATTRIBUTES/VARIABLE_VALUEB'variables/37/.ATTRIBUTES/VARIABLE_VALUEB'variables/38/.ATTRIBUTES/VARIABLE_VALUEB'variables/39/.ATTRIBUTES/VARIABLE_VALUEB'variables/40/.ATTRIBUTES/VARIABLE_VALUEB'variables/41/.ATTRIBUTES/VARIABLE_VALUEB'variables/42/.ATTRIBUTES/VARIABLE_VALUEB'variables/43/.ATTRIBUTES/VARIABLE_VALUEB'variables/44/.ATTRIBUTES/VARIABLE_VALUEB'variables/45/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2/tensor_names?
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:6*
dtype0*
valuevBt6B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
RestoreV2/shape_and_slices?
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*?
_output_shapes?
?::::::::::::::::::::::::::::::::::::::::::::::::::::::*D
dtypes:
8262
	RestoreV2g
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:2

Identity?
AssignVariableOpAssignVariableOp!assignvariableop_conv2d_55_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOpk

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:2

Identity_1?
AssignVariableOp_1AssignVariableOp!assignvariableop_1_conv2d_55_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_1k

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:2

Identity_2?
AssignVariableOp_2AssignVariableOp#assignvariableop_2_conv2d_53_kernelIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_2k

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:2

Identity_3?
AssignVariableOp_3AssignVariableOp!assignvariableop_3_conv2d_53_biasIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_3k

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:2

Identity_4?
AssignVariableOp_4AssignVariableOp!assignvariableop_4_dense_3_kernelIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_4k

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:2

Identity_5?
AssignVariableOp_5AssignVariableOp#assignvariableop_5_conv2d_54_kernelIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_5k

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:2

Identity_6?
AssignVariableOp_6AssignVariableOp!assignvariableop_6_conv2d_54_biasIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_6k

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:2

Identity_7?
AssignVariableOp_7AssignVariableOp#assignvariableop_7_conv2d_28_kernelIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_7k

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:2

Identity_8?
AssignVariableOp_8AssignVariableOp!assignvariableop_8_conv2d_28_biasIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_8k

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:2

Identity_9?
AssignVariableOp_9AssignVariableOp/assignvariableop_9_batch_normalization_12_gammaIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_9n
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:2
Identity_10?
AssignVariableOp_10AssignVariableOp/assignvariableop_10_batch_normalization_12_betaIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_10n
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:2
Identity_11?
AssignVariableOp_11AssignVariableOp6assignvariableop_11_batch_normalization_12_moving_meanIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_11n
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:2
Identity_12?
AssignVariableOp_12AssignVariableOp:assignvariableop_12_batch_normalization_12_moving_varianceIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_12n
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:2
Identity_13?
AssignVariableOp_13AssignVariableOp$assignvariableop_13_conv2d_29_kernelIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_13n
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:2
Identity_14?
AssignVariableOp_14AssignVariableOp"assignvariableop_14_conv2d_29_biasIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_14n
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:2
Identity_15?
AssignVariableOp_15AssignVariableOp$assignvariableop_15_conv2d_30_kernelIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_15n
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:2
Identity_16?
AssignVariableOp_16AssignVariableOp"assignvariableop_16_conv2d_30_biasIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_16n
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:2
Identity_17?
AssignVariableOp_17AssignVariableOp0assignvariableop_17_batch_normalization_13_gammaIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_17n
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:2
Identity_18?
AssignVariableOp_18AssignVariableOp/assignvariableop_18_batch_normalization_13_betaIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_18n
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:2
Identity_19?
AssignVariableOp_19AssignVariableOp6assignvariableop_19_batch_normalization_13_moving_meanIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_19n
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:2
Identity_20?
AssignVariableOp_20AssignVariableOp:assignvariableop_20_batch_normalization_13_moving_varianceIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_20n
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:2
Identity_21?
AssignVariableOp_21AssignVariableOp$assignvariableop_21_conv2d_31_kernelIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_21n
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:2
Identity_22?
AssignVariableOp_22AssignVariableOp"assignvariableop_22_conv2d_31_biasIdentity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_22n
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:2
Identity_23?
AssignVariableOp_23AssignVariableOp$assignvariableop_23_conv2d_32_kernelIdentity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_23n
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:2
Identity_24?
AssignVariableOp_24AssignVariableOp"assignvariableop_24_conv2d_32_biasIdentity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_24n
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:2
Identity_25?
AssignVariableOp_25AssignVariableOp0assignvariableop_25_batch_normalization_14_gammaIdentity_25:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_25n
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:2
Identity_26?
AssignVariableOp_26AssignVariableOp/assignvariableop_26_batch_normalization_14_betaIdentity_26:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_26n
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:2
Identity_27?
AssignVariableOp_27AssignVariableOp6assignvariableop_27_batch_normalization_14_moving_meanIdentity_27:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_27n
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:2
Identity_28?
AssignVariableOp_28AssignVariableOp:assignvariableop_28_batch_normalization_14_moving_varianceIdentity_28:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_28n
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:2
Identity_29?
AssignVariableOp_29AssignVariableOp$assignvariableop_29_conv2d_33_kernelIdentity_29:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_29n
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:2
Identity_30?
AssignVariableOp_30AssignVariableOp"assignvariableop_30_conv2d_33_biasIdentity_30:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_30n
Identity_31IdentityRestoreV2:tensors:31"/device:CPU:0*
T0*
_output_shapes
:2
Identity_31?
AssignVariableOp_31AssignVariableOp$assignvariableop_31_conv2d_34_kernelIdentity_31:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_31n
Identity_32IdentityRestoreV2:tensors:32"/device:CPU:0*
T0*
_output_shapes
:2
Identity_32?
AssignVariableOp_32AssignVariableOp"assignvariableop_32_conv2d_34_biasIdentity_32:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_32n
Identity_33IdentityRestoreV2:tensors:33"/device:CPU:0*
T0*
_output_shapes
:2
Identity_33?
AssignVariableOp_33AssignVariableOp0assignvariableop_33_batch_normalization_15_gammaIdentity_33:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_33n
Identity_34IdentityRestoreV2:tensors:34"/device:CPU:0*
T0*
_output_shapes
:2
Identity_34?
AssignVariableOp_34AssignVariableOp/assignvariableop_34_batch_normalization_15_betaIdentity_34:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_34n
Identity_35IdentityRestoreV2:tensors:35"/device:CPU:0*
T0*
_output_shapes
:2
Identity_35?
AssignVariableOp_35AssignVariableOp6assignvariableop_35_batch_normalization_15_moving_meanIdentity_35:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_35n
Identity_36IdentityRestoreV2:tensors:36"/device:CPU:0*
T0*
_output_shapes
:2
Identity_36?
AssignVariableOp_36AssignVariableOp:assignvariableop_36_batch_normalization_15_moving_varianceIdentity_36:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_36n
Identity_37IdentityRestoreV2:tensors:37"/device:CPU:0*
T0*
_output_shapes
:2
Identity_37?
AssignVariableOp_37AssignVariableOp$assignvariableop_37_conv2d_35_kernelIdentity_37:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_37n
Identity_38IdentityRestoreV2:tensors:38"/device:CPU:0*
T0*
_output_shapes
:2
Identity_38?
AssignVariableOp_38AssignVariableOp"assignvariableop_38_conv2d_35_biasIdentity_38:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_38n
Identity_39IdentityRestoreV2:tensors:39"/device:CPU:0*
T0*
_output_shapes
:2
Identity_39?
AssignVariableOp_39AssignVariableOp$assignvariableop_39_conv2d_36_kernelIdentity_39:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_39n
Identity_40IdentityRestoreV2:tensors:40"/device:CPU:0*
T0*
_output_shapes
:2
Identity_40?
AssignVariableOp_40AssignVariableOp"assignvariableop_40_conv2d_36_biasIdentity_40:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_40n
Identity_41IdentityRestoreV2:tensors:41"/device:CPU:0*
T0*
_output_shapes
:2
Identity_41?
AssignVariableOp_41AssignVariableOp0assignvariableop_41_batch_normalization_16_gammaIdentity_41:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_41n
Identity_42IdentityRestoreV2:tensors:42"/device:CPU:0*
T0*
_output_shapes
:2
Identity_42?
AssignVariableOp_42AssignVariableOp/assignvariableop_42_batch_normalization_16_betaIdentity_42:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_42n
Identity_43IdentityRestoreV2:tensors:43"/device:CPU:0*
T0*
_output_shapes
:2
Identity_43?
AssignVariableOp_43AssignVariableOp6assignvariableop_43_batch_normalization_16_moving_meanIdentity_43:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_43n
Identity_44IdentityRestoreV2:tensors:44"/device:CPU:0*
T0*
_output_shapes
:2
Identity_44?
AssignVariableOp_44AssignVariableOp:assignvariableop_44_batch_normalization_16_moving_varianceIdentity_44:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_44n
Identity_45IdentityRestoreV2:tensors:45"/device:CPU:0*
T0*
_output_shapes
:2
Identity_45?
AssignVariableOp_45AssignVariableOp$assignvariableop_45_conv2d_37_kernelIdentity_45:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_45n
Identity_46IdentityRestoreV2:tensors:46"/device:CPU:0*
T0*
_output_shapes
:2
Identity_46?
AssignVariableOp_46AssignVariableOp"assignvariableop_46_conv2d_37_biasIdentity_46:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_46n
Identity_47IdentityRestoreV2:tensors:47"/device:CPU:0*
T0*
_output_shapes
:2
Identity_47?
AssignVariableOp_47AssignVariableOp$assignvariableop_47_conv2d_38_kernelIdentity_47:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_47n
Identity_48IdentityRestoreV2:tensors:48"/device:CPU:0*
T0*
_output_shapes
:2
Identity_48?
AssignVariableOp_48AssignVariableOp"assignvariableop_48_conv2d_38_biasIdentity_48:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_48n
Identity_49IdentityRestoreV2:tensors:49"/device:CPU:0*
T0*
_output_shapes
:2
Identity_49?
AssignVariableOp_49AssignVariableOp0assignvariableop_49_batch_normalization_17_gammaIdentity_49:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_49n
Identity_50IdentityRestoreV2:tensors:50"/device:CPU:0*
T0*
_output_shapes
:2
Identity_50?
AssignVariableOp_50AssignVariableOp/assignvariableop_50_batch_normalization_17_betaIdentity_50:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_50n
Identity_51IdentityRestoreV2:tensors:51"/device:CPU:0*
T0*
_output_shapes
:2
Identity_51?
AssignVariableOp_51AssignVariableOp6assignvariableop_51_batch_normalization_17_moving_meanIdentity_51:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_51n
Identity_52IdentityRestoreV2:tensors:52"/device:CPU:0*
T0*
_output_shapes
:2
Identity_52?
AssignVariableOp_52AssignVariableOp:assignvariableop_52_batch_normalization_17_moving_varianceIdentity_52:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_529
NoOpNoOp"/device:CPU:0*
_output_shapes
 2
NoOp?	
Identity_53Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_52^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2
Identity_53f
Identity_54IdentityIdentity_53:output:0^NoOp_1*
T0*
_output_shapes
: 2
Identity_54?	
NoOp_1NoOp^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_52^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*"
_acd_function_control_output(*
_output_shapes
 2
NoOp_1"#
identity_54Identity_54:output:0*
_input_shapesn
l: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2$
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
AssignVariableOp_24AssignVariableOp_242*
AssignVariableOp_25AssignVariableOp_252*
AssignVariableOp_26AssignVariableOp_262*
AssignVariableOp_27AssignVariableOp_272*
AssignVariableOp_28AssignVariableOp_282*
AssignVariableOp_29AssignVariableOp_292(
AssignVariableOp_3AssignVariableOp_32*
AssignVariableOp_30AssignVariableOp_302*
AssignVariableOp_31AssignVariableOp_312*
AssignVariableOp_32AssignVariableOp_322*
AssignVariableOp_33AssignVariableOp_332*
AssignVariableOp_34AssignVariableOp_342*
AssignVariableOp_35AssignVariableOp_352*
AssignVariableOp_36AssignVariableOp_362*
AssignVariableOp_37AssignVariableOp_372*
AssignVariableOp_38AssignVariableOp_382*
AssignVariableOp_39AssignVariableOp_392(
AssignVariableOp_4AssignVariableOp_42*
AssignVariableOp_40AssignVariableOp_402*
AssignVariableOp_41AssignVariableOp_412*
AssignVariableOp_42AssignVariableOp_422*
AssignVariableOp_43AssignVariableOp_432*
AssignVariableOp_44AssignVariableOp_442*
AssignVariableOp_45AssignVariableOp_452*
AssignVariableOp_46AssignVariableOp_462*
AssignVariableOp_47AssignVariableOp_472*
AssignVariableOp_48AssignVariableOp_482*
AssignVariableOp_49AssignVariableOp_492(
AssignVariableOp_5AssignVariableOp_52*
AssignVariableOp_50AssignVariableOp_502*
AssignVariableOp_51AssignVariableOp_512*
AssignVariableOp_52AssignVariableOp_522(
AssignVariableOp_6AssignVariableOp_62(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_9:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
?
L
0__inference_activation_25_layer_call_fn_22007883

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
GPU 2J 8? *T
fORM
K__inference_activation_25_layer_call_and_return_conditional_losses_220032372
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
T__inference_batch_normalization_14_layer_call_and_return_conditional_losses_22008024

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
?
?
T__inference_batch_normalization_13_layer_call_and_return_conditional_losses_22002500

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
G__inference_conv2d_28_layer_call_and_return_conditional_losses_22003081

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
?
?
G__inference_conv2d_55_layer_call_and_return_conditional_losses_22007363

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
?
I
-__inference_dropout_32_layer_call_fn_22008290

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
GPU 2J 8? *Q
fLRJ
H__inference_dropout_32_layer_call_and_return_conditional_losses_220038882
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
I
-__inference_dropout_28_layer_call_fn_22007845

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
GPU 2J 8? *Q
fLRJ
H__inference_dropout_28_layer_call_and_return_conditional_losses_220032142
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
9__inference_batch_normalization_12_layer_call_fn_22007499

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
GPU 2J 8? *]
fXRV
T__inference_batch_normalization_12_layer_call_and_return_conditional_losses_220023302
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
?
?
T__inference_batch_normalization_12_layer_call_and_return_conditional_losses_22007610

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
?
?
G__inference_conv2d_33_layer_call_and_return_conditional_losses_22003314

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
g
K__inference_activation_23_layer_call_and_return_conditional_losses_22003149

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
?
?
G__inference_conv2d_53_layer_call_and_return_conditional_losses_22007402

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
?
?
T__inference_batch_normalization_14_layer_call_and_return_conditional_losses_22008006

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
L
0__inference_activation_24_layer_call_fn_22007711

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
GPU 2J 8? *T
fORM
K__inference_activation_24_layer_call_and_return_conditional_losses_220031802
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
?
d
H__inference_dropout_29_layer_call_and_return_conditional_losses_22007907

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
T__inference_batch_normalization_12_layer_call_and_return_conditional_losses_22007592

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
f
H__inference_dropout_26_layer_call_and_return_conditional_losses_22003126

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
?
?
G__inference_conv2d_35_layer_call_and_return_conditional_losses_22008318

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
?
L
0__inference_activation_23_layer_call_fn_22007663

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
GPU 2J 8? *T
fORM
K__inference_activation_23_layer_call_and_return_conditional_losses_220031492
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
G__inference_conv2d_29_layer_call_and_return_conditional_losses_22007658

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
input_88
serving_default_input_8:0?????????E
	conv2d_548
StatefulPartitionedCall:0?????????I;
dense_30
StatefulPartitionedCall:1?????????tensorflow/serving/predict:Ҷ
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

	variables
regularization_losses
trainable_variables
	keras_api

signatures
?__call__
?_default_save_signature
+?&call_and_return_all_conditional_losses"
_tf_keras_network
"
_tf_keras_input_layer
?	
layer-0
layer_with_weights-0
layer-1
layer_with_weights-1
layer-2
layer-3
layer-4
layer_with_weights-2
layer-5
layer-6
layer-7
layer_with_weights-3
layer-8
layer-9
layer-10
layer_with_weights-4
layer-11
layer-12
layer_with_weights-5
layer-13
layer-14
layer-15
layer_with_weights-6
layer-16
 layer-17
!layer-18
"layer_with_weights-7
"layer-19
#layer-20
$layer_with_weights-8
$layer-21
%layer-22
&layer-23
'layer_with_weights-9
'layer-24
(layer-25
)layer-26
*layer_with_weights-10
*layer-27
+layer-28
,layer_with_weights-11
,layer-29
-layer-30
.layer-31
/layer_with_weights-12
/layer-32
0layer-33
1layer-34
2layer_with_weights-13
2layer-35
3layer-36
4layer_with_weights-14
4layer-37
5layer-38
6layer-39
7layer_with_weights-15
7layer-40
8layer-41
9layer-42
:layer_with_weights-16
:layer-43
;layer-44
<	variables
=regularization_losses
>trainable_variables
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_network
?

@kernel
Abias
B	variables
Cregularization_losses
Dtrainable_variables
E	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?
F	variables
Gregularization_losses
Htrainable_variables
I	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?

Jkernel
Kbias
L	variables
Mregularization_losses
Ntrainable_variables
O	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?
P	variables
Qregularization_losses
Rtrainable_variables
S	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?
T	variables
Uregularization_losses
Vtrainable_variables
W	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?

Xkernel
Y	variables
Zregularization_losses
[trainable_variables
\	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?

]kernel
^bias
_	variables
`regularization_losses
atrainable_variables
b	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?
c0
d1
e2
f3
g4
h5
i6
j7
k8
l9
m10
n11
o12
p13
q14
r15
s16
t17
u18
v19
w20
x21
y22
z23
{24
|25
}26
~27
28
?29
?30
?31
?32
?33
?34
?35
?36
?37
?38
?39
?40
?41
?42
?43
?44
?45
@46
A47
J48
K49
X50
]51
^52"
trackable_list_wrapper
 "
trackable_list_wrapper
?
c0
d1
e2
f3
i4
j5
k6
l7
m8
n9
q10
r11
s12
t13
u14
v15
y16
z17
{18
|19
}20
~21
?22
?23
?24
?25
?26
?27
?28
?29
?30
?31
?32
?33
@34
A35
J36
K37
X38
]39
^40"
trackable_list_wrapper
?

	variables
?layer_metrics
regularization_losses
 ?layer_regularization_losses
trainable_variables
?metrics
?layers
?non_trainable_variables
?__call__
?_default_save_signature
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
-
?serving_default"
signature_map
"
_tf_keras_input_layer
?

ckernel
dbias
?	variables
?regularization_losses
?trainable_variables
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?
	?axis
	egamma
fbeta
gmoving_mean
hmoving_variance
?	variables
?regularization_losses
?trainable_variables
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?
?	variables
?regularization_losses
?trainable_variables
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?
?	variables
?regularization_losses
?trainable_variables
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?

ikernel
jbias
?	variables
?regularization_losses
?trainable_variables
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?
?	variables
?regularization_losses
?trainable_variables
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?
?	variables
?regularization_losses
?trainable_variables
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?

kkernel
lbias
?	variables
?regularization_losses
?trainable_variables
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
)
?	keras_api"
_tf_keras_layer
?
?	variables
?regularization_losses
?trainable_variables
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?
	?axis
	mgamma
nbeta
omoving_mean
pmoving_variance
?	variables
?regularization_losses
?trainable_variables
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?
?	variables
?regularization_losses
?trainable_variables
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?

qkernel
rbias
?	variables
?regularization_losses
?trainable_variables
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?
?	variables
?regularization_losses
?trainable_variables
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?
?	variables
?regularization_losses
?trainable_variables
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?

skernel
tbias
?	variables
?regularization_losses
?trainable_variables
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
)
?	keras_api"
_tf_keras_layer
?
?	variables
?regularization_losses
?trainable_variables
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?
	?axis
	ugamma
vbeta
wmoving_mean
xmoving_variance
?	variables
?regularization_losses
?trainable_variables
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?
?	variables
?regularization_losses
?trainable_variables
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?

ykernel
zbias
?	variables
?regularization_losses
?trainable_variables
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?
?	variables
?regularization_losses
?trainable_variables
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?
?	variables
?regularization_losses
?trainable_variables
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?

{kernel
|bias
?	variables
?regularization_losses
?trainable_variables
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
)
?	keras_api"
_tf_keras_layer
?
?	variables
?regularization_losses
?trainable_variables
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?
	?axis
	}gamma
~beta
moving_mean
?moving_variance
?	variables
?regularization_losses
?trainable_variables
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?
?	variables
?regularization_losses
?trainable_variables
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?
?kernel
	?bias
?	variables
?regularization_losses
?trainable_variables
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?
?	variables
?regularization_losses
?trainable_variables
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?
?	variables
?regularization_losses
?trainable_variables
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?
?kernel
	?bias
?	variables
?regularization_losses
?trainable_variables
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
)
?	keras_api"
_tf_keras_layer
?
?	variables
?regularization_losses
?trainable_variables
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?
	?axis

?gamma
	?beta
?moving_mean
?moving_variance
?	variables
?regularization_losses
?trainable_variables
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?
?	variables
?regularization_losses
?trainable_variables
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?
?kernel
	?bias
?	variables
?regularization_losses
?trainable_variables
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?
?	variables
?regularization_losses
?trainable_variables
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?
?	variables
?regularization_losses
?trainable_variables
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?
?kernel
	?bias
?	variables
?regularization_losses
?trainable_variables
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
)
?	keras_api"
_tf_keras_layer
?
?	variables
?regularization_losses
?trainable_variables
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?
	?axis

?gamma
	?beta
?moving_mean
?moving_variance
?	variables
?regularization_losses
?trainable_variables
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?
?	variables
?regularization_losses
?trainable_variables
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?
c0
d1
e2
f3
g4
h5
i6
j7
k8
l9
m10
n11
o12
p13
q14
r15
s16
t17
u18
v19
w20
x21
y22
z23
{24
|25
}26
~27
28
?29
?30
?31
?32
?33
?34
?35
?36
?37
?38
?39
?40
?41
?42
?43
?44
?45"
trackable_list_wrapper
 "
trackable_list_wrapper
?
c0
d1
e2
f3
i4
j5
k6
l7
m8
n9
q10
r11
s12
t13
u14
v15
y16
z17
{18
|19
}20
~21
?22
?23
?24
?25
?26
?27
?28
?29
?30
?31
?32
?33"
trackable_list_wrapper
?
<	variables
?layer_metrics
=regularization_losses
 ?layer_regularization_losses
>trainable_variables
?metrics
?layers
?non_trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
+:)?2conv2d_55/kernel
:2conv2d_55/bias
.
@0
A1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
@0
A1"
trackable_list_wrapper
?
B	variables
?layer_metrics
Cregularization_losses
 ?layer_regularization_losses
Dtrainable_variables
?metrics
?layers
?non_trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
F	variables
?layer_metrics
Gregularization_losses
 ?layer_regularization_losses
Htrainable_variables
?metrics
?layers
?non_trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
,:*??2conv2d_53/kernel
:?2conv2d_53/bias
.
J0
K1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
J0
K1"
trackable_list_wrapper
?
L	variables
?layer_metrics
Mregularization_losses
 ?layer_regularization_losses
Ntrainable_variables
?metrics
?layers
?non_trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
P	variables
?layer_metrics
Qregularization_losses
 ?layer_regularization_losses
Rtrainable_variables
?metrics
?layers
?non_trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
T	variables
?layer_metrics
Uregularization_losses
 ?layer_regularization_losses
Vtrainable_variables
?metrics
?layers
?non_trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 :@2dense_3/kernel
'
X0"
trackable_list_wrapper
 "
trackable_list_wrapper
'
X0"
trackable_list_wrapper
?
Y	variables
?layer_metrics
Zregularization_losses
 ?layer_regularization_losses
[trainable_variables
?metrics
?layers
?non_trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
+:)?I2conv2d_54/kernel
:I2conv2d_54/bias
.
]0
^1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
]0
^1"
trackable_list_wrapper
?
_	variables
?layer_metrics
`regularization_losses
 ?layer_regularization_losses
atrainable_variables
?metrics
?layers
?non_trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
+:)?2conv2d_28/kernel
:?2conv2d_28/bias
+:)?2batch_normalization_12/gamma
*:(?2batch_normalization_12/beta
3:1? (2"batch_normalization_12/moving_mean
7:5? (2&batch_normalization_12/moving_variance
,:*??2conv2d_29/kernel
:?2conv2d_29/bias
,:*??2conv2d_30/kernel
:?2conv2d_30/bias
+:)?2batch_normalization_13/gamma
*:(?2batch_normalization_13/beta
3:1? (2"batch_normalization_13/moving_mean
7:5? (2&batch_normalization_13/moving_variance
,:*??2conv2d_31/kernel
:?2conv2d_31/bias
,:*??2conv2d_32/kernel
:?2conv2d_32/bias
+:)?2batch_normalization_14/gamma
*:(?2batch_normalization_14/beta
3:1? (2"batch_normalization_14/moving_mean
7:5? (2&batch_normalization_14/moving_variance
,:*??2conv2d_33/kernel
:?2conv2d_33/bias
,:*??2conv2d_34/kernel
:?2conv2d_34/bias
+:)?2batch_normalization_15/gamma
*:(?2batch_normalization_15/beta
3:1? (2"batch_normalization_15/moving_mean
7:5? (2&batch_normalization_15/moving_variance
,:*??2conv2d_35/kernel
:?2conv2d_35/bias
,:*??2conv2d_36/kernel
:?2conv2d_36/bias
+:)?2batch_normalization_16/gamma
*:(?2batch_normalization_16/beta
3:1? (2"batch_normalization_16/moving_mean
7:5? (2&batch_normalization_16/moving_variance
,:*??2conv2d_37/kernel
:?2conv2d_37/bias
,:*??2conv2d_38/kernel
:?2conv2d_38/bias
+:)?2batch_normalization_17/gamma
*:(?2batch_normalization_17/beta
3:1? (2"batch_normalization_17/moving_mean
7:5? (2&batch_normalization_17/moving_variance
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
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
{
g0
h1
o2
p3
w4
x5
6
?7
?8
?9
?10
?11"
trackable_list_wrapper
.
c0
d1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
c0
d1"
trackable_list_wrapper
?
?	variables
?layer_metrics
?regularization_losses
 ?layer_regularization_losses
?trainable_variables
?metrics
?layers
?non_trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
<
e0
f1
g2
h3"
trackable_list_wrapper
 "
trackable_list_wrapper
.
e0
f1"
trackable_list_wrapper
?
?	variables
?layer_metrics
?regularization_losses
 ?layer_regularization_losses
?trainable_variables
?metrics
?layers
?non_trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?	variables
?layer_metrics
?regularization_losses
 ?layer_regularization_losses
?trainable_variables
?metrics
?layers
?non_trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?	variables
?layer_metrics
?regularization_losses
 ?layer_regularization_losses
?trainable_variables
?metrics
?layers
?non_trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
.
i0
j1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
i0
j1"
trackable_list_wrapper
?
?	variables
?layer_metrics
?regularization_losses
 ?layer_regularization_losses
?trainable_variables
?metrics
?layers
?non_trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?	variables
?layer_metrics
?regularization_losses
 ?layer_regularization_losses
?trainable_variables
?metrics
?layers
?non_trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?	variables
?layer_metrics
?regularization_losses
 ?layer_regularization_losses
?trainable_variables
?metrics
?layers
?non_trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
.
k0
l1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
k0
l1"
trackable_list_wrapper
?
?	variables
?layer_metrics
?regularization_losses
 ?layer_regularization_losses
?trainable_variables
?metrics
?layers
?non_trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
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
?	variables
?layer_metrics
?regularization_losses
 ?layer_regularization_losses
?trainable_variables
?metrics
?layers
?non_trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
<
m0
n1
o2
p3"
trackable_list_wrapper
 "
trackable_list_wrapper
.
m0
n1"
trackable_list_wrapper
?
?	variables
?layer_metrics
?regularization_losses
 ?layer_regularization_losses
?trainable_variables
?metrics
?layers
?non_trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?	variables
?layer_metrics
?regularization_losses
 ?layer_regularization_losses
?trainable_variables
?metrics
?layers
?non_trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
.
q0
r1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
q0
r1"
trackable_list_wrapper
?
?	variables
?layer_metrics
?regularization_losses
 ?layer_regularization_losses
?trainable_variables
?metrics
?layers
?non_trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?	variables
?layer_metrics
?regularization_losses
 ?layer_regularization_losses
?trainable_variables
?metrics
?layers
?non_trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?	variables
?layer_metrics
?regularization_losses
 ?layer_regularization_losses
?trainable_variables
?metrics
?layers
?non_trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
.
s0
t1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
s0
t1"
trackable_list_wrapper
?
?	variables
?layer_metrics
?regularization_losses
 ?layer_regularization_losses
?trainable_variables
?metrics
?layers
?non_trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
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
?	variables
?layer_metrics
?regularization_losses
 ?layer_regularization_losses
?trainable_variables
?metrics
?layers
?non_trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
<
u0
v1
w2
x3"
trackable_list_wrapper
 "
trackable_list_wrapper
.
u0
v1"
trackable_list_wrapper
?
?	variables
?layer_metrics
?regularization_losses
 ?layer_regularization_losses
?trainable_variables
?metrics
?layers
?non_trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?	variables
?layer_metrics
?regularization_losses
 ?layer_regularization_losses
?trainable_variables
?metrics
?layers
?non_trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
.
y0
z1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
y0
z1"
trackable_list_wrapper
?
?	variables
?layer_metrics
?regularization_losses
 ?layer_regularization_losses
?trainable_variables
?metrics
?layers
?non_trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?	variables
?layer_metrics
?regularization_losses
 ?layer_regularization_losses
?trainable_variables
?metrics
?layers
?non_trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?	variables
?layer_metrics
?regularization_losses
 ?layer_regularization_losses
?trainable_variables
?metrics
?layers
?non_trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
.
{0
|1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
{0
|1"
trackable_list_wrapper
?
?	variables
?layer_metrics
?regularization_losses
 ?layer_regularization_losses
?trainable_variables
?metrics
?layers
?non_trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
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
?	variables
?layer_metrics
?regularization_losses
 ?layer_regularization_losses
?trainable_variables
?metrics
?layers
?non_trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
=
}0
~1
2
?3"
trackable_list_wrapper
 "
trackable_list_wrapper
.
}0
~1"
trackable_list_wrapper
?
?	variables
?layer_metrics
?regularization_losses
 ?layer_regularization_losses
?trainable_variables
?metrics
?layers
?non_trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?	variables
?layer_metrics
?regularization_losses
 ?layer_regularization_losses
?trainable_variables
?metrics
?layers
?non_trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
0
?0
?1"
trackable_list_wrapper
 "
trackable_list_wrapper
0
?0
?1"
trackable_list_wrapper
?
?	variables
?layer_metrics
?regularization_losses
 ?layer_regularization_losses
?trainable_variables
?metrics
?layers
?non_trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?	variables
?layer_metrics
?regularization_losses
 ?layer_regularization_losses
?trainable_variables
?metrics
?layers
?non_trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?	variables
?layer_metrics
?regularization_losses
 ?layer_regularization_losses
?trainable_variables
?metrics
?layers
?non_trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
0
?0
?1"
trackable_list_wrapper
 "
trackable_list_wrapper
0
?0
?1"
trackable_list_wrapper
?
?	variables
?layer_metrics
?regularization_losses
 ?layer_regularization_losses
?trainable_variables
?metrics
?layers
?non_trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
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
?	variables
?layer_metrics
?regularization_losses
 ?layer_regularization_losses
?trainable_variables
?metrics
?layers
?non_trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
@
?0
?1
?2
?3"
trackable_list_wrapper
 "
trackable_list_wrapper
0
?0
?1"
trackable_list_wrapper
?
?	variables
?layer_metrics
?regularization_losses
 ?layer_regularization_losses
?trainable_variables
?metrics
?layers
?non_trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?	variables
?layer_metrics
?regularization_losses
 ?layer_regularization_losses
?trainable_variables
?metrics
?layers
?non_trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
0
?0
?1"
trackable_list_wrapper
 "
trackable_list_wrapper
0
?0
?1"
trackable_list_wrapper
?
?	variables
?layer_metrics
?regularization_losses
 ?layer_regularization_losses
?trainable_variables
?metrics
?layers
?non_trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?	variables
?layer_metrics
?regularization_losses
 ?layer_regularization_losses
?trainable_variables
?metrics
?layers
?non_trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?	variables
?layer_metrics
?regularization_losses
 ?layer_regularization_losses
?trainable_variables
?metrics
?layers
?non_trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
0
?0
?1"
trackable_list_wrapper
 "
trackable_list_wrapper
0
?0
?1"
trackable_list_wrapper
?
?	variables
?layer_metrics
?regularization_losses
 ?layer_regularization_losses
?trainable_variables
?metrics
?layers
?non_trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
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
?	variables
?layer_metrics
?regularization_losses
 ?layer_regularization_losses
?trainable_variables
?metrics
?layers
?non_trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
@
?0
?1
?2
?3"
trackable_list_wrapper
 "
trackable_list_wrapper
0
?0
?1"
trackable_list_wrapper
?
?	variables
?layer_metrics
?regularization_losses
 ?layer_regularization_losses
?trainable_variables
?metrics
?layers
?non_trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?	variables
?layer_metrics
?regularization_losses
 ?layer_regularization_losses
?trainable_variables
?metrics
?layers
?non_trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
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
14
15
16
 17
!18
"19
#20
$21
%22
&23
'24
(25
)26
*27
+28
,29
-30
.31
/32
033
134
235
336
437
538
639
740
841
942
:43
;44"
trackable_list_wrapper
{
g0
h1
o2
p3
w4
x5
6
?7
?8
?9
?10
?11"
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
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
.
g0
h1"
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
.
o0
p1"
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
w0
x1"
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
/
0
?1"
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
0
?0
?1"
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
0
?0
?1"
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
?2?
*__inference_model_7_layer_call_fn_22005278
*__inference_model_7_layer_call_fn_22006276
*__inference_model_7_layer_call_fn_22006389
*__inference_model_7_layer_call_fn_22005810?
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
#__inference__wrapped_model_22002308input_8"?
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
E__inference_model_7_layer_call_and_return_conditional_losses_22006600
E__inference_model_7_layer_call_and_return_conditional_losses_22006798
E__inference_model_7_layer_call_and_return_conditional_losses_22005929
E__inference_model_7_layer_call_and_return_conditional_losses_22006048?
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
*__inference_model_4_layer_call_fn_22003664
*__inference_model_4_layer_call_fn_22006895
*__inference_model_4_layer_call_fn_22006992
*__inference_model_4_layer_call_fn_22004700?
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
E__inference_model_4_layer_call_and_return_conditional_losses_22007173
E__inference_model_4_layer_call_and_return_conditional_losses_22007343
E__inference_model_4_layer_call_and_return_conditional_losses_22004840
E__inference_model_4_layer_call_and_return_conditional_losses_22004980?
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
,__inference_conv2d_55_layer_call_fn_22007352?
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
G__inference_conv2d_55_layer_call_and_return_conditional_losses_22007363?
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
-__inference_dropout_51_layer_call_fn_22007368
-__inference_dropout_51_layer_call_fn_22007373?
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
H__inference_dropout_51_layer_call_and_return_conditional_losses_22007378
H__inference_dropout_51_layer_call_and_return_conditional_losses_22007382?
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
,__inference_conv2d_53_layer_call_fn_22007391?
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
G__inference_conv2d_53_layer_call_and_return_conditional_losses_22007402?
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
,__inference_flatten_3_layer_call_fn_22007407?
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
G__inference_flatten_3_layer_call_and_return_conditional_losses_22007413?
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
-__inference_dropout_50_layer_call_fn_22007418
-__inference_dropout_50_layer_call_fn_22007423?
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
H__inference_dropout_50_layer_call_and_return_conditional_losses_22007428
H__inference_dropout_50_layer_call_and_return_conditional_losses_22007432?
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
*__inference_dense_3_layer_call_fn_22007439?
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
E__inference_dense_3_layer_call_and_return_conditional_losses_22007447?
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
,__inference_conv2d_54_layer_call_fn_22007456?
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
G__inference_conv2d_54_layer_call_and_return_conditional_losses_22007467?
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
&__inference_signature_wrapper_22006163input_8"?
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
,__inference_conv2d_28_layer_call_fn_22007476?
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
G__inference_conv2d_28_layer_call_and_return_conditional_losses_22007486?
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
9__inference_batch_normalization_12_layer_call_fn_22007499
9__inference_batch_normalization_12_layer_call_fn_22007512
9__inference_batch_normalization_12_layer_call_fn_22007525
9__inference_batch_normalization_12_layer_call_fn_22007538?
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
T__inference_batch_normalization_12_layer_call_and_return_conditional_losses_22007556
T__inference_batch_normalization_12_layer_call_and_return_conditional_losses_22007574
T__inference_batch_normalization_12_layer_call_and_return_conditional_losses_22007592
T__inference_batch_normalization_12_layer_call_and_return_conditional_losses_22007610?
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
0__inference_activation_22_layer_call_fn_22007615?
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
K__inference_activation_22_layer_call_and_return_conditional_losses_22007620?
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
-__inference_dropout_26_layer_call_fn_22007625
-__inference_dropout_26_layer_call_fn_22007630?
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
H__inference_dropout_26_layer_call_and_return_conditional_losses_22007635
H__inference_dropout_26_layer_call_and_return_conditional_losses_22007639?
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
,__inference_conv2d_29_layer_call_fn_22007648?
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
G__inference_conv2d_29_layer_call_and_return_conditional_losses_22007658?
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
0__inference_activation_23_layer_call_fn_22007663?
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
K__inference_activation_23_layer_call_and_return_conditional_losses_22007668?
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
-__inference_dropout_27_layer_call_fn_22007673
-__inference_dropout_27_layer_call_fn_22007678?
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
H__inference_dropout_27_layer_call_and_return_conditional_losses_22007683
H__inference_dropout_27_layer_call_and_return_conditional_losses_22007687?
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
,__inference_conv2d_30_layer_call_fn_22007696?
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
G__inference_conv2d_30_layer_call_and_return_conditional_losses_22007706?
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
0__inference_activation_24_layer_call_fn_22007711?
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
K__inference_activation_24_layer_call_and_return_conditional_losses_22007716?
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
9__inference_batch_normalization_13_layer_call_fn_22007729
9__inference_batch_normalization_13_layer_call_fn_22007742
9__inference_batch_normalization_13_layer_call_fn_22007755
9__inference_batch_normalization_13_layer_call_fn_22007768?
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
T__inference_batch_normalization_13_layer_call_and_return_conditional_losses_22007786
T__inference_batch_normalization_13_layer_call_and_return_conditional_losses_22007804
T__inference_batch_normalization_13_layer_call_and_return_conditional_losses_22007822
T__inference_batch_normalization_13_layer_call_and_return_conditional_losses_22007840?
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
-__inference_dropout_28_layer_call_fn_22007845
-__inference_dropout_28_layer_call_fn_22007850?
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
H__inference_dropout_28_layer_call_and_return_conditional_losses_22007855
H__inference_dropout_28_layer_call_and_return_conditional_losses_22007859?
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
,__inference_conv2d_31_layer_call_fn_22007868?
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
G__inference_conv2d_31_layer_call_and_return_conditional_losses_22007878?
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
0__inference_activation_25_layer_call_fn_22007883?
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
K__inference_activation_25_layer_call_and_return_conditional_losses_22007888?
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
-__inference_dropout_29_layer_call_fn_22007893
-__inference_dropout_29_layer_call_fn_22007898?
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
H__inference_dropout_29_layer_call_and_return_conditional_losses_22007903
H__inference_dropout_29_layer_call_and_return_conditional_losses_22007907?
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
,__inference_conv2d_32_layer_call_fn_22007916?
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
G__inference_conv2d_32_layer_call_and_return_conditional_losses_22007926?
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
0__inference_activation_26_layer_call_fn_22007931?
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
K__inference_activation_26_layer_call_and_return_conditional_losses_22007936?
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
9__inference_batch_normalization_14_layer_call_fn_22007949
9__inference_batch_normalization_14_layer_call_fn_22007962
9__inference_batch_normalization_14_layer_call_fn_22007975
9__inference_batch_normalization_14_layer_call_fn_22007988?
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
T__inference_batch_normalization_14_layer_call_and_return_conditional_losses_22008006
T__inference_batch_normalization_14_layer_call_and_return_conditional_losses_22008024
T__inference_batch_normalization_14_layer_call_and_return_conditional_losses_22008042
T__inference_batch_normalization_14_layer_call_and_return_conditional_losses_22008060?
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
-__inference_dropout_30_layer_call_fn_22008065
-__inference_dropout_30_layer_call_fn_22008070?
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
H__inference_dropout_30_layer_call_and_return_conditional_losses_22008075
H__inference_dropout_30_layer_call_and_return_conditional_losses_22008079?
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
,__inference_conv2d_33_layer_call_fn_22008088?
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
G__inference_conv2d_33_layer_call_and_return_conditional_losses_22008098?
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
0__inference_activation_27_layer_call_fn_22008103?
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
K__inference_activation_27_layer_call_and_return_conditional_losses_22008108?
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
-__inference_dropout_31_layer_call_fn_22008113
-__inference_dropout_31_layer_call_fn_22008118?
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
H__inference_dropout_31_layer_call_and_return_conditional_losses_22008123
H__inference_dropout_31_layer_call_and_return_conditional_losses_22008127?
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
,__inference_conv2d_34_layer_call_fn_22008136?
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
G__inference_conv2d_34_layer_call_and_return_conditional_losses_22008146?
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
0__inference_activation_28_layer_call_fn_22008151?
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
K__inference_activation_28_layer_call_and_return_conditional_losses_22008156?
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
9__inference_batch_normalization_15_layer_call_fn_22008169
9__inference_batch_normalization_15_layer_call_fn_22008182
9__inference_batch_normalization_15_layer_call_fn_22008195
9__inference_batch_normalization_15_layer_call_fn_22008208?
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
T__inference_batch_normalization_15_layer_call_and_return_conditional_losses_22008226
T__inference_batch_normalization_15_layer_call_and_return_conditional_losses_22008244
T__inference_batch_normalization_15_layer_call_and_return_conditional_losses_22008262
T__inference_batch_normalization_15_layer_call_and_return_conditional_losses_22008280?
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
-__inference_dropout_32_layer_call_fn_22008285
-__inference_dropout_32_layer_call_fn_22008290?
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
H__inference_dropout_32_layer_call_and_return_conditional_losses_22008295
H__inference_dropout_32_layer_call_and_return_conditional_losses_22008299?
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
,__inference_conv2d_35_layer_call_fn_22008308?
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
G__inference_conv2d_35_layer_call_and_return_conditional_losses_22008318?
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
0__inference_activation_29_layer_call_fn_22008323?
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
K__inference_activation_29_layer_call_and_return_conditional_losses_22008328?
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
-__inference_dropout_33_layer_call_fn_22008333
-__inference_dropout_33_layer_call_fn_22008338?
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
H__inference_dropout_33_layer_call_and_return_conditional_losses_22008343
H__inference_dropout_33_layer_call_and_return_conditional_losses_22008347?
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
,__inference_conv2d_36_layer_call_fn_22008356?
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
G__inference_conv2d_36_layer_call_and_return_conditional_losses_22008366?
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
0__inference_activation_30_layer_call_fn_22008371?
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
K__inference_activation_30_layer_call_and_return_conditional_losses_22008376?
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
9__inference_batch_normalization_16_layer_call_fn_22008389
9__inference_batch_normalization_16_layer_call_fn_22008402
9__inference_batch_normalization_16_layer_call_fn_22008415
9__inference_batch_normalization_16_layer_call_fn_22008428?
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
T__inference_batch_normalization_16_layer_call_and_return_conditional_losses_22008446
T__inference_batch_normalization_16_layer_call_and_return_conditional_losses_22008464
T__inference_batch_normalization_16_layer_call_and_return_conditional_losses_22008482
T__inference_batch_normalization_16_layer_call_and_return_conditional_losses_22008500?
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
-__inference_dropout_34_layer_call_fn_22008505
-__inference_dropout_34_layer_call_fn_22008510?
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
H__inference_dropout_34_layer_call_and_return_conditional_losses_22008515
H__inference_dropout_34_layer_call_and_return_conditional_losses_22008519?
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
,__inference_conv2d_37_layer_call_fn_22008528?
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
G__inference_conv2d_37_layer_call_and_return_conditional_losses_22008538?
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
0__inference_activation_31_layer_call_fn_22008543?
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
K__inference_activation_31_layer_call_and_return_conditional_losses_22008548?
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
-__inference_dropout_35_layer_call_fn_22008553
-__inference_dropout_35_layer_call_fn_22008558?
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
H__inference_dropout_35_layer_call_and_return_conditional_losses_22008563
H__inference_dropout_35_layer_call_and_return_conditional_losses_22008567?
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
,__inference_conv2d_38_layer_call_fn_22008576?
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
G__inference_conv2d_38_layer_call_and_return_conditional_losses_22008586?
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
0__inference_activation_32_layer_call_fn_22008591?
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
K__inference_activation_32_layer_call_and_return_conditional_losses_22008596?
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
9__inference_batch_normalization_17_layer_call_fn_22008609
9__inference_batch_normalization_17_layer_call_fn_22008622
9__inference_batch_normalization_17_layer_call_fn_22008635
9__inference_batch_normalization_17_layer_call_fn_22008648?
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
T__inference_batch_normalization_17_layer_call_and_return_conditional_losses_22008666
T__inference_batch_normalization_17_layer_call_and_return_conditional_losses_22008684
T__inference_batch_normalization_17_layer_call_and_return_conditional_losses_22008702
T__inference_batch_normalization_17_layer_call_and_return_conditional_losses_22008720?
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
-__inference_dropout_36_layer_call_fn_22008725
-__inference_dropout_36_layer_call_fn_22008730?
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
H__inference_dropout_36_layer_call_and_return_conditional_losses_22008735
H__inference_dropout_36_layer_call_and_return_conditional_losses_22008739?
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
 ?
#__inference__wrapped_model_22002308?Fcdefghijklmnopqrstuvwxyz{|}~?????????????????@AJK]^X8?5
.?+
)?&
input_8?????????
? "k?h
8
	conv2d_54+?(
	conv2d_54?????????I
,
dense_3!?
dense_3??????????
K__inference_activation_22_layer_call_and_return_conditional_losses_22007620j8?5
.?+
)?&
inputs??????????
? ".?+
$?!
0??????????
? ?
0__inference_activation_22_layer_call_fn_22007615]8?5
.?+
)?&
inputs??????????
? "!????????????
K__inference_activation_23_layer_call_and_return_conditional_losses_22007668j8?5
.?+
)?&
inputs??????????
? ".?+
$?!
0??????????
? ?
0__inference_activation_23_layer_call_fn_22007663]8?5
.?+
)?&
inputs??????????
? "!????????????
K__inference_activation_24_layer_call_and_return_conditional_losses_22007716j8?5
.?+
)?&
inputs??????????
? ".?+
$?!
0??????????
? ?
0__inference_activation_24_layer_call_fn_22007711]8?5
.?+
)?&
inputs??????????
? "!????????????
K__inference_activation_25_layer_call_and_return_conditional_losses_22007888j8?5
.?+
)?&
inputs??????????
? ".?+
$?!
0??????????
? ?
0__inference_activation_25_layer_call_fn_22007883]8?5
.?+
)?&
inputs??????????
? "!????????????
K__inference_activation_26_layer_call_and_return_conditional_losses_22007936j8?5
.?+
)?&
inputs??????????
? ".?+
$?!
0??????????
? ?
0__inference_activation_26_layer_call_fn_22007931]8?5
.?+
)?&
inputs??????????
? "!????????????
K__inference_activation_27_layer_call_and_return_conditional_losses_22008108j8?5
.?+
)?&
inputs??????????
? ".?+
$?!
0??????????
? ?
0__inference_activation_27_layer_call_fn_22008103]8?5
.?+
)?&
inputs??????????
? "!????????????
K__inference_activation_28_layer_call_and_return_conditional_losses_22008156j8?5
.?+
)?&
inputs??????????
? ".?+
$?!
0??????????
? ?
0__inference_activation_28_layer_call_fn_22008151]8?5
.?+
)?&
inputs??????????
? "!????????????
K__inference_activation_29_layer_call_and_return_conditional_losses_22008328j8?5
.?+
)?&
inputs??????????
? ".?+
$?!
0??????????
? ?
0__inference_activation_29_layer_call_fn_22008323]8?5
.?+
)?&
inputs??????????
? "!????????????
K__inference_activation_30_layer_call_and_return_conditional_losses_22008376j8?5
.?+
)?&
inputs??????????
? ".?+
$?!
0??????????
? ?
0__inference_activation_30_layer_call_fn_22008371]8?5
.?+
)?&
inputs??????????
? "!????????????
K__inference_activation_31_layer_call_and_return_conditional_losses_22008548j8?5
.?+
)?&
inputs??????????
? ".?+
$?!
0??????????
? ?
0__inference_activation_31_layer_call_fn_22008543]8?5
.?+
)?&
inputs??????????
? "!????????????
K__inference_activation_32_layer_call_and_return_conditional_losses_22008596j8?5
.?+
)?&
inputs??????????
? ".?+
$?!
0??????????
? ?
0__inference_activation_32_layer_call_fn_22008591]8?5
.?+
)?&
inputs??????????
? "!????????????
T__inference_batch_normalization_12_layer_call_and_return_conditional_losses_22007556?efghN?K
D?A
;?8
inputs,????????????????????????????
p 
? "@?=
6?3
0,????????????????????????????
? ?
T__inference_batch_normalization_12_layer_call_and_return_conditional_losses_22007574?efghN?K
D?A
;?8
inputs,????????????????????????????
p
? "@?=
6?3
0,????????????????????????????
? ?
T__inference_batch_normalization_12_layer_call_and_return_conditional_losses_22007592tefgh<?9
2?/
)?&
inputs??????????
p 
? ".?+
$?!
0??????????
? ?
T__inference_batch_normalization_12_layer_call_and_return_conditional_losses_22007610tefgh<?9
2?/
)?&
inputs??????????
p
? ".?+
$?!
0??????????
? ?
9__inference_batch_normalization_12_layer_call_fn_22007499?efghN?K
D?A
;?8
inputs,????????????????????????????
p 
? "3?0,?????????????????????????????
9__inference_batch_normalization_12_layer_call_fn_22007512?efghN?K
D?A
;?8
inputs,????????????????????????????
p
? "3?0,?????????????????????????????
9__inference_batch_normalization_12_layer_call_fn_22007525gefgh<?9
2?/
)?&
inputs??????????
p 
? "!????????????
9__inference_batch_normalization_12_layer_call_fn_22007538gefgh<?9
2?/
)?&
inputs??????????
p
? "!????????????
T__inference_batch_normalization_13_layer_call_and_return_conditional_losses_22007786?mnopN?K
D?A
;?8
inputs,????????????????????????????
p 
? "@?=
6?3
0,????????????????????????????
? ?
T__inference_batch_normalization_13_layer_call_and_return_conditional_losses_22007804?mnopN?K
D?A
;?8
inputs,????????????????????????????
p
? "@?=
6?3
0,????????????????????????????
? ?
T__inference_batch_normalization_13_layer_call_and_return_conditional_losses_22007822tmnop<?9
2?/
)?&
inputs??????????
p 
? ".?+
$?!
0??????????
? ?
T__inference_batch_normalization_13_layer_call_and_return_conditional_losses_22007840tmnop<?9
2?/
)?&
inputs??????????
p
? ".?+
$?!
0??????????
? ?
9__inference_batch_normalization_13_layer_call_fn_22007729?mnopN?K
D?A
;?8
inputs,????????????????????????????
p 
? "3?0,?????????????????????????????
9__inference_batch_normalization_13_layer_call_fn_22007742?mnopN?K
D?A
;?8
inputs,????????????????????????????
p
? "3?0,?????????????????????????????
9__inference_batch_normalization_13_layer_call_fn_22007755gmnop<?9
2?/
)?&
inputs??????????
p 
? "!????????????
9__inference_batch_normalization_13_layer_call_fn_22007768gmnop<?9
2?/
)?&
inputs??????????
p
? "!????????????
T__inference_batch_normalization_14_layer_call_and_return_conditional_losses_22008006?uvwxN?K
D?A
;?8
inputs,????????????????????????????
p 
? "@?=
6?3
0,????????????????????????????
? ?
T__inference_batch_normalization_14_layer_call_and_return_conditional_losses_22008024?uvwxN?K
D?A
;?8
inputs,????????????????????????????
p
? "@?=
6?3
0,????????????????????????????
? ?
T__inference_batch_normalization_14_layer_call_and_return_conditional_losses_22008042tuvwx<?9
2?/
)?&
inputs??????????
p 
? ".?+
$?!
0??????????
? ?
T__inference_batch_normalization_14_layer_call_and_return_conditional_losses_22008060tuvwx<?9
2?/
)?&
inputs??????????
p
? ".?+
$?!
0??????????
? ?
9__inference_batch_normalization_14_layer_call_fn_22007949?uvwxN?K
D?A
;?8
inputs,????????????????????????????
p 
? "3?0,?????????????????????????????
9__inference_batch_normalization_14_layer_call_fn_22007962?uvwxN?K
D?A
;?8
inputs,????????????????????????????
p
? "3?0,?????????????????????????????
9__inference_batch_normalization_14_layer_call_fn_22007975guvwx<?9
2?/
)?&
inputs??????????
p 
? "!????????????
9__inference_batch_normalization_14_layer_call_fn_22007988guvwx<?9
2?/
)?&
inputs??????????
p
? "!????????????
T__inference_batch_normalization_15_layer_call_and_return_conditional_losses_22008226?}~?N?K
D?A
;?8
inputs,????????????????????????????
p 
? "@?=
6?3
0,????????????????????????????
? ?
T__inference_batch_normalization_15_layer_call_and_return_conditional_losses_22008244?}~?N?K
D?A
;?8
inputs,????????????????????????????
p
? "@?=
6?3
0,????????????????????????????
? ?
T__inference_batch_normalization_15_layer_call_and_return_conditional_losses_22008262u}~?<?9
2?/
)?&
inputs??????????
p 
? ".?+
$?!
0??????????
? ?
T__inference_batch_normalization_15_layer_call_and_return_conditional_losses_22008280u}~?<?9
2?/
)?&
inputs??????????
p
? ".?+
$?!
0??????????
? ?
9__inference_batch_normalization_15_layer_call_fn_22008169?}~?N?K
D?A
;?8
inputs,????????????????????????????
p 
? "3?0,?????????????????????????????
9__inference_batch_normalization_15_layer_call_fn_22008182?}~?N?K
D?A
;?8
inputs,????????????????????????????
p
? "3?0,?????????????????????????????
9__inference_batch_normalization_15_layer_call_fn_22008195h}~?<?9
2?/
)?&
inputs??????????
p 
? "!????????????
9__inference_batch_normalization_15_layer_call_fn_22008208h}~?<?9
2?/
)?&
inputs??????????
p
? "!????????????
T__inference_batch_normalization_16_layer_call_and_return_conditional_losses_22008446?????N?K
D?A
;?8
inputs,????????????????????????????
p 
? "@?=
6?3
0,????????????????????????????
? ?
T__inference_batch_normalization_16_layer_call_and_return_conditional_losses_22008464?????N?K
D?A
;?8
inputs,????????????????????????????
p
? "@?=
6?3
0,????????????????????????????
? ?
T__inference_batch_normalization_16_layer_call_and_return_conditional_losses_22008482x????<?9
2?/
)?&
inputs??????????
p 
? ".?+
$?!
0??????????
? ?
T__inference_batch_normalization_16_layer_call_and_return_conditional_losses_22008500x????<?9
2?/
)?&
inputs??????????
p
? ".?+
$?!
0??????????
? ?
9__inference_batch_normalization_16_layer_call_fn_22008389?????N?K
D?A
;?8
inputs,????????????????????????????
p 
? "3?0,?????????????????????????????
9__inference_batch_normalization_16_layer_call_fn_22008402?????N?K
D?A
;?8
inputs,????????????????????????????
p
? "3?0,?????????????????????????????
9__inference_batch_normalization_16_layer_call_fn_22008415k????<?9
2?/
)?&
inputs??????????
p 
? "!????????????
9__inference_batch_normalization_16_layer_call_fn_22008428k????<?9
2?/
)?&
inputs??????????
p
? "!????????????
T__inference_batch_normalization_17_layer_call_and_return_conditional_losses_22008666?????N?K
D?A
;?8
inputs,????????????????????????????
p 
? "@?=
6?3
0,????????????????????????????
? ?
T__inference_batch_normalization_17_layer_call_and_return_conditional_losses_22008684?????N?K
D?A
;?8
inputs,????????????????????????????
p
? "@?=
6?3
0,????????????????????????????
? ?
T__inference_batch_normalization_17_layer_call_and_return_conditional_losses_22008702x????<?9
2?/
)?&
inputs??????????
p 
? ".?+
$?!
0??????????
? ?
T__inference_batch_normalization_17_layer_call_and_return_conditional_losses_22008720x????<?9
2?/
)?&
inputs??????????
p
? ".?+
$?!
0??????????
? ?
9__inference_batch_normalization_17_layer_call_fn_22008609?????N?K
D?A
;?8
inputs,????????????????????????????
p 
? "3?0,?????????????????????????????
9__inference_batch_normalization_17_layer_call_fn_22008622?????N?K
D?A
;?8
inputs,????????????????????????????
p
? "3?0,?????????????????????????????
9__inference_batch_normalization_17_layer_call_fn_22008635k????<?9
2?/
)?&
inputs??????????
p 
? "!????????????
9__inference_batch_normalization_17_layer_call_fn_22008648k????<?9
2?/
)?&
inputs??????????
p
? "!????????????
G__inference_conv2d_28_layer_call_and_return_conditional_losses_22007486mcd7?4
-?*
(?%
inputs?????????
? ".?+
$?!
0??????????
? ?
,__inference_conv2d_28_layer_call_fn_22007476`cd7?4
-?*
(?%
inputs?????????
? "!????????????
G__inference_conv2d_29_layer_call_and_return_conditional_losses_22007658nij8?5
.?+
)?&
inputs??????????
? ".?+
$?!
0??????????
? ?
,__inference_conv2d_29_layer_call_fn_22007648aij8?5
.?+
)?&
inputs??????????
? "!????????????
G__inference_conv2d_30_layer_call_and_return_conditional_losses_22007706nkl8?5
.?+
)?&
inputs??????????
? ".?+
$?!
0??????????
? ?
,__inference_conv2d_30_layer_call_fn_22007696akl8?5
.?+
)?&
inputs??????????
? "!????????????
G__inference_conv2d_31_layer_call_and_return_conditional_losses_22007878nqr8?5
.?+
)?&
inputs??????????
? ".?+
$?!
0??????????
? ?
,__inference_conv2d_31_layer_call_fn_22007868aqr8?5
.?+
)?&
inputs??????????
? "!????????????
G__inference_conv2d_32_layer_call_and_return_conditional_losses_22007926nst8?5
.?+
)?&
inputs??????????
? ".?+
$?!
0??????????
? ?
,__inference_conv2d_32_layer_call_fn_22007916ast8?5
.?+
)?&
inputs??????????
? "!????????????
G__inference_conv2d_33_layer_call_and_return_conditional_losses_22008098nyz8?5
.?+
)?&
inputs??????????
? ".?+
$?!
0??????????
? ?
,__inference_conv2d_33_layer_call_fn_22008088ayz8?5
.?+
)?&
inputs??????????
? "!????????????
G__inference_conv2d_34_layer_call_and_return_conditional_losses_22008146n{|8?5
.?+
)?&
inputs??????????
? ".?+
$?!
0??????????
? ?
,__inference_conv2d_34_layer_call_fn_22008136a{|8?5
.?+
)?&
inputs??????????
? "!????????????
G__inference_conv2d_35_layer_call_and_return_conditional_losses_22008318p??8?5
.?+
)?&
inputs??????????
? ".?+
$?!
0??????????
? ?
,__inference_conv2d_35_layer_call_fn_22008308c??8?5
.?+
)?&
inputs??????????
? "!????????????
G__inference_conv2d_36_layer_call_and_return_conditional_losses_22008366p??8?5
.?+
)?&
inputs??????????
? ".?+
$?!
0??????????
? ?
,__inference_conv2d_36_layer_call_fn_22008356c??8?5
.?+
)?&
inputs??????????
? "!????????????
G__inference_conv2d_37_layer_call_and_return_conditional_losses_22008538p??8?5
.?+
)?&
inputs??????????
? ".?+
$?!
0??????????
? ?
,__inference_conv2d_37_layer_call_fn_22008528c??8?5
.?+
)?&
inputs??????????
? "!????????????
G__inference_conv2d_38_layer_call_and_return_conditional_losses_22008586p??8?5
.?+
)?&
inputs??????????
? ".?+
$?!
0??????????
? ?
,__inference_conv2d_38_layer_call_fn_22008576c??8?5
.?+
)?&
inputs??????????
? "!????????????
G__inference_conv2d_53_layer_call_and_return_conditional_losses_22007402nJK8?5
.?+
)?&
inputs??????????
? ".?+
$?!
0??????????
? ?
,__inference_conv2d_53_layer_call_fn_22007391aJK8?5
.?+
)?&
inputs??????????
? "!????????????
G__inference_conv2d_54_layer_call_and_return_conditional_losses_22007467m]^8?5
.?+
)?&
inputs??????????
? "-?*
#? 
0?????????I
? ?
,__inference_conv2d_54_layer_call_fn_22007456`]^8?5
.?+
)?&
inputs??????????
? " ??????????I?
G__inference_conv2d_55_layer_call_and_return_conditional_losses_22007363m@A8?5
.?+
)?&
inputs??????????
? "-?*
#? 
0?????????
? ?
,__inference_conv2d_55_layer_call_fn_22007352`@A8?5
.?+
)?&
inputs??????????
? " ???????????
E__inference_dense_3_layer_call_and_return_conditional_losses_22007447[X/?,
%?"
 ?
inputs?????????@
? "%?"
?
0?????????
? |
*__inference_dense_3_layer_call_fn_22007439NX/?,
%?"
 ?
inputs?????????@
? "???????????
H__inference_dropout_26_layer_call_and_return_conditional_losses_22007635n<?9
2?/
)?&
inputs??????????
p 
? ".?+
$?!
0??????????
? ?
H__inference_dropout_26_layer_call_and_return_conditional_losses_22007639n<?9
2?/
)?&
inputs??????????
p
? ".?+
$?!
0??????????
? ?
-__inference_dropout_26_layer_call_fn_22007625a<?9
2?/
)?&
inputs??????????
p 
? "!????????????
-__inference_dropout_26_layer_call_fn_22007630a<?9
2?/
)?&
inputs??????????
p
? "!????????????
H__inference_dropout_27_layer_call_and_return_conditional_losses_22007683n<?9
2?/
)?&
inputs??????????
p 
? ".?+
$?!
0??????????
? ?
H__inference_dropout_27_layer_call_and_return_conditional_losses_22007687n<?9
2?/
)?&
inputs??????????
p
? ".?+
$?!
0??????????
? ?
-__inference_dropout_27_layer_call_fn_22007673a<?9
2?/
)?&
inputs??????????
p 
? "!????????????
-__inference_dropout_27_layer_call_fn_22007678a<?9
2?/
)?&
inputs??????????
p
? "!????????????
H__inference_dropout_28_layer_call_and_return_conditional_losses_22007855n<?9
2?/
)?&
inputs??????????
p 
? ".?+
$?!
0??????????
? ?
H__inference_dropout_28_layer_call_and_return_conditional_losses_22007859n<?9
2?/
)?&
inputs??????????
p
? ".?+
$?!
0??????????
? ?
-__inference_dropout_28_layer_call_fn_22007845a<?9
2?/
)?&
inputs??????????
p 
? "!????????????
-__inference_dropout_28_layer_call_fn_22007850a<?9
2?/
)?&
inputs??????????
p
? "!????????????
H__inference_dropout_29_layer_call_and_return_conditional_losses_22007903n<?9
2?/
)?&
inputs??????????
p 
? ".?+
$?!
0??????????
? ?
H__inference_dropout_29_layer_call_and_return_conditional_losses_22007907n<?9
2?/
)?&
inputs??????????
p
? ".?+
$?!
0??????????
? ?
-__inference_dropout_29_layer_call_fn_22007893a<?9
2?/
)?&
inputs??????????
p 
? "!????????????
-__inference_dropout_29_layer_call_fn_22007898a<?9
2?/
)?&
inputs??????????
p
? "!????????????
H__inference_dropout_30_layer_call_and_return_conditional_losses_22008075n<?9
2?/
)?&
inputs??????????
p 
? ".?+
$?!
0??????????
? ?
H__inference_dropout_30_layer_call_and_return_conditional_losses_22008079n<?9
2?/
)?&
inputs??????????
p
? ".?+
$?!
0??????????
? ?
-__inference_dropout_30_layer_call_fn_22008065a<?9
2?/
)?&
inputs??????????
p 
? "!????????????
-__inference_dropout_30_layer_call_fn_22008070a<?9
2?/
)?&
inputs??????????
p
? "!????????????
H__inference_dropout_31_layer_call_and_return_conditional_losses_22008123n<?9
2?/
)?&
inputs??????????
p 
? ".?+
$?!
0??????????
? ?
H__inference_dropout_31_layer_call_and_return_conditional_losses_22008127n<?9
2?/
)?&
inputs??????????
p
? ".?+
$?!
0??????????
? ?
-__inference_dropout_31_layer_call_fn_22008113a<?9
2?/
)?&
inputs??????????
p 
? "!????????????
-__inference_dropout_31_layer_call_fn_22008118a<?9
2?/
)?&
inputs??????????
p
? "!????????????
H__inference_dropout_32_layer_call_and_return_conditional_losses_22008295n<?9
2?/
)?&
inputs??????????
p 
? ".?+
$?!
0??????????
? ?
H__inference_dropout_32_layer_call_and_return_conditional_losses_22008299n<?9
2?/
)?&
inputs??????????
p
? ".?+
$?!
0??????????
? ?
-__inference_dropout_32_layer_call_fn_22008285a<?9
2?/
)?&
inputs??????????
p 
? "!????????????
-__inference_dropout_32_layer_call_fn_22008290a<?9
2?/
)?&
inputs??????????
p
? "!????????????
H__inference_dropout_33_layer_call_and_return_conditional_losses_22008343n<?9
2?/
)?&
inputs??????????
p 
? ".?+
$?!
0??????????
? ?
H__inference_dropout_33_layer_call_and_return_conditional_losses_22008347n<?9
2?/
)?&
inputs??????????
p
? ".?+
$?!
0??????????
? ?
-__inference_dropout_33_layer_call_fn_22008333a<?9
2?/
)?&
inputs??????????
p 
? "!????????????
-__inference_dropout_33_layer_call_fn_22008338a<?9
2?/
)?&
inputs??????????
p
? "!????????????
H__inference_dropout_34_layer_call_and_return_conditional_losses_22008515n<?9
2?/
)?&
inputs??????????
p 
? ".?+
$?!
0??????????
? ?
H__inference_dropout_34_layer_call_and_return_conditional_losses_22008519n<?9
2?/
)?&
inputs??????????
p
? ".?+
$?!
0??????????
? ?
-__inference_dropout_34_layer_call_fn_22008505a<?9
2?/
)?&
inputs??????????
p 
? "!????????????
-__inference_dropout_34_layer_call_fn_22008510a<?9
2?/
)?&
inputs??????????
p
? "!????????????
H__inference_dropout_35_layer_call_and_return_conditional_losses_22008563n<?9
2?/
)?&
inputs??????????
p 
? ".?+
$?!
0??????????
? ?
H__inference_dropout_35_layer_call_and_return_conditional_losses_22008567n<?9
2?/
)?&
inputs??????????
p
? ".?+
$?!
0??????????
? ?
-__inference_dropout_35_layer_call_fn_22008553a<?9
2?/
)?&
inputs??????????
p 
? "!????????????
-__inference_dropout_35_layer_call_fn_22008558a<?9
2?/
)?&
inputs??????????
p
? "!????????????
H__inference_dropout_36_layer_call_and_return_conditional_losses_22008735n<?9
2?/
)?&
inputs??????????
p 
? ".?+
$?!
0??????????
? ?
H__inference_dropout_36_layer_call_and_return_conditional_losses_22008739n<?9
2?/
)?&
inputs??????????
p
? ".?+
$?!
0??????????
? ?
-__inference_dropout_36_layer_call_fn_22008725a<?9
2?/
)?&
inputs??????????
p 
? "!????????????
-__inference_dropout_36_layer_call_fn_22008730a<?9
2?/
)?&
inputs??????????
p
? "!????????????
H__inference_dropout_50_layer_call_and_return_conditional_losses_22007428n<?9
2?/
)?&
inputs??????????
p 
? ".?+
$?!
0??????????
? ?
H__inference_dropout_50_layer_call_and_return_conditional_losses_22007432n<?9
2?/
)?&
inputs??????????
p
? ".?+
$?!
0??????????
? ?
-__inference_dropout_50_layer_call_fn_22007418a<?9
2?/
)?&
inputs??????????
p 
? "!????????????
-__inference_dropout_50_layer_call_fn_22007423a<?9
2?/
)?&
inputs??????????
p
? "!????????????
H__inference_dropout_51_layer_call_and_return_conditional_losses_22007378l;?8
1?.
(?%
inputs?????????
p 
? "-?*
#? 
0?????????
? ?
H__inference_dropout_51_layer_call_and_return_conditional_losses_22007382l;?8
1?.
(?%
inputs?????????
p
? "-?*
#? 
0?????????
? ?
-__inference_dropout_51_layer_call_fn_22007368_;?8
1?.
(?%
inputs?????????
p 
? " ???????????
-__inference_dropout_51_layer_call_fn_22007373_;?8
1?.
(?%
inputs?????????
p
? " ???????????
G__inference_flatten_3_layer_call_and_return_conditional_losses_22007413`7?4
-?*
(?%
inputs?????????
? "%?"
?
0?????????@
? ?
,__inference_flatten_3_layer_call_fn_22007407S7?4
-?*
(?%
inputs?????????
? "??????????@?
E__inference_model_4_layer_call_and_return_conditional_losses_22004840??cdefghijklmnopqrstuvwxyz{|}~?????????????????@?=
6?3
)?&
input_5?????????
p 

 
? ".?+
$?!
0??????????
? ?
E__inference_model_4_layer_call_and_return_conditional_losses_22004980??cdefghijklmnopqrstuvwxyz{|}~?????????????????@?=
6?3
)?&
input_5?????????
p

 
? ".?+
$?!
0??????????
? ?
E__inference_model_4_layer_call_and_return_conditional_losses_22007173??cdefghijklmnopqrstuvwxyz{|}~???????????????????<
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
E__inference_model_4_layer_call_and_return_conditional_losses_22007343??cdefghijklmnopqrstuvwxyz{|}~???????????????????<
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
*__inference_model_4_layer_call_fn_22003664??cdefghijklmnopqrstuvwxyz{|}~?????????????????@?=
6?3
)?&
input_5?????????
p 

 
? "!????????????
*__inference_model_4_layer_call_fn_22004700??cdefghijklmnopqrstuvwxyz{|}~?????????????????@?=
6?3
)?&
input_5?????????
p

 
? "!????????????
*__inference_model_4_layer_call_fn_22006895??cdefghijklmnopqrstuvwxyz{|}~???????????????????<
5?2
(?%
inputs?????????
p 

 
? "!????????????
*__inference_model_4_layer_call_fn_22006992??cdefghijklmnopqrstuvwxyz{|}~???????????????????<
5?2
(?%
inputs?????????
p

 
? "!????????????
E__inference_model_7_layer_call_and_return_conditional_losses_22005929?Fcdefghijklmnopqrstuvwxyz{|}~?????????????????@AJK]^X@?=
6?3
)?&
input_8?????????
p 

 
? "S?P
I?F
?
0/0?????????
%?"
0/1?????????I
? ?
E__inference_model_7_layer_call_and_return_conditional_losses_22006048?Fcdefghijklmnopqrstuvwxyz{|}~?????????????????@AJK]^X@?=
6?3
)?&
input_8?????????
p

 
? "S?P
I?F
?
0/0?????????
%?"
0/1?????????I
? ?
E__inference_model_7_layer_call_and_return_conditional_losses_22006600?Fcdefghijklmnopqrstuvwxyz{|}~?????????????????@AJK]^X??<
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
? ?
E__inference_model_7_layer_call_and_return_conditional_losses_22006798?Fcdefghijklmnopqrstuvwxyz{|}~?????????????????@AJK]^X??<
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
? ?
*__inference_model_7_layer_call_fn_22005278?Fcdefghijklmnopqrstuvwxyz{|}~?????????????????@AJK]^X@?=
6?3
)?&
input_8?????????
p 

 
? "E?B
?
0?????????
#? 
1?????????I?
*__inference_model_7_layer_call_fn_22005810?Fcdefghijklmnopqrstuvwxyz{|}~?????????????????@AJK]^X@?=
6?3
)?&
input_8?????????
p

 
? "E?B
?
0?????????
#? 
1?????????I?
*__inference_model_7_layer_call_fn_22006276?Fcdefghijklmnopqrstuvwxyz{|}~?????????????????@AJK]^X??<
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
*__inference_model_7_layer_call_fn_22006389?Fcdefghijklmnopqrstuvwxyz{|}~?????????????????@AJK]^X??<
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
1?????????I?
&__inference_signature_wrapper_22006163?Fcdefghijklmnopqrstuvwxyz{|}~?????????????????@AJK]^XC?@
? 
9?6
4
input_8)?&
input_8?????????"k?h
8
	conv2d_54+?(
	conv2d_54?????????I
,
dense_3!?
dense_3?????????