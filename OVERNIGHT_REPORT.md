# Overnight run starting 2026-04-25T21:09:09+00:00

## dyck_T128_deltanet  (success)

```
GPU: NVIDIA GeForce RTX 5090

[deltanet]  T=128  max_depth=15  params=1,078,528
  step     tloss     vloss   val_acc   end_acc  q1/q2/q3/q4
   500    0.2535    0.2630     0.894     0.664  1.00/0.97/0.88/0.73
  1000    0.3976    0.1873     0.923     0.723  1.00/0.96/0.92/0.81
  1500    0.0088    0.0020     0.999     0.996  1.00/1.00/1.00/1.00
  2000    0.0003    0.0133     0.996     0.971  1.00/1.00/1.00/0.99
  2500    0.0007    0.0011     1.000     0.994  1.00/1.00/1.00/1.00
  3000    0.0013    0.0003     1.000     1.000  1.00/1.00/1.00/1.00
  3500    0.0000    0.0000     1.000     1.000  1.00/1.00/1.00/1.00
  4000    0.0000    0.0000     1.000     1.000  1.00/1.00/1.00/1.00
  4500    0.0000    0.0000     1.000     1.000  1.00/1.00/1.00/1.00
  5000    0.0000    0.0000     1.000     1.000  1.00/1.00/1.00/1.00

==========================================================================================
arch                                T  end_acc  per_tok  val_loss     params    secs
------------------------------------------------------------------------------------------
deltanet                          128    1.000    1.000    0.0000  1,078,528   117.5
```

## dyck_T128_hybrid_v2  (success)

```
GPU: NVIDIA GeForce RTX 5090

[ortho,deltanet,ortho,deltanet]  T=128  max_depth=15  params=958,656
  step     tloss     vloss   val_acc   end_acc  q1/q2/q3/q4
   500    0.2420    0.3109     0.860     0.740  0.99/0.90/0.80/0.75
  1000    0.0149    0.0293     0.990     0.941  1.00/1.00/1.00/0.97
  1500    0.0041    0.0095     0.997     0.973  1.00/1.00/1.00/0.99
  2000    0.0010    0.0018     1.000     0.998  1.00/1.00/1.00/1.00
  2500    0.0011    0.0003     1.000     1.000  1.00/1.00/1.00/1.00
  3000    0.0002    0.0001     1.000     0.998  1.00/1.00/1.00/1.00
  3500    0.0022    0.0010     1.000     1.000  1.00/1.00/1.00/1.00
  4000    0.0000    0.0000     1.000     1.000  1.00/1.00/1.00/1.00
  4500    0.0000    0.0000     1.000     1.000  1.00/1.00/1.00/1.00
  5000    0.0000    0.0000     1.000     1.000  1.00/1.00/1.00/1.00

==========================================================================================
arch                                T  end_acc  per_tok  val_loss     params    secs
------------------------------------------------------------------------------------------
ortho,deltanet,ortho,deltanet     128    1.000    1.000    0.0000    958,656   182.4
```

## dyck_T512_deltanet  (success)

```
GPU: NVIDIA GeForce RTX 5090

[deltanet]  T=512  max_depth=15  params=1,127,680
  step     tloss     vloss   val_acc   end_acc  q1/q2/q3/q4
   500    0.9160    0.8121     0.672     0.598  0.85/0.65/0.60/0.59
  1000    0.1887    0.2758     0.891     0.865  0.93/0.88/0.88/0.86
  1500    0.1512    0.0901     0.968     0.975  1.00/0.97/0.95/0.96
  2000    0.1542    0.0576     0.978     0.947  0.99/0.99/0.98/0.96
  2500    0.1289    0.2500     0.904     0.828  0.98/0.93/0.87/0.84
  3000    0.0129    0.0039     0.999     1.000  1.00/1.00/1.00/1.00
  3500    0.0012    0.0011     1.000     1.000  1.00/1.00/1.00/1.00
  4000    0.0013    0.0013     1.000     0.996  1.00/1.00/1.00/1.00
  4500    0.0002    0.0002     1.000     1.000  1.00/1.00/1.00/1.00
  5000    0.0006    0.0001     1.000     1.000  1.00/1.00/1.00/1.00

==========================================================================================
arch                                T  end_acc  per_tok  val_loss     params    secs
------------------------------------------------------------------------------------------
deltanet                          512    1.000    1.000    0.0001  1,127,680   362.4
```

## code_135M_deltanet  (success)

```
[transformers] Token indices sequence length is longer than the specified maximum sequence length for this model (23199 > 8192). Running this sequence through the model will result in indexing errors
        VAL  loss=4.4623  ppl=86.69
  2200     28431    4.0551   1.90e-04
  2400     29466    4.0222   1.73e-04
  2600     29348    3.8269   1.57e-04
  2800     29456    3.8742   1.40e-04
  3000     29321    3.8587   1.23e-04
[transformers] Token indices sequence length is longer than the specified maximum sequence length for this model (23199 > 8192). Running this sequence through the model will result in indexing errors
        VAL  loss=4.1776  ppl=65.21
  3200     28638    3.5872   1.08e-04
  3400     29314    3.8436   9.27e-05
  3600     29420    3.5674   7.89e-05
  3800     29258    3.7702   6.66e-05
  4000     29333    3.8292   5.58e-05
[transformers] Token indices sequence length is longer than the specified maximum sequence length for this model (23199 > 8192). Running this sequence through the model will result in indexing errors
        VAL  loss=4.0243  ppl=55.94
  4200     28271    3.5063   4.67e-05
  4400     29352    3.5136   3.95e-05
  4600     29227    3.7103   3.42e-05
  4800     29333    3.6589   3.11e-05
  5000     29207    3.6220   3.00e-05
[transformers] Token indices sequence length is longer than the specified maximum sequence length for this model (23199 > 8192). Running this sequence through the model will result in indexing errors
        VAL  loss=3.9318  ppl=51.00

Done in 701s (140 ms/step avg).
```

## dyck_T512_hybrid_v2  (success)

```
GPU: NVIDIA GeForce RTX 5090

[ortho,deltanet,ortho,deltanet]  T=512  max_depth=15  params=1,007,808
  step     tloss     vloss   val_acc   end_acc  q1/q2/q3/q4
   500    0.9495    2.2504     0.391     0.264  0.64/0.37/0.28/0.27
  1000    0.8260    0.9894     0.631     0.664  0.66/0.56/0.64/0.66
  1500    0.2554    0.2268     0.908     0.904  0.96/0.88/0.89/0.90
  2000    0.1335    0.1614     0.934     0.908  0.99/0.93/0.91/0.90
  2500    0.0538    0.0442     0.984     0.979  1.00/0.99/0.97/0.98
  3000    0.3517    1.0996     0.762     0.643  0.97/0.80/0.66/0.62
  3500    0.0081    0.0132     0.995     0.982  1.00/1.00/1.00/0.99
  4000    0.0029    0.0236     0.992     0.977  1.00/1.00/0.99/0.98
  4500    0.0018    0.0021     0.999     1.000  1.00/1.00/1.00/1.00
  5000    0.0001    0.0031     0.999     1.000  1.00/1.00/1.00/1.00

==========================================================================================
arch                                T  end_acc  per_tok  val_loss     params    secs
------------------------------------------------------------------------------------------
ortho,deltanet,ortho,deltanet     512    1.000    1.000    0.0010  1,007,808   520.6
```

## code_135M_hybrid_50_50  (success)

```
[transformers] Token indices sequence length is longer than the specified maximum sequence length for this model (23199 > 8192). Running this sequence through the model will result in indexing errors
        VAL  loss=4.6484  ppl=104.42
  2200     16113    4.2648   1.90e-04
  2400     16363    4.2414   1.73e-04
  2600     16449    4.0471   1.57e-04
  2800     16463    4.0830   1.40e-04
  3000     16493    4.0707   1.23e-04
[transformers] Token indices sequence length is longer than the specified maximum sequence length for this model (23199 > 8192). Running this sequence through the model will result in indexing errors
        VAL  loss=4.3772  ppl=79.62
  3200     16081    3.8179   1.08e-04
  3400     16472    4.0750   9.27e-05
  3600     16472    3.7727   7.89e-05
  3800     16428    4.0076   6.66e-05
  4000     16483    4.0601   5.58e-05
[transformers] Token indices sequence length is longer than the specified maximum sequence length for this model (23199 > 8192). Running this sequence through the model will result in indexing errors
        VAL  loss=4.2170  ppl=67.83
  4200     16158    3.7610   4.67e-05
  4400     16455    3.7549   3.95e-05
  4600     16379    3.9426   3.42e-05
  4800     16405    3.8782   3.11e-05
  5000     16399    3.8471   3.00e-05
[transformers] Token indices sequence length is longer than the specified maximum sequence length for this model (23199 > 8192). Running this sequence through the model will result in indexing errors
        VAL  loss=4.1292  ppl=62.13

Done in 1257s (251 ms/step avg).
```

## code_135M_hybrid_25_75  (success)

```
[transformers] Token indices sequence length is longer than the specified maximum sequence length for this model (23199 > 8192). Running this sequence through the model will result in indexing errors
        VAL  loss=4.5286  ppl=92.63
  2200     22481    4.1331   1.90e-04
  2400     23017    4.0929   1.73e-04
  2600     23115    3.9040   1.57e-04
  2800     22763    3.9451   1.40e-04
  3000     22944    3.9210   1.23e-04
[transformers] Token indices sequence length is longer than the specified maximum sequence length for this model (23199 > 8192). Running this sequence through the model will result in indexing errors
        VAL  loss=4.2536  ppl=70.36
  3200     22453    3.6690   1.08e-04
  3400     23016    3.9197   9.27e-05
  3600     23022    3.6373   7.89e-05
  3800     22995    3.8528   6.66e-05
  4000     23057    3.9050   5.58e-05
[transformers] Token indices sequence length is longer than the specified maximum sequence length for this model (23199 > 8192). Running this sequence through the model will result in indexing errors
        VAL  loss=4.0894  ppl=59.71
  4200     22478    3.6028   4.67e-05
  4400     22959    3.6008   3.95e-05
  4600     23053    3.7925   3.42e-05
  4800     23069    3.7381   3.11e-05
  5000     22930    3.7007   3.00e-05
[transformers] Token indices sequence length is longer than the specified maximum sequence length for this model (23199 > 8192). Running this sequence through the model will result in indexing errors
        VAL  loss=4.0024  ppl=54.73

Done in 895s (179 ms/step avg).
```

## code_135M_hybrid_75_25  (success)

```
[transformers] Token indices sequence length is longer than the specified maximum sequence length for this model (23199 > 8192). Running this sequence through the model will result in indexing errors
        VAL  loss=4.8337  ppl=125.68
  2200     12695    4.5007   1.90e-04
  2400     12862    4.4850   1.73e-04
  2600     12860    4.2877   1.57e-04
  2800     12860    4.3154   1.40e-04
  3000     12899    4.3366   1.23e-04
[transformers] Token indices sequence length is longer than the specified maximum sequence length for this model (23199 > 8192). Running this sequence through the model will result in indexing errors
        VAL  loss=4.5989  ppl=99.38
  3200     12656    4.0742   1.08e-04
  3400     12937    4.3239   9.27e-05
  3600     12903    4.0082   7.89e-05
  3800     12873    4.2585   6.66e-05
  4000     12873    4.3005   5.58e-05
[transformers] Token indices sequence length is longer than the specified maximum sequence length for this model (23199 > 8192). Running this sequence through the model will result in indexing errors
        VAL  loss=4.4577  ppl=86.29
  4200     12671    4.0273   4.67e-05
  4400     12887    4.0178   3.95e-05
  4600     12812    4.1827   3.42e-05
  4800     12895    4.1148   3.11e-05
  5000     12934    4.0863   3.00e-05
[transformers] Token indices sequence length is longer than the specified maximum sequence length for this model (23199 > 8192). Running this sequence through the model will result in indexing errors
        VAL  loss=4.3662  ppl=78.74

Done in 1597s (319 ms/step avg).
```


Total wall-clock: 4489s (74 min)
