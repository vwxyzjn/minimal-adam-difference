# minimal-adam-layer-norm-bug-repro

This is a minimal repro for a bug in Adam when using LayerNorm for RLHF. 

**TL;DR.** I found that the gradients of the `LayerNorm` parameters seem "off" compared to those in OpenAI's tensorflow codebase. These gradients further cause an larger effect on all the parameters in the model when using the Adam optimizer. However, if we use the `SGD` optimizer, then the undesirable effects on all the parameters are significantly reduced.

## Setup

I record the original query, response, and rewards from https://github.com/openai/lm-human-preferences and save them in https://huggingface.co/datasets/vwxyzjn/lm-human-preferences-debug/tree/main . I also record the weights and gradeint of the first two epochs of training with Adam and GradientDecent optimizers. 

Here is a table summarizing the files and their contents:

| File | Contents |
| --- | --- |
| `query.npy` | The original query (batch of 64) |
| `response.npy` | The original response (batch of 64) |
| `rewards.npy` | The original rewards (batch of 64) |
| `params_and_grads.pkl` | The params and grads of the first two epochs of training with `tf.train.AdamOptimizer(learning_rate=0.00001, epsilon=1e-5)` |
| `params_and_grads_sgd.pkl` | The params and grads of the first two epochs of training with `tf.train.GradientDescentOptimizer(learning_rate=0.00001)` |

## Repro

```
poetry install
poetry run python main.py
```

It should output something like below.

```
working with Adam:
epoch=0, ratio mean=1.0
epoch=0, ratio var=0.0
epoch=0, ratio max=1.0
epoch=0, ratio min=1.0
approxkl 0.0 pg_loss -1.2417634032146907e-08 pg_clipfrac 0.0
epoch=1, ratio mean=1.013449788093567
epoch=1, ratio var=0.006566493306308985
epoch=1, ratio max=1.7290376424789429
epoch=1, ratio min=0.3672836720943451
approxkl 0.0030072603840380907 pg_loss -0.03935058042407036 pg_clipfrac 0.02604166604578495
=============================================
working with SGD:
epoch=0, ratio mean=1.0
epoch=0, ratio var=0.0
epoch=0, ratio max=1.0
epoch=0, ratio min=1.0
approxkl 0.0 pg_loss -1.2417634032146907e-08 pg_clipfrac 0.0
epoch=1, ratio mean=1.0000768899917603
epoch=1, ratio var=1.899234462143795e-06
epoch=1, ratio max=1.006998062133789
epoch=1, ratio min=0.9929711818695068
approxkl 9.512250471743755e-07 pg_loss -0.0003744984569493681 pg_clipfrac 0.0
```

**Observation 1**: The `ratio` of the `SGD` updates have significantly smaller variance than those of the `Adam` updates. Also, this result in a much smaller `approxkl` and `pg_loss` and `clipfrac` for `SGD` than `Adam`.

What's going on? I plot the parameter difference between `main.py` and `lm-human-preferences` before the training, which shows the parameters are identical.

![](diffs/Adam/param_diffs_0.png)

After the first epoch of update, I then plot the gradient difference between `main.py` and `lm-human-preferences`:

![](diffs/Adam/grad_diffs_0.png)

**Observation 2**: Notice how the gradients of the `LayerNorm` parameters are significantly different. This is the root cause of the problem. 

As an example, we print some gradients below. Notice how the first gradient for `h0/ln_1/b:0` is `2.4899011e-05` in OAI's codebase, but `1.8152525e-05` in `main.py`. This is a difference of `0.6746486e-05`, which is quite significant.

In comparison, the gradients of the other layers are much more similar. For example, the first gradient for `h0/attn/c_attn/w:0` is `2.88992633e-05` in OAI's codebase, but `2.88992633e-05` in `main.py`. This is a difference of `0.0`, which is much smaller than the difference in the `LayerNorm` parameters.



```
(Pdb) oname, ograd[:10], name, param.grad.detach().numpy()[:10]
('policy/model/h0/ln_1/b:0', array([ 2.4899011e-05, -1.1588502e-03,  1.7985557e-03,  7.4343453e-03,
       -2.5840786e-03, -3.5906259e-03, -6.6465489e-04,  1.8007826e-03,
       -1.6414827e-03, -6.6386913e-03], dtype=float32), 'transformer.h.0.ln_1.bias', array([ 1.8152525e-05, -1.1576341e-03,  1.7961735e-03,  7.4219629e-03,
       -2.5832835e-03, -3.5855419e-03, -6.7265466e-04,  1.8039590e-03,
       -1.6386800e-03, -6.6277790e-03], dtype=float32))
(Pdb) oname, ograd[:10], name, param.grad.detach().numpy()[:10]
('policy/model/h0/attn/c_attn/w:0', array([[[ 2.88992633e-05, -6.70402551e-06, -1.57610848e-05, ...,
         -1.05873929e-04, -9.40704340e-05,  1.00523466e-04],
        [ 7.87996178e-05, -5.04239551e-07, -8.35032733e-06, ...,
         -4.07231477e-04,  4.93751504e-05, -2.81412737e-04],
        [ 8.21374197e-05, -1.94475469e-05, -1.36382323e-05, ...,
         -1.95847577e-04, -4.09606873e-04,  2.84076581e-04],
        ...,
        [-6.02674390e-06,  4.23970641e-06, -7.39748998e-07, ...,
          1.90844381e-04, -8.59782376e-05, -6.60822116e-05],
        [ 3.50006849e-05, -1.32066066e-06, -3.52823263e-05, ...,
         -1.33828435e-04,  1.01715421e-04,  3.40739585e-04],
        [ 1.05423496e-04, -2.66656862e-05, -4.54609835e-05, ...,
         -4.23200603e-04, -1.64171652e-04,  2.63288792e-04]]],
      dtype=float32), 'transformer.h.0.attn.c_attn.weight', array([[ 2.88245592e-05, -6.68141320e-06, -1.57281083e-05, ...,
        -1.05716754e-04, -9.39845631e-05,  1.00422243e-04],
       [ 7.86117525e-05, -4.96559778e-07, -8.27262829e-06, ...,
        -4.06837964e-04,  4.93464249e-05, -2.81286135e-04],
       [ 8.19143170e-05, -1.94303120e-05, -1.35097052e-05, ...,
        -1.95316272e-04, -4.09374770e-04,  2.83872039e-04],
       ...,
       [-1.33238527e-05, -6.14432452e-08,  7.30297143e-06, ...,
         8.94646073e-05, -1.24311875e-04,  1.05930310e-04],
       [-1.15070456e-04,  1.79788076e-05,  3.04212826e-05, ...,
         6.06048678e-04,  3.23058601e-04, -4.77053138e-04],
       [-5.75132690e-05,  2.93947778e-05,  3.10599062e-05, ...,
         2.26184493e-05,  1.36010476e-05,  9.29407452e-06]], dtype=float32))
```

Then after a gradient pass (i.e., `optimizer.step()`) in `main.py`, I plot the parameter difference between `main.py` and `lm-human-preferences`:

![](diffs/Adam/param_diffs_1.png)

**Observation 3**: Even though the gradient of most layers are similar between pytorch and OAI's codebase, the Adam optimizer causes a significant difference in the parameters. For example, `('transformer.wte.weight', 'policy/model/wte:0')` have near identical gradients as indicated in the last section, but their weights become quite different after a gradient pass.

I wonder if some global regularization / normalization is applied with Adam, which impacts the parameters.

Then I did a same setup but with the SGD optimizer. The gradient difference is of course the same, but the parameter difference is much smaller and only relavent to the `LayerNorm` parameters:

![](diffs/SGD/param_diffs_1.png)


## End-to-end

I then did an end-to-end testing with a toy RLHF codebase (https://gist.github.com/vwxyzjn/010e0f92e057ef1a779028d656ab4705) using SGD and Adam, respetively, with 10 random seeds.


```bash
# the only difference is the optmizer used
diff train_policy.py train_policy_adam.py 
292c292
<     optimizer = optim.SGD(policy.parameters(), lr=args.ppo.lr)
---
>     optimizer = optim.Adam(policy.parameters(), lr=args.ppo.lr)
```

The results are staggering, with SGD converging to a good policy and Adam experiencing significant instability similar to the negative KL divergence issue we have been facing:

![](e2e_results.png)

